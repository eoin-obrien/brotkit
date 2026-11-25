"""One cold build in a fresh process, with stage timings (throwaway; run by cold.py).

Stages (seconds):
  import_numba   import numba_cuda_mlir.cuda (one-time per process)
  import_core    import cuda.core + Device(0).set_current() (one-time: context creation)
  exec           exec the generated .py (module body, decorators; no compile)
  frontend       Numba bytecode -> IR, typing, lowering to MLIR (mlir_compiler.compile_mlir)
  optimize       MLIR pipeline (mlir_optimization.optimize), excluding PTX generation
  ptx            MLIR -> LLVM 7 / NVVM IR -> libnvvm -> PTX
  compile_other  rest of compile_ptx (dispatcher setup, target resolution)
  link           nvJitLink PTX -> cubin + verbose log (ptxas inside), compute cache bypassed
  load           cuda.core ObjectCode.from_cubin + get_kernel (module load)
  first_launch   one 64x64 launch + sync
Timers patch private numba-cuda-mlir functions for measurement only.

usage: python cold_child.py <tier> <kernel> <gen_dir>  -> one JSON line
"""

from __future__ import annotations

import time

T0 = time.perf_counter()

import importlib.util  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

t = {}
t_ = time.perf_counter()
from numba_cuda_mlir import cuda  # noqa: E402

t["import_numba"] = time.perf_counter() - t_
t_ = time.perf_counter()
from cuda.core import Device, LaunchConfig, ObjectCode, launch  # noqa: E402

dev = Device(0)
dev.set_current()
stream = dev.create_stream()
t["import_core"] = time.perf_counter() - t_

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_s2  # noqa: E402
import s2lib  # noqa: E402
from numba_cuda_mlir import mlir_compiler, mlir_optimization  # noqa: E402

acc = {"frontend": 0.0, "optimize": 0.0, "ptx": 0.0}
_depth = {"n": 0}


_active: set[str] = set()


def timed(mod, name, bucket, subtract_ptx=False):
    """Time the outermost call only: compile_mlir re-enters itself for device callees."""
    orig = getattr(mod, name)

    def wrap(*a, **k):
        if bucket in _active:
            return orig(*a, **k)
        _active.add(bucket)
        p0 = acc["ptx"]
        s = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            dt = time.perf_counter() - s
            if subtract_ptx:
                dt -= acc["ptx"] - p0
            acc[bucket] += dt
            _active.discard(bucket)

    setattr(mod, name, wrap)


timed(mlir_compiler, "compile_mlir", "frontend")
timed(mlir_optimization, "optimize", "optimize", subtract_ptx=True)
for nm in ("_call_llvm70_capi", "_compile_to_ptx"):
    if hasattr(mlir_optimization, nm):
        timed(mlir_optimization, nm, "ptx")


def _warmup(cc) -> None:
    """A trivial kernel through the same path, to separate one-time compiler setup."""
    from numba_cuda_mlir import types

    def tiny(p, n):
        a = cuda.carray(p, (n,), np.float32)
        i = cuda.grid(1)
        if i < n:
            a[i] = np.float32(1.0)

    ptx, _ = cuda.compile_ptx(tiny, (types.CPointer(types.float32), types.int64), cc=cc)
    s2lib.link(str(ptx).rstrip("\x00"), cc)


def main() -> None:
    tier, kern, gdir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    cc0 = dev.compute_capability
    t_ = time.perf_counter()
    _warmup((cc0.major, cc0.minor))
    t["compiler_init"] = time.perf_counter() - t_
    for b in acc:
        acc[b] = 0.0
    path = gen_s2.write(tier, kern, "ptr64", out_dir=gdir)
    t_ = time.perf_counter()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec
    assert spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    k = getattr(mod, gen_s2.entry_name(tier, kern))
    t["exec"] = time.perf_counter() - t_

    cc = dev.compute_capability
    cc = (cc.major, cc.minor)
    sig = gen_s2.signature(tier, "ptr64")
    t_ = time.perf_counter()
    ptx, _ = cuda.compile_ptx(k.py_func, sig, cc=cc)
    ptx = str(ptx).rstrip("\x00")
    tot = time.perf_counter() - t_
    t.update(acc)
    t["compile_other"] = tot - sum(acc.values())

    t_ = time.perf_counter()
    cubin, _log = s2lib.link(ptx, cc)
    t["link"] = time.perf_counter() - t_

    entry = re.search(r"\.visible \.entry (\S+)\(", ptx).group(1)
    t_ = time.perf_counter()
    kfn = ObjectCode.from_cubin(cubin).get_kernel(entry)
    t["load"] = time.perf_counter() - t_

    h = w = 64
    bf = dev.memory_resource.allocate(gen_s2.NLF * h * w * 4, stream=stream)
    bu = dev.memory_resource.allocate(gen_s2.NLU * h * w * 4, stream=stream)
    word, nw, _ = gen_s2.TIERS[tier]
    ft = getattr(np, word)
    args = (
        *[ft(0.0)] * (2 * nw),
        ft(1e-3),
        np.int32(50),
        np.int32(0),
        bf,
        bu,
        np.int64(h),
        np.int64(w),
        np.int64(w),
    )
    t_ = time.perf_counter()
    launch(stream, LaunchConfig(grid=(2, 8), block=(32, 8)), kfn, *args)
    stream.sync()
    t["first_launch"] = time.perf_counter() - t_
    t["total_process"] = time.perf_counter() - T0
    t["per_kernel_cold"] = sum(
        t[s]
        for s in (
            "exec",
            "frontend",
            "optimize",
            "ptx",
            "compile_other",
            "link",
            "load",
            "first_launch",
        )
    )
    print(
        json.dumps(
            dict(
                tier=tier,
                kernel=kern,
                ptx_bytes=len(ptx),
                cubin_bytes=len(cubin),
                src_bytes=len(path.read_text()),
                **{k_: round(v, 4) for k_, v in t.items()},
            )
        )
    )


if __name__ == "__main__":
    main()

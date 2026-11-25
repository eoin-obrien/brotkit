"""One warm start in a fresh process (throwaway; run by warm.py).

mech=b: brotkit KernelCache + CudaCacheBackend (cubin in brotkit's store, cuda.core load).
mech=a: Numba cache=True, generated .py in its own per-key directory, dispatcher launch.

Times (s):
  one_time      imports + device/context/stream (b: brotkit.backends.cuda, cuda.core;
                a: numba_cuda_mlir + context)
  spec          read the generated source and build the KernelSpec (b) / exec the module (a)
  get           b: KernelCache.get (key, read + verify entry, ObjectCode.from_cubin,
                   get_kernel, ResourceUsage from the manifest)
                a: dispatcher.compile(sig) served from Numba's cache
  first_launch  one 64x64 launch + sync
  render        (render=1) full 3840x2160, max_iter 1000 render of the S4 view + sync
usage: python warm_child.py <mech> <tier> <kernel> <root> <render 0|1>
"""

from __future__ import annotations

import time

T0 = time.perf_counter()

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
mech, tier, kern, root, do_render = sys.argv[1:6]
t: dict[str, float] = {}

W, H, MAX_ITER = 3840, 2160, 1000
CENTRE, WIDTH = (-0.743643887037151, 0.131825904205330), 0.05


def view_args(word: str, nw: int):
    from fractions import Fraction

    dt = np.dtype(word).type
    out = []
    for x in CENTRE:
        r = Fraction(x)
        for _ in range(nw):
            w = dt(float(r))
            out.append(w)
            r -= Fraction(float(w))
    return (*out, dt(WIDTH / W))


def main() -> None:
    import gen_s2

    word, nw, _ = gen_s2.TIERS[tier]
    if mech == "b":
        t_ = time.perf_counter()
        from brotkit.backends.cuda import CudaCacheBackend
        from brotkit.kernels.cache import KernelCache, KernelStore

        be = CudaCacheBackend()
        t["one_time"] = time.perf_counter() - t_
        t_ = time.perf_counter()
        src = (pathlib.Path(root) / "src" / f"{tier}_{kern}.py").read_text()
        import warm

        spec = be.spec(source=src, entry=gen_s2.entry_name(tier, kern), signature=warm.sig(tier))
        cache = KernelCache(be, KernelStore(pathlib.Path(root) / "store"))
        t["spec"] = time.perf_counter() - t_
        t_ = time.perf_counter()
        k = cache.get(spec)
        t["get"] = time.perf_counter() - t_
        assert cache.stats.builds == 0, "not a warm hit"
        assert cache.stats.disk_hits == 1, "not a warm hit"
        regs = k.resources.registers
        dev, stream = be.device, be.stream

        def alloc(n):
            return dev.memory_resource.allocate(n, stream=stream)

        def launch(grid, block, args, bf, bu, h, w):
            k.launch(grid, block, *args, bf, bu, h, w, w)

        def sync():
            stream.sync()
    else:
        t_ = time.perf_counter()
        from numba_cuda_mlir import cuda

        cuda.current_context()
        t["one_time"] = time.perf_counter() - t_
        t_ = time.perf_counter()
        import importlib.util

        p = pathlib.Path(root) / "a" / f"{tier}_{kern}" / "kmod.py"
        spec_ = importlib.util.spec_from_file_location("kmod", p)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        disp = cuda.jit(cache=True)(getattr(mod, gen_s2.entry_name(tier, kern)).py_func)
        t["spec"] = time.perf_counter() - t_
        t_ = time.perf_counter()
        disp.compile(gen_s2.signature(tier, "ptr64"))
        t["get"] = time.perf_counter() - t_
        assert sum(disp._cache_hits.values()) == 1, "not a Numba cache hit"
        regs = None

        class Buf:
            def __init__(self, n):
                self.a = cuda.device_array(n // 4, np.uint32)

        def alloc(n):
            return Buf(n)

        def launch(grid, block, args, bf, bu, h, w):
            disp[grid, block](
                *args,
                bf.a.__cuda_array_interface__["data"][0],
                bu.a.__cuda_array_interface__["data"][0],
                np.int64(h),
                np.int64(w),
                np.int64(w),
            )

        def sync():
            cuda.synchronize()

    args0 = view_args(word, nw)
    h = w = 64
    bf, bu = alloc(gen_s2.NLF * h * w * 4), alloc(gen_s2.NLU * h * w * 4)
    t_ = time.perf_counter()
    launch((2, 8), (32, 8), (*args0, np.int32(50), np.int32(0)), bf, bu, h, w)
    sync()
    t["first_launch"] = time.perf_counter() - t_
    if do_render == "1":
        bf, bu = alloc(gen_s2.NLF * H * W * 4), alloc(gen_s2.NLU * H * W * 4)
        t_ = time.perf_counter()
        band = (
            2160
            if word == "float32" and nw == 1
            else (540 if nw <= 2 and word == "float32" else 72)
        )
        for r0 in range(0, H, band):
            launch(
                (W // 32, (band + 7) // 8),
                (32, 8),
                (*args0, np.int32(MAX_ITER), np.int32(r0)),
                bf,
                bu,
                H,
                W,
            )
            sync()
        t["render"] = time.perf_counter() - t_
    t["in_process_total"] = time.perf_counter() - T0
    print(
        json.dumps(
            {
                "mech": mech,
                "tier": tier,
                "kernel": kern,
                "registers": regs,
                "numba_imported": "numba_cuda_mlir" in sys.modules,
                **{k_: round(v, 5) for k_, v in t.items()},
            }
        )
    )


if __name__ == "__main__":
    main()

"""Q1: what compiles, and what reports resources, with no GPU and no driver?

Runs every step in its own try block and records OK / FAIL with the exception text. Run it
(a) in the no-GPU container (see README: Dockerfile.nogpu), and (b) on the GPU host for
comparison. Never launches a kernel.

usage: python q1_nogpu.py <label>          -> results/q1_<label>.json
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULTS: dict[str, dict] = {}
TMP = pathlib.Path(tempfile.mkdtemp(prefix="s4q1_"))


def step(name: str):
    def deco(fn):
        try:
            out = fn()
            RESULTS[name] = {"ok": True, "detail": out}
            print(f"OK    {name}: {str(out)[:160]}", flush=True)
        except BaseException as e:
            msg = f"{type(e).__name__}: {e}".strip()
            RESULTS[name] = {"ok": False, "error": msg[:2000], "tb": traceback.format_exc()[-3000:]}
            print(f"FAIL  {name}: {msg[:300]}", flush=True)
        return fn

    return deco


def nvbin(tool: str) -> str:
    import nvidia  # namespace package from the cu13 wheels

    for root in nvidia.__path__:
        p = pathlib.Path(root) / "cu13" / "bin" / tool
        if p.exists():
            return str(p)
    raise FileNotFoundError(tool)


@step("env")
def _env():
    def can_load(name):
        try:
            ctypes.CDLL(name)
            return True
        except OSError as e:
            return f"no ({e})"

    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "/dev/dxg": os.path.exists("/dev/dxg"),
        "/dev/nvidia0": os.path.exists("/dev/nvidia0"),
        "/usr/lib/wsl/lib": os.path.exists("/usr/lib/wsl/lib"),
        "libcuda.so.1": can_load("libcuda.so.1"),
        "nvidia-smi": shutil.which("nvidia-smi"),
        "python": sys.version.split()[0],
    }


@step("import numba_cuda_mlir.cuda")
def _imp():
    import numba_cuda_mlir
    from numba_cuda_mlir import cuda  # noqa: F401

    return numba_cuda_mlir.__version__


if os.environ.get("S4_PATCH_HOSTCC"):
    # DIAGNOSTIC ONLY (never shipped): stub the host-cc lookup that resolve_gpu_target makes
    # unconditionally, to see whether anything else in the compile path needs a driver.
    @step("PATCH numba_cuda_mlir.tools.get_gpu_compute_capability -> (0, 0) when no driver")
    def _patch():
        from numba_cuda_mlir import tools

        orig = tools.get_gpu_compute_capability

        def stub(fmt=tuple):
            try:
                return orig(fmt)
            except Exception:
                return (0, 0) if fmt is tuple else "sm_00"

        tools.get_gpu_compute_capability = stub
        return "patched"


import gen  # noqa: E402

CASES = [("f32", "A"), ("dd", "C"), ("qf", "C")]
PTX: dict[tuple[str, str, int], str] = {}


def _ccs():
    return [(8, 6), (10, 0), (12, 0)]


for tier, kern in CASES:
    for cc in _ccs():

        @step(f"compile_ptx {tier}/{kern} cc={cc[0]}{cc[1]}")
        def _cp(tier=tier, kern=kern, cc=cc):
            from numba_cuda_mlir import cuda

            k = gen.load(tier, kern)
            ptx, resty = cuda.compile_ptx(k.py_func, gen.signature(tier), cc=cc)
            ptx = ptx.rstrip("\x00")
            PTX[(tier, kern, cc[0] * 10 + cc[1])] = ptx
            target = next(line for line in ptx.splitlines() if line.startswith(".target"))
            return f"{len(ptx)} chars, {target}, restype={resty}"


@step("compile_ptx f32/A cc=None (current device)")
def _cp_none():
    from numba_cuda_mlir import cuda

    k = gen.load("f32", "A")
    ptx, _ = cuda.compile_ptx(k.py_func, gen.signature("f32"))
    return f"{len(ptx)} chars"


@step("compile_ptx dd/C cc=86 launch_bounds=(256, 2)")
def _cp_lb():
    from numba_cuda_mlir import cuda

    k = gen.load("dd", "C")
    ptx, _ = cuda.compile_ptx(k.py_func, gen.signature("dd"), cc=(8, 6), launch_bounds=(256, 2))
    return [
        line.strip() for line in ptx.splitlines() if ".maxntid" in line or ".minnctapersm" in line
    ]


for out in ("ltoir", "cubin"):

    @step(f"cuda.compile output={out} f32/A cc=86")
    def _cc(out=out):
        from numba_cuda_mlir import cuda

        k = gen.load("f32", "A")
        res = cuda.compile(k.py_func, gen.signature("f32"), device=False, cc=(8, 6), output=out)
        code = res[0]
        return f"{type(code).__name__} {len(code)} bytes"


def _ptxas(tier, kern, cc, extra=()):
    ptx = PTX[(tier, kern, cc)]
    src = TMP / f"{tier}_{kern}_{cc}.ptx"
    cub = TMP / f"{tier}_{kern}_{cc}{'_'.join(extra)}.cubin"
    src.write_text(ptx)
    r = subprocess.run(
        [nvbin("ptxas"), f"-arch=sm_{cc}", "-v", *extra, str(src), "-o", str(cub)],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise RuntimeError(r.stderr[-500:])
    return cub, [
        ln.split("info    : ")[-1] for ln in r.stderr.splitlines() if "Used" in ln or "spill" in ln
    ]


for tier, kern in CASES:
    for cc in (86, 100, 120):

        @step(f"ptxas(wheel) -v {tier}/{kern} sm_{cc}")
        def _px(tier=tier, kern=kern, cc=cc):
            return _ptxas(tier, kern, cc)[1]


@step("ptxas(wheel) -v -maxrregcount=32 dd/C sm_86")
def _px_cap():
    return _ptxas("dd", "C", 86, ("-maxrregcount=32",))[1]


def _nvjitlink(ptx: str, cc: int, opts=()):
    from cuda.bindings import nvjitlink

    o = [f"-arch=sm_{cc}", "-verbose", *opts]
    h = nvjitlink.create(len(o), o)
    try:
        data = ptx.encode()
        nvjitlink.add_data(h, nvjitlink.InputType.PTX, data, len(data), "k.ptx")
        nvjitlink.complete(h)
        n = nvjitlink.get_linked_cubin_size(h)
        buf = bytearray(n)
        nvjitlink.get_linked_cubin(h, buf)
        ln = nvjitlink.get_info_log_size(h)
        log = bytearray(ln)
        nvjitlink.get_info_log(h, log)
        return bytes(buf), log.decode(errors="replace").rstrip("\x00")
    finally:
        nvjitlink.destroy(h)


for tier, kern in CASES:

    @step(f"nvJitLink -verbose {tier}/{kern} sm_86")
    def _nj(tier=tier, kern=kern):
        cub, log = _nvjitlink(PTX[(tier, kern, 86)], 86)
        (TMP / f"nj_{tier}_{kern}.cubin").write_bytes(cub)
        return {"cubin_bytes": len(cub), "info_log": log.strip().splitlines()}


@step("nvJitLink -verbose -maxrregcount=32 dd/C sm_86")
def _nj_cap():
    _, log = _nvjitlink(PTX[("dd", "C", 86)], 86, ("-maxrregcount=32",))
    return log.strip().splitlines()


@step("cuobjdump(wheel) --dump-resource-usage dd/C sm_86 cubin")
def _cuobj():
    cub, _ = _ptxas("dd", "C", 86)
    r = subprocess.run(
        [nvbin("cuobjdump"), "--dump-resource-usage", str(cub)], capture_output=True, text=True
    )
    if r.returncode:
        raise RuntimeError(r.stderr)
    return [ln.strip() for ln in r.stdout.splitlines() if "REG" in ln or "Resource" in ln]


@step("nvdisasm(wheel) qf/C sm_120 cubin")
def _nvdis():
    cub, _ = _ptxas("qf", "C", 120)
    r = subprocess.run([nvbin("nvdisasm"), "-c", str(cub)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return f"{len(r.stdout.splitlines())} SASS lines"


@step("cuda.jit dispatcher .compile(sig) f32/A")
def _disp():
    from numba_cuda_mlir import cuda

    k = cuda.jit(gen.load("f32", "A").py_func)
    k.compile(gen.signature("f32"))
    return "compiled"


@step("cuda.jit(chip='sm_86') dispatcher .compile + inspect_asm f32/A")
def _disp_chip():
    from numba_cuda_mlir import cuda

    k = cuda.jit(chip="sm_86")(gen.load("f32", "A").py_func)
    k.compile(gen.signature("f32"))
    return f"{len(k.inspect_asm(gen.signature('f32')))} chars"


@step("dispatcher.get_regs_per_thread() f32/A")
def _regs():
    from numba_cuda_mlir import cuda

    k = cuda.jit(gen.load("f32", "A").py_func)
    k.compile(gen.signature("f32"))
    return k.get_regs_per_thread(gen.signature("f32"))


@step("brotkit.backends.cuda.resource_usage f32/A + dd/C sm_86/100/120")
def _bk():
    from brotkit.backends.cuda import resource_usage

    out = {}
    for tier, kern in (("f32", "A"), ("dd", "C")):
        k = gen.load(tier, kern)
        for cc in ((8, 6), (10, 0), (12, 0)):
            out[f"{tier}/{kern}/{cc}"] = str(resource_usage(k, gen.signature(tier), cc=cc))
    return out


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "host"
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"q1_{label}.json").write_text(json.dumps(RESULTS, indent=1))
    print(f"wrote results/q1_{label}.json")

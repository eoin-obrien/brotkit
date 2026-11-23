"""Spike S1 smoke test (throwaway): versions, device facts, two tiny kernels, cache=True.

Run twice: the second run should load `cached_kernel` from the on-disk cache.
Set NUMBA_DEBUG_CACHE=1 to see where numba-cuda reads and writes cache files.
"""

import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, distributions, version

MIN_FREE_MIB = 2048  # stop and ask a human if less VRAM is free than this

t0 = time.perf_counter()
free = int(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    ).split()[0]
)
print(f"free VRAM: {free} MiB")
if free < MIN_FREE_MIB:
    sys.exit(f"ABORT: only {free} MiB free (< {MIN_FREE_MIB}); GPU looks busy")

import numpy as np  # noqa: E402

BACKEND = os.environ.get("S1_BACKEND", "numba-cuda")
if BACKEND == "mlir":
    from numba_cuda_mlir import cuda
else:
    from numba import cuda
print("backend:", BACKEND)

print(
    f"python {sys.version.split()[0]} ({platform.python_implementation()}) on {platform.release()}"
)
for pkg in ("numba", "numba-cuda", "numba-cuda-mlir", "numpy", "llvmlite"):
    try:
        print(f"{pkg:24s} {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:24s} -")
nv = sorted(
    (d.metadata["Name"], d.version)
    for d in distributions()
    if d.metadata["Name"].lower().startswith(("nvidia", "cuda"))
)
for name, ver in nv:
    print(f"{name:24s} {ver}")

print("import time:", f"{time.perf_counter() - t0:.2f}s")
print("cuda.is_available():", cuda.is_available())
from cuda.bindings import driver as cu_drv  # noqa: E402
from cuda.bindings import nvrtc as cu_nvrtc  # noqa: E402
from cuda.bindings import runtime as cu_rt  # noqa: E402

d = cu_drv.cuDriverGetVersion()[1]
r = cu_rt.cudaRuntimeGetVersion()[1]
print(f"driver CUDA version: {d // 1000}.{d % 1000 // 10} | runtime: {r // 1000}.{r % 1000 // 10}")
print("NVRTC version:", cu_nvrtc.nvrtcVersion()[1:])

dev = cuda.get_current_device()
print("device:", dev.name.decode() if isinstance(dev.name, bytes) else dev.name)
print("compute capability:", dev.compute_capability)
print("SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO:", dev.SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)
print("NUMBA_CACHE_DIR:", os.environ.get("NUMBA_CACHE_DIR", "(unset)"))


@cuda.jit
def saxpy(a, x, y, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a * x[i] + y[i]


@cuda.jit
def atomic_sum(x, acc):
    i = cuda.grid(1)
    if i < x.size:
        cuda.atomic.add(acc, 0, x[i])


@cuda.jit(cache=True)
def cached_kernel(x, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = x[i] * x[i] + 1.0


n = 1 << 18
tpb = 256
blocks = (n + tpb - 1) // tpb
rng = np.random.default_rng(0)

x = rng.standard_normal(n).astype(np.float32)
y = rng.standard_normal(n).astype(np.float32)
out = cuda.device_array(n, dtype=np.float32)
t = time.perf_counter()
saxpy[blocks, tpb](np.float32(2.5), cuda.to_device(x), cuda.to_device(y), out)
cuda.synchronize()
print(f"saxpy f32 compile+run: {time.perf_counter() - t:.2f}s")
ok1 = np.allclose(out.copy_to_host(), 2.5 * x + y, rtol=1e-6, atol=1e-6)
print("saxpy f32 matches NumPy:", ok1)

xd = rng.standard_normal(n)
acc = cuda.to_device(np.zeros(1, dtype=np.float64))
t = time.perf_counter()
atomic_sum[blocks, tpb](cuda.to_device(xd), acc)
cuda.synchronize()
print(f"atomic_sum f64 compile+run: {time.perf_counter() - t:.2f}s")
got, want = acc.copy_to_host()[0], xd.sum()
ok2 = abs(got - want) < 1e-9 * n
print(f"atomic_sum f64 matches NumPy: {ok2} (got {got:.12f}, want {want:.12f})")

xd32 = rng.standard_normal(n).astype(np.float32)
outc = cuda.device_array(n, dtype=np.float32)
t = time.perf_counter()
cached_kernel[blocks, tpb](cuda.to_device(xd32), outc)
cuda.synchronize()
print(f"cached_kernel compile-or-load+run: {time.perf_counter() - t:.2f}s")
ok3 = np.allclose(outc.copy_to_host(), xd32 * xd32 + 1.0, rtol=1e-6)
print("cached_kernel matches NumPy:", ok3)
stats = getattr(cached_kernel, "stats", None)
if stats is not None:
    print("cached_kernel cache_hits:", dict(stats.cache_hits), "misses:", dict(stats.cache_misses))

print("RESULT:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")
sys.exit(0 if (ok1 and ok2 and ok3) else 1)

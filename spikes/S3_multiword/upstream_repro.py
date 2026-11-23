"""Minimal repro for the numba-cuda-mlir upstream issue draft (upstream_issue.md).

1. fastmath=True has no effect below sm_100 (the LLVM 7 bridge drops the libnvvm options).
2. There is no way to disable FMA contraction: a*b + c becomes fma.rn with fastmath=False.
"""

import re
from importlib.metadata import version

from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import float32, void


def kernel(a, b, c, out):
    out[0] = a[0] * b[0] + c[0]  # contractible mul + add
    out[1] = a[0] / b[0]  # prec-div
    out[2] = a[0] + b[0]  # ftz candidate


sig = void(float32[:], float32[:], float32[:], float32[:])
print("numba-cuda-mlir", version("numba-cuda-mlir"))
for cc in ((8, 6), (9, 0), (10, 0), (12, 0)):
    for fastmath in (False, True):
        ptx, _ = cuda.compile_ptx(kernel, sig, fastmath=fastmath, cc=cc)
        ops = re.findall(r"^\s*((?:fma|mul|add|div)\.[a-z0-9.]*f32)", ptx, re.M)
        print(f"sm_{cc[0]}{cc[1]:<2} fastmath={fastmath!s:<5} {ops}")

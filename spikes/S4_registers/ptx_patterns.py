"""Diagnostic: instruction patterns in PTX of generated kernels, for designing the PTX tests.
Runs in the no-GPU container with S4_PATCH_HOSTCC=1 (see q1_nogpu.py) or on a GPU host."""

import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("S4_PATCH_HOSTCC"):
    from numba_cuda_mlir import tools

    _o = tools.get_gpu_compute_capability

    def _stub(fmt=tuple):
        try:
            return _o(fmt)
        except Exception:
            return (0, 0) if fmt is tuple else "sm_00"

    tools.get_gpu_compute_capability = _stub
import gen
from numba_cuda_mlir import cuda

PAT = re.compile(r"^\s*(?:@%p\d+\s+)?((?:fma|mad|add|sub|mul|neg|cvt)\.[\w.]+)", re.M)
for tier, kern in [(t, k) for t in sys.argv[1].split(",") for k in sys.argv[2].split(",")]:
    for cc in ((8, 6), (12, 0)):
        ptx, _ = cuda.compile_ptx(gen.load(tier, kern).py_func, gen.signature(tier), cc=cc)
        c = collections.Counter(m for m in PAT.findall(ptx) if re.search(r"\.f(32|64)", m))
        bad = {
            k: v
            for k, v in c.items()
            if re.fullmatch(r"(add|sub|mul)\.f(32|64)", k)
            or k.startswith("mad")
            or (k.startswith("fma") and ".rn" not in k)
        }
        f64 = sum(v for k, v in c.items() if "f64" in k)
        fma = {k: v for k, v in c.items() if k.startswith("fma")}
        print(f"{tier}/{kern} sm_{cc[0]}{cc[1]}: f64-ops={f64} fma={fma} non-rn/fused={bad}")

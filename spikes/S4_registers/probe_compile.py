import os
import re
import subprocess
import sys
import tempfile
import time

import gen
from numba_cuda_mlir import cuda

PTXAS = os.path.join(
    os.path.dirname(__file__), ".venv/lib/python3.15/site-packages/nvidia/cu13/bin/ptxas"
)
_a = sys.argv[1] if len(sys.argv) > 1 else "86"
cc = (int(_a[:-1]), int(_a[-1]))
TF = sys.argv[2].split(",") if len(sys.argv) > 2 else list(gen.TIERS)
for tier in TF:
    for k in gen.KERNELS:
        kd = gen.load(tier, k)
        t0 = time.perf_counter()
        ptx, _ = cuda.compile_ptx(kd.py_func, gen.signature(tier), cc=cc)
        t = time.perf_counter() - t0
        with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
            f.write(ptx.rstrip("\x00"))
        r = subprocess.run(
            [PTXAS, f"-arch=sm_{cc[0]}{cc[1]}", "-v", f.name, "-o", os.devnull],
            capture_output=True,
            text=True,
        )
        info = " | ".join(
            ln.split("info    : ")[-1]
            for ln in r.stderr.splitlines()
            if "Used" in ln or "spill" in ln
        )
        print(
            f"{tier:6} {k:3} {t:5.1f}s f64ops={len(re.findall(r'\.f64', ptx)):4d} "
            f"{info or r.stderr[:200]}",
            flush=True,
        )

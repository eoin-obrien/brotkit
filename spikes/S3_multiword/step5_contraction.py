"""Step 5 (revised): FMA contraction test.

5a  naive (bare * + -) multi-word code: PTX/SASS show fused multiply-adds we never wrote,
    and the Dekker-split dd_mul / two_prod fail (counts come from step2/step3 runs with
    variant=naive; this script does the instruction evidence).
5b  rn variant: separate mul.rn/add.rn (DMUL/DADD), the only fma is the explicit cuda.fma.
5c  the same kernels compiled for cc=(10,0) with compile_ptx, fastmath False and True.

usage: uv run step5_contraction.py
"""

import json
import os
import re
import subprocess
import tempfile

import guard

PTX_OPS = r"^\s*((?:fma|mul|add|sub|neg)\.[a-z0-9.]*f(?:32|64))\b"
SASS_OPS = r"\b((?:D|F)(?:FMA|MUL|ADD)(?:\.[A-Z0-9]+)*)\b"

# Source-level count of explicit cuda.fma per kernel (what we *wrote*).
EXPLICIT_FMA = {"k_one_mul_dekker": 0, "k_one_mul": 1, "k_one_two_sum": 0}


def count(pattern, text):
    out: dict[str, int] = {}
    for op in re.findall(pattern, text, re.M):
        out[op] = out.get(op, 0) + 1
    return dict(sorted(out.items()))


def body(ptx: str, name_hint: str) -> str:
    """The .entry body for the kernel (strip other functions)."""
    m = re.search(r"\.entry[^{]*" + ".*?" + r"\{(.*?)\n\}", ptx, re.S)
    return m.group(1) if m else ptx


def sass_offline(ptx: str, arch: str) -> str | None:
    ptxas = os.path.join(
        os.path.dirname(__file__), ".venv/lib/python3.15/site-packages/nvidia/cu13/bin/ptxas"
    )
    cuobjdump = "/usr/local/cuda/bin/cuobjdump"
    if not (os.path.exists(ptxas) and os.path.exists(cuobjdump)):
        return None
    with tempfile.TemporaryDirectory() as d:
        src, cub = os.path.join(d, "k.ptx"), os.path.join(d, "k.cubin")
        with open(src, "w") as fh:
            fh.write(ptx.rstrip("\x00"))
        r = subprocess.run([ptxas, f"-arch={arch}", src, "-o", cub], capture_output=True, text=True)
        if r.returncode:
            return f"ptxas failed: {r.stderr[:300]}"
        r = subprocess.run([cuobjdump, "-sass", cub], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else f"cuobjdump failed: {r.stderr[:300]}"


def excerpt(ptx: str, n: int = 14) -> str:
    lines = [
        ln.strip()
        for ln in ptx.splitlines()
        if re.match(r"\s*(fma|mul|add|sub|neg)\.[a-z0-9.]*f(32|64)", ln)
    ]
    return "\n".join(lines[:n])


def main():
    guard.check("step5")
    import gen
    import numpy as np
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.types import float32, float64, void

    res = {}
    for prec, ty, dt in (("f64", float64, np.float64), ("f32", float32, np.float32)):
        sig = void(ty[:, :], ty[:, :], ty[:, :])
        for variant in ("naive", "rn"):
            mod = gen.build(prec, variant)
            for kname in EXPLICIT_FMA:
                k = getattr(mod, kname)
                # sm_86: compile by launching, then PTX + SASS from the dispatcher
                a = cuda.to_device(np.ones((4, 4), dt))
                k[1, 32](a, a, a)
                cuda.synchronize()
                key = next(iter(k.overloads.keys()))
                ptx86 = k.inspect_asm(key)
                sass86 = k.inspect_sass(key)
                entry = dict(
                    explicit_fma_in_source=EXPLICIT_FMA[kname],
                    sm_86=dict(
                        ptx=count(PTX_OPS, ptx86),
                        sass=count(SASS_OPS, sass86),
                        ptx_excerpt=excerpt(ptx86),
                    ),
                )
                # 5c: Blackwell, offline, fastmath off and on
                for fm in (False, True):
                    ptx100, _ = cuda.compile_ptx(k.py_func, sig, fastmath=fm, cc=(10, 0))
                    s100 = sass_offline(ptx100, "sm_100")
                    entry[f"sm_100_fastmath_{fm}"] = dict(
                        ptx=count(PTX_OPS, ptx100),
                        sass=count(SASS_OPS, s100)
                        if s100 and not s100.startswith(("ptxas", "cuobj"))
                        else s100,
                        ptx_excerpt=excerpt(ptx100),
                    )
                res[f"{prec}/{variant}/{kname}"] = entry
                print(
                    prec,
                    variant,
                    kname,
                    json.dumps(
                        {
                            k2: (
                                v
                                if not isinstance(v, dict)
                                else {kk: vv for kk, vv in v.items() if kk != "ptx_excerpt"}
                            )
                            for k2, v in entry.items()
                        }
                    ),
                    flush=True,
                )
    with open("results/step5_instructions.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

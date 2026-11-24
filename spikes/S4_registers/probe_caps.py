"""Q4 mechanics: is max_registers per kernel or per module, does it survive both MLIR paths,
does launch_bounds reach PTX, and does cache=True key on max_registers?

usage: python probe_caps.py mech | cache_write | cache_read
(for the cache test: empty probe/cache first, then cache_write, then cache_read in a new process)
"""

import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "probe"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "probe", "cache"))

import gen  # noqa: E402
from numba_cuda_mlir import cuda, types  # noqa: E402

SIG = (types.float64[::1], types.float64[::1])


def regs(d, sig):
    d.compile(sig)
    return d.get_regs_per_thread(sig)


mode = sys.argv[1]
if mode == "mech":
    tk = importlib.import_module("two_kernels")
    out = {
        "two_kernels same module": {
            "k_capped(max_registers=24)": regs(tk.k_capped, SIG),
            "k_default": regs(tk.k_default, SIG),
        }
    }
    sig = gen.signature("dd")
    py = gen.load("dd", "F6").py_func
    out["dd/F6 dispatcher regs by max_registers (sm_86, LLVM-7 bridge)"] = {
        str(c): regs(cuda.jit(max_registers=c)(py) if c else cuda.jit(py), sig)
        for c in (None, 32, 64, 96, 128)
    }
    out["dd/F6 dispatcher regs launch_bounds=(256,4)"] = regs(
        cuda.jit(launch_bounds=(256, 4))(py), sig
    )
    import subprocess
    import tempfile

    cuobjdump = os.path.join(HERE, ".venv/lib/python3.15/site-packages/nvidia/cu13/bin/cuobjdump")
    for chip in ("sm_100", "sm_120"):
        for cap in (None, 48):
            opts = {"chip": chip} | ({"max_registers": cap} if cap else {})
            d = cuda.jit(**opts)(py)
            key = f"dispatcher chip={chip} max_registers={cap}: linked cubin resource usage"
            try:
                d.compile(sig)
                cubin = d._find_overload(sig).metadata["cubin"]
                with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
                    f.write(cubin)
                    f.flush()
                    r = subprocess.run(
                        [cuobjdump, "--dump-resource-usage", f.name], capture_output=True, text=True
                    )
                out[key] = [
                    ln.strip() for ln in r.stdout.splitlines() if "REG:" in ln or "arch" in ln
                ]
            except Exception as e:
                out[key] = f"{type(e).__name__}: {e}"[:400]
    for cc in ((8, 6), (10, 0), (12, 0)):
        ptx, _ = cuda.compile_ptx(py, sig, cc=cc, launch_bounds=(256, 4))
        out[f"compile_ptx launch_bounds=(256,4) cc={cc}"] = [
            ln.strip() for ln in ptx.splitlines() if ".maxntid" in ln or ".minnctapersm" in ln
        ]
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "results", "q4_mechanics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
elif mode == "cache_write":
    tk = importlib.import_module("two_kernels")
    print(
        "cache_write k_cached_cap(max_registers=24) regs:",
        regs(tk.k_cached_cap, SIG),
        tk.k_cached_cap.stats,
    )
elif mode == "cache_read":
    tk = importlib.import_module("two_kernels")
    print("cache_read k_cached_nocap regs:", regs(tk.k_cached_nocap, SIG), tk.k_cached_nocap.stats)
    print("cache_read k_cached_cap regs:", regs(tk.k_cached_cap, SIG), tk.k_cached_cap.stats)

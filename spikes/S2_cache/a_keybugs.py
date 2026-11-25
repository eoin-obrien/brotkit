"""Mechanism (a): which compile options does numba-cuda-mlir's cache=True key ignore? (throwaway)

For each option, three fresh processes share one kernel file in its own directory:
  1. build variant X with cache=True  (writes __pycache__/*.nbi/*.nbc)
  2. build variant Y with cache=True  (should miss; a hit here is the bug)
  3. build variant Y with cache=False (the reference build of Y)
A stale hit = process 2 reports a cache hit AND its build differs from process 3's
(registers, or PTX text for options that don't change registers).

Kernel: S2's ptr64 dd/C (59-60 registers uncapped), plus an extra-module-global case.

usage: .venv/bin/python a_keybugs.py -> results/a_keybugs.json
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import s2lib

HERE = pathlib.Path(__file__).parent

CASES = {
    # name: (options X, options Y)
    "max_registers": ({}, {"max_registers": 32}),
    "ptxas_options": ({}, {"ptxas_options": "--maxrregcount=40"}),
    "fastmath": ({"fastmath": False}, {"fastmath": True}),
    "lineinfo": ({}, {"lineinfo": True}),
    "opt": ({}, {"opt": False}),
    "launch_bounds (control: in the key)": ({}, {"launch_bounds": (256, 4)}),
}

CHILD = r"""
import json, sys, importlib.util, hashlib
sys.path.insert(0, {here!r})
import gen_s2
from numba_cuda_mlir import cuda
opts = json.loads(sys.argv[1]); cache = sys.argv[2] == "1"; path = sys.argv[3]; name = sys.argv[4]
if "launch_bounds" in opts: opts["launch_bounds"] = tuple(opts["launch_bounds"])
spec = importlib.util.spec_from_file_location(name, path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
k = getattr(mod, "k_dd_C")
k = cuda.jit(cache=cache, **opts)(k.py_func) if (opts or cache) else k
sig = gen_s2.signature("dd", "ptr64")
cres = k.compile(sig)
args = tuple(sig)
hits = sum(k._cache_hits.values()) if hasattr(k, "_cache_hits") else None
regs = k.get_regs_per_thread(sig) if hasattr(k, "get_regs_per_thread") else None
ptx = cres.metadata.get("ptx") if hasattr(cres, "metadata") else None
cubin = cres.metadata.get("cubin") if hasattr(cres, "metadata") else None
print(json.dumps(dict(hits=hits, regs=regs,
    ptx_sha=hashlib.sha256(ptx.encode() if isinstance(ptx, str) else (ptx or b"")).hexdigest()[:16],
    ptx_has_loc=(".loc" in ptx) if isinstance(ptx, str) else None,
    cubin_sha=hashlib.sha256(cubin or b"").hexdigest()[:16])))
"""

GLOBAL_CHILD = r"""
import json, sys, importlib.util
sys.path.insert(0, sys.argv[2])
from numba_cuda_mlir import cuda, types
import numpy as np
spec = importlib.util.spec_from_file_location("kmod", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
out = cuda.device_array(1, np.float32)
mod.k[1, 1](out)
print(json.dumps(dict(value=float(out.copy_to_host()[0]), hits=sum(mod.k._cache_hits.values()))))
"""


def run(code: str, *args: str) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env)
    if r.returncode:
        return {"error": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "rc!=0"}
    return json.loads(r.stdout.strip().splitlines()[-1])


def option_case(name: str, x: dict, y: dict) -> dict:
    d = pathlib.Path(tempfile.mkdtemp(prefix="s2a_"))
    gen_src = None
    import gen_s2

    gen_src = gen_s2.emit("dd", "C", "ptr64")
    f = d / "kmod.py"
    f.write_text(gen_src)
    code = CHILD.format(here=str(HERE))
    r1 = run(code, json.dumps(x), "1", str(f), "kmod")
    r2 = run(code, json.dumps(y), "1", str(f), "kmod")
    r3 = run(code, json.dumps(y), "0", str(f), "kmod")
    differs = (r2.get("regs"), r2.get("ptx_sha"), r2.get("cubin_sha")) != (
        r3.get("regs"),
        r3.get("ptx_sha"),
        r3.get("cubin_sha"),
    )
    shutil.rmtree(d)
    return {
        "x": x,
        "y": y,
        "build_x_cached": r1,
        "build_y_cached": r2,
        "build_y_reference": r3,
        "y_hit": bool(r2.get("hits")),
        "stale": bool(r2.get("hits")) and differs,
        "y_differs_from_x_when_fresh": (r1.get("regs"), r1.get("ptx_sha"))
        != (r3.get("regs"), r3.get("ptx_sha")),
    }


def global_case() -> dict:
    """Kernel bytecode and file unchanged; a constant imported from another module changes."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="s2g_"))
    (d / "consts.py").write_text("import numpy as np\nVAL = np.float32(1.0)\n")
    (d / "kmod.py").write_text(
        "from numba_cuda_mlir import cuda\nfrom consts import VAL\n\n"
        "@cuda.jit(cache=True)\ndef k(out):\n    out[0] = VAL\n"
    )
    r1 = run(GLOBAL_CHILD, str(d / "kmod.py"), str(d))
    (d / "consts.py").write_text("import numpy as np\nVAL = np.float32(2.0)\n")
    r2 = run(GLOBAL_CHILD, str(d / "kmod.py"), str(d))
    shutil.rmtree(d)
    return {"before": r1, "after_const_change": r2, "stale": r2.get("value") == 1.0}


def main() -> None:
    out = {}
    for name, (x, y) in CASES.items():
        out[name] = option_case(name, x, y)
        r = out[name]
        print(
            f"{name:36} y_hit={r['y_hit']} stale={r['stale']} "
            f"x={r['build_x_cached'].get('regs')}/{r['build_x_cached'].get('ptx_sha')} "
            f"y_cached={r['build_y_cached'].get('regs')}/{r['build_y_cached'].get('ptx_sha')} "
            f"y_ref={r['build_y_reference'].get('regs')}/{r['build_y_reference'].get('ptx_sha')} "
            f"{r['build_y_cached'].get('error', '')}",
            flush=True,
        )
    out["imported_global"] = global_case()
    print("imported_global", out["imported_global"], flush=True)
    (s2lib.RESULTS / "a_keybugs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

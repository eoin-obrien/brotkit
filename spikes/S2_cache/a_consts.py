"""Mechanism (a): an in-function constant changes while the file's size and mtime stay the same.

Numba's index key hashes co_code (not co_consts); the source stamp (mtime, size) is what
normally invalidates the index. This checks whether a same-size edit with the mtime restored
(e.g. a tool that preserves timestamps) is served stale. (throwaway)

usage: .venv/bin/python a_consts.py <scratch dir> -> results/a_consts.json
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import s2lib

KMOD = """from numba_cuda_mlir import cuda
import numpy as np

@cuda.jit(cache=True)
def k(out):
    out[0] = np.float32({v})
"""
RUN = r"""
import sys, importlib.util, json
from numba_cuda_mlir import cuda
import numpy as np
spec = importlib.util.spec_from_file_location("kmod", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
o = cuda.device_array(1, np.float32); m.k[1, 1](o)
print(json.dumps(dict(value=float(o.copy_to_host()[0]), hits=sum(m.k._cache_hits.values()))))
"""


def run(f: pathlib.Path) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", RUN, str(f)], capture_output=True, text=True, env=env)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> None:
    d = pathlib.Path(sys.argv[1])
    d.mkdir(parents=True, exist_ok=True)
    f = d / "kmod.py"
    f.write_text(KMOD.format(v="1.0"))
    out = {"first": run(f)}
    st = f.stat()
    f.write_text(KMOD.format(v="2.0"))
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    out["same_stamp_const_2"] = run(f)
    f.touch()
    out["after_touch"] = run(f)
    out["stale"] = out["same_stamp_const_2"]["value"] == 1.0
    print(out)
    (s2lib.RESULTS / "a_consts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

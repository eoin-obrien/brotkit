"""Cold cost: source string -> launchable kernel, median of N fresh processes (throwaway).

Each sample: a fresh interpreter, a fresh generated-source dir (no .pyc), the nvJitLink compute
cache bypassed by the PTX salt (s2lib.link). Samples interleave configs (round-robin) so drift
spreads evenly.

usage: .venv/bin/python cold.py [N] -> results/cold.json
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile

import s2lib

HERE = pathlib.Path(__file__).parent
TIERS = ("f32", "df64", "f64", "dd", "qf")
KERNELS = ("A", "C", "F6")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    samples: dict[str, list[dict]] = {}
    for i in range(n):
        for kern in KERNELS:
            for tier in TIERS:
                d = tempfile.mkdtemp(prefix="s2cold_")
                r = subprocess.run(
                    [sys.executable, str(HERE / "cold_child.py"), tier, kern, d],
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=HERE,
                )
                shutil.rmtree(d)
                if r.returncode:
                    print(r.stderr[-2000:])
                    raise SystemExit(f"{tier}/{kern} failed")
                rec = json.loads(r.stdout.strip().splitlines()[-1])
                samples.setdefault(f"{tier}/{kern}", []).append(rec)
                print(f"[{i}] {tier}/{kern} cold={rec['per_kernel_cold']:.2f}s", flush=True)
    med = {}
    for key, recs in samples.items():
        fields = [k for k, v in recs[0].items() if isinstance(v, float)]
        med[key] = {f: round(statistics.median(r[f] for r in recs), 4) for f in fields}
        med[key]["n"] = len(recs)
        med[key]["per_kernel_cold_min_max"] = (
            min(r["per_kernel_cold"] for r in recs),
            max(r["per_kernel_cold"] for r in recs),
        )
        for b in ("ptx_bytes", "cubin_bytes", "src_bytes"):
            med[key][b] = recs[0][b]
    (s2lib.RESULTS / "cold.json").write_text(
        json.dumps({"median": med, "samples": samples}, indent=1)
    )
    for key, m in med.items():
        print(
            key,
            {
                k: m[k]
                for k in (
                    "exec",
                    "frontend",
                    "optimize",
                    "ptx",
                    "compile_other",
                    "link",
                    "load",
                    "first_launch",
                    "per_kernel_cold",
                    "import_numba",
                    "import_core",
                    "compiler_init",
                )
            },
        )


if __name__ == "__main__":
    main()

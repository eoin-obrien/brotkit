"""Warm-up kernel during a kernel build, on an idle GPU (throwaway).

Each trial points the runner at an empty kernel cache, waits for the idle clock, and renders
qf/whole at 4K: the lookup builds the qf kernel (about 2 s). With `warm_on_miss` a helper thread
queues ~300 ms of warm kernel before the build; without it the GPU stays idle. No prior
exists, so the render calibrates. Compares the render's kernel time (real chunks and
calibration), which carries the cold-clock penalty, and the first chunks' speed.

usage: .venv/bin/python warm_build.py <scratch dir> [reps] -> results/warm_build.json
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import s5lib

from brotkit.kernels.cache import KernelCache, KernelStore


def main() -> None:
    scratch = pathlib.Path(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    s5lib.s4_on_path()
    import q3_throughput as q

    gpu = s5lib.Gpu.open(scratch / "warm_build_base")
    baseline = int(q.smi("memory.used")[0])
    out = []
    for rep in range(reps):
        for warm in (False, True):
            root = scratch / f"warm_build_{rep}_{int(warm)}"
            shutil.rmtree(root, ignore_errors=True)
            gpu.cache = KernelCache(gpu.backend, KernelStore(root))
            gpu.kernels.clear()
            gpu.specs.clear()
            gpu.warm_on_miss = warm
            s5lib.wait_idle(q, baseline, f"warm_build {rep} {warm}")
            s5lib.wait_cold()
            mhz0 = s5lib.sm_mhz()
            r = s5lib.render(gpu, "qf", "C", "whole", record_prior=False)
            s = r.summary()
            calib_s = s["calibration_s"]
            first = [x[4] for x in s["records"] if not x[0]][:3]
            row = {
                "rep": rep,
                "warm_on_miss": warm,
                "mhz_before": mhz0,
                "queue_warm_ms": round(r.warmed_ms, 1),  # type: ignore[attr-defined]
                "prep_s": round(r.prep_s, 3),  # type: ignore[attr-defined]
                "wall_s": s["wall_s"],
                "kernel_s": s["kernel_s"],
                "calibration_s": calib_s,
                "gpu_s": round(s["kernel_s"] + calib_s, 4),
                "first_chunks_s": first,
                "chunks": s["chunks"],
            }
            out.append(row)
            print(row, flush=True)
            (s5lib.HERE / "results" / "warm_build.json").write_text(json.dumps(out, indent=1))
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

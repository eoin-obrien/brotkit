"""Main sweep: 4K kernel C, every tier x view, default ChunkPolicy, one render each (throwaway).

Records per render: chunks, wall and kernel time, overhead (1 - kernel / wall), the longest
chunk, the planner's worst-case bound against the actual time, and chunks over the ceiling.
Each config renders twice: cold (full calibration) and warm (the cold render's estimates as the
planner's prior, as a re-render in the same session would). GPU idle check as S4 before each.

usage: .venv/bin/python sweep.py <cache root> [wait] [tiers] [views] -> results/sweep_<wait>.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import s5lib

from brotkit.backends.chunks import ChunkPolicy

MAX_ITER: dict[str, dict[str, int]] = {}  # every tier at the view default (10000 worst case)


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    wait = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    tiers = sys.argv[3].split(",") if len(sys.argv) > 3 else ["f32", "df64", "f64", "dd", "qf"]
    views = sys.argv[4].split(",") if len(sys.argv) > 4 else ["seahorse", "whole", "interior"]
    s5lib.s4_on_path()
    import q3_throughput as q

    gpu = s5lib.Gpu.open(root)
    baseline = int(q.smi("memory.used")[0])
    out = []
    for tier in tiers:
        gpu.kernel(tier, "C")  # build/load once; not timed
        for view in views:
            prior = None
            for start in ("cold", "warm"):
                s5lib.wait_idle(q, baseline, f"{tier}/{view}")
                mi = MAX_ITER.get(tier, {}).get(view, s5lib.VIEWS[view][3])
                r = s5lib.render(
                    gpu, tier, "C", view, max_iter=mi, policy=ChunkPolicy(), wait=wait, prior=prior
                )
                prior = r.estimates  # type: ignore[attr-defined]
                s = r.summary() | {"start": start}
                out.append(s)
                rest = r.wall_s - r.kernel_s - s["calibration_s"]
                print(
                    f"{tier:5} {view:9} {start} it={mi:5} chunks={s['chunks']:4} "
                    f"calib={s['calibration_chunks']:2}/{s['calibration_s'] * 1e3:6.1f}ms "
                    f"rest={rest / s['wall_s'] * 100:5.2f}% "
                    f"wall={s['wall_s']:8.3f}s "
                    f"overhead={s['overhead'] * 100:6.3f}% "
                    f"({(r.wall_s - r.kernel_s) * 1e3:7.1f} ms) "
                    f"max={s['chunk_s_max']:.3f}s p90={s['chunk_s_p90']:.3f}s "
                    f"pred_max={s['pred_worst_max_s']} act/pred<={s['actual_over_pred_max']} "
                    f"calib/pred<={s['calib_over_pred_max']} "
                    f"over_ceil={s['over_ceiling']} >2xtarget={s['over_target_x2']} "
                    f"cpu/wall={s['cpu_s'] / s['wall_s']:.3f}",
                    flush=True,
                )
                name = f"sweep_{wait}.json"
                (s5lib.HERE / "results" / name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

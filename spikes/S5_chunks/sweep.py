"""Main sweep: 4K kernel C, every tier x view, default ChunkPolicy (throwaway).

Each config renders three times:
    first  no prior (the kernel cache's perf record deleted): the calibration launches run
    cold   the prior the first render stored, after the GPU has dropped to its idle clock
    warm   the same, straight after (clock up)
`cold` and `warm` are the revised overhead gate's two cases. Records per render: chunks, wall
and kernel time, overhead (wall - kernel - clock warm-up; calibration launches count), host setup
before the first launch (`prep`, outside the wall time), the longest chunk, the planner's
worst-case bound against the actual time, and chunks over the ceiling. GPU idle check as S4.

usage: .venv/bin/python sweep.py <cache root> [warmup_ms] [tiers] [views] [wait]
       -> results/sweep_prior.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import s5lib

from brotkit.backends.chunks import ChunkPolicy

GATE_LONG_S = 3.7  # renders this long: total overhead < 1%; shorter: <= 50 ms
GATE_SHORT_MS = 50.0


def gate(wall_s: float, overhead_s: float) -> bool:
    if wall_s >= GATE_LONG_S:
        return overhead_s < 0.01 * wall_s
    return overhead_s * 1e3 <= GATE_SHORT_MS


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    warmup_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    tiers = sys.argv[3].split(",") if len(sys.argv) > 3 else ["f32", "df64", "f64", "dd", "qf"]
    views = sys.argv[4].split(",") if len(sys.argv) > 4 else ["seahorse", "whole", "interior"]
    wait = sys.argv[5] if len(sys.argv) > 5 else "hybrid"
    s5lib.s4_on_path()
    import q3_throughput as q

    gpu = s5lib.Gpu.open(root)
    baseline = int(q.smi("memory.used")[0])
    if warmup_ms:
        gpu.calibrate_warm(idle_mhz=210, now_mhz=s5lib.load_mhz(gpu))
    out = []
    name = f"sweep_prior_w{warmup_ms:g}.json"
    for tier in tiers:
        gpu.kernel(tier, "C")  # build/load once; not timed
        for view in views:
            gpu.forget_prior(tier, "C")
            for start in ("first", "cold", "warm"):
                s5lib.wait_idle(q, baseline, f"{tier}/{view}")
                waited = s5lib.wait_cold() if start != "warm" else 0.0
                mhz0 = s5lib.sm_mhz()
                r = s5lib.render(
                    gpu,
                    tier,
                    "C",
                    view,
                    policy=ChunkPolicy(),
                    wait=wait,
                    warmup_ms=warmup_ms if start != "first" else 0.0,
                )
                s = r.summary()
                oh = r.wall_s - r.kernel_s - r.warmup_s
                prep = r.prep_s  # type: ignore[attr-defined]
                s |= {
                    "start": start,
                    "warmup_ms": warmup_ms if start != "first" else 0.0,
                    "had_prior": r.prior is not None,  # type: ignore[attr-defined]
                    "mhz_before": mhz0,
                    "waited_cold_s": round(waited, 1),
                    "prep_s": round(prep, 5),
                    "overhead_ms": round(oh * 1e3, 2),
                    "cold_factor_end": round(r.cold_factor, 3),  # type: ignore[attr-defined]
                    "gate": gate(r.wall_s, oh),
                    "gate_with_prep": gate(r.wall_s + prep, oh + prep),
                }
                out.append(s)
                print(
                    f"{tier:5} {view:9} {start:5} mhz0={mhz0:4} chunks={s['chunks']:4} "
                    f"calib={s['calibration_chunks']:2}/{s['calibration_s'] * 1e3:6.1f}ms "
                    f"warmup={s['warmup_chunks']}/{s['warmup_s'] * 1e3:5.1f}ms "
                    f"wall={s['wall_s']:8.3f}s overhead={s['overhead'] * 100:6.3f}% "
                    f"({s['overhead_ms']:7.1f} ms) prep={prep * 1e3:5.1f}ms "
                    f"max={s['chunk_s_max']:.3f}s act/pred<={s['actual_over_pred_max']} "
                    f"over_ceil={s['over_ceiling']} c_end={s['cold_factor_end']} "
                    f"gate={'PASS' if s['gate'] else 'FAIL'} "
                    f"cpu/wall={s['cpu_s'] / s['wall_s']:.3f}",
                    flush=True,
                )
                (s5lib.HERE / "results" / name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

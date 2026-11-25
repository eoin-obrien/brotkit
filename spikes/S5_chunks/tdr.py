"""Single long launches up to 75% of TdrDelay, never past it (throwaway).

TdrDelay isn't set on Xenon, so Windows' default of 2 s applies; this goes up to 1.5 s. Each
launch is one dd "interior" band (every pixel at max_iter) sized to last about 0.25, 0.5, 1.0
and 1.5 s. The GPU's clock drops when it idles and the first launches after idle run 4-9x slower
(S5), so a launch sized from an earlier measurement could overrun TDR by that factor. Each
long launch is therefore sized from short launches on the same rows run immediately before it
(about 1.5 s of 8-row launches, the first ones at a tenth of max_iter, at the clock the long
launch will start at), and the series stops if a
launch overruns its prediction by more than 1.3x. Records the kernel time, that the launch
completed, and that a follow-up launch still works (no reset, no lost context).

usage: .venv/bin/python tdr.py <cache root> -> results/tdr.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import gen_s5
import numpy as np
import s5lib

TDR_DELAY_S = 2.0  # Windows default; HKLM\\...\\GraphicsDrivers has no TdrDelay on Xenon
LIMIT_S = 0.75 * TDR_DELAY_S
MARGIN = 1.15  # bands are sized as if rows cost 15% more than the dearest measured row


def launch(gpu, k, args0, mi, row0, rows, bf, bu, w, h) -> float:
    from cuda.core import EventOptions

    opts = EventOptions(timing_enabled=True)
    stream = gpu.backend.stream
    ea = stream.record(options=opts)
    k.launch(
        ((w + 31) // 32, (rows + 7) // 8),
        (32, 8),
        *args0,
        np.int64(mi),
        np.int64(row0),
        np.int64(0),
        np.int64(rows),
        np.int64(w),
        bf,
        bu,
        np.int64(h),
        np.int64(w),
        np.int64(w),
    )
    eb = stream.record(options=opts)
    stream.sync()
    return (eb - ea) / 1e3


def main() -> None:
    gpu = s5lib.Gpu.open(pathlib.Path(sys.argv[1]))
    s5lib.s4_on_path()
    import q3_throughput as q

    baseline = int(q.smi("memory.used")[0])
    w, h, mi = s5lib.W4K, s5lib.H4K, 5000
    k = gpu.kernel("dd", "C")
    bf = gpu.alloc(gen_s5.NLF * 4 * h * w)
    bu = gpu.alloc(gen_s5.NLU * 4 * h * w)
    args0 = s5lib.view_args("dd", "interior", w)
    row0 = h // 3
    out = []
    prev: tuple[float, float] | None = None  # (long launch's s/row, warm-up s/row then)
    for goal in (0.25, 0.5, 1.0, 1.5):
        s5lib.wait_idle(q, baseline, f"tdr {goal}")
        # Measure at the clock the long launch will start at: short launches, same rows.
        # 8-row launches (warm ~0.1 s); the first three at a tenth of max_iter, so even on a
        # cold clock (up to 9x) none comes near the ceiling, let alone TDR.
        for _ in range(3):
            launch(gpu, k, args0, mi // 10, row0, 8, bf, bu, w, h)
        t_end = time.perf_counter() + 1.5
        per_row = []
        while time.perf_counter() < t_end:
            per_row.append(launch(gpu, k, args0, mi, row0, 8, bf, bu, w, h) / 8)
        warm = max(per_row[-3:])
        # An 8-row launch's cost per row includes the latency floor, so it overestimates a long
        # band; once a long launch has run, size from its cost per row, scaled by the clock
        # ratio between the two warm-ups (never below what the long launch showed).
        s_per_row = warm if prev is None else prev[0] * max(1.0, warm / prev[1])
        rows = min(int(goal / (MARGIN * s_per_row)) // 8 * 8, h - row0)
        predicted = rows * s_per_row * MARGIN
        if rows < 8 or predicted > LIMIT_S:
            print(f"skip {goal}: rows {rows}, predicted {predicted:.2f}s", flush=True)
            continue
        t0 = time.perf_counter()
        ks = launch(gpu, k, args0, mi, row0, rows, bf, bu, w, h)
        wall = time.perf_counter() - t0
        after = s5lib.render(gpu, "f32", "C", "seahorse", w=640, h=360)  # context still works
        rec = dict(
            goal_s=goal,
            rows=rows,
            warmup_launches=len(per_row),
            warmup_first_over_last=round(per_row[0] / per_row[-1], 2),
            predicted_s=round(predicted, 3),
            kernel_s=round(ks, 4),
            wall_s=round(wall, 4),
            completed=True,
            followup_ok=after.complete,
        )
        out.append(rec)
        print(rec, flush=True)
        (s5lib.HERE / "results" / "tdr.json").write_text(json.dumps(out, indent=1))
        prev = (ks / rows, warm)
        if ks > 1.3 * predicted:
            print("overran the prediction by >1.3x: stopping the series", flush=True)
            break


if __name__ == "__main__":
    main()

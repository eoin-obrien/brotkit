"""Chunk time against chunk size for all-interior pixels, per tier (throwaway).

Every pixel runs to max_iter, so t(P) shows the two regimes: below one wave of resident threads
a chunk costs about max_iter x per-thread iteration latency whatever its size; above it, time
grows with P / throughput. One wave = SMs x resident threads per SM at the kernel's occupancy
(from brotkit.backends.resources). GPU idle check before each tier; 3 repeats per size, median.

usage: .venv/bin/python curve.py <cache root> [max_iter] -> results/curve.json
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

import numpy as np
import s5lib

from brotkit.backends.chunks import Tile
from brotkit.backends.resources import occupancy

ROW0 = 900  # rows 900+ of the "interior" view are all interior at 4K
BLOCK = 256


def main() -> None:
    gpu = s5lib.Gpu.open(pathlib.Path(sys.argv[1]))
    mi = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    s5lib.s4_on_path()
    import q3_throughput as q
    from cuda.core import EventOptions

    sms = gpu.backend.device.properties.multiprocessor_count
    baseline = int(q.smi("memory.used")[0])
    w, h = s5lib.W4K, s5lib.H4K
    stream = gpu.backend.stream
    opts = EventOptions(timing_enabled=True)
    bf, bu = gpu.alloc(5 * 4 * h * w * 2), gpu.alloc(3 * 4 * h * w)
    out = {}
    for tier in ("f32", "df64", "f64", "dd", "qf"):
        k = gpu.kernel(tier, "C")
        occ = occupancy(k.resources, BLOCK)
        wave = sms * occ.blocks_per_sm * BLOCK
        args0 = s5lib.view_args(tier, "interior", w)
        s5lib.wait_idle(q, baseline, f"curve {tier}")
        # Warm the GPU to boost clock first: the idle check leaves it at 210 MHz, and the first
        # launches would otherwise measure the clock ramp, not the kernel.
        warm = Tile(ROW0, 0, 64, w)
        t_end = time.perf_counter() + 0.6
        while time.perf_counter() < t_end:
            k.launch(
                ((w + 31) // 32, 8),
                (32, 8),
                *args0,
                np.int64(mi),
                np.int64(warm.row0),
                np.int64(0),
                np.int64(warm.th),
                np.int64(w),
                bf,
                bu,
                np.int64(h),
                np.int64(w),
                np.int64(w),
            )
            stream.sync()
        sizes = sorted(
            {256, 1024, 4096, 16384, wave // 8, wave // 4, wave // 2, wave, 2 * wave, 4 * wave}
        )
        rows = []
        for p in sizes:
            th = 8 if p <= 8 * w else (p // w) // 8 * 8
            tw = min(w, max(32, p // th // 32 * 32))
            tile = Tile(ROW0, 0, th, tw)
            ts = []
            for _ in range(5):
                ea = stream.record(options=opts)
                k.launch(
                    ((tw + 31) // 32, (th + 7) // 8),
                    (32, 8),
                    *args0,
                    np.int64(mi),
                    np.int64(tile.row0),
                    np.int64(0),
                    np.int64(th),
                    np.int64(tw),
                    bf,
                    bu,
                    np.int64(h),
                    np.int64(w),
                    np.int64(w),
                )
                eb = stream.record(options=opts)
                stream.sync()
                ts.append((eb - ea) / 1e3)
            t = statistics.median(ts[1:])  # first launch at this size discarded
            clk = int(q.smi("clocks.sm")[0])
            rows.append(
                dict(
                    pixels=tile.pixels,
                    waves=round(tile.pixels / wave, 3),
                    s=round(t, 6),
                    per_thread_iter_ns=round(t / mi * 1e9, 2),
                    giter_s=round(tile.pixels * mi / t / 1e9, 2),
                    sm_mhz=clk,
                )
            )
            print(tier, rows[-1], flush=True)
        out[tier] = dict(
            max_iter=mi,
            registers=k.resources.registers,
            blocks_per_sm=occ.blocks_per_sm,
            sms=sms,
            wave_pixels=wave,
            rows=rows,
        )
        (s5lib.HERE / "results" / "curve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Fixed cost of one chunk's pipeline, by wait mode (throwaway).

The runner's per-chunk sequence (event, kernel, event, tile_sum, event, wait, 16-byte readback)
on 200 tiny 8x32 tiles of exterior pixels, so the kernels themselves take microseconds. Reports
wall time per chunk, kernel and tile_sum time per chunk (GPU events), the host remainder, and
process CPU time over wall time.

usage: .venv/bin/python fixed_cost.py <cache root> -> results/fixed_cost.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import s5lib

N = 200


def main() -> None:
    from cuda.core import EventOptions

    gpu = s5lib.Gpu.open(pathlib.Path(sys.argv[1]))
    k = gpu.kernel("f32", "C")
    stream = gpu.backend.stream
    h, w = 64, 4096
    bf, bu = gpu.alloc(7 * 4 * h * w), gpu.alloc(3 * 4 * h * w)
    acc = gpu.alloc(16 * N)
    args0 = s5lib.view_args("f32", "seahorse", w)
    two = np.zeros(2, np.uint64)
    out = {}
    modes = (
        ("spin", 0.0),
        ("block", 0.0),
        ("thread", 0.0),
        ("poll", 0.002),
        ("poll", 0.0005),
        ("poll", 0.0001),
        ("hybrid", 0.0001),
    )
    for wait, poll in modes:
        opts = EventOptions(timing_enabled=True, blocking_sync=wait in ("block", "thread"))
        s5lib._memset(acc, 0, 16 * N, stream)
        stream.sync()
        ks = ss = 0.0
        cpu0, t0 = time.process_time(), time.perf_counter()
        for i in range(N):
            r0, c0 = (i % 8) * 8, (i // 8) * 32 % w
            ea = stream.record(options=opts)
            k.launch(
                (1, 1),
                (32, 8),
                *args0,
                np.int64(50),
                np.int64(r0),
                np.int64(c0),
                np.int64(8),
                np.int64(32),
                bf,
                bu,
                np.int64(h),
                np.int64(w),
                np.int64(w),
            )
            eb = stream.record(options=opts)
            gpu.tile_sum.launch(
                64,
                256,
                bu,
                int(acc.handle) + 16 * i,
                np.int64(r0),
                np.int64(c0),
                np.int64(8),
                np.int64(32),
                np.int64(h),
                np.int64(w),
            )
            ec = stream.record(options=opts)
            s5lib._wait(stream, ec, wait, poll, 0.0)
            s5lib._dtoh(two, acc, 16, offset=16 * i)
            ks += (eb - ea) / 1e3
            ss += (ec - eb) / 1e3
        wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        key = wait if not poll else f"{wait} {poll * 1e3:g}ms"
        out[key] = dict(
            wall_per_chunk_us=round(wall / N * 1e6, 1),
            kernel_per_chunk_us=round(ks / N * 1e6, 1),
            sum_per_chunk_us=round(ss / N * 1e6, 1),
            host_per_chunk_us=round((wall - ks - ss) / N * 1e6, 1),
            cpu_per_wall=round(cpu / wall, 3),
        )
        print(key, out[key], flush=True)
    (s5lib.HERE / "results" / "fixed_cost.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

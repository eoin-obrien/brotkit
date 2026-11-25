"""Is a NaN-coordinate probe a safe way to measure tau? (throwaway)

tau (per-thread seconds per iteration) must be measured on pixels that run to max_iter. NaN
coordinates do that for any formula: NaN propagates and `r2 > BAIL` is false for NaN. The risk
is libdevice fast paths on NaN (F6's atan2/sin) making NaN iterations cheaper than real ones.
Compares tau from a sub-knee tile of NaN pixels against a sub-knee tile of real interior pixels.

usage: .venv/bin/python nan_tau.py <cache root> -> results/nan_tau.json
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time

import numpy as np
import s5lib

ROW0 = 1000


def main() -> None:
    gpu = s5lib.Gpu.open(pathlib.Path(sys.argv[1]))
    s5lib.s4_on_path()
    import q3_throughput as q
    from cuda.core import EventOptions

    baseline = int(q.smi("memory.used")[0])
    w, h, mi = s5lib.W4K, s5lib.H4K, 2000
    stream = gpu.backend.stream
    opts = EventOptions(timing_enabled=True)
    bf, bu = gpu.alloc(7 * 4 * h * w), gpu.alloc(3 * 4 * h * w)
    out = []
    for kern in ("C", "F6"):
        for tier in ("f32", "df64", "f64", "dd", "qf"):
            k = gpu.kernel(tier, kern)
            real = s5lib.view_args(tier, "interior", w)
            word = np.dtype(gen_word(tier)).type
            nan = (*(word(np.nan) for _ in real[:-1]), real[-1])
            s5lib.wait_idle(q, baseline, f"nan {tier}/{kern}")

            def launch(args, th=8, tw=1024, k=k):
                ea = stream.record(options=opts)
                k.launch(
                    ((tw + 31) // 32, th // 8),
                    (32, 8),
                    *args,
                    np.int64(mi),
                    np.int64(ROW0),
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
                return (eb - ea) / 1e3

            t_end = time.perf_counter() + 0.5  # warm to boost clock
            while time.perf_counter() < t_end:
                launch(real, th=64, tw=w)
            res = {}
            for name, args in (("real", real), ("nan", nan), ("real2", real), ("nan2", nan)):
                ts = [launch(args) for _ in range(5)]
                res[name] = statistics.median(ts[1:])
            n_real = int(_n_max(gpu, bu, h, w))
            rec = dict(
                tier=tier,
                kernel=kern,
                tau_real_ns=round(min(res["real"], res["real2"]) / mi * 1e9, 1),
                tau_nan_ns=round(min(res["nan"], res["nan2"]) / mi * 1e9, 1),
                nan_over_real=round(
                    min(res["nan"], res["nan2"]) / min(res["real"], res["real2"]), 3
                ),
                real_tile_max_n=n_real,
            )
            out.append(rec)
            print(rec, flush=True)
    (s5lib.HERE / "results" / "nan_tau.json").write_text(json.dumps(out, indent=1))


def gen_word(tier: str) -> str:
    return s5lib.gen_s5.TIERS[tier][0]


def _n_max(gpu, bu, h, w) -> int:
    u = np.empty((h, w), np.uint32)
    s5lib._dtoh(u, bu, u.nbytes)  # layer 0 = escape counts (last launch was NaN: all max_iter)
    return int(u[ROW0 : ROW0 + 8, :1024].max())


if __name__ == "__main__":
    main()

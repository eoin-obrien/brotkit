"""Q3/Q4 throughput on the GPU: Giter/s at 3840x2160, max_iter 1000, S3's seahorse view.

Giter/s = sum of escape counts (pixel-iterations actually executed) / kernel wall time.
Launches are split into row bands of ~0.25 s (WSL2/WDDM watchdog). Each config: compile and
one full warm-up render, then the MEDIAN of >= 5 timed renders (a render that takes < 0.2 s is
repeated until it does and averaged). A background sampler records SM clock and temperature
during every render; the run is flagged if the per-render median clock varies > 5 %.

Before every config the GPU is checked: if utilisation > 5 % or memory used has grown more
than 1.5 GiB above the start-of-run baseline (plus our own allocation), the script stops and
says so instead of measuring (maintainer's rule, S4 brief).

usage: uv run q3_throughput.py main|caps|probe   -> results/q3_<set>.json (incremental)
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
W, H, MAX_ITER = 3840, 2160, 1000
CENTRE = (-0.743643887037151, 0.131825904205330)  # S3 seahorse valley
WIDTH = 0.05
BLOCKS = {128: (32, 4), 256: (32, 8)}


def smi(fields: str) -> list[str]:
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True
    )
    return [f.strip() for f in out.strip().split(",")]


class Sampler(threading.Thread):
    """Polls SM clock / temperature / utilisation every ~100 ms."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, int, int, int]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                clk, temp, util = smi("clocks.sm,temperature.gpu,utilization.gpu")
                self.samples.append((time.perf_counter(), int(clk), int(temp), int(util)))
            except Exception:
                pass
            time.sleep(0.1)

    def window(self, t0: float, t1: float) -> tuple[float, float]:
        inside = [s for s in self.samples if t0 <= s[0] <= t1]
        if not inside:  # render shorter than the poll interval: nearest sample after t0
            after = [s for s in self.samples if s[0] >= t0] or self.samples[-1:]
            inside = after[:1]
        return statistics.median(s[1] for s in inside), statistics.median(s[2] for s in inside)


class Busy(SystemExit):
    pass


def gpu_idle_check(baseline_mib: int, own_mib: int, label: str) -> dict:
    time.sleep(
        1.5
    )  # utilization.gpu averages over the last sample period: let our own work age out
    utils = []
    # 8 samples over ~1.2 s; pause on SUSTAINED load (median > 5 %), as agreed with the maintainer.
    # The WSL2 host's desktop compositor gives single-sample spikes of 5-30 % at the 210 MHz idle
    # clock with no compute process on the card; the per-render clock check still flags contention.
    for _ in range(8):
        u, used = smi("utilization.gpu,memory.used")
        utils.append(int(u))
        time.sleep(0.15)
    used_i = int(used)
    rec = dict(
        util_max=max(utils),
        util_median=statistics.median(utils),
        util_samples=utils,
        mem_used_mib=used_i,
        baseline_mib=baseline_mib,
    )
    if statistics.median(utils) > 5 or used_i - baseline_mib - own_mib > 1536:
        raise Busy(
            f"PAUSED before {label}: GPU busy (utilisation {utils}%, memory used {used_i} MiB, "
            f"baseline {baseline_mib} MiB + ours ~{own_mib} MiB). Not measuring."
        )
    return rec


def view_args(tier: str):
    import gen
    import mpmath as mp

    mp.mp.dps = 50
    word, nw, _ = gen.TIERS[tier]
    dt = np.dtype(word).type

    def words(x):
        ws, r = [], mp.mpf(x)
        for _ in range(nw):
            w = dt(float(r))
            ws.append(w)
            r = r - mp.mpf(float(w))
        return ws

    return (*words(CENTRE[0]), *words(CENTRE[1]), dt(WIDTH / W))


def time_config(k, args, block, out_f, out_u, sampler, runs=5):
    from numba_cuda_mlir import cuda

    bx, by = block

    def render(band):
        for r0 in range(0, H, band):
            nb = min(band, H - r0)
            k[(W // bx, (nb + by - 1) // by), (bx, by)](
                *args, np.int32(MAX_ITER), np.int32(r0), out_f, out_u
            )
            cuda.synchronize()

    # compile + band sizing on one band of 32 rows at mid-height
    k[(W // bx, 32 // by), (bx, by)](*args, np.int32(MAX_ITER), np.int32(H // 2), out_f, out_u)
    cuda.synchronize()
    t0 = time.perf_counter()
    k[(W // bx, 32 // by), (bx, by)](*args, np.int32(MAX_ITER), np.int32(H // 2), out_f, out_u)
    cuda.synchronize()
    per32 = time.perf_counter() - t0
    band = int(max(32, min(H, 32 * (0.25 / max(per32, 1e-6)))) // 32 * 32)
    t0 = time.perf_counter()
    render(band)  # full warm-up render
    warm = time.perf_counter() - t0
    reps = max(1, math.ceil(0.2 / max(warm, 1e-6)))
    t_end = time.perf_counter() + 1.0  # spin >= 1 s more so the SM clock settles before timing
    while time.perf_counter() < t_end:
        render(band)
    times, clocks, temps = [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(reps):
            render(band)
        t1 = time.perf_counter()
        times.append((t1 - t0) / reps)
        c, tp = sampler.window(t0, t1)
        clocks.append(c)
        temps.append(tp)
    n = out_u.copy_to_host()[0].astype(np.int64)
    it = int(n.sum())
    med = statistics.median(times)
    spread = (max(clocks) - min(clocks)) / max(clocks) if clocks else 0.0
    return dict(
        seconds_median=round(med, 5),
        seconds_all=[round(t, 5) for t in times],
        reps_per_run=reps,
        pixel_iterations=it,
        giter_s=round(it / med / 1e9, 3),
        mean_iter=round(it / (W * H), 2),
        interior_frac=round(float((n == MAX_ITER).mean()), 4),
        band_rows=band,
        launches=-(-H // band),
        sm_clock_mhz=clocks,
        temp_c=temps,
        clock_spread=round(spread, 4),
        clock_flag=spread > 0.05,
    )


def configs(which: str):
    tiers5 = ("f32", "df64", "f64", "dd", "qf")
    if which == "probe":
        return [(t, "A", None, 256) for t in ("f32", "dd", "qf")]
    if which == "main":
        out = []
        for bs in (256, 128):
            for t in tiers5:
                for kern in ("A", "B", "C", "F6", "S2"):
                    out.append((t, kern, None, bs))
            out.append(("f32rn", "A", None, bs))
            out.append(("qfb", "C", None, bs))
        return out
    if which == "caps":
        out = []
        for t in ("dd", "qf", "df64", "f64"):
            for kern in ("C", "F6"):
                for cap in (32, 64, 96, 128):
                    out.append((t, kern, {"max_registers": cap}, 256))
            out.append(
                (t, "F6", {"launch_bounds": (256, 3)}, 256)
            )  # ask for 3 blocks/SM (<=85 regs)
            out.append((t, "F6", {"launch_bounds": (256, 4)}, 256))  # 4 blocks/SM (<=64 regs)
        return out
    raise SystemExit(f"unknown set {which}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "probe"
    sys.path.insert(0, str(HERE))
    import guard

    baseline = int(smi("memory.used")[0])
    guard.check(f"q3 {which}")
    import gen
    from numba_cuda_mlir import cuda

    from brotkit.backends.cuda import resource_usage
    from brotkit.backends.resources import occupancy

    out_f = cuda.device_array((gen.NLF, H, W), np.float32)
    out_u = cuda.device_array((gen.NLU, H, W), np.uint32)
    own_mib = 600 + (out_f.nbytes + out_u.nbytes) // 2**20  # context + outputs, roughly
    path = HERE / "results" / f"q3_{which}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res.setdefault(
        "_env",
        dict(
            gpu=smi("name,driver_version,clocks.max.sm,power.limit"),
            view=dict(centre=CENTRE, width=WIDTH, W=W, H=H, max_iter=MAX_ITER),
            method="median of 5 renders after warm-up; Giter/s = sum(n)/time",
        ),
    )
    sampler = Sampler()
    sampler.start()
    try:
        for tier, kern, opts, bs in configs(which):
            key = f"{tier}/{kern}/{json.dumps(opts, sort_keys=True) if opts else 'default'}/{bs}"
            if key in res:
                continue
            # Maintainer's choice: wait out desktop bursts (20 s re-checks, <= 10 min) rather
            # than exit; nothing is measured while the check fails, and each wait is recorded.
            waits: list[str] = []
            deadline = time.monotonic() + 600
            while True:
                try:
                    idle = gpu_idle_check(baseline, own_mib, key)
                    break
                except Busy as e:
                    if time.monotonic() > deadline:
                        raise
                    waits.append(str(e))
                    print(f"waiting 20 s: {e}", flush=True)
                    time.sleep(20)
            idle["waits"] = waits
            guard.check(key)
            sig = gen.signature(tier)
            k = gen.load(tier, kern, jit_options=opts)
            ru = resource_usage(
                k,
                sig,
                (8, 6),
                max_registers=(opts or {}).get("max_registers"),
                launch_bounds=(opts or {}).get("launch_bounds"),
            )
            occ = occupancy(ru, bs)
            row = dict(
                tier=tier,
                kernel=kern,
                features=gen.FEATURE_COUNT[kern],
                opts=opts,
                block=bs,
                regs=ru.registers,
                spill_st=ru.spill_store_bytes,
                spill_ld=ru.spill_load_bytes,
                stack=ru.stack_bytes,
                occupancy=round(occ.fraction, 4),
                occ_limiter=str(occ.limiter),
                pre_check=idle,
            )
            row |= time_config(k, view_args(tier), BLOCKS[bs], out_f, out_u, sampler)
            res[key] = row
            path.write_text(json.dumps(res, indent=1))
            print(
                f"{key:44} regs={row['regs']:3} spill={row['spill_st']}/{row['spill_ld']} "
                f"occ={row['occupancy']:.3f} {row['giter_s']:9.3f} Giter/s  "
                f"t={row['seconds_median']:.4f}s mean_iter={row['mean_iter']} "
                f"clk={min(row['sm_clock_mhz'])}-{max(row['sm_clock_mhz'])} T={max(row['temp_c'])}"
                f"{'  CLOCK-FLAG' if row['clock_flag'] else ''}",
                flush=True,
            )
    except Busy as e:
        print(str(e), flush=True)
        sys.exit(3)
    finally:
        sampler.stop.set()


if __name__ == "__main__":
    main()

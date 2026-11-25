"""Clock ramp after idle, and how much a warm-up kernel launched before host work hides (throwaway).

Each trial waits for the GPU to drop to its idle clock, then:
    launch the warm kernel on a side stream for H ms (sized at the idle clock), sleep H ms
    (standing in for the host's render setup: kernel lookup, codegen, buffers), then queue
    `PROBES` identical latency probes (qf kernel C, NaN coordinates, 1024 px, ~1 ms warm)
    back to back on the render stream and time each with events.
Slowdown is each probe's time over the warm median. Reported per head start H: the first
probe's slowdown, the mean over the first 10, the GPU time until a probe runs under 1.2x warm,
and the cold penalty (sum of probe time over warm, in ms).
Also: the host setup a real render does before its first launch (memory-hit and disk-hit kernel
lookup, source generation, buffers), to say how long a head start the host work gives for free.

usage: .venv/bin/python warmup.py <cache root> [reps] -> results/warmup.json
"""

from __future__ import annotations

import itertools
import json
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np
import s5lib

PROBES = 300
HEADS_MS = (0, 5, 20, 50, 100, 200)
TIER = "qf"
PROBE_ITERS = 600  # ~1 ms warm at qf's 1.6 us tau


def sm_mhz() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.split()[0])


def wait_cold(limit_mhz: int = 300, timeout_s: float = 180.0) -> float:
    """Wait for two consecutive idle-clock readings, then 2 s more. Returns seconds waited."""
    t0 = time.monotonic()
    ok = 0
    while ok < 2:
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"clock stayed above {limit_mhz} MHz for {timeout_s} s")
        ok = ok + 1 if sm_mhz() <= limit_mhz else 0
        time.sleep(0.5)
    time.sleep(2.0)
    return time.monotonic() - t0


def probes(gpu: s5lib.Gpu, k, args, sf, su, n: int) -> list[float]:
    from cuda.core import EventOptions

    opt = EventOptions(timing_enabled=True)
    stream = gpu.backend.stream
    evs = [stream.record(options=opt)]
    for _ in range(n):
        k.launch(
            (4, 1),
            (32, 8),
            *args,
            np.int64(PROBE_ITERS),
            np.int64(0),
            np.int64(0),
            np.int64(8),
            np.int64(128),
            sf,
            su,
            np.int64(8),
            np.int64(128),
            np.int64(128),
        )
        evs.append(stream.record(options=opt))
    evs[-1].sync()
    return [(b - a) for a, b in itertools.pairwise(evs)]  # ms


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    s5lib.s4_on_path()
    import q3_throughput as q
    from cuda.core import EventOptions

    gpu = s5lib.Gpu.open(root)
    baseline = int(q.smi("memory.used")[0])
    import gen_s5

    k = gpu.kernel(TIER, "C")
    word = np.dtype(gen_s5.TIERS[TIER][0]).type
    a0 = s5lib.view_args(TIER, "interior", 3840)
    args = (*(word(np.nan) for _ in a0[:-1]), a0[-1])
    sf = gpu.alloc(gen_s5.NLF * 8 * 128 * 4)
    su = gpu.alloc(gen_s5.NLU * 8 * 128 * 4)

    # Host setup a render does before its first launch.
    prep: dict[str, float] = {}
    t = time.perf_counter()
    gen_s5.emit(TIER, "C")
    prep["emit_source_ms"] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    gpu.cache.get(gpu.spec(TIER, "C"))
    prep["lookup_memory_hit_ms"] = (time.perf_counter() - t) * 1e3
    gpu.cache.clear_memory()
    t = time.perf_counter()
    gpu.cache.get(gpu.spec(TIER, "C"))
    prep["lookup_disk_hit_ms"] = (time.perf_counter() - t) * 1e3
    nf, nu = gen_s5.NLF * 3840 * 2160 * 4, gen_s5.NLU * 3840 * 2160 * 4
    t = time.perf_counter()
    bufs = (gpu.alloc(nf), gpu.alloc(nu))
    gpu.backend.stream.sync()
    prep["alloc_4k_layers_ms"] = (time.perf_counter() - t) * 1e3
    del bufs
    print("prep", {kk: round(v, 3) for kk, v in prep.items()}, flush=True)

    # Warm reference and warm-kernel sizing (clock read under load).
    s5lib.wait_idle(q, baseline, "warmup")
    gpu.warm_iters_per_ms = 1  # placeholder so launch_warm runs for the load below
    import cuda.core as cc

    sms = gpu.backend.device.properties.multiprocessor_count
    cfg = cc.LaunchConfig(grid=sms, block=32)
    cc.launch(
        gpu.warm_stream,
        cfg,
        gpu.warm._kernel,
        np.int64(20_000_000),
        np.uintp(int(gpu.warm_out.handle)),
    )
    time.sleep(0.3)
    load_mhz = sm_mhz()
    gpu.warm_stream.sync()
    gpu.calibrate_warm(idle_mhz=210, now_mhz=load_mhz)
    probes(gpu, k, args, sf, su, 200)  # keep it warm
    warm = statistics.median(probes(gpu, k, args, sf, su, 100))
    print(
        f"load clock {load_mhz} MHz, warm probe {warm:.4f} ms, "
        f"warm iters/ms at idle {gpu.warm_iters_per_ms}",
        flush=True,
    )

    out: dict[str, object] = {
        "tier": TIER,
        "probe_iters": PROBE_ITERS,
        "probes": PROBES,
        "warm_probe_ms": warm,
        "load_mhz": load_mhz,
        "prep": prep,
        "trials": [],
    }
    trials: list[dict[str, object]] = out["trials"]  # type: ignore[assignment]
    opt = EventOptions(timing_enabled=True)
    for rep in range(reps):
        for head in HEADS_MS:
            s5lib.wait_idle(q, baseline, f"warmup H={head}")
            waited = wait_cold()
            mhz0 = sm_mhz()
            wa = gpu.warm_stream.record(options=opt)
            t0 = time.perf_counter()
            gpu.launch_warm(head)
            wb = gpu.warm_stream.record(options=opt)
            time.sleep(head / 1e3)
            t_issue = time.perf_counter() - t0
            ts = probes(gpu, k, args, sf, su, PROBES)
            gpu.warm_stream.sync()
            slow = [x / warm for x in ts]
            busy = np.cumsum(ts)
            under = next((float(busy[i]) for i, s in enumerate(slow) if s < 1.2), float("nan"))
            tr = {
                "rep": rep,
                "head_ms": head,
                "waited_s": round(waited, 1),
                "mhz_before": mhz0,
                "warm_kernel_gpu_ms": round((wb - wa), 3) if head else 0.0,
                "host_head_ms": round(t_issue * 1e3, 2),
                "first": round(slow[0], 2),
                "mean_first10": round(float(np.mean(slow[:10])), 2),
                "busy_ms_until_1_2": round(under, 1),
                "penalty_ms": round(float(sum(ts) - warm * len(ts)), 1),
                "slowdowns": [round(x, 3) for x in slow],
            }
            trials.append(tr)
            print(
                f"rep {rep} H={head:3} ms  mhz0={mhz0} warmk={tr['warm_kernel_gpu_ms']} ms "
                f"first={tr['first']}x first10={tr['mean_first10']}x "
                f"until<1.2x={tr['busy_ms_until_1_2']} ms penalty={tr['penalty_ms']} ms",
                flush=True,
            )
            (s5lib.HERE / "results" / "warmup.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

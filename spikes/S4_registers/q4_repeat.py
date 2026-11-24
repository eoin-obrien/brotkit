"""Q4 follow-up: interleaved A/B repeats, to separate real cap effects from run-to-run drift.

The cap sweep showed f64/C at the default build and at max_registers=64 differing by ~20 %
although their SASS is byte-identical, so a single cross-run comparison isn't trustworthy on
this shared card. Here each pair is timed alternately (A B A B ...) for ROUNDS rounds, with the
same idle check and timing method as q3_throughput.py.

usage: uv run python q4_repeat.py -> results/q4_repeat.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np
import q3_throughput as q

ROUNDS = 3
PAIRS = [  # (tier, kernel, opts_a, opts_b)
    ("f64", "C", None, {"max_registers": 64}),  # identical SASS: the noise floor
    ("dd", "C", None, {"max_registers": 128}),  # 59 -> 66 registers
    ("qf", "C", None, {"max_registers": 128}),  # 64 -> 117 registers
    ("qf", "F6", None, {"max_registers": 128}),  # 72 -> 118
    ("dd", "F6", None, {"max_registers": 64}),  # 72 -> 64, small spill
    ("df64", "F6", None, {"max_registers": 32}),  # sweep ran at 1800 MHz
]


def main() -> None:
    sys.path.insert(0, str(q.HERE))
    import gen
    import guard
    from numba_cuda_mlir import cuda

    baseline = int(q.smi("memory.used")[0])
    guard.check("q4 repeat")
    out_f = cuda.device_array((gen.NLF, q.H, q.W), np.float32)
    out_u = cuda.device_array((gen.NLU, q.H, q.W), np.uint32)
    own = 600 + (out_f.nbytes + out_u.nbytes) // 2**20
    sampler = q.Sampler()
    sampler.start()
    res: dict = {}
    try:
        for tier, kern, oa, ob in PAIRS:
            ks = {n: gen.load(tier, kern, jit_options=o) for n, o in (("a", oa), ("b", ob))}
            rec: dict = {"opts_a": oa, "opts_b": ob, "a": [], "b": []}
            for _ in range(ROUNDS):
                for n in ("a", "b"):
                    deadline = time.monotonic() + 600
                    while True:
                        try:
                            q.gpu_idle_check(baseline, own, f"{tier}/{kern}/{n}")
                            break
                        except q.Busy as e:
                            if time.monotonic() > deadline:
                                raise
                            print(f"waiting 20 s: {e}", flush=True)
                            time.sleep(20)
                    r = q.time_config(
                        ks[n], q.view_args(tier), q.BLOCKS[256], out_f, out_u, sampler
                    )
                    rec[n].append(
                        dict(giter_s=r["giter_s"], clk=r["sm_clock_mhz"], flag=r["clock_flag"])
                    )
            ga = [x["giter_s"] for x in rec["a"]]
            gb = [x["giter_s"] for x in rec["b"]]
            rec["median_a"], rec["median_b"] = statistics.median(ga), statistics.median(gb)
            rec["b_over_a"] = rec["median_b"] / rec["median_a"]
            rec["spread_a"] = (max(ga) - min(ga)) / min(ga)
            rec["spread_b"] = (max(gb) - min(gb)) / min(gb)
            key = f"{tier}/{kern}/{json.dumps(ob, sort_keys=True)}"
            res[key] = rec
            print(
                f"{key:36} a={[round(x, 3) for x in ga]} b={[round(x, 3) for x in gb]} "
                f"b/a={rec['b_over_a']:.3f}",
                flush=True,
            )
            (q.HERE / "results" / "q4_repeat.json").write_text(json.dumps(res, indent=1))
    finally:
        sampler.stop.set()


if __name__ == "__main__":
    main()

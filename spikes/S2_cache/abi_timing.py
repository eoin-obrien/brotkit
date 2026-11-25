"""ABI cost, part 2: throughput struct vs ptr ABI on kernel C, alternating (throwaway).

Uses S4's q3_throughput timing (idle check, warm-up, >= 1 s spin, median of 5, clock sampler)
and S4's q4_repeat interleaving: a b a b a b per tier, same process.

usage: .venv/bin/python abi_timing.py [rounds] -> results/abi_timing.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import gen_s2
import numpy as np
import s2lib

s2lib.s4_on_path()
import guard  # noqa: E402
import q3_throughput as q  # noqa: E402

KERNELS = tuple(__import__("os").environ.get("S2_KERNELS", "C").split(","))
VARIANTS = tuple(__import__("os").environ.get("S2_VARIANTS", "struct,ptr").split(","))
TIERS = tuple(__import__("os").environ.get("S2_TIERS", "f32,df64,f64,dd,qf").split(","))


def main() -> None:
    from numba_cuda_mlir import cuda

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    baseline = int(q.smi("memory.used")[0])
    guard.check("s2 abi")
    out_f = cuda.device_array((gen_s2.NLF, q.H, q.W), np.float32)
    out_u = cuda.device_array((gen_s2.NLU, q.H, q.W), np.uint32)
    own = 600 + (out_f.nbytes + out_u.nbytes) // 2**20
    sampler = q.Sampler()
    sampler.start()
    res: dict = {}
    try:
        for kern, tier in [(k, t) for k in KERNELS for t in TIERS]:

            def mk(n, tier=tier, kern=kern):
                if n == "struct":
                    return gen_s2.load(tier, kern, "struct")
                return gen_s2.PtrAdapter(gen_s2.load(tier, kern, n), tier, n)

            ks = {n: mk(n) for n in VARIANTS}
            rec: dict = {n: [] for n in VARIANTS}
            for _ in range(rounds):
                for n in VARIANTS:
                    deadline = time.monotonic() + 900
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
                        dict(
                            giter_s=r["giter_s"],
                            clk=r["sm_clock_mhz"],
                            flag=r["clock_flag"],
                            iters=r["pixel_iterations"],
                        )
                    )
            P = VARIANTS[1]
            gs = [x["giter_s"] for x in rec["struct"]]
            gp = [x["giter_s"] for x in rec[P]]
            rec["median_struct"], rec["median_ptr"] = statistics.median(gs), statistics.median(gp)
            rec["ptr_over_struct"] = rec["median_ptr"] / rec["median_struct"]
            if "ptr64" in rec:
                rec["median_ptr64"] = statistics.median(x["giter_s"] for x in rec["ptr64"])
                rec["ptr64_over_struct"] = rec["median_ptr64"] / rec["median_struct"]
                print(
                    f"  ptr64={[round(x['giter_s'], 3) for x in rec['ptr64']]} "
                    f"ptr64/struct={rec['ptr64_over_struct']:.3f}",
                    flush=True,
                )
            rec["same_iterations"] = (
                len({x["iters"] for v in rec.values() if isinstance(v, list) for x in v}) == 1
            )
            res[f"{tier}/{kern}"] = rec
            print(
                f"{tier:5}/{kern:3} {P} struct={[round(x, 3) for x in gs]} "
                f"ptr={[round(x, 3) for x in gp]} "
                f"ptr/struct={rec['ptr_over_struct']:.3f} same_iters={rec['same_iterations']}",
                flush=True,
            )
            (s2lib.RESULTS / __import__("os").environ.get("S2_OUT", "abi_timing.json")).write_text(
                json.dumps(res, indent=1)
            )
    finally:
        sampler.stop.set()


if __name__ == "__main__":
    main()

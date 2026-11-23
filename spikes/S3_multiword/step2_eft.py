"""Step 2 (and 5a/5b primitives): exactness of error-free transformations on the GPU.

usage: uv run step2_eft.py [variant=rn] [n=10000000]
"""

import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import guard


def main():
    guard.check("step2")

    import eftcheck as ec
    import gen
    import numpy as np

    variant = sys.argv[1] if len(sys.argv) > 1 else "rn"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000
    ops = ["two_sum", "fast_two_sum", "two_prod", "two_prod_dekker"]

    results = {"variant": variant, "n_per_op_base": N, "rows": []}
    with ProcessPoolExecutor(24) as pool:
        for prec in ("f64", "f32"):
            mod = gen.build(prec, variant)
            for op in ops:
                guard.check(f"{prec} {op}")
                groups = ec.inputs(prec, op, N, seed=zlib.crc32(f"{prec}{op}".encode()))
                for gname, (a, b) in groups.items():
                    if op == "two_prod_dekker" and gname in ("subnormal", "prod_edge"):
                        continue  # Dekker split has its own (tighter) domain
                    keep = ec.finite_inputs(a, b, op)
                    a, b = a[keep], b[keep]
                    t0 = time.perf_counter()
                    s, e = ec.run_gpu(mod, op, a, b)
                    bad = ec.verify(pool, op, a, b, s, e)
                    row = dict(prec=prec, op=op, group=gname, n=int(a.size))
                    if op == "two_prod":
                        dom = ec.in_domain_two_prod(a, b, prec)
                        row.update(
                            n_in_domain=int(dom.sum()),
                            fail_in_domain=int((bad & dom).sum()),
                            n_out_domain=int((~dom).sum()),
                            inexact_out_domain=int((bad & ~dom).sum()),
                        )
                        if gname == "prod_edge":
                            f = ec.FMT[prec]
                            a64, b64 = np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64))
                            off = (
                                np.maximum(np.floor(np.log2(a64)), f["emin"])
                                + np.maximum(np.floor(np.log2(b64)), f["emin"])
                                - (f["emin"] + f["p"] - 1)
                            ).astype(int)
                            row["inexact_rate_by_offset"] = {
                                int(o): round(float(bad[off == o].mean()), 4)
                                for o in range(-4, 4)
                                if (off == o).any()
                            }
                    else:
                        row.update(fail=int(bad.sum()))
                    if bad.any():
                        i = (
                            int(np.flatnonzero(bad if op != "two_prod" else bad & dom)[0])
                            if (op != "two_prod" or (bad & dom).any())
                            else int(np.flatnonzero(bad)[0])
                        )
                        row["example"] = dict(
                            a=float(a[i]).hex(),
                            b=float(b[i]).hex(),
                            s=float(s[i]).hex(),
                            e=float(e[i]).hex(),
                        )
                    row["secs"] = round(time.perf_counter() - t0, 1)
                    print(json.dumps(row), flush=True)
                    results["rows"].append(row)

    out = f"results/step2_{variant}.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

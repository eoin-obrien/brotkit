"""Step 3 (and 5a/5b dd_mul): df64 / dd composite arithmetic vs mpmath at 60 digits.

usage: uv run step3_composite.py [variant=rn] [n=1000000]
Errors are relative (normwise for complex), in units of u = 2^-48 (df64) or 2^-106 (dd).
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

OPS = {0: "add", 1: "mul", 2: "sqr", 3: "cmul", 4: "csqr", 5: "mul_dekker"}
U = {"f64": 2.0**-106, "f32": 2.0**-48}


def make_inputs(prec: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    dt = np.float64 if prec == "f64" else np.float32
    p = 53 if prec == "f64" else 24
    rng = np.random.default_rng(seed)

    def word_pairs():
        hi = rng.choice([-1.0, 1.0], n) * np.ldexp(rng.uniform(1, 2, n), rng.integers(-4, 2, n))
        hi = hi.astype(dt)
        lo = (hi.astype(np.float64) * 2.0**-p * rng.uniform(-1, 1, n)).astype(dt)
        s = (hi + lo).astype(dt)  # fast_two_sum renormalisation in the word dtype
        e = (lo - (s - hi)).astype(dt)
        return s, e

    a = np.empty((n, 4), dt)
    b = np.empty((n, 4), dt)
    for arr in (a, b):
        arr[:, 0], arr[:, 1] = word_pairs()
        arr[:, 2], arr[:, 3] = word_pairs()
    return a, b


def _ref_chunk(args):
    import mpmath as mp

    fn, a, b, out = args
    mp.mp.dps = 60
    errs = []
    for x, y, r in zip(a.tolist(), b.tolist(), out.tolist(), strict=True):
        ar = mp.mpf(x[0]) + mp.mpf(x[1])
        ai = mp.mpf(x[2]) + mp.mpf(x[3])
        br = mp.mpf(y[0]) + mp.mpf(y[1])
        bi = mp.mpf(y[2]) + mp.mpf(y[3])
        if fn == 0:
            ref, got = ar + br, mp.mpf(r[0]) + mp.mpf(r[1])
        elif fn in (1, 5):
            ref, got = ar * br, mp.mpf(r[0]) + mp.mpf(r[1])
        elif fn == 2:
            ref, got = ar * ar, mp.mpf(r[0]) + mp.mpf(r[1])
        elif fn == 3:
            ref = mp.mpc(ar, ai) * mp.mpc(br, bi)
            got = mp.mpc(mp.mpf(r[0]) + mp.mpf(r[1]), mp.mpf(r[2]) + mp.mpf(r[3]))
        else:
            ref = mp.mpc(ar, ai) ** 2
            got = mp.mpc(mp.mpf(r[0]) + mp.mpf(r[1]), mp.mpf(r[2]) + mp.mpf(r[3]))
        errs.append(float(abs(got - ref) / abs(ref)) if ref != 0 else float(abs(got)))
    return errs


def run(variant: str, n: int, fns=(0, 1, 2, 3, 4, 5), pool=None, cc_label="sm_86"):
    import gen
    import guard
    from numba_cuda_mlir import cuda

    rows = []
    for prec in ("f64", "f32"):
        guard.check(f"step3 {prec}")
        mod = gen.build(prec, variant)
        a, b = make_inputs(prec, n, seed=3 if prec == "f64" else 4)
        da, db = cuda.to_device(a), cuda.to_device(b)
        for fn in fns:
            dout = cuda.device_array_like(a)
            mod.k_wop[(n + 255) // 256, 256](fn, da, db, dout)
            cuda.synchronize()
            out = dout.copy_to_host()
            idx = np.array_split(np.arange(n), 96)
            errs = (
                np.concatenate(
                    [
                        np.asarray(e)
                        for e in pool.map(_ref_chunk, [(fn, a[i], b[i], out[i]) for i in idx])
                    ]
                )
                / U[prec]
            )
            row = dict(
                fmt="dd" if prec == "f64" else "df64",
                op=OPS[fn],
                variant=variant,
                n=n,
                max_u=float(errs.max()),
                median_u=float(np.median(errs)),
                p99_u=float(np.percentile(errs, 99)),
                over_16u=int((errs > 16).sum()),
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "rn"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    with ProcessPoolExecutor(24) as pool:
        rows = run(variant, n, pool=pool)
    with open(f"results/step3_{variant}.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()

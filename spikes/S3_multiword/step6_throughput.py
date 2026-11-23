"""Step 6: rough throughput of z^2+c escape time at 3840x2160, max_iter 1000, one view.

Giter/s = sum of escape counts (pixel-iterations actually executed) / kernel wall time.
Launches are split into row bands of ~0.25 s (WSL2/WDDM watchdog); timing covers all bands
with a synchronize after each. Best of 3 after a warm-up.

usage: uv run step6_throughput.py
"""

import json
import time

import guard
import numpy as np

W, H, MAX_ITER = 3840, 2160, 1000
CENTRE = (-0.743643887037151, 0.131825904205330)  # seahorse valley
WIDTH = 0.05  # complex-plane width; spacing ~1.3e-5, inside f32's range


def main():
    guard.check("step6")
    import gen
    import mpmath as mp
    from numba_cuda_mlir import cuda

    mp.mp.dps = 40
    dx = WIDTH / W
    cr, ci = mp.mpf(CENTRE[0]), mp.mpf(CENTRE[1])

    def words(x, dt):
        hi = dt(float(x))
        return hi, dt(float(x - mp.mpf(float(hi))))

    f32n, f64n = gen.build("f32", "naive"), gen.build("f64", "naive")
    df, dd = gen.build("f32", "rn"), gen.build("f64", "rn")
    cases = {
        "f32": (
            f32n.k_mandel_plain_view,
            (np.float32(CENTRE[0]), np.float32(CENTRE[1]), np.float32(dx)),
        ),
        "df64": (
            df.k_mandel_w_view,
            (*words(cr, np.float32), *words(ci, np.float32), np.float32(dx)),
        ),
        "f64": (
            f64n.k_mandel_plain_view,
            (np.float64(CENTRE[0]), np.float64(CENTRE[1]), np.float64(dx)),
        ),
        "dd": (
            dd.k_mandel_w_view,
            (*words(cr, np.float64), *words(ci, np.float64), np.float64(dx)),
        ),
    }
    counts = cuda.device_array((H, W), np.int32)
    rows = {}
    for name, (k, args) in cases.items():
        guard.check(f"step6 {name}")
        # warm-up compile + band sizing on 16 rows
        k[(W // 16, 1), (16, 16)](*args, np.int32(MAX_ITER), np.int32(H // 2), counts)
        cuda.synchronize()
        t0 = time.perf_counter()
        k[(W // 16, 1), (16, 16)](*args, np.int32(MAX_ITER), np.int32(H // 2), counts)
        cuda.synchronize()
        per16 = time.perf_counter() - t0
        band = int(max(16, min(H, 16 * (0.25 / max(per16, 1e-6)))) // 16 * 16)
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            for r0 in range(0, H, band):
                nb = min(band, H - r0)
                k[(W // 16, (nb + 15) // 16), (16, 16)](
                    *args, np.int32(MAX_ITER), np.int32(r0), counts
                )
                cuda.synchronize()
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        c = counts.copy_to_host()
        it = int(c.astype(np.int64).sum())
        rows[name] = dict(
            seconds=round(best, 4),
            pixel_iterations=it,
            giter_s=round(it / best / 1e9, 3),
            mean_iter=round(it / (W * H), 1),
            interior_frac=round(float((c == MAX_ITER).mean()), 4),
            band_rows=band,
            launches=(H + band - 1) // band,
        )
        print(name, json.dumps(rows[name]), flush=True)
    base = rows["f64"]["giter_s"]
    for r in rows.values():
        r["vs_f64"] = round(r["giter_s"] / base, 3)
    rows["_ratios"] = dict(
        f32_over_df64=round(rows["f32"]["giter_s"] / rows["df64"]["giter_s"], 2),
        df64_over_f64=round(rows["df64"]["giter_s"] / rows["f64"]["giter_s"], 2),
        f64_over_dd=round(rows["f64"]["giter_s"] / rows["dd"]["giter_s"], 2),
        df64_over_dd=round(rows["df64"]["giter_s"] / rows["dd"]["giter_s"], 2),
    )
    print(json.dumps(rows["_ratios"]))
    with open("results/step6.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()

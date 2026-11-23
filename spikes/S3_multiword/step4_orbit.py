"""Step 4: deep orbit test. dd at pixel spacing 1e-20, df64 at 1e-12 (df64 can't reach 1e-20).

Centre: Misiurewicz point M(4,1) (preperiod 4, period 1, |multiplier| ~ 1.328), solved with
mpmath findroot at 90 digits. 64x64 grid, max_iter 5000, escape when |z|^2 > 4 (tested before
the update; count = completed iterations). The CPU reference iterates the *exact* value of the
multi-word c the GPU received, so input rounding is not counted as error.

usage: uv run step4_orbit.py
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CENTRE_RE = "-0.101096363845622161025785445738622565463805442826253483876931"
CENTRE_IM = "0.956286510809141500771096057729977435809833336510529170034314"
N = 64
MAX_ITER = 5000


def _ref(args):
    """Escape count and distance estimate (in pixels) at `dps` digits for exact c = sum(words)."""
    import mpmath as mp

    re_words, im_words, dps, spacing = args
    mp.mp.dps = dps
    cr = mp.fsum(mp.mpf(w) for w in re_words)
    ci = mp.fsum(mp.mpf(w) for w in im_words)
    zr = zi = mp.mpf(0)
    dzr = dzi = mp.mpf(0)
    n = 0
    while n < MAX_ITER:
        x2, y2 = zr * zr, zi * zi
        if x2 + y2 > 4:
            break
        # dz <- 2 z dz + 1
        dzr, dzi = 2 * (zr * dzr - zi * dzi) + 1, 2 * (zr * dzi + zi * dzr)
        zr, zi = x2 - y2 + cr, 2 * zr * zi + ci
        n += 1
    if n == MAX_ITER:
        return n, None
    az = mp.sqrt(zr * zr + zi * zi)
    adz = mp.sqrt(dzr * dzr + dzi * dzi)
    de = float(az * mp.log(az) / adz) / spacing if adz != 0 else None
    return n, de


def pixel_c(spacing: str):
    import mpmath as mp

    mp.mp.dps = 80
    s = mp.mpf(spacing)
    cr, ci = mp.mpf(CENTRE_RE), mp.mpf(CENTRE_IM)
    re = [[cr + (x - (N - 1) / mp.mpf(2)) * s for x in range(N)] for _ in range(N)]
    im = [[ci + ((N - 1) / mp.mpf(2) - y) * s for _ in range(N)] for y in range(N)]
    return re, im


def split_words(vals, dt, nwords=2):
    import mpmath as mp

    out = np.zeros((nwords, N, N), dt)
    for y in range(N):
        for x in range(N):
            r = vals[y][x]
            for k in range(nwords):
                w = dt(float(r))
                out[k, y, x] = w
                r = r - mp.mpf(float(w))
    return out


def gpu_counts(kernel, *arrays):
    from numba_cuda_mlir import cuda

    counts = cuda.device_array((N, N), np.int32)
    dev = [cuda.to_device(np.ascontiguousarray(a)) for a in arrays]
    kernel[(N // 16, N // 16), (16, 16)](*dev, np.int32(MAX_ITER), np.int32(0), counts)
    cuda.synchronize()
    return counts.copy_to_host()


def nudge_sensitive(pool, words_re, words_im, ref_n, u, spacing):
    """Does moving c by one unit of the format's precision (u*|c|) change the 60-digit count?"""
    jobs = []
    for dr, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        jobs.append(([*words_re, dr * u], [*words_im, di * u], 60, spacing))
    return any(r[0] != ref_n for r in pool.map(_ref, jobs))


def case(pool, label, spacing, wdt, u, mod_w, mod_plain_ok, mod_plain_fail, ok_dt, fail_dt):
    import guard

    guard.check(f"step4 {label}")
    re, im = pixel_c(spacing)
    wr, wi = split_words(re, wdt), split_words(im, wdt)
    t0 = time.perf_counter()
    cw = gpu_counts(mod_w.k_mandel_w, wr[0], wr[1], wi[0], wi[1])
    sp = float(spacing)
    jobs = [
        ([float(wr[0, y, x]), float(wr[1, y, x])], [float(wi[0, y, x]), float(wi[1, y, x])], 60, sp)
        for y in range(N)
        for x in range(N)
    ]
    ref = pool.map(_ref, jobs, chunksize=16)
    ref = list(ref)
    ref_n = np.array([r[0] for r in ref]).reshape(N, N)
    de = np.array([np.nan if r[1] is None else r[1] for r in ref]).reshape(N, N)
    mism = np.argwhere(cw != ref_n)
    details = []
    for y, x in mism:
        j = y * N + x
        hi = _ref((jobs[j][0], jobs[j][1], 120, sp))
        c_abs = abs(complex(jobs[j][0][0], jobs[j][1][0]))
        sens = nudge_sensitive(pool, jobs[j][0], jobs[j][1], int(ref_n[y, x]), u * c_abs, sp)
        details.append(
            dict(
                px=[int(x), int(y)],
                gpu=int(cw[y, x]),
                ref60=int(ref_n[y, x]),
                ref120=int(hi[0]),
                de_px=None if np.isnan(de[y, x]) else float(de[y, x]),
                one_ulp_nudge_changes_count=bool(sens),
            )
        )
    out = dict(
        label=label,
        spacing=spacing,
        n_pixels=N * N,
        identical=int((cw == ref_n).sum()),
        identical_frac=float((cw == ref_n).mean()),
        mismatches=details,
        counts_range=[int(ref_n.min()), int(ref_n.max())],
        distinct_counts_ref=int(np.unique(ref_n).size),
        de_px_median_all=float(np.nanmedian(de)),
        de_px_p5_all=float(np.nanpercentile(de, 5)),
        max_iter_pixels=int((ref_n == MAX_ITER).sum()),
    )
    # plain-precision grids
    for tag, mod, dt in (("agree", mod_plain_ok, ok_dt), ("fail", mod_plain_fail, fail_dt)):
        if mod is None:
            continue
        pr = np.array([[dt(float(v)) for v in row] for row in re], dt)
        pi = np.array([[dt(float(v)) for v in row] for row in im], dt)
        cp = gpu_counts(mod.k_mandel_plain, pr, pi)
        out[f"plain_{tag}"] = dict(
            dtype=np.dtype(dt).name,
            distinct_c=int(np.unique(pr + 1j * pi).size),
            distinct_counts=int(np.unique(cp).size),
            identical_to_ref=int((cp == ref_n).sum()),
            identical_to_multiword=int((cp == cw).sum()),
            mismatch_vs_multiword_de_px=[
                None if np.isnan(de[y, x]) else round(float(de[y, x]), 3)
                for y, x in np.argwhere(cp != cw)
            ][:200],
        )
    out["secs"] = round(time.perf_counter() - t0, 1)
    np.save(f"results/step4_{label}_counts.npy", np.stack([cw, ref_n]))
    return out


def main():
    import guard

    guard.check("step4")
    import gen

    dd, df = gen.build("f64", "rn"), gen.build("f32", "rn")
    f64n, f32n = gen.build("f64", "naive"), gen.build("f32", "naive")  # plain kernels only
    res = {"centre": [CENTRE_RE, CENTRE_IM], "max_iter": MAX_ITER, "grid": N}
    with ProcessPoolExecutor(24) as pool:
        # dd at 1e-20; plain f64 should collapse (fail). No "agree" grid exists at this depth.
        res["dd"] = case(
            pool, "dd_1e-20", "1e-20", np.float64, 2.0**-106, dd, None, f64n, None, np.float64
        )
        print(json.dumps({k: v for k, v in res["dd"].items() if k != "mismatches"}), flush=True)
        # df64 at 1e-12; f32 should collapse (fail); f64 is the agreement check.
        res["df64"] = case(
            pool,
            "df64_1e-12",
            "1e-12",
            np.float32,
            2.0**-48,
            df,
            f64n,
            f32n,
            np.float64,
            np.float32,
        )
        print(json.dumps({k: v for k, v in res["df64"].items() if k != "mismatches"}), flush=True)
    with open("results/step4.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

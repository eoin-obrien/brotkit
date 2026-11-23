"""Shared helpers for steps 2 and 5: inputs, GPU launch, exact verification (throwaway)."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np

FN = {"two_sum": 0, "fast_two_sum": 1, "two_prod": 2, "two_prod_dekker": 3}
FMT = {  # p (mantissa bits incl. hidden), emin, emax, smallest subnormal exponent
    "f64": dict(dt=np.float64, p=53, emin=-1022, emax=1023),
    "f32": dict(dt=np.float32, p=24, emin=-126, emax=127),
}


def _rand_float(rng, n, dt, elo, ehi):
    """Random sign * full random mantissa * 2^U[elo, ehi]."""
    m = rng.uniform(1.0, 2.0, n)
    e = rng.integers(elo, ehi + 1, n)
    s = rng.choice([-1.0, 1.0], n)
    return (s * np.ldexp(m, e)).astype(dt)


def _subnormals(rng, n, prec):
    f = FMT[prec]
    if prec == "f64":
        bits = rng.integers(1, 2**52, n, dtype=np.uint64)
        x = bits.view(np.float64)
    else:
        bits = rng.integers(1, 2**23, n, dtype=np.uint32)
        x = bits.view(np.float32)
    return (x * rng.choice([-1, 1], n).astype(f["dt"])).astype(f["dt"])


def inputs(prec: str, op: str, n: int, seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Input groups. Sizes are fractions of n (sum = n for sums; products add prod_edge)."""
    rng = np.random.default_rng(seed)
    f = FMT[prec]
    dt, p, emin, emax = f["dt"], f["p"], f["emin"], f["emax"]
    g: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    k = n // 10
    # uniform: exponents in [-20, 20]
    g["uniform"] = (_rand_float(rng, 3 * k, dt, -20, 20), _rand_float(rng, 3 * k, dt, -20, 20))
    # mixed magnitudes: b = a-scale * 2^-gap, gap in [0, 2p+10] (past the mantissa width)
    a = _rand_float(rng, 3 * k, dt, -20, 20)
    gap = rng.integers(0, 2 * p + 11, 3 * k)
    b = (
        np.ldexp(rng.uniform(1, 2, 3 * k), np.frexp(a.astype(np.float64))[1] - 1 - gap)
        * rng.choice([-1.0, 1.0], 3 * k)
    ).astype(dt)
    g["mixed"] = (a, b)
    # near-cancellation: b = -(a + j ulp(a)), j in [-8, 8]
    a = _rand_float(rng, 2 * k, dt, -20, 20)
    j = rng.integers(-8, 9, 2 * k).astype(dt)
    b = (-(a + j * np.spacing(a))).astype(dt)
    g["cancel"] = (a, b)
    # subnormal edges: both subnormal / one subnormal + one near min-normal
    m = k
    a1, b1 = _subnormals(rng, m, prec), _subnormals(rng, m, prec)
    a2 = _rand_float(rng, m, dt, emin, emin + 3)
    b2 = _subnormals(rng, m, prec)
    g["subnormal"] = (np.concatenate([a1, a2]), np.concatenate([b1, b2]))
    if op in ("two_prod", "two_prod_dekker"):
        # around the FMA two_prod domain threshold Ea + Eb = emin + p - 1, +-40 binades;
        # half near the overflow edge Ea + Eb ~ emax
        thr = emin + p - 1
        ea = rng.integers(max(emin, thr - emax), min(emax, thr - emin) + 1, 2 * k // 2)
        eb = np.clip(thr + rng.integers(-40, 41, ea.size) - ea, emin, emax)
        a = rng.uniform(1, 2, ea.size) * rng.choice([-1, 1], ea.size)
        b = rng.uniform(1, 2, ea.size) * rng.choice([-1, 1], ea.size)
        a = np.ldexp(a, ea).astype(dt)
        b = np.ldexp(b, eb).astype(dt)
        ea2 = rng.integers(0, emax + 1, k)
        eb2 = np.clip(emax - ea2 + rng.integers(-3, 2, k), emin, emax)
        a3 = np.ldexp(rng.uniform(1, 2, k), ea2).astype(dt)
        b3 = np.ldexp(rng.uniform(1, 2, k), eb2).astype(dt)
        with np.errstate(over="ignore"):
            g["prod_edge"] = (np.concatenate([a, a3]), np.concatenate([b, b3]))
    if op == "fast_two_sum":  # precondition |a| >= |b|
        for key, (a, b) in g.items():
            sw = np.abs(a) < np.abs(b)
            g[key] = (np.where(sw, b, a), np.where(sw, a, b))
    return g


def in_domain_two_prod(a: np.ndarray, b: np.ndarray, prec: str) -> np.ndarray:
    """Sufficient condition for fma two_prod to be error-free (derived in S3.md):
    max(Ea, emin) + max(Eb, emin) >= emin + p - 1, and fl(a*b) finite.
    Zero operands are trivially exact."""
    f = FMT[prec]
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        ea = np.maximum(np.floor(np.log2(np.abs(a64))), f["emin"])
        eb = np.maximum(np.floor(np.log2(np.abs(b64))), f["emin"])
        prod = a * b
    finite = np.isfinite(prod)
    zero = (a == 0) | (b == 0)
    return (zero | (ea + eb >= f["emin"] + f["p"] - 1)) & finite


def run_gpu(mod, op: str, a: np.ndarray, b: np.ndarray):
    from numba_cuda_mlir import cuda

    da, db = cuda.to_device(a), cuda.to_device(b)
    ds, de = cuda.device_array_like(a), cuda.device_array_like(a)
    tpb = 256
    mod.k_eft2[(a.size + tpb - 1) // tpb, tpb](FN[op], da, db, ds, de)
    cuda.synchronize()
    return ds.copy_to_host(), de.copy_to_host()


def _check_chunk(args):
    kind, a, b, s, e = args
    bad = []
    for i, (x, y, u, v) in enumerate(
        zip(a.tolist(), b.tolist(), s.tolist(), e.tolist(), strict=True)
    ):
        if not (math.isfinite(u) and math.isfinite(v)):
            bad.append(i)
            continue
        exact = Fraction(x) + Fraction(y) if kind == "sum" else Fraction(x) * Fraction(y)
        if Fraction(u) + Fraction(v) != exact:
            bad.append(i)
    return bad


def verify(pool: ProcessPoolExecutor, op: str, a, b, s, e, chunks: int = 96) -> np.ndarray:
    """Boolean mask of failures: (s + e) != (a op b) in exact rational arithmetic."""
    kind = "prod" if "prod" in op else "sum"
    idx = np.array_split(np.arange(a.size), chunks)
    jobs = [(kind, a[ix], b[ix], s[ix], e[ix]) for ix in idx]
    mask = np.zeros(a.size, bool)
    for ix, bad in zip(idx, pool.map(_check_chunk, jobs), strict=True):
        mask[ix[bad]] = True
    return mask


def finite_inputs(a, b, op):
    with np.errstate(over="ignore", invalid="ignore"):
        r = a + b if "sum" in op else a * b
    return np.isfinite(a) & np.isfinite(b) & (np.isfinite(r) | ("prod" in op))

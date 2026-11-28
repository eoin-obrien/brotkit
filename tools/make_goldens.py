#!/usr/bin/env python3
"""Generate the golden reference views in tests/golden/ (dev tool; never run in CI).

    uv run tools/make_goldens.py                 # every view (~10 min on 24 cores)
    uv run tools/make_goldens.py elephant_61x37  # one view; the manifest keeps the others
    uv run tools/make_goldens.py --check         # regenerate into a temp dir, compare sha256

Every pixel is computed exactly with mpmath (ADR-0011's mapping, brotkit.testing.reference)
at a working precision P = max(256, zoom bits + 128) and again at 2P; the run fails unless
both agree on outcome, n, and the float64 values of z, mu and DE on every pixel not flagged
sensitive. Commit generator changes first: the manifest records the generator's git SHA, and
the run refuses uncommitted changes to the generator (override: --allow-dirty).
docs/design/goldens.md describes the views, layers and tolerances.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mpmath
import numpy as np
import numpy.typing as npt
from mpmath import mpf

from brotkit.testing import golden as g
from brotkit.testing.reference import (
    RefEmbedding,
    RefFrame,
    distance_estimate,
    escapes_at,
    iterate,
    smooth_mu,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "golden"
PREVIEWS = ROOT / "docs" / "_static" / "golden"
GENERATOR_FILES = ("tools/make_goldens.py", "src/brotkit/testing")

ADR = "ADR-0011 v1"
ESCAPE_RADIUS = "1e6"
LEVELS_PX = ("0.001", "0.003", "0.01", "0.03", "0.1", "0.3")  # stable_px ladder
RADIUS_REL = "1e-6"

# Seahorse valley: the classic deep-zoom point; spirals at every scale.
SEAHORSE = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139")
# Misiurewicz point M(4,1), from S3 (findroot at 100 digits, residual 2e-101): asymptotically
# self-similar, so filaments cross the view at every depth and it is never all-interior.
M41 = (
    "-0.101096363845622161025785445738622565463805442826253483876931",
    "0.956286510809141500771096057729977435809833336510529170034314",
)

PASS_ALL: dict[str, g.Expect] = dict.fromkeys(g.TIERS, "pass")


@dataclass(frozen=True, slots=True, kw_only=True)
class View:
    name: str
    title: str
    frame: RefFrame
    max_iter: int
    embedding: RefEmbedding = field(default_factory=RefEmbedding.mandelbrot)
    expect: dict[str, g.Expect] = field(default_factory=lambda: dict(PASS_ALL))
    standin: bool = True  # the S4 stand-in kernels can render it (Mandelbrot, no rotation)


def _expect(**kw: g.Expect) -> dict[str, g.Expect]:
    out = dict(PASS_ALL)
    out.update(kw)
    return out


VIEWS: tuple[View, ...] = (
    View(
        name="overview",
        title="Full Mandelbrot overview",
        frame=RefFrame(center=("-0.75", "0"), width="3.2", pixels=(64, 64)),
        max_iter=256,
    ),
    View(
        name="seahorse",
        title="Seahorse valley, spacing 1.6e-5 (f32's last passing view)",
        frame=RefFrame(center=SEAHORSE, width="1.024e-3", pixels=(64, 64)),
        max_iter=1000,
    ),
    View(
        name="seahorse_2e-6",
        title="Seahorse valley, spacing 2e-6 (past f32)",
        frame=RefFrame(center=SEAHORSE, width="1.28e-4", pixels=(64, 64)),
        max_iter=2000,
        expect=_expect(f32="fail"),
    ),
    View(
        name="m41_1e-12",
        title="M(4,1), spacing 1e-12 (S3's df64 check)",
        frame=RefFrame(center=M41, width="6.4e-11", pixels=(64, 64)),
        max_iter=1000,
        expect=_expect(f32="n/a"),
    ),
    View(
        name="m41_1e-20",
        title="M(4,1), spacing 1e-20 (dd and qf only)",
        frame=RefFrame(center=M41, width="6.4e-19", pixels=(64, 64)),
        max_iter=5000,
        expect=_expect(f32="n/a", f64="n/a", df64="fail"),
    ),
    View(
        name="julia",
        title="Julia set, c = -0.8 + 0.156i",
        frame=RefFrame(center=("0", "0"), width="3.2", pixels=(64, 64)),
        embedding=RefEmbedding.julia(("-0.8", "0.156")),
        max_iter=1000,
        standin=False,
    ),
    View(
        name="seahorse_rotated",
        title="Seahorse valley rotated by 0.5 rad",
        frame=RefFrame(center=SEAHORSE, width="1.024e-3", pixels=(64, 64), rotation=0.5),
        max_iter=1000,
        standin=False,
    ),
    View(
        name="elephant_61x37",
        title="Elephant valley, 61 x 37 (non-square, odd)",
        frame=RefFrame(center=("0.2925", "0.0145"), width="0.0305", pixels=(61, 37)),
        max_iter=500,
    ),
    View(
        name="period2_bulb",
        title="Period-2 bulb, mostly interior",
        frame=RefFrame(center=("-1", "0"), width="0.5", pixels=(64, 64)),
        max_iter=5000,
    ),
)
BY_NAME = {v.name: v for v in VIEWS}


def precision_bits(v: View) -> int:
    """P = max(256, zoom bits + 128), zoom bits = log2(max(1, |centre|) / spacing)."""
    c = max(1.0, *(abs(float(x)) for x in v.frame.center))
    s = float(v.frame.spacing)
    return max(256, math.ceil(math.log2(c / s)) + 128)


@dataclass(frozen=True, slots=True)
class PixelOut:
    n: int
    escaped: bool
    z: tuple[float, float]
    mu: float
    de: float
    abs_dz: float
    sensitive: bool
    stable_px: float
    check_n: bool  # 2P agrees on n and outcome
    check_f64: bool  # 2P agrees on the f64 values of z, mu, DE


def _main_orbit(v: View, i: int, j: int) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    r = mpf(ESCAPE_RADIUS)
    u, w = v.frame.pixel_centre(i, j)
    p = v.embedding.point(u, w)
    rel = mpf(RADIUS_REL)
    o = iterate(
        p,
        max_iter=v.max_iter,
        escape_radius=r,
        derivative=v.embedding.derivative,
        extra_radii=(r * (1 - rel), r * (1 + rel)),
    )
    return o, (u, w), r, p


def _values(o: Any, r: Any) -> tuple[tuple[float, float], float, float]:
    z = (float(o.z[0]), float(o.z[1]))
    if not o.escaped:
        return z, math.nan, math.nan
    return z, float(smooth_mu(o, r)), float(distance_estimate(o))


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def pixel(task: tuple[str, int, int, int]) -> PixelOut:
    name, i, j, prec = task
    v = BY_NAME[name]
    with mpmath.workprec(prec):
        o, (u, w), r, _ = _main_orbit(v, i, j)
        z, mu, de = _values(o, r)
        abs_dz = float(mpmath.hypot(*o.dz)) if o.escaped else math.nan
        r_sensitive = any(nr != o.n for nr in o.n_radii)
        s = v.frame.spacing_mpf()
        r2 = r * r
        stable = 0.0
        for level in LEVELS_PX:
            d = mpf(level) * s
            nudged = ((u + d, w), (u - d, w), (u, w + d), (u, w - d))
            if not all(
                escapes_at(v.embedding.point(a, b), n=o.n, max_iter=v.max_iter, r2=r2)
                for a, b in nudged
            ):
                break
            stable = float(level)
    with mpmath.workprec(2 * prec):
        o2, _, r_2, _ = _main_orbit(v, i, j)
        z2, mu2, de2 = _values(o2, r_2)
    return PixelOut(
        n=o.n,
        escaped=o.escaped,
        z=z,
        mu=mu,
        de=de,
        abs_dz=abs_dz,
        sensitive=r_sensitive or stable == 0.0,
        stable_px=stable,
        check_n=(o2.n, o2.escaped) == (o.n, o.escaped),
        check_f64=z2 == z and _same(mu2, mu) and _same(de2, de),
    )


@dataclass(slots=True)
class Generated:
    layers: dict[str, np.ndarray]
    abs_dz: npt.NDArray[np.float64]
    check_n: npt.NDArray[np.bool_]
    check_f64: npt.NDArray[np.bool_]
    prec: int
    seconds: float


def generate(v: View, *, jobs: int | None = None) -> Generated:
    w, h = v.frame.pixels
    prec = precision_bits(v)
    tasks = [(v.name, i, j, prec) for j in range(h) for i in range(w)]
    t0 = time.perf_counter()
    outs: list[PixelOut]
    if jobs == 1:
        outs = [pixel(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(jobs) as pool:
            outs = pool.map(pixel, tasks, chunksize=8)
    seconds = time.perf_counter() - t0

    def grid(f: Callable[[PixelOut], Any], dtype: Any) -> np.ndarray:
        return np.array([f(o) for o in outs], dtype=dtype).reshape(h, w)

    layers = {
        "outcome": grid(lambda o: g.ESCAPED if o.escaped else g.MAX_ITER, np.uint8),
        "n": grid(lambda o: o.n, np.uint32),
        "z_final": grid(lambda o: complex(*o.z), np.complex128),
        "mu": grid(lambda o: o.mu, np.float64),
        "de": grid(lambda o: o.de, np.float64),
        "sensitive": grid(lambda o: o.sensitive, np.bool_),
        "stable_px": grid(lambda o: o.stable_px, np.float32),
    }
    return Generated(
        layers=layers,
        abs_dz=grid(lambda o: o.abs_dz, np.float64),
        check_n=grid(lambda o: o.check_n, np.bool_),
        check_f64=grid(lambda o: o.check_f64, np.bool_),
        prec=prec,
        seconds=seconds,
    )


def view_params(v: View) -> dict[str, Any]:
    e = v.embedding
    return {
        "title": v.title,
        "center": list(v.frame.center),
        "width": v.frame.width,
        "spacing": str(v.frame.spacing),
        "pixels": list(v.frame.pixels),
        "rotation": v.frame.rotation,
        "embedding": {"origin": list(e.origin), "e1": list(e.e1), "e2": list(e.e2)},
        "derivative": e.derivative,
        "max_iter": v.max_iter,
        "escape_radius": ESCAPE_RADIUS,
        "power": 2,
        "de_convention": "estimate",
        "standin": v.standin,
    }


def _floor2(x: float) -> float:
    return math.floor(x * 100) / 100


def view_meta(v: View, gen: Generated, sha: str) -> dict[str, Any]:
    lay = gen.layers
    sens = lay["sensitive"]
    escaped = lay["outcome"] == g.ESCAPED
    sens_frac = float(sens.mean())
    max_dz = float(np.nanmax(gen.abs_dz)) if escaped.any() else 0.0
    meta: dict[str, Any] = {
        "file": f"{v.name}.npz",
        "sha256": sha,
        **view_params(v),
        "precision_bits": gen.prec,
        "check_precision_bits": 2 * gen.prec,
        "convergence": {
            "non_sensitive": int((~sens).sum()),
            "n_or_outcome_mismatches": int((~gen.check_n & ~sens).sum()),
            "f64_value_mismatches": int((~gen.check_f64 & ~sens).sum()),
            "sensitive_n_mismatches": int((~gen.check_n & sens).sum()),
        },
        "escaped_fraction": float(escaped.mean()),
        "sensitive_fraction": sens_frac,
        "max_sensitive_fraction": math.ceil((sens_frac + 0.01) * 100) / 100,
        "n_range": [int(lay["n"][escaped].min()), int(lay["n"][escaped].max())]
        if escaped.any()
        else None,
        "max_abs_dz": max_dz,
        # float32 DE (the stand-ins square |dz| in float32) is checked where |dz|^2 fits, and
        # is not applicable to the view when fewer than half its escaped pixels do.
        "f32_de_in_range_fraction": None,  # filled in below, once the layers are wrapped
        "f32_de_ok": None,
        "runtime_s": round(gen.seconds, 1),
    }
    gold = g.Golden(v.name, {**meta, "tiers": {}}, dict(lay))
    in_range = round(float(g.f32_de_in_range(gold)[escaped].mean()), 4) if escaped.any() else 0.0
    meta["f32_de_in_range_fraction"] = in_range
    meta["f32_de_ok"] = in_range >= 0.5
    fracs = {t: float(g.decidable(gold, t).mean()) for t in g.TIERS}
    passing = [_floor2(fracs[t] - 0.01) for t in g.TIERS if v.expect[t] == "pass"]
    view_min = min(passing) if passing else 0.5
    tiers: dict[str, Any] = {}
    for t in g.TIERS:
        ex = v.expect[t]
        entry: dict[str, Any] = {"expect": ex, "decidable_fraction": round(fracs[t], 4)}
        if ex == "pass":
            entry["min_decidable"] = _floor2(fracs[t] - 0.01)
        elif ex == "fail":
            entry["min_decidable"] = view_min  # what the passing tiers on this view decide
        tiers[t] = entry
    meta["tiers"] = tiers
    return meta


# --- previews (humans only; tests never read them) ------------------------------------------


def _png(path: Path, rgb: npt.NDArray[np.uint8]) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def preview(path: Path, lay: dict[str, np.ndarray], scale: int = 4) -> None:
    mu = lay["mu"]
    t = np.where(np.isfinite(mu), np.log1p(np.nan_to_num(mu)), 0.0)
    phase = (t * 1.7) % 1.0
    k = np.array([0.0, 1 / 3, 2 / 3])
    rgb = 0.5 + 0.5 * np.cos(2 * np.pi * (phase[..., None] + k))
    rgb[lay["outcome"] != g.ESCAPED] = 0.0
    sens = lay["sensitive"].astype(bool)
    rgb[sens] = 0.5 * rgb[sens] + 0.5 * np.array([1.0, 0.0, 1.0])
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = img.repeat(scale, axis=0).repeat(scale, axis=1)
    _png(path, img)


# --- driver ---------------------------------------------------------------------------------


def git_state() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *GENERATOR_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return sha, bool(dirty)


def versions() -> dict[str, str]:
    return {
        "mpmath": mpmath.__version__,
        "mpmath_backend": mpmath.libmp.BACKEND,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def run(names: list[str], *, out: Path, previews: Path | None, jobs: int | None) -> int:
    sha, dirty = git_state()
    manifest_path = out / "manifest.json"
    old: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    views: dict[str, Any] = dict(old.get("views", {}))
    ok = True
    for name in names:
        v = BY_NAME[name]
        print(
            f"{name}: {v.frame.pixels[0]}x{v.frame.pixels[1]}, max_iter {v.max_iter} ...",
            flush=True,
        )
        gen = generate(v, jobs=jobs)
        digest = g.write_npz(out / f"{name}.npz", gen.layers)
        meta = view_meta(v, gen, digest)
        views[name] = meta
        conv = meta["convergence"]
        bad = conv["n_or_outcome_mismatches"] + conv["f64_value_mismatches"]
        ok &= bad == 0
        dec = ", ".join(f"{t} {meta['tiers'][t]['decidable_fraction']:.3f}" for t in g.TIERS)
        print(
            f"  {gen.seconds:.1f} s at P={gen.prec}; P vs 2P mismatches on non-sensitive: {bad}; "
            f"sensitive {meta['sensitive_fraction']:.3f}; escaped {meta['escaped_fraction']:.3f}; "
            f"n {meta['n_range']}; max|dz| {meta['max_abs_dz']:.3g}; decidable: {dec}",
            flush=True,
        )
        if previews is not None:
            previews.mkdir(parents=True, exist_ok=True)
            preview(previews / f"{name}.png", gen.layers)
    manifest: dict[str, Any] = {
        "format": 1,
        "pixel_convention": ADR,
        "generator": {"path": "tools/make_goldens.py", "git_sha": sha, "dirty": dirty},
        "versions": versions(),
        "cpu_count": os.cpu_count(),
        "escape_radius": ESCAPE_RADIUS,
        "sensitivity": {"nudge_px": str(g.SENSITIVE_NUDGE_PX), "radius_rel": RADIUS_REL},
        "stable_px_levels": list(LEVELS_PX),
        "k_decidable": g.K_DECIDABLE,
        "tier_eps": dict(g.TIER_EPS),
        "views": {n: views[n] for n in BY_NAME if n in views},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    if not ok:
        print("FAILED: P and 2P disagree on non-sensitive pixels", file=sys.stderr)
    return 0 if ok else 1


def check(names: list[str], *, jobs: int | None) -> int:
    manifest = g.load_manifest(OUT)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            digest = g.write_npz(Path(tmp) / "x.npz", generate(BY_NAME[name], jobs=jobs).layers)
            want = manifest["views"][name]["sha256"]
            print(f"{name}: {'ok' if digest == want else f'MISMATCH {digest} != {want}'}")
            bad += digest != want
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("views", nargs="*", help=f"default: all ({', '.join(BY_NAME)})")
    ap.add_argument("--out", type=Path, default=OUT, help="directory for the NPZs and manifest")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--check", action="store_true", help="regenerate and compare sha256 only")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="run with uncommitted generator changes"
    )
    ap.add_argument("--no-previews", action="store_true")
    args = ap.parse_args(argv)
    names: list[str] = list(args.views) or list(BY_NAME)
    if unknown := [n for n in names if n not in BY_NAME]:
        ap.error(f"unknown views {unknown}; have {list(BY_NAME)}")
    if args.check:
        return check(names, jobs=args.jobs)
    _, dirty = git_state()
    if dirty and not args.allow_dirty:
        print("refusing: uncommitted changes to the generator (commit first, or --allow-dirty)")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    previews = None if args.no_previews else PREVIEWS if args.out == OUT else args.out / "previews"
    return run(names, out=args.out, previews=previews, jobs=args.jobs)


if __name__ == "__main__":
    sys.exit(main())

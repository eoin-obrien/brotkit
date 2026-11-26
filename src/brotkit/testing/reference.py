"""Exact reference for the pixel convention and the escape-time iteration (ADR-0011).

This is the executable form of ADR-0011. The golden generator (`tools/make_goldens.py`) and the
tests share it, and phase 1's `Frame` and `Embedding` must agree with it. It is pure Python on
mpmath at hundreds of bits: slow by design, never used to render.

Conventions (ADR-0011):

- Pixel (i, j) covers [i, i+1) x [j, j+1) in pixel space; its sample sits at its centre
  (i + 1/2, j + 1/2). Row 0 is the top row and +v points up.
- Pixel spacing is s = width / W, so `width` is the x extent and the height is s * H.
- A pixel-space point (x, y) has the unrotated offset du = (x - W/2) s, dv = -(y - H/2) s from
  the frame centre. `rotation` (radians) turns that offset counter-clockwise about the centre:
  u = cu + cos(t) du - sin(t) dv, v = cv + sin(t) du + cos(t) dv. The sampled window turns
  counter-clockwise, so the picture appears turned clockwise.
- An embedding maps (u, v) to p = origin + u e1 + v e2 in C^2 = R^4, ordered
  (Re z0, Im z0, Re c, Im c). Mandelbrot: origin 0, e1 = Re c, e2 = Im c. Julia(c): origin
  (0, 0, Re c, Im c), e1 = Re z0, e2 = Im z0.
- Iteration: z_{k+1} = z_k^2 + c. The escape count n is the first k with |z_k| > R (strict),
  testing z_0 .. z_{max_iter-1}; n = max_iter if none escapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

import mpmath
from mpmath import mpf

__all__ = [
    "Orbit",
    "RefEmbedding",
    "RefFrame",
    "distance_estimate",
    "escapes_at",
    "iterate",
    "smooth_mu",
]

type Mpf = mpf
type Vec4 = tuple[float, float, float, float]
Derivative = Literal["dc", "dz"]


def _decimal(s: str, what: str) -> Decimal:
    try:
        d = Decimal(s)
    except InvalidOperation:
        raise ValueError(f"{what} must be a decimal string, got {s!r}") from None
    if not d.is_finite():
        raise ValueError(f"{what} must be finite, got {s!r}")
    return d


@dataclass(frozen=True, slots=True, kw_only=True)
class RefFrame:
    """The pixel grid in plane coordinates. Centres and width are decimal strings, so deep
    coordinates are exact; mpmath rounds them at the caller's working precision."""

    center: tuple[str, str]
    width: str
    pixels: tuple[int, int]  # (W, H)
    rotation: float = 0.0  # radians, counter-clockwise turn of the sampled window

    def __post_init__(self) -> None:
        _decimal(self.center[0], "center[0]")
        _decimal(self.center[1], "center[1]")
        if _decimal(self.width, "width") <= 0:
            raise ValueError("width must be positive")
        w, h = self.pixels
        if w < 1 or h < 1:
            raise ValueError(f"pixels must be positive, got {self.pixels}")

    @property
    def spacing(self) -> Decimal:
        """s = width / W (Decimal, for display; mapping uses `spacing_mpf`)."""
        return Decimal(self.width) / self.pixels[0]

    def spacing_mpf(self) -> Mpf:
        return mpf(self.width) / self.pixels[0]

    def to_plane(self, x: Mpf | float | int, y: Mpf | float | int) -> tuple[Mpf, Mpf]:
        """Pixel-space point (x, y) to plane (u, v). The centre of pixel (i, j) is
        (i + 1/2, j + 1/2); use `pixel_centre` for that."""
        w, h = self.pixels
        s = self.spacing_mpf()
        du = (mpf(x) - mpf(w) / 2) * s
        dv = -(mpf(y) - mpf(h) / 2) * s
        if self.rotation:
            t = mpf(self.rotation)  # the float's exact binary value
            ct, st = mpmath.cos(t), mpmath.sin(t)
            du, dv = ct * du - st * dv, st * du + ct * dv
        return mpf(self.center[0]) + du, mpf(self.center[1]) + dv

    def pixel_centre(self, i: int, j: int) -> tuple[Mpf, Mpf]:
        return self.to_plane(mpf(i) + mpf(0.5), mpf(j) + mpf(0.5))


@dataclass(frozen=True, slots=True, kw_only=True)
class RefEmbedding:
    """Plane (u, v) to (Re z0, Im z0, Re c, Im c): p = origin + u e1 + v e2."""

    origin: tuple[str, str, str, str]
    e1: Vec4
    e2: Vec4

    def __post_init__(self) -> None:
        for k, s in enumerate(self.origin):
            _decimal(s, f"origin[{k}]")

    @classmethod
    def mandelbrot(cls) -> RefEmbedding:
        return cls(origin=("0", "0", "0", "0"), e1=(0.0, 0.0, 1.0, 0.0), e2=(0.0, 0.0, 0.0, 1.0))

    @classmethod
    def julia(cls, c: tuple[str, str]) -> RefEmbedding:
        return cls(origin=("0", "0", c[0], c[1]), e1=(1.0, 0.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0, 0.0))

    @property
    def derivative(self) -> Derivative:
        """Which derivative the distance estimate uses: d/dc for the c-plane (seed 0), d/dz0
        for the z0-plane (seed 1). Other slices aren't supported by the reference."""
        if self.e1 == (0.0, 0.0, 1.0, 0.0) and self.e2 == (0.0, 0.0, 0.0, 1.0):
            return "dc"
        if self.e1 == (1.0, 0.0, 0.0, 0.0) and self.e2 == (0.0, 1.0, 0.0, 0.0):
            return "dz"
        raise ValueError("the reference supports only the Mandelbrot and Julia planes")

    def point(self, u: Mpf, v: Mpf) -> tuple[Mpf, Mpf, Mpf, Mpf]:
        o = [mpf(s) for s in self.origin]
        p = [o[k] + u * mpf(self.e1[k]) + v * mpf(self.e2[k]) for k in range(4)]
        return p[0], p[1], p[2], p[3]


@dataclass(frozen=True, slots=True)
class Orbit:
    n: int  # escape count; max_iter if it didn't escape
    escaped: bool
    z: tuple[Mpf, Mpf]  # z_n at escape, else z_max_iter
    dz: tuple[Mpf, Mpf]  # the derivative alongside z
    n_radii: tuple[int, ...]  # the escape count for each of `extra_radii`


def iterate(
    point: tuple[Mpf, Mpf, Mpf, Mpf],
    *,
    max_iter: int,
    escape_radius: Mpf,
    derivative: Derivative,
    extra_radii: tuple[Mpf, ...] = (),
) -> Orbit:
    """Iterate z^2 + c from z0 with its derivative. `extra_radii` get their own escape counts
    from the same orbit (it runs on past R until they've escaped too)."""
    zr, zi, cr, ci = point
    dr, di = (mpf(0), mpf(0)) if derivative == "dc" else (mpf(1), mpf(0))
    one = mpf(1) if derivative == "dc" else mpf(0)
    r2 = escape_radius * escape_radius
    extra2 = [r * r for r in extra_radii]
    n_extra: list[int | None] = [None] * len(extra2)
    hit: tuple[int, Mpf, Mpf, Mpf, Mpf] | None = None
    k = 0
    while k < max_iter:
        m2 = zr * zr + zi * zi
        for e, ex2 in enumerate(extra2):
            if n_extra[e] is None and m2 > ex2:
                n_extra[e] = k
        if hit is None and m2 > r2:
            hit = (k, zr, zi, dr, di)
        if hit is not None and all(x is not None for x in n_extra):
            break
        if hit is None:  # the derivative is only needed up to the main escape
            dr, di = 2 * (zr * dr - zi * di) + one, 2 * (zr * di + zi * dr)
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
        k += 1
    radii_n = tuple(max_iter if x is None else x for x in n_extra)
    if hit is None:
        return Orbit(max_iter, False, (zr, zi), (dr, di), radii_n)
    n, hzr, hzi, hdr, hdi = hit
    return Orbit(n, True, (hzr, hzi), (hdr, hdi), radii_n)


def escapes_at(point: tuple[Mpf, Mpf, Mpf, Mpf], *, n: int, max_iter: int, r2: Mpf) -> bool:
    """True if the orbit's escape count is exactly `n` (max_iter: never escapes). Stops as
    soon as the answer is known, so it costs at most n + 1 iterations."""
    zr, zi, cr, ci = point
    for k in range(min(n, max_iter - 1) + 1):
        if zr * zr + zi * zi > r2:
            return k == n
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
    return n == max_iter


def smooth_mu(o: Orbit, escape_radius: Mpf, power: int = 2) -> Mpf:
    """mu = n + 1 - log(log|z| / log R) / log p (spec, "Core vocabulary")."""
    lz = mpmath.log(mpmath.hypot(*o.z))
    lr = mpmath.log(escape_radius)
    return o.n + 1 - mpmath.log(lz / lr) / mpmath.log(power)


def distance_estimate(o: Orbit) -> Mpf:
    """The spec's `estimate` convention: 2 |z| ln|z| / |z'|."""
    az = mpmath.hypot(*o.z)
    return 2 * az * mpmath.log(az) / mpmath.hypot(*o.dz)

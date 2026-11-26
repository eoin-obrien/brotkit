"""Golden references and the rules that define "matches the golden" (docs/design/goldens.md).

Pure NumPy: no CUDA import, so CPU tests and any renderer can use it. A golden is an exact
mpmath computation (tools/make_goldens.py) of one small view, stored as an uncompressed NPZ in
`tests/golden/` (Git LFS) and described by `tests/golden/manifest.json`.

Rules, per tier:

- **Decidable pixels** (`stable_px >= K_DECIDABLE * c_rounding_px` and not `sensitive`):
  outcome and n match exactly. `c_rounding_px` is how far rounding the pixel's plane
  coordinates to the tier moves them, in pixels: eps * max(|u|, |v|) / s.
- **Decidable fraction:** at least the manifest's `min_decidable` for that tier and view,
  and always reported, so the mask can't excuse everything.
- **Sensitive pixels** (n or outcome changes under a 1/1000 px nudge or a 1e-6 change of R)
  aren't checked pixel by pixel: the report counts those within +-1 or with a flipped
  outcome, and the rest as outliers. The view's sensitive fraction must stay under the
  manifest's `max_sensitive_fraction`.
- **mu and DE** on decidable escaped pixels where n matches: relative error within the tier's
  budget, in units of the eps of the precision they were computed in (`TOLERANCES`). With a
  float32 derivative, DE is checked only where |dz|^2 fits in float32 (`f32_de_in_range`),
  and is not applicable to a view where fewer than half its escaped pixels do
  (`f32_de_ok` in the manifest).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "GOLDEN_DIR",
    "K_DECIDABLE",
    "TIERS",
    "TIER_EPS",
    "TOLERANCES",
    "Golden",
    "GoldenError",
    "Report",
    "Tolerance",
    "c_rounding_px",
    "compare",
    "decidable",
    "f32_de_in_range",
    "load_golden",
    "load_manifest",
    "plane_coords_f64",
    "write_npz",
]

GOLDEN_DIR: Final = Path(__file__).resolve().parents[3] / "tests" / "golden"

TIERS: Final = ("f32", "df64", "f64", "dd", "qf")
# Unit roundoff per tier (spec: Precision.eps). df64 and qf are the S3/S4 word counts:
# 2 x 24 and about 96 bits; dd is 2 x 53.
TIER_EPS: Final[Mapping[str, float]] = {
    "f32": 2.0**-24,
    "df64": 2.0**-48,
    "f64": 2.0**-53,
    "dd": 2.0**-106,
    "qf": 2.0**-96,
}
COMPUTE_EPS: Final[Mapping[str, float]] = {"f32": 2.0**-24, "f64": 2.0**-53}
K_DECIDABLE: Final = 4.0
SENSITIVE_NUDGE_PX: Final = 1e-3
F32_MAX: Final = float(np.finfo(np.float32).max)

LAYERS: Final = ("outcome", "n", "z_final", "mu", "de", "sensitive", "stable_px")
ESCAPED: Final = 0  # Outcome.ESCAPED (spec)
MAX_ITER: Final = 3  # Outcome.MAX_ITER

Expect = Literal["pass", "fail", "n/a"]


class GoldenError(RuntimeError):
    """A golden file is missing, is an LFS pointer, or doesn't match the manifest."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Tolerance:
    """Relative error budgets for mu and DE, in units of `computed_in`'s eps."""

    mu_eps: float
    de_eps: float
    computed_in: Literal["f32", "f64"]


# Calibrated against the S4 stand-in kernels (smooth + DE), about 4x the largest error seen
# on any view the tier passes; docs/design/goldens.md has the table and the measurements.
# The stand-ins compute mu and DE in float32 from the hi word on every tier. Recalibrate
# when the phase 1 renderer replaces them.
TOLERANCES: Final[Mapping[str, Tolerance]] = {
    "f32": Tolerance(mu_eps=25_000, de_eps=1.7e6, computed_in="f32"),
    "df64": Tolerance(mu_eps=24_000, de_eps=11_000, computed_in="f32"),
    "f64": Tolerance(mu_eps=24_000, de_eps=15, computed_in="f32"),
    "dd": Tolerance(mu_eps=4, de_eps=15, computed_in="f32"),
    "qf": Tolerance(mu_eps=600, de_eps=11_000, computed_in="f32"),
}


def load_manifest(directory: Path = GOLDEN_DIR) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.exists():
        raise GoldenError(f"no golden manifest at {path}")
    return json.loads(path.read_text())


class Golden:
    """One golden view: its manifest entry and its layers (read-only arrays, shape (H, W))."""

    __slots__ = ("_layers", "meta", "name")

    def __init__(self, name: str, meta: Mapping[str, Any], layers: Mapping[str, np.ndarray]):
        self.name = name
        self.meta = meta
        for a in layers.values():
            a.setflags(write=False)
        self._layers = dict(layers)

    def __getitem__(self, layer: str) -> np.ndarray:
        return self._layers[layer]

    @property
    def shape(self) -> tuple[int, int]:
        w, h = self.meta["pixels"]
        return (h, w)

    @property
    def max_iter(self) -> int:
        return int(self.meta["max_iter"])

    def tier(self, tier: str) -> Mapping[str, Any]:
        return self.meta["tiers"][tier]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_golden(name: str, directory: Path = GOLDEN_DIR) -> Golden:
    """Load a golden view and check its sha256 against the manifest."""
    manifest = load_manifest(directory)
    if name not in manifest["views"]:
        raise GoldenError(f"no golden view {name!r}; have {sorted(manifest['views'])}")
    meta = manifest["views"][name]
    path = directory / meta["file"]
    if not path.exists():
        raise GoldenError(f"missing {path}")
    if path.read_bytes()[:40].startswith(b"version https://git-lfs"):
        raise GoldenError(f"{path} is a Git LFS pointer; run `git lfs pull`")
    if (got := _sha256(path)) != meta["sha256"]:
        raise GoldenError(f"{path}: sha256 {got} != manifest {meta['sha256']}")
    with np.load(path, allow_pickle=False) as f:
        layers = {k: f[k] for k in LAYERS}
    return Golden(name, meta, layers)


def write_npz(path: Path, layers: Mapping[str, np.ndarray]) -> str:
    """Write an uncompressed NPZ byte-for-byte reproducibly (fixed zip timestamps, sorted
    names, no compression, whose output could vary with the zlib build). Returns sha256."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(layers):
            arr = np.ascontiguousarray(layers[name])
            body = io.BytesIO()
            np.lib.format.write_array(body, arr, version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, body.getvalue())
    data = buf.getvalue()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def plane_coords_f64(meta: Mapping[str, Any]) -> tuple[npt.NDArray[np.float64], ...]:
    """Pixel-centre plane coordinates (u, v) in float64, shape (H, W). Only for magnitudes
    (c rounding), never for rendering: f64 can't resolve deep views."""
    w, h = int(meta["pixels"][0]), int(meta["pixels"][1])
    s = float(meta["width"]) / w
    du: npt.NDArray[np.float64] = (np.arange(w, dtype=np.float64) + 0.5 - w / 2) * s
    dv: npt.NDArray[np.float64] = -(np.arange(h, dtype=np.float64) + 0.5 - h / 2) * s
    dug = np.broadcast_to(du[np.newaxis, :], (h, w))
    dvg = np.broadcast_to(dv[:, np.newaxis], (h, w))
    t = float(meta.get("rotation", 0.0))
    ct, st = math.cos(t), math.sin(t)
    u = float(meta["center"][0]) + ct * dug - st * dvg
    v = float(meta["center"][1]) + st * dug + ct * dvg
    return u, v


def c_rounding_px(meta: Mapping[str, Any], tier: str) -> npt.NDArray[np.float64]:
    """How far rounding each pixel's plane coordinates to `tier` can move them, in pixels."""
    u, v = plane_coords_f64(meta)
    s = float(meta["width"]) / meta["pixels"][0]
    return TIER_EPS[tier] * np.maximum(np.abs(u), np.abs(v)) / s


def f32_de_in_range(golden: Golden) -> npt.NDArray[np.bool_]:
    """Escaped pixels whose |dz|^2 fits in float32 with a factor-4 margin, so a float32
    derivative can't have overflowed. |dz| comes from the golden: 2 |z| ln|z| / DE."""
    z = np.abs(golden["z_final"])
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_dz = 2.0 * z * np.log(z) / golden["de"]
    return (golden["outcome"] == ESCAPED) & (abs_dz < math.sqrt(F32_MAX) / 2)


def decidable(golden: Golden, tier: str) -> npt.NDArray[np.bool_]:
    """Pixels whose outcome and n a correct `tier` renderer must reproduce exactly."""
    need = K_DECIDABLE * c_rounding_px(golden.meta, tier)
    return (golden["stable_px"] >= need) & ~golden["sensitive"].astype(bool)


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    view: str
    tier: str
    ok: bool
    failures: tuple[str, ...]
    decidable_fraction: float
    min_decidable: float
    sensitive_fraction: float
    n_mismatches: int  # on decidable pixels
    sensitive_within_one: int  # sensitive pixels with |dn| <= 1 or a flipped outcome
    sensitive_outliers: int
    resolves_nudge: bool
    mu_max_eps: float | None  # largest relative mu error, in computed_in eps
    de_max_eps: float | None
    de_status: Literal["checked", "not applicable", "missing"]
    mu_pixels: int
    de_pixels: int

    def summary(self) -> str:
        head = f"{self.view} / {self.tier}: {'PASS' if self.ok else 'FAIL'}"
        mu = "-" if self.mu_max_eps is None else f"{self.mu_max_eps:.3g}"
        de = "-" if self.de_max_eps is None else f"{self.de_max_eps:.3g}"
        lines = [
            head,
            f"  decidable {self.decidable_fraction:.3f} (min {self.min_decidable:.3f}), "
            f"sensitive {self.sensitive_fraction:.3f}",
            f"  n mismatches on decidable pixels: {self.n_mismatches}",
            f"  sensitive: {self.sensitive_within_one} within +-1 or flipped, "
            f"{self.sensitive_outliers} outliers (counted, not failed)",
            f"  mu max err {mu} eps on {self.mu_pixels} px; DE {self.de_status}, max err {de} eps "
            f"on {self.de_pixels} px",
        ]
        lines += [f"  - {f}" for f in self.failures]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _rel_err_eps(
    got: npt.NDArray[np.floating[Any]], want: npt.NDArray[np.float64], eps: float, floor: float
) -> npt.NDArray[np.float64]:
    g = got.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs(g - want) / np.maximum(np.abs(want), floor) / eps
    return np.where(np.isfinite(err), err, np.inf)


def compare(
    result_layers: Mapping[str, np.ndarray],
    golden: Golden,
    tier: str,
    *,
    tolerance: Tolerance | None = None,
) -> Report:
    """Compare a renderer's layers (`n` required; `outcome`, `mu`, `de` optional; all shape
    (H, W)) with `golden` under `tier`'s rules. `tolerance` overrides `TOLERANCES[tier]`."""
    tol = tolerance or TOLERANCES[tier]
    tm = golden.tier(tier)
    failures: list[str] = []
    h, w = golden.shape

    def report(**kw: Any) -> Report:
        base: dict[str, Any] = {
            "view": golden.name,
            "tier": tier,
            "ok": not failures,
            "failures": tuple(failures),
            "decidable_fraction": 0.0,
            "min_decidable": float(tm.get("min_decidable", 0.0)),
            "sensitive_fraction": float(golden.meta["sensitive_fraction"]),
            "n_mismatches": 0,
            "sensitive_within_one": 0,
            "sensitive_outliers": 0,
            "resolves_nudge": False,
            "mu_max_eps": None,
            "de_max_eps": None,
            "de_status": "missing",
            "mu_pixels": 0,
            "de_pixels": 0,
        }
        base.update(kw)
        return Report(**base)

    if tm["expect"] == "n/a":
        failures.append(f"{tier} is not applicable to {golden.name} (manifest)")
        return report()
    n = np.asarray(result_layers["n"])
    for name, arr in result_layers.items():
        if np.shape(arr) != (h, w):
            failures.append(f"layer {name!r} has shape {np.shape(arr)}, golden is {(h, w)}")
    if failures:
        return report()

    n = n.astype(np.int64)
    max_iter = golden.max_iter
    outcome = np.asarray(result_layers.get("outcome", np.where(n < max_iter, ESCAPED, MAX_ITER)))
    gn = golden["n"].astype(np.int64)
    gout = golden["outcome"]
    sens = golden["sensitive"].astype(bool)
    dec = decidable(golden, tier)
    frac = float(dec.mean())
    min_dec = float(tm["min_decidable"])
    if frac < min_dec:
        failures.append(f"decidable fraction {frac:.3f} < minimum {min_dec:.3f}")
    if golden.meta["sensitive_fraction"] > golden.meta["max_sensitive_fraction"]:
        failures.append(
            f"sensitive fraction {golden.meta['sensitive_fraction']:.3f} > "
            f"max {golden.meta['max_sensitive_fraction']:.3f}"
        )

    wrong = dec & ((n != gn) | (outcome != gout))
    n_bad = int(wrong.sum())
    if n_bad:
        jj, ii = np.nonzero(wrong)
        ex = ", ".join(
            f"(i={i}, j={j}) n={n[j, i]} want {gn[j, i]}"
            for j, i in zip(jj[:3], ii[:3], strict=True)
        )
        failures.append(f"{n_bad} decidable pixels differ in n or outcome: {ex}")

    flipped = outcome != gout
    within = sens & (flipped | (np.abs(n - gn) <= 1))
    outliers = sens & ~within
    # Counted, never failed: a sensitive pixel's n can jump by more than 1 under any
    # perturbation, orbit rounding included (stand-ins that pass everything else do this on
    # up to 26% of a view's sensitive pixels; docs/design/goldens.md).
    resolves = bool((K_DECIDABLE * c_rounding_px(golden.meta, tier)).max() <= SENSITIVE_NUDGE_PX)

    cmp_px = dec & (gout == ESCAPED) & (n == gn) & (outcome == gout)
    eps = COMPUTE_EPS[tol.computed_in]
    mu_max: float | None = None
    mu_px = 0
    if "mu" in result_layers:
        e = _rel_err_eps(np.asarray(result_layers["mu"]), golden["mu"], eps, 1.0)[cmp_px]
        mu_px = int(e.size)
        mu_max = float(e.max()) if e.size else 0.0
        if mu_max > tol.mu_eps:
            failures.append(f"mu error {mu_max:.3g} eps > budget {tol.mu_eps:g}")
    de_max: float | None = None
    de_px = 0
    de_status: Literal["checked", "not applicable", "missing"] = "missing"
    if "de" in result_layers:
        de_px_mask = cmp_px
        if tol.computed_in == "f32":
            de_px_mask = cmp_px & f32_de_in_range(golden)
        if tol.computed_in == "f32" and not golden.meta["f32_de_ok"]:
            de_status = "not applicable"
        else:
            de_status = "checked"
            e = _rel_err_eps(np.asarray(result_layers["de"]), golden["de"], eps, 0.0)[de_px_mask]
            de_px = int(e.size)
            de_max = float(e.max()) if e.size else 0.0
            if de_max > tol.de_eps:
                failures.append(f"DE error {de_max:.3g} eps > budget {tol.de_eps:g}")

    return report(
        decidable_fraction=frac,
        min_decidable=min_dec,
        n_mismatches=n_bad,
        sensitive_within_one=int(within.sum()),
        sensitive_outliers=int(outliers.sum()),
        resolves_nudge=resolves,
        mu_max_eps=mu_max,
        de_max_eps=de_max,
        de_status=de_status,
        mu_pixels=mu_px,
        de_pixels=de_px,
    )

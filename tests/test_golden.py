"""The pixel convention (ADR-0011), the golden set, and the comparison rules (CPU only)."""

from __future__ import annotations

import hashlib
import json
import math
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

import mpmath
import numpy as np
import pytest
from mpmath import mpf

from brotkit.testing import golden as g
from brotkit.testing.reference import RefEmbedding, RefFrame, escapes_at, iterate
from tests.golden_standin import kernel_origin

MANIFEST = g.load_manifest()
VIEWS = list(MANIFEST["views"])
STANDIN_VIEWS = [v for v in VIEWS if MANIFEST["views"][v]["standin"]]


def frame_of(meta: dict[str, Any]) -> RefFrame:
    return RefFrame(
        center=(meta["center"][0], meta["center"][1]),
        width=meta["width"],
        pixels=(meta["pixels"][0], meta["pixels"][1]),
        rotation=meta["rotation"],
    )


def embedding_of(meta: dict[str, Any]) -> RefEmbedding:
    e = meta["embedding"]
    return RefEmbedding(origin=tuple(e["origin"]), e1=tuple(e["e1"]), e2=tuple(e["e2"]))


@pytest.fixture(scope="module")
def goldens() -> dict[str, g.Golden]:
    return {v: g.load_golden(v) for v in VIEWS}


# --- the pixel convention ------------------------------------------------------------------


def close(a: Any, b: Any, bits: int = 200) -> bool:
    """Equal up to rounding at `bits` (two computations of one value in different orders)."""
    return abs(a - b) <= mpf(2) ** -bits * max(1, abs(b))


def test_pixel_centres_have_the_half_pixel_offset() -> None:
    f = RefFrame(center=("0", "0"), width="4", pixels=(4, 2))  # s = 1
    assert f.pixel_centre(0, 0) == (mpf(-1.5), mpf(0.5))
    assert f.pixel_centre(3, 1) == (mpf(1.5), mpf(-0.5))


def test_row_zero_is_the_top_and_v_points_up() -> None:
    f = RefFrame(center=("0", "0"), width="1", pixels=(8, 8))
    assert f.pixel_centre(0, 0)[1] > f.pixel_centre(0, 7)[1]
    assert f.pixel_centre(0, 0)[0] < f.pixel_centre(7, 0)[0]


def test_width_is_the_x_extent_for_non_square_frames() -> None:
    f = RefFrame(center=("0.2925", "0.0145"), width="0.0305", pixels=(61, 37))
    assert f.spacing == Decimal("0.0005")
    with mpmath.workprec(256):
        left, top = f.to_plane(0, 0)
        right, bottom = f.to_plane(61, 37)
        assert close(right - left, mpf("0.0305"))
        assert close(top - bottom, mpf("0.0005") * 37)


def test_odd_frames_put_the_centre_on_a_pixel_centre() -> None:
    f = RefFrame(center=("0.2925", "0.0145"), width="0.0305", pixels=(61, 37))
    with mpmath.workprec(256):
        assert f.pixel_centre(30, 18) == (mpf("0.2925"), mpf("0.0145"))


def test_rotation_turns_the_window_counter_clockwise_about_the_centre() -> None:
    f0 = RefFrame(center=("1", "2"), width="2", pixels=(2, 2))
    f90 = RefFrame(center=("1", "2"), width="2", pixels=(2, 2), rotation=math.pi / 2)
    with mpmath.workprec(256):
        # the right-hand edge midpoint (+du) lands above the centre (+dv) after +90 degrees
        u, v = f90.to_plane(2, 1)
        assert abs(u - 1) < mpf(1e-15)
        assert abs(v - 3) < mpf(1e-15)
        assert f0.to_plane(1, 1) == f90.to_plane(1, 1) == (mpf(1), mpf(2))


def test_rotation_matches_the_formula() -> None:
    f = RefFrame(center=("0.5", "-0.25"), width="3", pixels=(6, 4), rotation=0.5)
    with mpmath.workprec(200):
        s = mpf(3) / 6
        du, dv = (mpf(1) + mpf(0.5) - 3) * s, -(mpf(2) + mpf(0.5) - 2) * s
        t = mpf(0.5)
        want = (
            mpf("0.5") + mpmath.cos(t) * du - mpmath.sin(t) * dv,
            mpf("-0.25") + mpmath.sin(t) * du + mpmath.cos(t) * dv,
        )
        got = f.pixel_centre(1, 2)
        assert close(got[0], want[0])
        assert close(got[1], want[1])


def test_embeddings_map_the_plane_to_z0_and_c() -> None:
    u, v = mpf("0.25"), mpf("-0.5")
    assert RefEmbedding.mandelbrot().point(u, v) == (0, 0, u, v)
    assert RefEmbedding.julia(("-0.8", "0.156")).point(u, v) == (u, v, mpf("-0.8"), mpf("0.156"))
    assert RefEmbedding.mandelbrot().derivative == "dc"
    assert RefEmbedding.julia(("0", "0")).derivative == "dz"


def test_escape_count_is_the_first_index_past_r() -> None:
    # c = 1: z = 0, 1, 2, 5, 26, ...; |z_3| = 5 is the first above R = 4.5
    o = iterate(
        (mpf(0), mpf(0), mpf(1), mpf(0)), max_iter=10, escape_radius=mpf(4.5), derivative="dc"
    )
    assert (o.n, o.escaped, o.z) == (3, True, (mpf(5), mpf(0)))
    assert o.dz == (mpf(13), mpf(0))  # dz_{k+1} = 2 z_k dz_k + 1: 0, 1, 3, 13
    assert escapes_at((mpf(0), mpf(0), mpf(1), mpf(0)), n=3, max_iter=10, r2=mpf(4.5) ** 2)
    inside = iterate(
        (mpf(0), mpf(0), mpf(0), mpf(0)), max_iter=7, escape_radius=mpf(2), derivative="dc"
    )
    assert (inside.n, inside.escaped) == (7, False)


@pytest.mark.parametrize("view", VIEWS)
def test_f64_plane_coordinates_match_the_reference(view: str) -> None:
    meta = MANIFEST["views"][view]
    frame = frame_of(meta)
    u, v = g.plane_coords_f64(meta)
    s = float(meta["width"]) / meta["pixels"][0]
    h, w = u.shape
    assert (w, h) == tuple(meta["pixels"])
    cu, cv = float(meta["center"][0]), float(meta["center"][1])
    f64_px = 4 * 2.0**-53 * max(1.0, abs(cu), abs(cv)) / s
    with mpmath.workprec(meta["precision_bits"]):
        for j in (0, h // 2, h - 1):
            for i in (0, w // 3, w - 1):
                ru, rv = frame.pixel_centre(i, j)
                # The magnitudes (all c_rounding_px uses) are right to f64 precision.
                assert u[j, i] == pytest.approx(float(ru), rel=1e-15, abs=1e-300)
                assert v[j, i] == pytest.approx(float(rv), rel=1e-15, abs=1e-300)
                # Offsets in pixels, where f64 can resolve them: f64 rounding of the centre
                # is f64_px pixels, so deep views skip this.
                if f64_px < 1e-3:
                    du = float((ru - mpf(meta["center"][0])) / mpf(s))
                    dv = float((rv - mpf(meta["center"][1])) / mpf(s))
                    assert (u[j, i] - cu) / s == pytest.approx(du, abs=f64_px + 1e-9)
                    assert (v[j, i] - cv) / s == pytest.approx(dv, abs=f64_px + 1e-9)


@pytest.mark.parametrize("view", STANDIN_VIEWS)
def test_the_standin_origin_reproduces_every_pixel_centre(view: str) -> None:
    # S4's kernels map (x, y) to c0 + ((x - W//2) dx, (H//2 - y) dx); with c0 from
    # kernel_origin that is exactly ADR-0011's pixel centre (the stand-in's SPIKE shift).
    meta = MANIFEST["views"][view]
    frame = frame_of(meta)
    w, h = meta["pixels"]
    with mpmath.workprec(meta["precision_bits"]):
        c0r, c0i = kernel_origin(meta)
        s = frame.spacing_mpf()
        for j in range(h):
            for i in (0, w // 2, w - 1):
                ru, rv = frame.pixel_centre(i, j)
                # equal up to rounding at the working precision, far below a pixel
                assert abs(c0r + (i - w // 2) * s - ru) < s * mpf(2) ** -100
                assert abs(c0i + (h // 2 - j) * s - rv) < s * mpf(2) ** -100


# --- the goldens ----------------------------------------------------------------------------


def test_manifest_records_parameters_and_provenance() -> None:
    assert MANIFEST["pixel_convention"].startswith("ADR-0011")
    assert MANIFEST["escape_radius"] == "1e6"
    assert {"mpmath", "numpy", "python"} <= set(MANIFEST["versions"])
    assert len(MANIFEST["generator"]["git_sha"]) == 40
    for meta in MANIFEST["views"].values():
        assert isinstance(meta["center"][0], str)
        assert meta["check_precision_bits"] == 2 * meta["precision_bits"] >= 512
        assert meta["convergence"]["n_or_outcome_mismatches"] == 0
        assert meta["convergence"]["f64_value_mismatches"] == 0
        assert set(meta["tiers"]) == set(g.TIERS)


def test_goldens_fit_the_size_budget() -> None:
    total = sum((g.GOLDEN_DIR / m["file"]).stat().st_size for m in MANIFEST["views"].values())
    assert total < 5_000_000


@pytest.mark.parametrize("view", VIEWS)
def test_golden_layers_are_consistent(view: str, goldens: dict[str, g.Golden]) -> None:
    gold = goldens[view]
    n, out = gold["n"], gold["outcome"]
    assert n.shape == gold.shape
    assert ((out == g.ESCAPED) == (n < gold.max_iter)).all()
    esc = out == g.ESCAPED
    assert np.isfinite(gold["mu"][esc]).all()
    assert np.isnan(gold["mu"][~esc]).all()
    # mu = n + 1 - log2(log|z| / log R) lies in [n, n + 1) for |z| in (R, R^2]
    assert ((gold["mu"][esc] >= n[esc]) & (gold["mu"][esc] < n[esc] + 1)).all()
    assert (gold["de"][esc] > 0).all()
    # a pixel that moves under the smallest nudge (stable_px 0) is always sensitive
    assert gold["sensitive"][gold["stable_px"] == 0].all()
    assert set(np.unique(gold["stable_px"])) <= {
        0.0,
        *(np.float32(x) for x in MANIFEST["stable_px_levels"]),
    }


@pytest.mark.parametrize("view", ["overview", "period2_bulb"])
def test_views_centred_on_the_real_axis_are_mirror_symmetric(
    view: str, goldens: dict[str, g.Golden]
) -> None:
    # Conjugate symmetry: with v-centre 0 and an even height, row j mirrors row H-1-j only
    # if pixel centres carry the half-pixel offset.
    n = goldens[view]["n"]
    assert (n == n[::-1]).all()


@pytest.mark.parametrize("view", VIEWS)
def test_golden_pixels_reproduce_from_the_reference(
    view: str, goldens: dict[str, g.Golden]
) -> None:
    # Guards the generator's (j, i) layout and mapping: spot pixels, recomputed from scratch.
    gold = goldens[view]
    meta = gold.meta
    frame, emb = frame_of(dict(meta)), embedding_of(dict(meta))
    h, w = gold.shape
    rng = random.Random(view)
    pixels = [(0, 0), (w - 1, h - 1), (w - 1, 0)] + [
        (rng.randrange(w), rng.randrange(h)) for _ in range(5)
    ]
    with mpmath.workprec(meta["precision_bits"]):
        for i, j in pixels:
            o = iterate(
                emb.point(*frame.pixel_centre(i, j)),
                max_iter=gold.max_iter,
                escape_radius=mpf(meta["escape_radius"]),
                derivative=emb.derivative,
            )
            assert o.n == gold["n"][j, i], (i, j)


@pytest.mark.parametrize("view", VIEWS)
def test_golden_matches_itself_on_every_passing_tier(
    view: str, goldens: dict[str, g.Golden]
) -> None:
    gold = goldens[view]
    layers = {"n": gold["n"], "outcome": gold["outcome"], "mu": gold["mu"], "de": gold["de"]}
    for tier in g.TIERS:
        r = g.compare(layers, gold, tier)
        expect = gold.tier(tier)["expect"]
        if expect == "pass":
            assert r.ok, r.summary()
            assert r.mu_max_eps == 0.0
        elif expect == "n/a":
            assert not r.ok
        print(r.summary())


def test_fail_tiers_cannot_decide_enough_pixels(goldens: dict[str, g.Golden]) -> None:
    # Even perfect data fails a tier the view is designed to defeat: rounding c to that tier
    # moves too many pixels across their stability radius (the decidable-fraction guard).
    for view, tier in (("seahorse_2e-6", "f32"), ("m41_1e-20", "df64")):
        gold = goldens[view]
        r = g.compare({"n": gold["n"]}, gold, tier)
        assert not r.ok
        assert r.decidable_fraction < r.min_decidable


def test_float32_de_is_not_applicable_at_depth(goldens: dict[str, g.Golden]) -> None:
    assert goldens["overview"].meta["f32_de_ok"]
    for view in ("m41_1e-12", "m41_1e-20"):
        gold = goldens[view]
        assert not gold.meta["f32_de_ok"]
        r = g.compare({"n": gold["n"], "de": gold["de"]}, gold, "dd")
        assert r.de_status == "not applicable"


# --- the comparison rules, on synthetic errors ----------------------------------------------


def pick(mask: np.ndarray) -> tuple[int, int]:
    jj, ii = np.nonzero(mask)
    return int(jj[len(jj) // 2]), int(ii[len(ii) // 2])


@pytest.fixture
def seahorse(goldens: dict[str, g.Golden]) -> g.Golden:
    return goldens["seahorse"]


def test_an_n_error_on_a_decidable_pixel_fails(seahorse: g.Golden) -> None:
    n = seahorse["n"].copy()
    j, i = pick(g.decidable(seahorse, "dd"))
    n[j, i] += 1
    r = g.compare({"n": n}, seahorse, "dd")
    assert not r.ok
    assert r.n_mismatches == 1


def test_an_off_by_one_on_a_sensitive_pixel_passes(seahorse: g.Golden) -> None:
    n = seahorse["n"].copy()
    j, i = pick(seahorse["sensitive"])
    n[j, i] += 1
    r = g.compare({"n": n}, seahorse, "dd")
    assert r.ok, r.summary()
    assert r.sensitive_within_one == int(seahorse["sensitive"].sum())


def test_a_large_error_on_a_sensitive_pixel_is_counted_not_failed(seahorse: g.Golden) -> None:
    n = seahorse["n"].copy()
    j, i = pick(seahorse["sensitive"])
    n[j, i] += 5
    r = g.compare({"n": n}, seahorse, "dd")
    assert r.ok, r.summary()
    assert r.sensitive_outliers == 1


@pytest.mark.parametrize("view", ["seahorse", "elephant_61x37"])
def test_transposed_output_fails(view: str, goldens: dict[str, g.Golden]) -> None:
    gold = goldens[view]
    r = g.compare({"n": np.ascontiguousarray(gold["n"].T)}, gold, "f64")
    assert not r.ok


def test_mirrored_rows_fail(seahorse: g.Golden) -> None:
    assert not g.compare({"n": seahorse["n"][::-1]}, seahorse, "f64").ok


def test_mu_and_de_errors_beyond_budget_fail(seahorse: g.Golden) -> None:
    eps = 2.0**-24
    tol = g.Tolerance(mu_eps=100, de_eps=100, computed_in="f32")
    base = {"n": seahorse["n"], "mu": seahorse["mu"], "de": seahorse["de"]}
    assert g.compare(base, seahorse, "f32", tolerance=tol).ok
    assert not g.compare(
        {**base, "mu": seahorse["mu"] * (1 + 200 * eps)}, seahorse, "f32", tolerance=tol
    ).ok
    assert not g.compare(
        {**base, "de": seahorse["de"] * (1 + 200 * eps)}, seahorse, "f32", tolerance=tol
    ).ok
    assert g.compare(
        {**base, "de": seahorse["de"] * (1 + 50 * eps)}, seahorse, "f32", tolerance=tol
    ).ok


def test_not_applicable_tiers_are_refused(goldens: dict[str, g.Golden]) -> None:
    gold = goldens["m41_1e-12"]
    r = g.compare({"n": gold["n"]}, gold, "f32")
    assert not r.ok
    assert "not applicable" in r.failures[0]


def test_the_report_always_prints_the_decidable_fraction(seahorse: g.Golden) -> None:
    text = g.compare({"n": seahorse["n"]}, seahorse, "f32").summary()
    assert "decidable" in text


# --- storage --------------------------------------------------------------------------------


def test_npz_writes_are_byte_reproducible(tmp_path: Path, seahorse: g.Golden) -> None:
    layers = {k: seahorse[k] for k in g.LAYERS}
    a = g.write_npz(tmp_path / "a.npz", layers)
    b = g.write_npz(tmp_path / "b.npz", layers)
    assert a == b == hashlib.sha256((tmp_path / "a.npz").read_bytes()).hexdigest()
    with np.load(tmp_path / "a.npz", allow_pickle=False) as f:
        assert (f["n"] == seahorse["n"]).all()


def test_load_golden_rejects_lfs_pointers_and_hash_mismatches(tmp_path: Path) -> None:
    views = {k: {"file": f"{k}.npz", "sha256": "00"} for k in ("x", "y")}
    (tmp_path / "manifest.json").write_text(json.dumps({"views": views}))
    (tmp_path / "x.npz").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:ab\n")
    (tmp_path / "y.npz").write_bytes(b"not it")
    with pytest.raises(g.GoldenError, match="git lfs pull"):
        g.load_golden("x", tmp_path)
    with pytest.raises(g.GoldenError, match="sha256"):
        g.load_golden("y", tmp_path)
    with pytest.raises(g.GoldenError, match="no golden view"):
        g.load_golden("z", tmp_path)


@pytest.mark.slow
def test_regenerating_the_61x37_view_reproduces_its_sha256(tmp_path: Path) -> None:
    import make_goldens

    gen = make_goldens.generate(make_goldens.BY_NAME["elephant_61x37"])
    digest = g.write_npz(tmp_path / "v.npz", gen.layers)
    assert digest == MANIFEST["views"]["elephant_61x37"]["sha256"]
    assert not (~gen.check_n & ~gen.layers["sensitive"]).any()  # P and 2P agree

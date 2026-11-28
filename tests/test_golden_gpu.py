"""S4 spike kernels against the goldens, per the tolerance table (GPU, correctness only).

STAND-INS: replace these with the phase 1 renderer when it lands (backlog 1.9), and delete
tests/golden_standin.py and its SPIKE adjustments with them. Expected-fail tiers are negative
tests: a pass there means the comparison is too loose, so it fails the test.
"""

from __future__ import annotations

import pytest

from brotkit.testing import golden as g

pytestmark = pytest.mark.gpu
pytest.importorskip("numba_cuda_mlir")

from tests.golden_standin import run_standin  # noqa: E402

MANIFEST = g.load_manifest()
CASES = [
    pytest.param(view, tier, meta["tiers"][tier]["expect"], id=f"{view}-{tier}")
    for view, meta in MANIFEST["views"].items()
    if meta["standin"]
    for tier in g.TIERS
    if meta["tiers"][tier]["expect"] != "n/a"
]


@pytest.mark.parametrize(("view", "tier", "expect"), CASES)
def test_standin_kernel_against_golden(
    view: str, tier: str, expect: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    gold = g.load_golden(view)
    layers = run_standin(tier, gold, tmp_path_factory.mktemp("kernels"))
    r = g.compare(layers, gold, tier)
    print(r.summary())
    if expect == "pass":
        assert r.ok, r.summary()
        return
    assert not r.ok, f"negative test passed, so the comparison is too loose:\n{r.summary()}"
    # ...and it rejects a render that really is wrong, not a good one: the tier gets n wrong
    # on more than 1% of the pixels that are stable to 1/1000 px.
    stable = ~gold["sensitive"]
    wrong = (layers["n"] != gold["n"]) & stable
    assert wrong.sum() > 0.01 * stable.sum(), r.summary()


@pytest.mark.parametrize("view", ["seahorse", "elephant_61x37"])
def test_a_renderer_without_the_half_pixel_offset_fails(
    view: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # A passing tier (dd) with a real bug: S4's own mapping, no half-pixel shift. For the
    # even-sized view every pixel moves by half a pixel; for 61 x 37 none does, which is the
    # control (odd sizes put the centre on a pixel centre, so the two conventions agree).
    gold = g.load_golden(view)
    r = g.compare(
        run_standin("dd", gold, tmp_path_factory.mktemp("k"), half_pixel=False), gold, "dd"
    )
    print(r.summary())
    if view == "elephant_61x37":
        assert r.ok, r.summary()
    else:
        assert not r.ok
        assert r.n_mismatches > 0.1 * r.decidable_fraction * gold["n"].size, r.summary()

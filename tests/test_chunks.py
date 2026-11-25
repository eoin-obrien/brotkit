"""Chunk planner (spike S5, ADR-0009): coverage, ceiling, target, on a simulated GPU.

The simulated GPU follows S5's measurements: a launch costs `fixed`; a chunk then takes about the
larger of the latency floor (its largest escape count x tau) and its throughput time (each
group of 32 pixels in a row runs to its slowest pixel, warp divergence, at R
lane-iterations/s), plus 0.4x the smaller, which matches the measured knee. For FP64-bound
tiers (`shared_block`) the 8 warps of a launch block share the SM's FP64 units, so a block's
floor is tau/8 x the sum of its warps' largest counts: a lone warp at max_iter runs 8x faster
than a full block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brotkit.backends.chunks import (
    CeilingInfeasibleError,
    Chunk,
    ChunkPlanner,
    ChunkPolicy,
    Estimates,
    Tile,
)

type Field = np.ndarray[tuple[int, int], np.dtype[np.int64]]


@dataclass(frozen=True)
class Gpu:
    rate: float = 6e11  # f32-like: 600 Giter/s full, 75 ns per thread-iteration floor
    tau: float = 75e-9
    fixed: float = 50e-6
    shared_block: bool = False


F32 = Gpu()
DD = Gpu(rate=1.5e9, tau=15e-6, shared_block=True)  # S5's dd numbers on an RTX 3090 Ti
QF = Gpu(rate=1.5e10, tau=1.6e-6)
F64 = Gpu(rate=1.8e10, tau=1.26e-6)


def prior_of(gpu: Gpu, slower: float = 1.0) -> Estimates:
    """A prior for `gpu` as it would be `slower` times slower (< 1: the prior is too fast)."""
    return Estimates(tau=gpu.tau * slower, rate=gpu.rate / slower)


def kernel_seconds(gpu: Gpu, field: Field, chunk: Chunk) -> tuple[float, int, int]:
    t = chunk.tile
    if chunk.calibration:  # NaN coordinates: every pixel runs to the probe's max_iter
        region = np.full((t.th, t.tw), chunk.max_iter, np.int64)
    else:
        region = np.minimum(field[t.row0 : t.row0 + t.th, t.col0 : t.col0 + t.tw], chunk.max_iter)
    pad = (-region.shape[1]) % 32
    groups = np.pad(region, ((0, 0), (0, pad))).reshape(region.shape[0], -1, 32).max(axis=2)
    n_max = int(region.max())
    if gpu.shared_block:  # blocks are 8 rows x 32 columns
        rpad = (-groups.shape[0]) % 8
        blocks = np.pad(groups, ((0, rpad), (0, 0))).reshape(-1, 8, groups.shape[1]).sum(axis=1)
        floor = float(blocks.max()) * gpu.tau / 8
    else:
        floor = n_max * gpu.tau
    thr = float(groups.sum()) * 32 / gpu.rate
    # S5's measured shape: time ~ the larger term, up to 1.4x it near the knee. Never above
    # floor + throughput, the additive model the planner bounds with.
    secs = gpu.fixed + max(floor, thr) + 0.4 * min(floor, thr)
    return secs, int(region.sum()), n_max


@dataclass
class Run:
    planner: ChunkPlanner
    times: list[float]
    chunks: list[Chunk]

    @property
    def total(self) -> float:
        return sum(self.times)


def run(
    gpu: Gpu,
    field: Field,
    max_iter: int,
    policy: ChunkPolicy | None = None,
    *,
    cold: tuple[float, float] = (1.0, 0.0),
    slow: tuple[float, float] = (1.0, float("inf")),
    prior: Estimates | None = None,
) -> Run:
    """`cold=(slowdown, warm_s)`: the GPU runs `slowdown` times slower until it has been busy
    for `warm_s` seconds (S5: clocks ramp up from idle during the first probes).
    `slow=(slowdown, after_s)`: it runs `slowdown` times slower once busy for `after_s`
    seconds (another process starts sharing it)."""
    h, w = field.shape
    p = ChunkPlanner(h=h, w=w, max_iter=max_iter, policy=policy or ChunkPolicy(), prior=prior)
    times: list[float] = []
    chunks: list[Chunk] = []
    slowdown, warm_s = cold
    while (c := p.next()) is not None:
        s, iters, n_max = kernel_seconds(gpu, field, c)
        busy = sum(times)
        if busy < warm_s:
            s *= slowdown
        if busy >= slow[1]:
            s *= slow[0]
        times.append(s)
        chunks.append(c)
        p.record(c, s, iters, n_max)
    return Run(p, times, chunks)


def coverage(p: ChunkPlanner) -> Field:
    seen = np.zeros((p.h, p.w), np.int64)
    for r in p.records:
        if not r.chunk.calibration:
            t = r.tile
            seen[t.row0 : t.row0 + t.th, t.col0 : t.col0 + t.tw] += 1
    return seen


def ideal_seconds(gpu: Gpu, field: Field) -> float:
    """One perfectly sized launch: throughput time only."""
    pad = (-field.shape[1]) % 32
    groups = np.pad(field, ((0, 0), (0, pad))).reshape(field.shape[0], -1, 32).max(axis=2)
    return float(groups.sum()) * 32 / gpu.rate


# --- fields ---------------------------------------------------------------------------------


def uniform(h: int, w: int, n: int) -> Field:
    return np.full((h, w), n, np.int64)


def split_worst(h: int, w: int, max_iter: int) -> Field:
    """Cheap exterior on top, all interior below: the first chunks see only the cheap part."""
    f = np.full((h, w), 2, np.int64)
    f[h // 3 :, :] = max_iter
    return f


def disc(h: int, w: int, max_iter: int) -> Field:
    """An interior disc in an exterior that gets slower towards it (escape-time-like)."""
    y, x = np.mgrid[0:h, 0:w]
    d = np.hypot((y - h / 2) / h, (x - w / 2) / w)
    f = np.clip((0.02 / np.maximum(d - 0.25, 1e-3)).astype(np.int64), 1, max_iter)
    f[d < 0.25] = max_iter
    return f


FIELDS: dict[str, Callable[[int, int, int], Field]] = {
    "uniform-max": lambda h, w, m: uniform(h, w, m),
    "split": split_worst,
    "disc": disc,
}

policies = st.builds(
    ChunkPolicy,
    target_s=st.sampled_from([0.01, 0.05, 0.1]),
    probe_pixels=st.sampled_from([256, 1024]),
    max_growth=st.sampled_from([2.0, 8.0]),
    tile=st.sampled_from([None, 64, 256]),
)


# --- coverage ----------------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(h=st.integers(1, 300), w=st.integers(1, 700), policy=policies, n=st.integers(1, 2000))
def test_tiles_partition_the_image(h: int, w: int, policy: ChunkPolicy, n: int) -> None:
    r = run(F32, uniform(h, w, n), 2000, policy)
    assert (coverage(r.planner) == 1).all()
    assert r.planner.done_fraction == 1.0
    assert r.planner.next() is None
    assert r.planner.outstanding == ()


def test_calibration_comes_first_and_is_not_coverage() -> None:
    r = run(F32, uniform(256, 512, 500), 500)
    calib = [c for c in r.chunks if c.calibration]
    assert calib
    assert r.chunks[: len(calib)] == calib  # all calibration before any real chunk
    assert all(c.max_iter <= 500 for c in calib)
    assert calib[0].max_iter == 64
    assert r.planner.calibrated


@settings(max_examples=40, deadline=None)
@given(h=st.integers(1, 300), w=st.integers(1, 5000), policy=policies)
def test_calibration_tiles_fit_the_scratch_buffer(h: int, w: int, policy: ChunkPolicy) -> None:
    """S5's first 4K sweep wrote the rate probe past a scratch buffer sized for the latency
    probe (CUDA_ERROR_ILLEGAL_ADDRESS). Runners size scratch from `scratch_shape`."""
    r = run(QF, uniform(h, w, 2000), 2000, policy)
    sh, sw = r.planner.scratch_shape
    calib = [c.tile for c in r.chunks if c.calibration]
    assert calib
    assert all(t.row0 == 0 and t.col0 == 0 and t.th <= sh and t.tw <= sw for t in calib)


def test_bands_are_full_width_once_past_the_ramp() -> None:
    r = run(F32, uniform(512, 1024, 100), 1000)
    real = [c.tile for c in r.chunks if not c.calibration]
    bands = [t for t in real if t.col0 == 0 and t.tw == 1024]
    assert len(bands) >= len(real) - 4
    assert all(t.th % 8 == 0 or t.row0 + t.th == 512 for t in bands)


def test_square_tiles_stay_inside_their_tile() -> None:
    r = run(F32, uniform(600, 900, 100), 1000, ChunkPolicy(tile=256))
    for c in r.chunks:
        if c.calibration:  # calibration tiles live in scratch, not on the image
            continue
        t = c.tile
        assert t.row0 // 256 == (t.row0 + t.th - 1) // 256
        assert t.col0 // 256 == (t.col0 + t.tw - 1) // 256


# --- hard ceiling ---------------------------------------------------------------------------


@pytest.mark.parametrize("gpu", [F32, QF, DD], ids=["f32", "qf", "dd"])
@pytest.mark.parametrize("name", sorted(FIELDS))
@pytest.mark.parametrize("tile", [None, 256])
def test_no_chunk_exceeds_the_ceiling(gpu: Gpu, name: str, tile: int | None) -> None:
    max_iter = 5000 if gpu is DD else 20000
    policy = ChunkPolicy(tile=tile)
    r = run(gpu, FIELDS[name](540, 960, max_iter), max_iter, policy)
    assert max(r.times) <= policy.ceiling_s
    calib = [t for c, t in zip(r.chunks, r.times, strict=True) if c.calibration]
    assert max(calib) <= policy.calib_budget_s  # calibration is cheap by construction
    assert r.times[0] <= 64 * gpu.tau + 1024 * 64 / gpu.rate + gpu.fixed  # a tiny first probe
    for rec in r.planner.records[1:]:  # every issued bound respected the ceiling
        assert rec.predicted_worst_s <= policy.ceiling_s * 1.0001


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1), max_iter=st.sampled_from([500, 5000, 50000]), policy=policies
)
def test_ceiling_holds_on_random_blocky_fields(
    seed: int, max_iter: int, policy: ChunkPolicy
) -> None:
    rng = np.random.default_rng(seed)
    coarse = rng.choice([1, 10, max_iter // 10, max_iter], size=(9, 16), p=[0.4, 0.2, 0.2, 0.2])
    field = np.kron(coarse, np.ones((40, 40), np.int64)).astype(np.int64)
    gpu = QF if max_iter == 50000 else F32
    r = run(gpu, field, max_iter, policy)
    assert max(r.times) <= policy.ceiling_s


def test_floor_over_the_ceiling_is_refused_after_calibration_and_confirmation() -> None:
    """dd at 50000 iterations: 50000 x 15 us = 0.75 s per thread, over the 0.5 s ceiling."""
    p = ChunkPlanner(h=540, w=960, max_iter=50000)
    field = uniform(540, 960, 50000)
    chunks: list[Chunk] = []
    times: list[float] = []

    def drive() -> None:
        while (c := p.next()) is not None:
            s, iters, n_max = kernel_seconds(DD, field, c)
            chunks.append(c)
            times.append(s)
            p.record(c, s, iters, n_max)

    with pytest.raises(CeilingInfeasibleError, match="iteration-sliced") as err:
        drive()
    assert "#76" in str(err.value)
    assert err.value.alternative is None  # no tier given: nothing to suggest
    assert all(c.calibration for c in chunks)  # nothing touched the image
    confirm = p.policy.confirm_probes
    assert all(t <= p.policy.calib_budget_s for t in times[:-confirm])  # calibration: cheap
    assert all(t <= p.policy.ceiling_s / 2 for t in times[-confirm:])  # confirmation: bounded
    assert 20000 < p.max_feasible_iter < 30000  # 0.5 / (1.25 x ~15 us)


def test_a_cold_clock_does_not_refuse_a_feasible_render() -> None:
    """S5's dd sweep at max_iter=10000 (floor ~0.15 s) was refused: the first probes after
    idle ran ~3x slow, so the cold tau put the floor over the ceiling. Confirmation probes
    warm the clock and re-measure."""
    field = uniform(270, 480, 10000)
    r = run(DD, field, 10000, cold=(3.4, 0.05))
    assert (coverage(r.planner) == 1).all()
    assert max(r.times) <= r.planner.policy.ceiling_s
    assert r.planner.tau == pytest.approx(DD.tau, rel=0.1)


def test_a_lone_slow_warp_does_not_lower_tau() -> None:
    """S5, dd interior: an 8x32 chunk with one warp at max_iter finished 7x under the block
    floor; taking it as tau let later bounds run 3x over. Ragged chunks don't measure tau."""
    field = uniform(64, 512, 2)
    field[::8, ::64] = 5000  # one pixel per block in a lone warp
    r = run(DD, field, 5000)
    assert r.planner.tau == pytest.approx(DD.tau, rel=0.15)
    for rec in r.planner.records:
        if np.isfinite(rec.predicted_worst_s):
            assert rec.seconds <= rec.predicted_worst_s


def test_a_floor_heavy_render_amortises_the_floor() -> None:
    """dd at max_iter=10000: the floor (0.15 s) is over the 0.1 s target, so chunks grow to the
    ceiling to keep the floor's share down, even from a cold start (S5: 1621 tiny chunks, 8.7x
    the work time, with the target-or-knee rule)."""
    field = uniform(540, 960, 10000)
    r = run(DD, field, 10000, cold=(5.0, 0.2))
    real = [t for c, t in zip(r.chunks, r.times, strict=True) if not c.calibration]
    assert max(r.times) <= r.planner.policy.ceiling_s
    assert float(np.median(real)) >= 0.25
    assert r.total <= 2.0 * ideal_seconds(DD, field)


def test_tau_is_rechecked_once_the_clock_is_warm() -> None:
    """S5, f64 interior: tau from a cold calibration stayed 5x high all render, because big
    chunks are throughput-bound and never show it; the floor-share rule then sized 240 ms
    chunks against a 100 ms target. A latency probe after 0.5 s of GPU time fixes tau."""
    gpu = Gpu(rate=1.8e10, tau=1.26e-6)  # f64-like
    r = run(gpu, uniform(1080, 1920, 10000), 10000, cold=(5.0, 0.3))
    assert r.planner.tau == pytest.approx(gpu.tau, rel=0.15)
    late = [t for c, t in zip(r.chunks, r.times, strict=True) if not c.calibration][-10:]
    assert max(late) <= 2 * r.planner.policy.target_s


def test_a_warm_start_skips_calibration() -> None:
    field = disc(1080, 1920, 5000)
    first = run(QF, field, 5000, cold=(4.0, 0.2))
    assert first.planner.estimates is not None
    second = run(QF, field, 5000, prior=first.planner.estimates)
    assert (coverage(second.planner) == 1).all()
    assert any(c.calibration for c in first.chunks)
    assert not any(c.calibration for c in second.chunks)
    assert max(second.times) <= second.planner.policy.ceiling_s


def test_a_warm_prior_on_a_cold_gpu_keeps_the_bound() -> None:
    """S5: the clock drops between renders, so a render's first chunks ran 4-5x slower than the
    previous render's (warm) estimates, overrunning their bounds by up to 4.9x. The first chunk
    now assumes `prior_cold` (10x) and later ones the slowdown measured on the chunks before."""
    field = uniform(2160, 3840, 10000)
    gpu = F64
    prior = run(gpu, field, 10000).planner.estimates
    r = run(gpu, field, 10000, prior=prior, cold=(5.0, 0.3))
    for rec in r.planner.records:
        if not rec.chunk.calibration:
            assert rec.seconds <= rec.predicted_worst_s
    assert max(r.times) <= r.planner.policy.ceiling_s


def test_a_stale_prior_stays_under_the_ceiling() -> None:
    """The prior came from an idle GPU; now another process takes two thirds of it."""
    field = uniform(1080, 1920, 5000)
    prior = run(QF, field, 5000).planner.estimates
    r = run(QF, field, 5000, prior=prior, slow=(3.0, 0.0))
    assert max(r.times) <= r.planner.policy.ceiling_s
    assert (coverage(r.planner) == 1).all()


def test_estimates_follow_a_gpu_that_slows_down() -> None:
    """R only ever rises and tau only falls otherwise; a chunk over its bound scales them back,
    so after a sudden 3x slowdown the bound holds again within a chunk or two."""
    field = uniform(2160, 3840, 5000)
    r = run(QF, field, 5000, slow=(3.0, 1.0))
    over = [
        i
        for i, rec in enumerate(r.planner.records)
        if not rec.chunk.calibration and rec.seconds > rec.predicted_worst_s
    ]
    assert over  # the slowdown was felt
    assert len(over) <= 2
    assert over[-1] - over[0] <= 2


def test_tau_tightens_from_real_chunks() -> None:
    """A slow calibration (cold clock) must not fix tau for the whole render."""
    r = run(QF, uniform(540, 960, 2000), 2000, cold=(3.0, 0.004))
    calib_tau = min(rec.seconds / rec.max_n for rec in r.planner.records if rec.chunk.calibration)
    assert r.planner.tau < calib_tau


# --- prior (warm start from the kernel cache) ------------------------------------------------


@pytest.mark.parametrize("gpu", [F32, F64, QF], ids=["f32", "f64", "qf"])
@pytest.mark.parametrize("name", sorted(FIELDS))
def test_a_prior_needs_no_calibration_launches(gpu: Gpu, name: str) -> None:
    max_iter = 10000
    r = run(gpu, FIELDS[name](540, 960, max_iter), max_iter, prior=prior_of(gpu))
    assert not any(c.calibration for c in r.chunks)
    assert (coverage(r.planner) == 1).all()
    assert max(r.times) <= r.planner.policy.ceiling_s


@pytest.mark.parametrize("gpu", [F32, F64, QF], ids=["f32", "f64", "qf"])
@pytest.mark.parametrize("slowdown", [3.0, 9.0])
def test_a_prior_on_a_cold_gpu_keeps_every_bound(gpu: Gpu, slowdown: float) -> None:
    """S5: after idle the first launches ran 3-9x slower than a warm render."""
    field = split_worst(1080, 1920, 10000)
    r = run(gpu, field, 10000, prior=prior_of(gpu), cold=(slowdown, 0.3))
    for rec in r.planner.records:
        assert rec.seconds <= rec.predicted_worst_s
    assert r.times[0] <= r.planner.policy.target_s  # the first chunk is sized for the target


@pytest.mark.parametrize("gpu", [F32, F64, QF], ids=["f32", "f64", "qf"])
@pytest.mark.parametrize("slower", [10.0, 40.0])
def test_a_prior_far_too_fast_stays_under_the_ceiling(gpu: Gpu, slower: float) -> None:
    """The prior is a hint: a GPU 10x or 40x slower than recorded (another process holding
    it, a power limit) still never sees a chunk over the ceiling."""
    field = uniform(1080, 1920, 10000)
    r = run(gpu, field, 10000, prior=prior_of(gpu, 1 / slower))
    assert max(r.times) <= r.planner.policy.ceiling_s
    assert (coverage(r.planner) == 1).all()


def test_a_prior_too_slow_is_overtaken_by_measurement() -> None:
    field = disc(1080, 1920, 5000)
    exact = run(QF, field, 5000, prior=prior_of(QF))
    slow = run(QF, field, 5000, prior=prior_of(QF, 5.0))
    assert slow.planner.rate >= 0.8 * QF.rate
    assert slow.total <= 1.3 * exact.total
    assert max(slow.times) <= slow.planner.policy.ceiling_s


def test_the_slowdown_follows_the_clock() -> None:
    r = run(F64, uniform(2160, 3840, 10000), 10000, prior=prior_of(F64), cold=(6.0, 0.3))
    assert r.planner.cold_factor <= 1.5
    late = r.times[-10:-1]
    assert min(late) >= 0.5 * r.planner.policy.target_s  # chunks grew back once warm


def test_a_deep_dd_render_with_a_prior_measures_before_its_first_chunk() -> None:
    """dd at 10,000 iterations: the floor at the assumed 10x slowdown (1.5 s) breaks the
    ceiling, so one cheap probe measures the slowdown first; on a warm GPU that is all."""
    field = uniform(270, 480, 10000)
    r = run(DD, field, 10000, prior=prior_of(DD))
    assert r.chunks[0].calibration
    assert sum(c.calibration for c in r.chunks) == 1
    assert r.times[0] <= r.planner.policy.calib_budget_s
    assert (coverage(r.planner) == 1).all()
    assert max(r.times) <= r.planner.policy.ceiling_s


def test_a_deep_dd_render_with_a_prior_on_a_cold_gpu() -> None:
    field = uniform(270, 480, 10000)
    r = run(DD, field, 10000, prior=prior_of(DD), cold=(5.0, 0.3))
    pol = r.planner.policy
    assert sum(c.calibration for c in r.chunks) <= 1 + pol.confirm_probes
    assert (coverage(r.planner) == 1).all()
    assert max(r.times) <= pol.ceiling_s


def test_refusing_dd_names_qf_and_the_backlog_row() -> None:
    p = ChunkPlanner(h=540, w=960, max_iter=50000, tier="dd", prior=prior_of(DD))
    field = uniform(540, 960, 50000)

    def drive() -> None:
        while (c := p.next()) is not None:
            p.record(c, *kernel_seconds(DD, field, c))

    with pytest.raises(CeilingInfeasibleError) as err:
        drive()
    msg = str(err.value)
    assert "qf" in msg
    assert "#76" in msg
    assert err.value.alternative == "qf"
    assert 20000 < err.value.max_feasible_iter < 30000
    assert all(r.chunk.calibration for r in p.records)


@given(tau=st.floats(1e-12, 1.0), rate=st.floats(1.0, 1e15))
def test_estimates_json_round_trip(tau: float, rate: float) -> None:
    e = Estimates(tau=tau, rate=rate)
    assert Estimates.from_json(e.to_json()) == e


@pytest.mark.parametrize(
    "obj",
    [
        None,
        [],
        "x",
        {},
        {"tau": 1e-6},
        {"tau": "1e-6", "rate": 1e9},
        {"tau": True, "rate": 1e9},
        {"tau": 0.0, "rate": 1e9},
        {"tau": -1e-6, "rate": 1e9},
        {"tau": 1e-6, "rate": float("inf")},
        {"tau": float("nan"), "rate": 1e9},
    ],
)
def test_damaged_estimates_read_as_none(obj: object) -> None:
    assert Estimates.from_json(obj) is None


# --- efficiency -----------------------------------------------------------------------------------


@pytest.mark.parametrize("gpu", [F32, QF, DD], ids=["f32", "qf", "dd"])
def test_no_small_chunk_trap(gpu: Gpu) -> None:
    """S5's first planner sized chunks below the latency knee in dd and ran 16x slow (6.7x
    ideal in this simulator). Chunks now sit at or above the knee; what remains is the floor
    each interior-crossing band pays and the simulator's 1.4x knee penalty (hardware: ~1x)."""
    max_iter = 5000
    field = disc(1080, 1920, max_iter)
    r = run(gpu, field, max_iter)
    assert r.total <= 2.5 * ideal_seconds(gpu, field) + 0.05


@pytest.mark.parametrize("n", [2000, 20000])
def test_chunks_settle_near_the_target_on_a_uniform_field(n: int) -> None:
    policy = ChunkPolicy(target_s=0.01)
    r = run(F32, uniform(2160, 3840, n), 5000, policy)
    real = [t for c, t in zip(r.chunks, r.times, strict=True) if not c.calibration]
    settled = real[len(real) // 2 : -1] or real[-1:]
    assert 0.004 <= float(np.median(settled)) <= 0.02


def test_expected_seconds_follows_the_recent_chunks() -> None:
    r = run(F32, uniform(64, 4096, 300), 300)
    p = r.planner
    last = r.chunks[-1].tile
    assert p.expected_seconds(last.pixels) == pytest.approx(r.times[-1], rel=0.5)


def test_outstanding_lists_issued_but_unrecorded_chunks() -> None:
    p = ChunkPlanner(h=64, w=64, max_iter=100)
    c = p.next()
    assert p.outstanding == (c,)


def test_tile_pixels() -> None:
    assert Tile(0, 0, 8, 32).pixels == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_s": 0.0},
        {"target_s": 1.0, "ceiling_s": 0.5},
        {"safety": 0.9},
        {"probe_pixels": 0},
        {"calib_budget_s": 0.0},
        {"confirm_probes": -1},
        {"floor_share": 0.0},
        {"uniform_frac": 1.5},
        {"max_growth": 1.0},
        {"tile": 0},
        {"prior_cold": 0.5},
        {"cold_window": 0},
        {"ratio_min_s": -1.0},
    ],
)
def test_policy_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"must|need"):
        ChunkPolicy(**kwargs)  # pyright: ignore[reportArgumentType]


def test_planner_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ChunkPlanner(h=0, w=10, max_iter=10)

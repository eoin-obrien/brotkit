"""Chunk planning for interruptible launches (spike S5, ADR-0009). GPU-free.

A render runs as a sequence of chunks so the host can show progress and stop between them. A
chunk is a tile `(row0, col0, th, tw)`; a band is a full-width tile. A running kernel can't be
cancelled, so a chunk's duration is also the worst-case cancel latency.

**Time model** (S5, measured on an RTX 3090 Ti for every tier). A chunk of `p` pixels whose
pixels run at most `m` iterations takes at most

    t(p) <= m * tau + p * m / R

`tau` is the time of one iteration for a launch block whose every thread runs to `m` (the
latency floor: below a knee of about `R * tau` pixels a chunk costs `m * tau` however small it
is), and `R` is the iteration throughput with the GPU full. Both are learned from finished
chunks and never from one probe: `R` as the best measured iterations per second, `tau` as the
least `seconds / largest escape count` over *uniform* chunks (every pixel at the chunk's largest
count: calibration probes, solid interior). Those measurements can only err slow (a cold clock,
another process's time slice), so the estimates tighten as the GPU warms up (S5: the first
probes after idle ran 3-6x slow). A ragged chunk can run faster than `tau` (S5, dd: a lone warp
at `max_iter` in an otherwise finished block ran 7x faster, since the block's warps share the
few FP64 units), so ragged chunks never lower `tau`. Calibration chunks run on NaN coordinates
so every pixel reaches the probe's iteration count whatever the formula (S5: NaN tau is within
6% of real interior tau, or above it; the safety margin covers the difference).

**Ceiling.** Every chunk satisfies `safety * t(p) <= ceiling_s` with `m = max_iter`, whatever
the image holds. The first chunks are calibration probes, in two phases, each growing its
iteration count 4x while the projected time stays under `calib_budget_s`: a tiny tile measures
`tau`, then a large tile (above the knee) measures `R`, repeated at the largest iteration count
the budget allows until `R` settles. If later chunks run well under the model while the cap
limits growth, `R` is stale (it was learned on a cold clock) and the planner slips in another
throughput probe. So a bad first guess can never come near
the driver watchdog, and the worst-case bound starts from a near-full throughput figure rather
than from cheap exterior pixels. Calibration output goes to scratch and is discarded.
If even the floor `max_iter * tau` breaks the ceiling, no spatial chunk can. The planner then
runs up to `confirm_probes` more latency probes, each bounded to half the ceiling, which warm
the clock and re-measure `tau`; only if the floor still breaks the ceiling does it raise
`CeilingInfeasibleError` rather than launch (that needs iteration-sliced kernels).

**Target.** Within the ceiling, chunks aim for `target_s` using the recent chunks' cost, and
never drop below the knee while the ceiling allows, since smaller chunks don't run faster. When
the floor is a large part of the target (deep iteration counts in dd), the aim rises to
`floor / floor_share` so the floor costs at most that share of each chunk; the ceiling still
caps it. One
probe can underestimate later chunks by 10x or more (bands that cross the interior run to
`max_iter` on every pixel), so the estimate follows every finished chunk.

**Adapting both ways.** The estimates tighten as measurements come in, and a chunk that runs
over its bound (a GPU now shared with another process, a lower clock) scales `R` and `tau` back
by the overrun, so the next chunks shrink. Between the overrun and that correction nothing can
help: a running kernel can't be cancelled.

**Warm start.** `prior=` takes a previous render's `estimates` (same kernel, same GPU) and skips
the throughput probes. The latency probes still run: the GPU drops its clock between renders, so
a render's first chunks run several times slower than the prior's warm numbers (S5: 4-9x), and
a first chunk sized from the prior alone could overrun the ceiling by that much. `R` starts at
the prior's, divided by the cold factor the latency probes measure, and the first chunk gets a
quarter of the ceiling's throughput room.

Chunks are carved from regions: the whole image (bands) or square tiles of `tile` pixels. Inside
a region the planner takes whole strips of `strip_rows` rows when it can and a column segment of
one strip when even one strip would be too big.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "DEFAULT_CEILING_S",
    "DEFAULT_TARGET_S",
    "CeilingInfeasibleError",
    "Chunk",
    "ChunkPlanner",
    "ChunkPolicy",
    "ChunkRecord",
    "Estimates",
    "Tile",
]

DEFAULT_TARGET_S: Final = 0.1
DEFAULT_CEILING_S: Final = 0.5
"""25% of the Windows default `TdrDelay` (2 s), the watchdog WDDM applies under WSL2 (S5)."""


class CeilingInfeasibleError(RuntimeError):
    """One thread's `max_iter` iterations alone would take longer than the ceiling."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkPolicy:
    target_s: float = DEFAULT_TARGET_S
    ceiling_s: float = DEFAULT_CEILING_S
    safety: float = 1.25  # margin on the time model (clock drift, model error)
    probe_pixels: int = 1024  # latency calibration tile (below the knee)
    rate_probe_rows: int = 64  # throughput calibration tile: rows x cols, above the knee
    rate_probe_cols: int = 4096
    calib_iter: int = 64  # first calibration iteration count
    calib_budget_s: float = 0.02  # a calibration chunk's projected worst case stays below this
    confirm_probes: int = 3  # re-measure tau this many times before refusing a render
    floor_share: float = 0.25  # aim for chunks where the floor is at most this share of the time
    uniform_frac: float = 0.95  # a chunk is uniform (it measures tau) above this fill
    rate_probe_max: int = 8  # at most this many throughput probes
    rate_settle: float = 1.1  # throughput probes stop once R improves by less than this factor
    reprobe_every: int = 4  # a stale-R throughput probe at most every this many chunks
    reprobe_max: int = 8  # and at most this many per render
    stale_ratio: float = 0.65  # actual / modelled below this: R is stale (S5: >= 0.8 when not)
    tau_recheck_s: tuple[float, ...] = (0.5, 2.0)  # re-measure tau after this much GPU time
    max_growth: float = 8.0  # chunk pixels grow at most this much per step
    strip_rows: int = 8  # band heights are multiples of this (the launch block height)
    col_align: int = 32  # column segments are multiples of this (the launch block width)
    tile: int | None = None  # None: full-width bands; else square tiles of this many pixels

    def __post_init__(self) -> None:
        if not 0 < self.target_s <= self.ceiling_s:
            raise ValueError("need 0 < target_s <= ceiling_s")
        if self.safety < 1:
            raise ValueError("safety must be >= 1")
        sizes = (
            self.probe_pixels,
            self.rate_probe_rows,
            self.rate_probe_cols,
            self.calib_iter,
            self.strip_rows,
            self.col_align,
        )
        if min(sizes) < 1:
            raise ValueError("probe sizes, calib_iter, strip_rows and col_align must be positive")
        if not 0 < self.calib_budget_s <= self.ceiling_s:
            raise ValueError("need 0 < calib_budget_s <= ceiling_s")
        if self.confirm_probes < 0:
            raise ValueError("confirm_probes must be >= 0")
        if not 0 < self.floor_share <= 1:
            raise ValueError("need 0 < floor_share <= 1")
        if not 0 < self.stale_ratio < 1:
            raise ValueError("need 0 < stale_ratio < 1")
        if self.reprobe_every < 1 or self.reprobe_max < 0:
            raise ValueError("need reprobe_every >= 1 and reprobe_max >= 0")
        if self.rate_probe_max < 1 or self.rate_settle <= 1:
            raise ValueError("need rate_probe_max >= 1 and rate_settle > 1")
        if not 0 < self.uniform_frac <= 1:
            raise ValueError("need 0 < uniform_frac <= 1")
        if self.max_growth <= 1:
            raise ValueError("max_growth must be > 1")
        if self.tile is not None and self.tile < 1:
            raise ValueError("tile must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class Estimates:
    """A planner's learned time model, to warm-start the next render of the same kernel."""

    tau: float  # seconds per iteration for a full launch block
    rate: float  # iterations per second with the GPU full

    def __post_init__(self) -> None:
        if not (self.tau > 0 and self.rate > 0):
            raise ValueError("tau and rate must be positive")


@dataclass(frozen=True, slots=True)
class Tile:
    row0: int
    col0: int
    th: int
    tw: int

    @property
    def pixels(self) -> int:
        return self.th * self.tw


@dataclass(frozen=True, slots=True)
class Chunk:
    tile: Tile
    max_iter: int  # iteration cap for this launch (below the render's for calibration)
    calibration: bool = False  # run on NaN coordinates into scratch; output discarded


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkRecord:
    chunk: Chunk
    seconds: float  # kernel time (GPU events), not wall time
    iterations: int  # sum of escape counts over the tile
    max_n: int  # largest escape count in the tile
    predicted_worst_s: float  # the planner's bound when it issued the chunk

    @property
    def tile(self) -> Tile:
        return self.chunk.tile


@dataclass(slots=True)
class _Region:
    row0: int
    col0: int
    h: int
    w: int
    row: int = 0  # cursor, relative to the region
    col: int = 0


@dataclass(slots=True)
class ChunkPlanner:
    """Issues chunks covering an `h` x `w` image exactly once; learns from `record`."""

    h: int
    w: int
    max_iter: int
    policy: ChunkPolicy = field(default_factory=ChunkPolicy)
    prior: Estimates | None = None  # a previous render's estimates: skip calibration
    records: list[ChunkRecord] = field(default_factory=list[ChunkRecord])
    _regions: list[_Region] = field(default_factory=list[_Region])
    _pending: dict[Chunk, float] = field(default_factory=dict[Chunk, float])
    _rate: float = 0.0  # best measured iterations / second (a lower bound on R)
    _tau: float = 0.0  # per-thread seconds per iteration: least seconds / max_n seen
    _confirmations: int = 0
    _rate_probes: int = 0
    _reprobes: int = 0
    _tau_rechecks: int = 0
    _busy_s: float = 0.0
    _since_probe: int = 0
    _last_loose: bool = False
    _reprobe_chunk: Chunk | None = None
    _reprobe_rate_before: float = 0.0
    _last_probe_rate: float = 0.0
    _calibrated: bool = False
    _calib_phase: int = 0  # 0: latency (tau), 1: throughput (R), then done
    _calib_iter: int = 0
    _recent_n_mean: float = 0.0
    _recent_n_max: int = 0
    _last_pixels: int = 0
    _done_pixels: int = 0

    def __post_init__(self) -> None:
        if self.h < 1 or self.w < 1 or self.max_iter < 1:
            raise ValueError("h, w and max_iter must be positive")
        t = self.policy.tile
        if t is None:
            self._regions = [_Region(0, 0, self.h, self.w)]
        else:
            self._regions = [
                _Region(r, c, min(t, self.h - r), min(t, self.w - c))
                for r in range(0, self.h, t)
                for c in range(0, self.w, t)
            ]
        self._regions.reverse()  # pop() from the end = row-major order
        self._calib_iter = min(self.policy.calib_iter, self.max_iter)
        if self.prior is not None:
            # The latency probes still run: the clock drops between renders, so the first chunks
            # run several times slower than the prior's warm numbers (S5: 4-9x). R starts from
            # the prior scaled by the cold factor the probes measure; see `_calibrate`.
            self._rate = self.prior.rate

    # -- the time model ------------------------------------------------------------------------

    def worst_case_seconds(self, pixels: int, max_iter: int | None = None) -> float:
        """Upper bound (with the safety margin) on a chunk of `pixels`; inf before data."""
        if self._rate <= 0 or self._tau <= 0:
            return math.inf
        m = self.max_iter if max_iter is None else max_iter
        return self.policy.safety * (m * self._tau + pixels * m / self._rate)

    def expected_seconds(self, pixels: int) -> float:
        """Expected kernel time for `pixels` like the recent ones (0 before any data)."""
        if self._rate <= 0 or self._tau <= 0:
            return 0.0
        return self._recent_n_max * self._tau + pixels * self._recent_n_mean / self._rate

    @property
    def floor_seconds(self) -> float:
        """`max_iter * tau`: the least any chunk can take (inf before calibration)."""
        return self.max_iter * self._tau if self._tau > 0 else math.inf

    @property
    def max_feasible_iter(self) -> int:
        """The largest max_iter whose floor fits under the ceiling with the safety margin."""
        if self._tau <= 0:
            return 0
        return int(self.policy.ceiling_s / (self.policy.safety * self._tau))

    def _cap_pixels(self) -> int:
        """Most pixels whose bound stays under the ceiling."""
        pol = self.policy
        room = pol.ceiling_s / pol.safety - self.max_iter * self._tau
        if self.prior is not None and self._done_pixels == 0:
            room *= 0.25  # the prior may be stale (a shared or cooler GPU): start small
        return int(room * self._rate / self.max_iter)

    def budget_pixels(self) -> int:
        """How many pixels the next (non-calibration) chunk may have."""
        pol = self.policy
        cap = self._cap_pixels()
        knee = int(self._rate * self._tau)
        if self._recent_n_mean > 0:
            floor = self._recent_n_max * self._tau
            room = max(pol.target_s, floor / pol.floor_share) - floor
            per_px = self._recent_n_mean / self._rate
            target = int(room / per_px) if room > 0 else 0
        else:
            target = cap
        want = max(target, knee)
        growth = int(pol.max_growth * max(self._last_pixels, 1))
        return max(1, min(cap, want, growth))

    # -- issuing -------------------------------------------------------------------------------

    def next(self) -> Chunk | None:
        """The next chunk, or None when every pixel has been rendered."""
        if not self._calibrated:
            tile = self._probe_tile() if self._calib_phase == 0 else self._rate_probe_tile()
            chunk = Chunk(tile, self._calib_iter, calibration=True)
            self._pending[chunk] = self.worst_case_seconds(tile.pixels, self._calib_iter)
            return chunk
        if self.floor_seconds * self.policy.safety > self.policy.ceiling_s:
            if self._confirmations < self.policy.confirm_probes:
                # tau may come from a cold clock or a shared GPU: measure again before refusing.
                self._confirmations += 1
                tile = self._probe_tile()
                per_iter = self.policy.safety * (self._tau + tile.pixels / self._rate)
                m = max(1, min(self.max_iter, int(0.5 * self.policy.ceiling_s / per_iter)))
                chunk = Chunk(tile, m, calibration=True)
                self._pending[chunk] = self.worst_case_seconds(tile.pixels, m)
                return chunk
            raise CeilingInfeasibleError(
                f"max_iter={self.max_iter} alone takes about {self.floor_seconds:.2f} s per "
                f"thread on this GPU, over the {self.policy.ceiling_s:.2f} s chunk ceiling; "
                f"the most this kernel can run in one launch is about {self.max_feasible_iter} "
                "iterations (iteration-sliced launches are backlog work)"
            )
        while self._regions and self._regions[-1].row >= self._regions[-1].h:
            self._regions.pop()
        if not self._regions:
            return None
        if self._tau_recheck_due():
            self._tau_rechecks += 1
            tile = self._probe_tile()
            per_iter = self.policy.safety * (self._tau + tile.pixels / self._rate)
            m = max(1, min(self.max_iter, int(self.policy.calib_budget_s / per_iter)))
            chunk = Chunk(tile, m, calibration=True)
            self._pending[chunk] = self.worst_case_seconds(tile.pixels, m)
            return chunk
        if self._reprobe_due():
            self._reprobes += 1
            self._since_probe = 0
            m = self._rate_probe_iter()
            tile = self._rate_probe_tile()
            chunk = Chunk(tile, m, calibration=True)
            self._pending[chunk] = self.worst_case_seconds(tile.pixels, m)
            self._reprobe_rate_before = self._rate
            self._reprobe_chunk = chunk
            return chunk
        reg = self._regions[-1]
        pol = self.policy
        budget = self.budget_pixels()
        strip = min(pol.strip_rows, reg.h - reg.row)
        if reg.col == 0 and budget >= strip * reg.w:
            rows = (budget // reg.w) // pol.strip_rows * pol.strip_rows
            # Round up to the next strip when that stays under the ceiling: rounding a knee-sized
            # budget down would leave every chunk below the knee, paying the floor for less work.
            up = rows + pol.strip_rows
            if self.worst_case_seconds(up * reg.w) <= pol.ceiling_s and up * reg.w <= 2 * budget:
                rows = up
            rows = max(strip, min(rows, reg.h - reg.row))
            tile = Tile(reg.row0 + reg.row, reg.col0, rows, reg.w)
            reg.row += rows
        else:
            cols = budget // strip // pol.col_align * pol.col_align
            cols = max(min(pol.col_align, reg.w - reg.col), min(cols, reg.w - reg.col))
            tile = Tile(reg.row0 + reg.row, reg.col0 + reg.col, strip, cols)
            reg.col += cols
            if reg.col >= reg.w:
                reg.col = 0
                reg.row += strip
        chunk = Chunk(tile, self.max_iter)
        self._pending[chunk] = self.worst_case_seconds(tile.pixels)
        return chunk

    @property
    def scratch_shape(self) -> tuple[int, int]:
        """(rows, cols) the caller's calibration scratch buffer must hold.

        Calibration tiles always start at (0, 0) and run on NaN coordinates in a scratch
        buffer of this shape (pitch = cols), never on the image.
        """
        probe, rate = self._probe_tile(), self._rate_probe_tile()
        return max(probe.th, rate.th), max(probe.tw, rate.tw)

    def _tau_recheck_due(self) -> bool:
        """Measure tau again once the GPU has been busy for each of `tau_recheck_s` seconds.

        Large chunks are throughput-bound and never show tau, so a tau learned on a cold clock
        would stay high for the whole render (S5, f64: 6.2 us against 1.26 us warm), inflating
        the floor and, through `floor_share`, the chunks.
        """
        marks = self.policy.tau_recheck_s
        return self._tau_rechecks < len(marks) and self._busy_s >= marks[self._tau_rechecks]

    def _reprobe_due(self) -> bool:
        """R is stale: the last chunk ran in under `stale_ratio` of the model's time for its own
        iterations (largest count and sum), while the cap limits growth.

        Real chunks below the knee only ever show a low R, so they can't correct an R learned
        on a cold clock; a fresh throughput probe can (S5: dd's GPU took ~0.3 s of load to
        reach its boost clock, far longer than calibration).
        """
        pol = self.policy
        return (
            self._last_loose
            and self._since_probe >= pol.reprobe_every
            and self._reprobes < pol.reprobe_max
            and self.budget_pixels() >= self._cap_pixels()
        )

    def _rate_probe_iter(self) -> int:
        """Largest iteration count whose projected cost on the rate probe fits the budget."""
        big = self._rate_probe_tile().pixels
        per_iter = self.policy.safety * (self._tau + big / self._rate)
        return max(1, min(self.max_iter, int(self.policy.calib_budget_s / per_iter)))

    def _rate_probe_tile(self) -> Tile:
        """Throughput probe; it runs in the caller's scratch buffer, not on the image."""
        return Tile(0, 0, self.policy.rate_probe_rows, self.policy.rate_probe_cols)

    def _probe_tile(self) -> Tile:
        pol = self.policy
        th = min(pol.strip_rows, self.h)
        cols = pol.probe_pixels // th // pol.col_align * pol.col_align
        return Tile(0, 0, th, min(self.w, max(pol.col_align, cols)))

    def record(self, chunk: Chunk, seconds: float, iterations: int, max_n: int) -> ChunkRecord:
        """Report a finished chunk's kernel time, iteration sum and largest escape count."""
        predicted = self._pending.pop(chunk)
        rec = ChunkRecord(
            chunk=chunk,
            seconds=seconds,
            iterations=iterations,
            max_n=max_n,
            predicted_worst_s=predicted,
        )
        self.records.append(rec)
        self._busy_s += max(seconds, 0.0)
        # What the model said this chunk's own work costs, before this chunk updates it.
        modelled = (
            max_n * self._tau + iterations / self._rate if self._rate > 0 and self._tau > 0 else 0.0
        )
        if seconds > 0 and iterations > 0:
            self._rate = max(self._rate, iterations / seconds)
        # Uniform: every pixel near max_n and at least one full launch block (a short tile's
        # blocks are part-empty and run faster).
        t = chunk.tile
        full = t.th >= min(self.policy.strip_rows, self.h) or chunk.calibration
        uniform = full and iterations >= self.policy.uniform_frac * t.pixels * max_n
        if seconds > 0 and max_n > 0 and uniform:
            # A uniform chunk takes at least max_n * tau, so this is an upper bound on tau.
            est = seconds / max_n
            self._tau = est if self._tau <= 0 else min(self._tau, est)
        if not chunk.calibration and seconds > predicted:
            # Slower than the bound allowed (a shared GPU, a lower clock): the estimates only
            # ever tighten otherwise, so scale them back by the overrun.
            f = seconds / predicted
            self._rate /= f
            self._tau *= f
        if chunk.calibration:
            if chunk == self._reprobe_chunk:
                self._reprobe_chunk = None
                if self._rate < self.policy.rate_settle * self._reprobe_rate_before:
                    self._reprobes = self.policy.reprobe_max  # R had settled: stop re-probing
            if self._calibrated:  # a confirmation, recheck or re-probe: nothing more to plan
                return rec
            self._calibrate(rec)
            return rec
        p = chunk.tile.pixels
        self._done_pixels += p
        self._last_pixels = p
        self._since_probe += 1
        self._last_loose = seconds < self.policy.stale_ratio * modelled
        self._recent_n_mean = iterations / p
        self._recent_n_max = max_n
        return rec

    def _calibrate(self, rec: ChunkRecord) -> None:
        m = rec.chunk.max_iter
        pol = self.policy
        budget = pol.calib_budget_s
        if self._calib_phase == 0:
            # Latency: probe again at 4x the iterations while that stays cheap; finish once it
            # ran at the render's own max_iter or the next probe would cost too much.
            nxt = min(self.max_iter, 4 * m)
            if m < self.max_iter and self.worst_case_seconds(rec.tile.pixels, nxt) <= budget:
                self._calib_iter = nxt
                return
            if self.prior is not None:
                cold = max(1.0, self._tau / self.prior.tau)
                self._rate = min(self._rate, self.prior.rate / cold)
                self._calibrated = True
                self._last_pixels = self._probe_tile().pixels
                return
            self._calib_phase = 1
            self._calib_iter = self._rate_probe_iter()
            return
        # Throughput: the first probes after idle run on a cold clock, and short ones are
        # dominated by per-thread setup, so R would come out several times low (S5: 6x in dd)
        # and keep every chunk below the knee. Probe again, as long as the budget allows at the
        # newest R, until R stops improving or the probe ran at the render's max_iter.
        rate = rec.iterations / rec.seconds if rec.seconds > 0 else 0.0
        improving = rate > pol.rate_settle * self._last_probe_rate
        self._last_probe_rate = max(self._last_probe_rate, rate)
        self._rate_probes += 1
        if m < self.max_iter and improving and self._rate_probes < pol.rate_probe_max:
            self._calib_iter = self._rate_probe_iter()
            return
        self._calibrated = True
        self._last_pixels = self._probe_tile().pixels

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def done_fraction(self) -> float:
        return self._done_pixels / (self.h * self.w)

    @property
    def outstanding(self) -> tuple[Chunk, ...]:
        """Chunks issued but not recorded (in flight, or lost to an interrupt)."""
        return tuple(self._pending)

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def estimates(self) -> Estimates | None:
        """The current time model, to pass as `prior` to the next render of this kernel."""
        if self._tau > 0 and self._rate > 0:
            return Estimates(tau=self._tau, rate=self._rate)
        return None

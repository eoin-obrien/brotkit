"""Spike S5 chunk runner and helpers (throwaway; the library version is ADR-0009's).

`render(...)` runs one image as chunks from `brotkit.backends.chunks.ChunkPlanner`:

    for each tile: event a -> kernel -> event b -> tile_sum -> event c -> wait -> read iterations

The kernel and the tile-sum kernel come from S2's KernelCache once per process and are reused
for every chunk. Waiting is one of:
    "spin"   stream.sync()           (cuda.core default: busy-waits in C; Ctrl-C waits for it)
    "block"  event with blocking_sync (the thread sleeps in the driver; Ctrl-C waits for it)
    "poll"   event.is_done + time.sleep(poll_s)   (Python stays responsive)
    "thread" a helper thread waits on a blocking-sync event; the main thread waits on a
             threading.Event (no CPU, no polling granularity, Ctrl-C at once)
    "hybrid" sleep through 80% of the planner's expected chunk time in slices of at most 2 ms,
             then poll every poll_s (Python stays responsive, little CPU, little added latency)
On KeyboardInterrupt the runner finishes the in-flight chunk (it can't be cancelled), records it,
and returns a partial result instead of leaving the buffers mid-write.
"""

from __future__ import annotations

import math
import pathlib
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import gen_s5  # noqa: E402

from brotkit.backends.chunks import Chunk, ChunkPlanner, ChunkPolicy, Estimates, Tile  # noqa: E402

VIEWS = {
    # name: (centre re, centre im, width, default max_iter)
    "seahorse": (-0.743643887037151, 0.131825904205330, 0.05, 1000),  # S3/S4: mean 437 iters
    "whole": (-0.5, 0.0, 3.0, 10000),  # top/bottom rows exterior, middle rows interior
    "interior": (-0.15, 0.0, 1.0, 10000),  # mostly the main cardioid
}
W4K, H4K = 3840, 2160


def s4_on_path() -> None:
    """S4's timing helpers (smi, idle check) without importing its generator as `gen`."""
    s4 = str(HERE.parent / "S4_registers")
    if s4 not in sys.path:
        sys.path.append(s4)


def wait_idle(q: Any, baseline: int, label: str, own_mib: int = 1200) -> None:
    """S4's GPU idle rule: median utilisation <= 5%, else wait 20 s and check again."""
    deadline = time.monotonic() + 900
    while True:
        try:
            q.gpu_idle_check(baseline, own_mib, label)
            return
        except q.Busy as e:
            if time.monotonic() > deadline:
                raise
            print(f"waiting 20 s: {e}", flush=True)
            time.sleep(20)


def view_args(tier: str, view: str, w: int) -> tuple[Any, ...]:
    word, nw, _ = gen_s5.TIERS[tier]
    dt = np.dtype(word).type
    cr, ci, width, _ = VIEWS[view]
    out = []
    for x in (cr, ci):
        r = Fraction(x)
        for _ in range(nw):
            v = dt(float(r))
            out.append(v)
            r -= Fraction(float(v))
    return (*out, dt(width / w))


@dataclass
class Gpu:
    backend: Any
    cache: Any
    tile_sum: Any
    kernels: dict[tuple[str, str], Any] = field(default_factory=dict)
    lookups: int = 0

    @classmethod
    def open(cls, root: pathlib.Path) -> Gpu:
        from brotkit.backends.cuda import CudaCacheBackend
        from brotkit.kernels.cache import KernelCache, KernelStore

        be = CudaCacheBackend()
        cache = KernelCache(be, KernelStore(root))
        ts = cache.get(
            be.spec(
                source=gen_s5.TILE_SUM_SOURCE, entry="tile_sum", signature=gen_s5.TILE_SUM_SIGNATURE
            )
        )
        return cls(be, cache, ts)

    def kernel(self, tier: str, kern: str) -> Any:
        key = (tier, kern)
        if key not in self.kernels:
            self.lookups += 1
            spec = self.backend.spec(
                source=gen_s5.emit(tier, kern),
                entry=gen_s5.entry_name(tier, kern),
                signature=gen_s5.signature(tier),
            )
            self.kernels[key] = self.cache.get(spec)
        return self.kernels[key]

    def alloc(self, nbytes: int) -> Any:
        return self.backend.device.memory_resource.allocate(nbytes, stream=self.backend.stream)


def _memset(buf: Any, value: int, nbytes: int, stream: Any) -> None:
    from cuda.bindings import driver

    (err,) = driver.cuMemsetD8Async(int(buf.handle), value, nbytes, int(stream.handle))
    assert int(err) == 0, err


def _dtoh(dst: np.ndarray, buf: Any, nbytes: int, offset: int = 0) -> None:
    from cuda.bindings import driver

    (err,) = driver.cuMemcpyDtoH(dst.ctypes.data, int(buf.handle) + offset, nbytes)
    assert int(err) == 0, err


class RenderInterrupted(KeyboardInterrupt):
    """Ctrl-C during a render. `result` holds the partial result: `complete` is False and
    `done` lists the tiles that finished (their pixels are valid; the rest are untouched)."""

    def __init__(self, result: Result) -> None:
        super().__init__("render interrupted")
        self.result = result


@dataclass
class Result:
    tier: str
    kernel: str
    view: str
    h: int
    w: int
    max_iter: int
    policy: ChunkPolicy
    wait: str
    complete: bool
    wall_s: float
    kernel_s: float  # sum of chunk kernel times
    sum_s: float  # sum of tile_sum times
    cpu_s: float  # process CPU time during the render
    records: list[Any]
    done: list[Tile]
    interrupted_at: float | None = None
    stopped_at: float | None = None

    @property
    def overhead(self) -> float:
        return 1 - self.kernel_s / self.wall_s

    @property
    def iterations(self) -> int:
        return sum(r.iterations for r in self.records if not r.chunk.calibration)

    def summary(self) -> dict[str, Any]:
        """Statistics over real chunks; calibration probes are counted separately. The bound
        check is over real chunks (`actual_over_pred_max`) and, apart, over calibration probes
        with a finite bound (`calib_over_pred_max`: sub-millisecond, so launch jitter shows)."""
        real = [r for r in self.records if not r.chunk.calibration]
        calib = [r for r in self.records if r.chunk.calibration]
        secs = [r.seconds for r in real]
        bounded = [r for r in real if math.isfinite(r.predicted_worst_s)]
        cb = [r for r in calib if math.isfinite(r.predicted_worst_s)]
        return {
            "tier": self.tier,
            "kernel": self.kernel,
            "view": self.view,
            "w": self.w,
            "h": self.h,
            "max_iter": self.max_iter,
            "wait": self.wait,
            "target_s": self.policy.target_s,
            "ceiling_s": self.policy.ceiling_s,
            "tile": self.policy.tile,
            "complete": self.complete,
            "chunks": len(real),
            "calibration_chunks": len(calib),
            "calibration_s": round(sum(r.seconds for r in calib), 5),
            "wall_s": round(self.wall_s, 4),
            "kernel_s": round(self.kernel_s, 4),
            "sum_s": round(self.sum_s, 5),
            "overhead": round(self.overhead, 5),
            "cpu_s": round(self.cpu_s, 3),
            "giter_s": round(self.iterations / self.kernel_s / 1e9, 3) if self.kernel_s else None,
            "chunk_s_max": round(max(secs), 4) if secs else None,
            "chunk_s_median": round(statistics.median(secs), 4) if secs else None,
            "chunk_s_p90": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 4) if secs else None,
            "pred_worst_max_s": round(max(r.predicted_worst_s for r in bounded), 4)
            if bounded
            else None,
            "actual_over_pred_max": round(
                max((r.seconds / r.predicted_worst_s for r in bounded), default=0), 4
            ),
            "calib_over_pred_max": round(
                max((r.seconds / r.predicted_worst_s for r in cb), default=0), 4
            ),
            "over_ceiling": sum(r.seconds > self.policy.ceiling_s for r in self.records),
            "over_target_x2": sum(s > 2 * self.policy.target_s for s in secs),
            # per chunk: [calibration, rows, cols, max_iter, seconds, predicted worst, iterations,
            # largest escape count]
            "records": [
                [
                    int(r.chunk.calibration),
                    r.tile.th,
                    r.tile.tw,
                    r.chunk.max_iter,
                    round(r.seconds, 6),
                    round(r.predicted_worst_s, 6) if r.predicted_worst_s != float("inf") else None,
                    r.iterations,
                    r.max_n,
                ]
                for r in self.records
            ],
        }


def render(
    gpu: Gpu,
    tier: str,
    kern: str,
    view: str,
    *,
    w: int = W4K,
    h: int = H4K,
    max_iter: int | None = None,
    policy: ChunkPolicy | None = None,
    wait: str = "hybrid",
    poll_s: float = 0.0001,
    out: tuple[Any, Any] | None = None,
    sentinel: bool = False,
    on_chunk: Any = None,
    reraise: bool = False,
    prior: Estimates | None = None,
) -> Result:
    from cuda.core import EventOptions

    be = gpu.backend
    stream = be.stream
    k = gpu.kernel(tier, kern)
    mi = max_iter if max_iter is not None else VIEWS[view][3]
    pol = policy or ChunkPolicy()
    planner = ChunkPlanner(h=h, w=w, max_iter=mi, policy=pol, prior=prior)
    nf, nu = gen_s5.NLF * h * w * 4, gen_s5.NLU * h * w * 4
    bf, bu = out if out is not None else (gpu.alloc(nf), gpu.alloc(nu))
    if sentinel:
        _memset(bu, 0xFF, nu, stream)
    max_chunks = 1 << 16
    acc = gpu.alloc(16 * max_chunks)
    _memset(acc, 0, 16 * max_chunks, stream)
    args0 = view_args(tier, view, w)
    # Calibration chunks run on NaN coordinates (every pixel reaches the probe's max_iter, for
    # any formula; S5: NaN tau is within 6% of real interior tau, or higher) and write into a
    # scratch buffer, so they never touch the output.
    word = np.dtype(gen_s5.TIERS[tier][0]).type
    args_nan = (*(word(np.nan) for _ in args0[:-1]), args0[-1])
    sh, sw = planner.scratch_shape
    sf, su = gpu.alloc(gen_s5.NLF * sh * sw * 4), gpu.alloc(gen_s5.NLU * sh * sw * 4)
    timing = EventOptions(timing_enabled=True, blocking_sync=wait in ("block", "thread"))
    two = np.zeros(2, np.uint64)
    kern_s = sum_s = 0.0
    done: list[Tile] = []
    interrupted_at = None
    stream.sync()
    cpu0, t0 = time.process_time(), time.perf_counter()
    i = 0
    # Deferred interrupt: the first Ctrl-C only sets a flag, checked between chunks, so the
    # chunk in flight is always finished and recorded (it can't be cancelled anyway) and no
    # pixel is written without being flagged done. A second Ctrl-C raises at once.
    stop: list[float] = []

    def on_sigint(signum: int, frame: Any) -> None:
        if stop:
            raise KeyboardInterrupt
        stop.append(time.perf_counter())

    main = threading.current_thread() is threading.main_thread()
    old_handler = signal.signal(signal.SIGINT, on_sigint) if main else None
    try:
        while not stop and (chunk := planner.next()) is not None:
            tile = chunk.tile
            grid = ((tile.tw + 31) // 32, (tile.th + 7) // 8)
            if chunk.calibration:
                assert tile.th <= sh, tile
                assert tile.tw <= sw, tile
                args, of, ou, r0, c0, hh, ww = args_nan, sf, su, 0, 0, sh, sw
            else:
                args, of, ou, r0, c0, hh, ww = args0, bf, bu, tile.row0, tile.col0, h, w
            ea = stream.record(options=timing)
            k.launch(
                grid,
                (32, 8),
                *args,
                np.int64(chunk.max_iter),
                np.int64(r0),
                np.int64(c0),
                np.int64(tile.th),
                np.int64(tile.tw),
                of,
                ou,
                np.int64(hh),
                np.int64(ww),
                np.int64(ww),
            )
            eb = stream.record(options=timing)
            accp = int(acc.handle) + 16 * i
            gpu.tile_sum.launch(
                64,
                256,
                ou,
                accp,
                np.int64(r0),
                np.int64(c0),
                np.int64(tile.th),
                np.int64(tile.tw),
                np.int64(hh),
                np.int64(ww),
            )
            ec = stream.record(options=timing)
            expect = planner.expected_seconds(tile.pixels)
            _wait(stream, ec, wait, poll_s, expect)
            ks, ss = _finish(planner, chunk, ea, eb, ec, two, acc, i, done)
            if not chunk.calibration:  # calibration output is discarded: it counts as overhead
                kern_s += ks
            sum_s += ss
            i += 1
            if on_chunk is not None:
                on_chunk(planner)
        complete = not stop
    except KeyboardInterrupt:  # the second Ctrl-C: the chunk in flight is not recorded
        complete = False
        stream.sync()
    finally:
        if main:
            signal.signal(signal.SIGINT, old_handler)
    if stop:
        interrupted_at = stop[0]
    wall = time.perf_counter() - t0
    res = Result(
        tier,
        kern,
        view,
        h,
        w,
        mi,
        pol,
        wait,
        complete,
        wall,
        kern_s,
        sum_s,
        time.process_time() - cpu0,
        list(planner.records),
        done,
        interrupted_at,
        time.perf_counter() if interrupted_at else None,
    )
    res.buffers = (bf, bu)  # type: ignore[attr-defined]
    res.estimates = planner.estimates  # type: ignore[attr-defined]
    if not complete and reraise:
        raise RenderInterrupted(res)
    return res


def _wait(stream: Any, ev: Any, wait: str, poll_s: float, expect: float = 0.0) -> None:
    if wait == "spin":
        stream.sync()
    elif wait == "block":
        ev.sync()
    elif wait == "thread":
        # A helper thread sleeps in the driver (blocking-sync event); the main thread waits on
        # a threading.Event, which Ctrl-C interrupts at once and which wakes with no polling.
        # Any CUDA error in the helper is re-raised here, never left as a dead thread.
        done = threading.Event()
        err: list[BaseException] = []

        def helper() -> None:
            try:
                ev.sync()
            except BaseException as e:
                err.append(e)
            finally:
                done.set()

        threading.Thread(target=helper, daemon=True).start()
        while not done.wait(0.5):
            pass
        if err:
            raise err[0]
    elif wait.startswith("hybrid"):
        # "hybrid" or "hybrid<frac>": sleep in 1 ms slices to frac x the expected time
        # (default 0.5), then poll every poll_s.
        frac = float(wait[6:]) if len(wait) > 6 else 0.5
        end = time.perf_counter() + frac * expect
        while not ev.is_done and (left := end - time.perf_counter()) > 0:
            time.sleep(min(left, 0.001))
        while not ev.is_done:
            time.sleep(poll_s)
    else:
        while not ev.is_done:
            time.sleep(poll_s)


def _finish(planner, chunk: Chunk, ea, eb, ec, two, acc, i, done):
    ks = (eb - ea) / 1e3
    ss = (ec - eb) / 1e3
    _dtoh(two, acc, 16, offset=16 * i)
    planner.record(chunk, ks, int(two[0]), int(two[1]))
    if not chunk.calibration:
        done.append(chunk.tile)
    return ks, ss


def host_layers(gpu: Gpu, res: Result) -> tuple[np.ndarray, np.ndarray]:
    bf, bu = res.buffers  # type: ignore[attr-defined]
    gpu.backend.stream.sync()
    f = np.empty((gen_s5.NLF, res.h, res.w), np.float32)
    u = np.empty((gen_s5.NLU, res.h, res.w), np.uint32)
    _dtoh(f, bf, f.nbytes)
    _dtoh(u, bu, u.nbytes)
    return f, u


def coverage(res: Result) -> np.ndarray:
    m = np.zeros((res.h, res.w), bool)
    for t in res.done:
        m[t.row0 : t.row0 + t.th, t.col0 : t.col0 + t.tw] = True
    return m


def notebook_progress(label: str = "render") -> Any:
    """An `on_chunk` callback that updates one line in place in Jupyter (text fallback)."""
    try:
        from IPython.display import display
    except ImportError:  # pragma: no cover
        display = None
    t0 = time.perf_counter()
    handle = display(f"{label}: starting", display_id=True) if display else None

    def update(planner: ChunkPlanner) -> None:
        f = planner.done_fraction
        el = time.perf_counter() - t0
        eta = el * (1 - f) / f if f > 0 else float("nan")
        n = sum(1 for r in planner.records if not r.chunk.calibration)
        msg = f"{label}: {f * 100:5.1f}%  chunk {n}  elapsed {el:5.1f} s  ETA {eta:5.1f} s"
        if handle is not None:
            handle.update(msg)
        else:
            print(msg, end="\r", flush=True)

    return update

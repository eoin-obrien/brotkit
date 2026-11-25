"""Kernel cache on a real GPU: CUDA build/load adapter, key correctness, processes (S2, ADR-0008).

Marked `gpu`: needs a CUDA device and the `[cuda]` extra. The GPU-free half of the cache is
tested in `tests/test_kernel_cache.py`.
"""

# cuda.core objects are untyped, and CPU CI type-checks this file without the [cuda] extra.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false
# pyright: reportMissingImports=false

from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.gpu

# Skip on the third-party module: brotkit's own guard raises ImportError, which importorskip
# doesn't treat as a skip.
pytest.importorskip("numba_cuda_mlir")

import numpy as np  # noqa: E402  (installed with the [cuda] extra)

import brotkit.backends.cuda as cuda_backend  # noqa: E402
from brotkit.kernels.cache import (  # noqa: E402
    KernelCache,
    KernelCacheWarning,
    KernelStore,
    decode_entry,
)

# A raw-pointer ABI kernel (ADR-0008): typed pointers, int64 shape/pitch. 16 independent f64
# accumulators keep ~40+ registers live, so a 32-register cap is visible.
SOURCE = textwrap.dedent(
    """
    import numpy as np
    from numba_cuda_mlir import cuda


    def kern(p_out, h, w, pitch, scale, n_iter):
        out = cuda.carray(p_out, (h, pitch), np.float64)
        x, y = cuda.grid(2)
        if x < w and y < h:
            a0 = np.float64(x) * scale
            a1 = a0 + 1.0; a2 = a0 + 2.0; a3 = a0 + 3.0; a4 = a0 + 4.0; a5 = a0 + 5.0
            a6 = a0 + 6.0; a7 = a0 + 7.0; a8 = a0 + 8.0; a9 = a0 + 9.0; a10 = a0 + 10.0
            a11 = a0 + 11.0; a12 = a0 + 12.0; a13 = a0 + 13.0; a14 = a0 + 14.0; a15 = a0 + 15.0
            for i in range(n_iter):
                t = np.float64(y + i) * 1e-3
                a0 = a0 * 0.999 + t; a1 = a1 * 0.998 + t; a2 = a2 * 0.997 + t
                a3 = a3 * 0.996 + t; a4 = a4 * 0.995 + t; a5 = a5 * 0.994 + t
                a6 = a6 * 0.993 + t; a7 = a7 * 0.992 + t; a8 = a8 * 0.991 + t
                a9 = a9 * 0.990 + t; a10 = a10 * 0.989 + t; a11 = a11 * 0.988 + t
                a12 = a12 * 0.987 + t; a13 = a13 * 0.986 + t; a14 = a14 * 0.985 + t
                a15 = a15 * 0.984 + t
            out[y, x] = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12
                         + a13 + a14 + a15)
    """
)
SIGNATURE = ("float64*", "int64", "int64", "int64", "float64", "int64")
H, W, PITCH = 37, 50, 64
GRID, BLOCK = ((W + 15) // 16, (H + 15) // 16), (16, 16)


@pytest.fixture(scope="module")
def backend() -> Any:
    return cuda_backend.CudaCacheBackend()


@pytest.fixture
def cache(tmp_path: Path, backend: Any) -> KernelCache[Any]:
    return KernelCache(backend, KernelStore(tmp_path / "cache"))


def spec(backend: Any, **options: object) -> Any:
    return backend.spec(source=SOURCE, entry="kern", signature=SIGNATURE, options=options)


class Device2D:
    """A device buffer exposing __cuda_array_interface__ (stands in for CuPy/Numba arrays)."""

    def __init__(self, backend: Any, shape: tuple[int, int], dtype: Any = np.float64) -> None:
        self.shape, self.dtype = shape, np.dtype(dtype)
        nbytes = int(np.prod(shape)) * self.dtype.itemsize
        self.buf = backend.device.memory_resource.allocate(nbytes, stream=backend.stream)
        self.nbytes = nbytes

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (int(self.buf.handle), False),
            "version": 3,
            "strides": None,
            "stream": None,
        }

    def to_host(self, backend: Any) -> Any:
        from cuda.bindings import driver

        backend.stream.sync()
        out = np.empty(self.shape, self.dtype)
        (err,) = driver.cuMemcpyDtoH(out.ctypes.data, int(self.buf.handle), self.nbytes)
        assert int(err) == 0
        return out


def run(kernel: Any, backend: Any) -> Any:
    out = Device2D(backend, (H, PITCH))
    kernel.launch(GRID, BLOCK, out, H, W, PITCH, 0.01, 200)
    return out.to_host(backend)[:, :W]


def test_build_load_launch(cache: KernelCache[Any], backend: Any) -> None:
    k = cache.get(spec(backend))
    res = run(k, backend)
    assert np.isfinite(res).all()
    assert k.resources.registers == k._kernel.attributes.num_regs  # pyright: ignore[reportPrivateUsage]
    assert k.resources.stack_bytes == k._kernel.attributes.local_size_bytes  # pyright: ignore[reportPrivateUsage]


def test_cache_path_matches_numba_dev_path(
    cache: KernelCache[Any], backend: Any, tmp_path: Path
) -> None:
    """One ABI: Numba compiles and launches the same raw-pointer kernel the cache loads."""
    s = spec(backend)
    via_cache = run(cache.get(s), backend)
    disp = backend.dev_dispatcher(s, tmp_path / "dev" / "k_dev.py")
    out = Device2D(backend, (H, PITCH))
    backend.stream.sync()
    disp[GRID, BLOCK](int(out.buf.handle), H, W, PITCH, 0.01, 200)
    from cuda.bindings import driver

    (err,) = driver.cuCtxSynchronize()
    assert int(err) == 0
    assert np.array_equal(out.to_host(backend)[:, :W], via_cache)


def test_s4_regression_cap_and_no_cap_are_two_entries(
    cache: KernelCache[Any], backend: Any, tmp_path: Path
) -> None:
    """S4: Numba's cache=True served a 24-register build to an uncapped dispatcher (#493)."""
    capped = cache.get(spec(backend, max_registers=32))
    uncapped = cache.get(spec(backend))
    assert len(cache.store.entries()) == 2
    driver_regs = lambda k: k._kernel.attributes.num_regs  # noqa: E731
    assert driver_regs(capped) <= 32 < driver_regs(uncapped)
    assert capped.resources.registers == driver_regs(capped)
    assert uncapped.resources.registers == driver_regs(uncapped)
    # Fresh processes load each build without compiling, with the right registers.
    for opts, expect in (({"max_registers": 32}, driver_regs(capped)), ({}, driver_regs(uncapped))):
        r = fresh_process_load(tmp_path / "cache", opts)
        assert r["builds"] == 0
        assert r["disk_hits"] == 1
        assert r["driver_regs"] == r["manifest_regs"] == expect


CHILD = textwrap.dedent(
    """
    import json, sys, time
    t0 = time.perf_counter()
    from brotkit.backends.cuda import CudaCacheBackend
    from brotkit.kernels.cache import KernelCache, KernelStore
    be = CudaCacheBackend()
    t1 = time.perf_counter()
    kernel = json.loads(open(sys.argv[3]).read())  # not the test module: it imports Numba
    SOURCE, SIGNATURE = kernel["source"], kernel["signature"]
    c = KernelCache(be, KernelStore(sys.argv[1]))
    s = be.spec(source=SOURCE, entry="kern", signature=SIGNATURE, options=json.loads(sys.argv[2]))
    t2 = time.perf_counter()
    k = c.get(s)
    t3 = time.perf_counter()
    print(json.dumps(dict(
        builds=c.stats.builds, disk_hits=c.stats.disk_hits,
        driver_regs=k._kernel.attributes.num_regs, manifest_regs=k.resources.registers,
        one_time_s=t1 - t0, get_s=t3 - t2,
        numba_imported="numba_cuda_mlir" in sys.modules,
    )))
    """
)


def fresh_process_load(root: Path, options: Mapping[str, object]) -> dict[str, Any]:
    kernel_file = root.parent / "kernel.json"
    kernel_file.write_text(json.dumps({"source": SOURCE, "signature": list(SIGNATURE)}))
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(root),
            json.dumps(dict(options)),
            str(kernel_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_warm_load_in_fresh_process_skips_numba(
    cache: KernelCache[Any], backend: Any, tmp_path: Path
) -> None:
    cache.get(spec(backend))
    r = fresh_process_load(tmp_path / "cache", {})
    assert r["builds"] == 0
    assert not r["numba_imported"]  # a warm start never imports the JIT
    assert r["get_s"] < 0.1


def test_toolchain_bump_misses(tmp_path: Path, backend: Any) -> None:
    store = KernelStore(tmp_path / "cache")
    KernelCache(backend, store).get(spec(backend))
    other = cuda_backend.CudaCacheBackend()
    other._toolchain = {**backend.toolchain(), "numba-cuda-mlir": "0.6.1"}  # pyright: ignore[reportPrivateUsage]
    c2: KernelCache[Any] = KernelCache(other, KernelStore(tmp_path / "cache"))
    c2.get(spec(other))
    assert c2.stats.builds == 1
    assert len(store.entries()) == 2


def test_corrupt_real_entry_rebuilds(cache: KernelCache[Any], backend: Any, tmp_path: Path) -> None:
    good = run(cache.get(spec(backend)), backend)
    d, _ = cache.key(spec(backend))
    p = cache.store.entry_path(d)
    p.write_bytes(p.read_bytes()[:-100])
    c2: KernelCache[Any] = KernelCache(backend, KernelStore(tmp_path / "cache"))
    with pytest.warns(KernelCacheWarning):
        k = c2.get(spec(backend))
    assert c2.stats.builds == 1
    assert np.array_equal(run(k, backend), good)


def test_cubin_the_driver_rejects_is_a_miss(
    cache: KernelCache[Any], backend: Any, tmp_path: Path
) -> None:
    """Stands in for a driver update that rejects a stored cubin: checksums are valid, the
    driver's module load fails, and the cache rebuilds instead of raising (ADR-0008)."""
    import dataclasses

    good = run(cache.get(spec(backend)), backend)
    d, _ = cache.key(spec(backend))
    entry = decode_entry(cache.store.entry_path(d).read_bytes())
    junk = b"\x7fELF" + bytes(252)
    cache.store.write(
        dataclasses.replace(entry, blobs=(("cubin", junk), ("ptx", entry.blob("ptx"))))
    )
    c2: KernelCache[Any] = KernelCache(backend, KernelStore(tmp_path / "cache"))
    with pytest.warns(KernelCacheWarning, match="couldn't load"):
        k = c2.get(spec(backend))
    assert (c2.stats.load_failures, c2.stats.builds) == (1, 1)
    assert np.array_equal(run(k, backend), good)


def test_launch_rejects_bad_arguments(cache: KernelCache[Any], backend: Any) -> None:
    k = cache.get(spec(backend))
    with pytest.raises(TypeError, match="takes 6"):
        k.launch(GRID, BLOCK, 1, 2)
    wrong = Device2D(backend, (H, PITCH), np.float32)
    with pytest.raises(TypeError, match="dtype"):
        k.launch(GRID, BLOCK, wrong, H, W, PITCH, 0.01, 10)


def _race_worker(root: str, q: mp.Queue[dict[str, Any]]) -> None:
    import brotkit.backends.cuda as cb

    be = cb.CudaCacheBackend()
    c: KernelCache[Any] = KernelCache(be, KernelStore(root))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        k = c.get(spec(be))
    res = run(k, be)
    cache_w = [str(x.message) for x in w if issubclass(x.category, KernelCacheWarning)]
    q.put(
        {
            "builds": c.stats.builds,
            "sum": float(res.sum()),
            "cache_warnings": cache_w,
            "other_warnings": sorted({x.category.__name__ for x in w} - {"KernelCacheWarning"}),
        }
    )


@pytest.mark.slow
@pytest.mark.parametrize("round_", range(20))
def test_four_processes_race_one_uncached_key(tmp_path: Path, round_: int) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue[dict[str, Any]] = ctx.Queue()
    root = tmp_path / "cache"
    ps = [ctx.Process(target=_race_worker, args=(str(root), q)) for _ in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(300)
        assert p.exitcode == 0
    rs = [q.get() for _ in ps]
    assert sum(r["builds"] for r in rs) == 1
    assert len({r["sum"] for r in rs}) == 1  # all four launched and agree
    assert all(not r["cache_warnings"] for r in rs), [r["cache_warnings"] for r in rs]
    store = KernelStore(root)
    assert len(store.entries()) == 1
    assert not list(store.dir.glob("??/.*.tmp"))

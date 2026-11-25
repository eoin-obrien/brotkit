"""Kernel cache: key, store, locking, corruption and eviction, with a fake backend (no GPU).

The GPU half (the CUDA adapter, real registers, cross-process builds of real kernels) is in
`tests/test_cuda_cache.py`, marked `gpu`. Spike S2, ADR-0008.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import threading
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brotkit.kernels import cache as kc
from brotkit.kernels.cache import (
    CacheEntry,
    CacheLocationError,
    CorruptEntryError,
    KernelCache,
    KernelCacheWarning,
    KernelSpec,
    KernelStore,
    decode_entry,
    digest,
    encode_entry,
    key_components,
)
from tests.kernel_cache_fake import UNCAPPED_REGISTERS, FakeBackend, FakeKernel, spec

TOOLS = {"fakecc": "1.0", "fakelink": "1.0"}


def make_cache(
    tmp_path: Path,
    *,
    versions: dict[str, str] | None = None,
    build_seconds: float = 0.0,
    fail_loads: int = 0,
) -> tuple[KernelCache[FakeKernel], FakeBackend]:
    be = FakeBackend(versions=versions, build_seconds=build_seconds, fail_loads=fail_loads)
    return KernelCache(be, KernelStore(tmp_path / "cache")), be


def key(s: KernelSpec, tools: dict[str, str] | None = None, codegen: int = 1) -> str:
    return digest(key_components(s, tools or TOOLS, codegen_version=codegen))


# --- key ------------------------------------------------------------------------------------

BASE = spec()
# One change per key component. Each must give a different key from BASE (and each other).
VARIANTS: dict[str, KernelSpec] = {
    "source": spec(source=BASE.source + "# changed\n"),
    "entry": spec(entry="k2"),
    "signature": spec(signature=("float32*", "int32")),
    "backend": spec(backend="other"),
    "target": spec(target="sm_89"),
    "fastmath": spec(options={"fastmath": True}),
    "max_registers": spec(options={"max_registers": 32}),
    "launch_bounds": spec(options={"launch_bounds": (256, 2)}),
    "lineinfo": spec(options={"lineinfo": True}),
    "debug": spec(options={"debug": True}),
    "opt": spec(options={"opt": False}),
    "ptxas_options": spec(options={"ptxas_options": ["-O1"]}),
    "nvjitlink_options": spec(options={"nvjitlink_options": ["-lto"]}),
}


def test_every_option_has_a_variant() -> None:
    assert set(kc.OPTION_DEFAULTS) <= set(VARIANTS)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_each_spec_component_changes_the_key(name: str) -> None:
    assert key(VARIANTS[name]) != key(BASE)


def test_all_variants_distinct() -> None:
    keys = [key(v) for v in VARIANTS.values()]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_each_toolchain_version_changes_the_key(tool: str) -> None:
    assert key(BASE, {**TOOLS, tool: "1.1"}) != key(BASE)


def test_adding_a_toolchain_entry_changes_the_key() -> None:
    assert key(BASE, {**TOOLS, "newtool": "0.1"}) != key(BASE)


def test_codegen_version_changes_the_key() -> None:
    assert key(BASE, codegen=2) != key(BASE)


def test_python_and_format_change_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    k0 = key(BASE)
    monkeypatch.setattr(kc, "python_tag", lambda: "cpython-3.99")
    assert key(BASE) != k0
    monkeypatch.undo()
    monkeypatch.setattr(kc, "CACHE_FORMAT", 99)
    assert key(BASE) != k0


def test_python_tag_is_minor_version_only() -> None:
    import re
    import sys

    tag = kc.python_tag()
    major, minor = sys.version_info[:2]
    assert re.fullmatch(rf"{sys.implementation.name}-{major}\.{minor}t?", tag)


def test_defaults_are_normalised_away() -> None:
    explicit = spec(options={"fastmath": False, "max_registers": None, "opt": True})
    assert explicit.options == ()
    assert key(explicit) == key(BASE)


def test_tuple_and_list_options_are_the_same_key() -> None:
    assert key(spec(options={"launch_bounds": (256, 2)})) == key(
        spec(options={"launch_bounds": [256, 2]})
    )


def test_unknown_option_rejected() -> None:
    with pytest.raises(ValueError, match="unknown compile option"):
        spec(options={"maxregs": 32})


def test_direct_construction_validates_options() -> None:
    with pytest.raises(ValueError, match="sorted"):
        KernelSpec(
            source="",
            entry="k",
            signature=(),
            backend="b",
            target="t",
            options=(("opt", "false"), ("fastmath", "true")),
        )
    with pytest.raises(ValueError, match="default"):
        KernelSpec(
            source="",
            entry="k",
            signature=(),
            backend="b",
            target="t",
            options=(("fastmath", "false"),),
        )


def test_spec_is_frozen_and_hashable() -> None:
    s = spec(options={"max_registers": 64})
    assert hash(s) == hash(spec(options={"max_registers": 64}))
    with pytest.raises(AttributeError):
        s.entry = "x"  # pyright: ignore[reportAttributeAccessIssue]


option_values = st.fixed_dictionaries(
    {},
    optional={
        "fastmath": st.booleans(),
        "max_registers": st.none() | st.integers(16, 255),
        "launch_bounds": st.none() | st.tuples(st.integers(32, 1024), st.integers(1, 8)),
        "lineinfo": st.booleans(),
        "opt": st.booleans(),
        "ptxas_options": st.lists(st.sampled_from(["-O1", "-O2", "--dlcm=ca"]), max_size=2),
    },
)


@given(a=option_values, b=option_values)
def test_key_equal_iff_effective_options_equal(a: dict[str, object], b: dict[str, object]) -> None:
    sa, sb = spec(options=a), spec(options=b)
    effective = all(sa.option(n) == sb.option(n) for n in kc.OPTION_DEFAULTS)
    assert (key(sa) == key(sb)) == effective


@given(opts=option_values)
def test_key_is_deterministic(opts: dict[str, object]) -> None:
    assert key(spec(options=opts)) == key(spec(options=dict(reversed(list(opts.items())))))


# --- the S4 bug, as a regression test ----------------------------------------------------------


def test_max_registers_32_and_uncapped_are_separate_entries(tmp_path: Path) -> None:
    """S4 found Numba's cache=True served a capped build to an uncapped dispatcher (#493)."""
    c, be = make_cache(tmp_path)
    capped = c.get(spec(options={"max_registers": 32}))
    uncapped = c.get(spec())
    assert be.builds == 2
    assert len(c.store.entries()) == 2
    assert capped.registers == 32
    assert uncapped.registers == UNCAPPED_REGISTERS
    # And in a fresh "process" (new cache, same directory), each loads its own build.
    c2, be2 = make_cache(tmp_path)
    assert c2.get(spec()).registers == UNCAPPED_REGISTERS
    assert c2.get(spec(options={"max_registers": 32})).registers == 32
    assert be2.builds == 0


# --- hits and misses ---------------------------------------------------------------------------


def test_memory_then_disk_hits(tmp_path: Path) -> None:
    c, be = make_cache(tmp_path)
    k1 = c.get(BASE)
    assert c.get(BASE) is k1
    assert (be.builds, c.stats.memory_hits) == (1, 1)
    c2, be2 = make_cache(tmp_path)
    assert c2.get(BASE).cubin == k1.cubin
    assert (be2.builds, c2.stats.disk_hits) == (0, 1)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_changing_one_component_misses_and_repeating_hits(tmp_path: Path, name: str) -> None:
    c, be = make_cache(tmp_path)
    variant = VARIANTS[name]
    if variant.backend != "fake":
        with pytest.raises(ValueError, match="backend"):
            c.get(variant)
        return
    c.get(BASE)
    c.get(variant)
    assert be.builds == 2
    c2, be2 = make_cache(tmp_path)
    c2.get(variant)
    c2.get(BASE)
    assert be2.builds == 0


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_version_bump_misses_never_stale(tmp_path: Path, tool: str) -> None:
    c1, _ = make_cache(tmp_path, versions=TOOLS)
    old = c1.get(BASE)
    c2, be2 = make_cache(tmp_path, versions={**TOOLS, tool: "2.0"})
    new = c2.get(BASE)
    assert be2.builds == 1
    assert new.cubin != old.cubin
    assert len(c2.store.entries()) == 2


def test_codegen_bump_misses(tmp_path: Path) -> None:
    store = KernelStore(tmp_path / "cache")
    KernelCache(FakeBackend(), store).get(BASE)
    be = FakeBackend()
    KernelCache(be, KernelStore(tmp_path / "cache"), codegen_version=2).get(BASE)
    assert be.builds == 1


def test_manifest_holds_key_components_and_meta(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    c.get(spec(options={"max_registers": 32}))
    d, comps = c.key(spec(options={"max_registers": 32}))
    entry = decode_entry(c.store.entry_path(d).read_bytes())
    assert entry.key == comps
    assert entry.key["options"] == {"max_registers": 32}
    meta = entry.meta
    assert meta["symbol"] == "_Z1k_k"
    assert meta["resources"] == {"registers": 32}
    assert isinstance(meta["build_seconds"], float)
    assert entry.created.endswith("+00:00")
    assert c.store.source_path(d).read_text() == BASE.source


# --- corruption ---------------------------------------------------------------------------------


def _corrupt_and_reload(
    tmp_path: Path, damage: Callable[[Path], None]
) -> tuple[KernelCache[FakeKernel], FakeBackend]:
    c, _ = make_cache(tmp_path)
    good = c.get(BASE)
    d, _ = c.key(BASE)
    path = c.store.entry_path(d)
    damage(path)
    c2, be2 = make_cache(tmp_path)
    with pytest.warns(KernelCacheWarning, match="unusable"):
        again = c2.get(BASE)
    assert be2.builds == 1
    assert again.cubin == good.cubin
    assert decode_entry(path.read_bytes()).digest == d  # rebuilt and valid
    assert c2.stats.corrupt == 1
    return c2, be2


def _truncate(p: Path) -> None:
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])


def _flip_last(p: Path) -> None:
    b = bytearray(p.read_bytes())
    b[-1] ^= 0xFF
    p.write_bytes(bytes(b))


def _garbage(p: Path) -> None:
    p.write_bytes(b"not a cache entry")


def _empty(p: Path) -> None:
    p.write_bytes(b"")


@pytest.mark.parametrize("damage", [_truncate, _flip_last, _garbage, _empty])
def test_corrupt_entry_is_deleted_rebuilt_and_warned_once(
    tmp_path: Path, damage: Callable[[Path], None]
) -> None:
    c2, _be2 = _corrupt_and_reload(tmp_path, damage)
    # Same process asks again: memory hit, no second warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c2.get(BASE)
    # A fresh process sees a good entry: no warning, no build.
    c3, be3 = make_cache(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c3.get(BASE)
    assert be3.builds == 0


def test_entry_that_fails_to_load_is_a_miss_and_rebuilds(tmp_path: Path) -> None:
    """E.g. a driver update that rejects a stored cubin: rebuild, don't raise (ADR-0008)."""
    c, _ = make_cache(tmp_path)
    good = c.get(BASE)
    c2, be2 = make_cache(tmp_path, fail_loads=1)
    with pytest.warns(KernelCacheWarning, match="couldn't load"):
        again = c2.get(BASE)
    assert be2.builds == 1
    assert again.cubin == good.cubin
    assert (c2.stats.load_failures, c2.stats.disk_hits) == (1, 0)
    c3, be3 = make_cache(tmp_path)
    c3.get(BASE)  # the rebuilt entry loads in a fresh process
    assert (be3.builds, c3.stats.disk_hits) == (0, 1)


def test_fresh_build_that_fails_to_load_raises(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    c.get(BASE)
    c2, be2 = make_cache(tmp_path, fail_loads=2)
    with pytest.warns(KernelCacheWarning), pytest.raises(RuntimeError, match="INVALID_SOURCE"):
        c2.get(BASE)
    assert be2.builds == 1  # one rebuild, then the real error


def test_miss_whose_build_fails_to_load_raises(tmp_path: Path) -> None:
    c, be = make_cache(tmp_path, fail_loads=1)
    with pytest.raises(RuntimeError, match="INVALID_SOURCE"):
        c.get(BASE)
    assert be.builds == 1


def test_warns_once_per_key_even_if_damaged_again(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    c.get(BASE)
    d, comps = c.key(BASE)
    store = KernelStore(tmp_path / "cache")
    _garbage(store.entry_path(d))
    with pytest.warns(KernelCacheWarning):
        assert store.read(d, comps) is None
    c.store.write(decode_entry(encode_entry(_entry(d, comps))))
    _garbage(store.entry_path(d))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert store.read(d, comps) is None
    assert store.corrupt_count == 2


def _entry(d: str, comps: dict[str, kc.JSON]) -> CacheEntry:
    return CacheEntry(
        digest=d,
        key_json=kc.canonical_json(comps),
        meta_json="{}",
        blobs=(("x", b"1"),),
        created="2026-01-01T00:00:00+00:00",
    )


def test_entry_stored_under_another_key_is_rejected(tmp_path: Path) -> None:
    """Never trust a build the cache didn't key itself: same digest, different components."""
    c, _ = make_cache(tmp_path)
    d, comps = c.key(BASE)
    forged = dict(comps, options={"max_registers": 32})
    c.store.write(_entry(d, forged))
    with pytest.warns(KernelCacheWarning, match="differ"):
        assert c.store.read(d, comps) is None
    assert not c.store.entry_path(d).exists()


def test_entry_under_wrong_file_name_is_rejected(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    d, comps = c.key(BASE)
    other = "f" * 64
    c.store.write(_entry(d, comps))
    path = c.store.entry_path(d)
    target = c.store.entry_path(other)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)
    with pytest.warns(KernelCacheWarning, match="file name"):
        assert c.store.read(other, comps) is None


@st.composite
def entries(draw: st.DrawFn) -> CacheEntry:
    n = draw(st.integers(0, 4))
    blobs = tuple((f"b{i}", draw(st.binary(max_size=300))) for i in range(n))
    return CacheEntry(
        digest=draw(st.text("0123456789abcdef", min_size=64, max_size=64)),
        key_json='{"k":1}',
        meta_json=kc.canonical_json({"m": draw(st.integers())}),
        blobs=blobs,
        created="2026-11-24T00:00:00+00:00",
    )


@given(e=entries())
def test_encode_decode_roundtrip(e: CacheEntry) -> None:
    assert decode_entry(encode_entry(e)) == e


@given(e=entries(), data=st.data())
def test_any_truncation_is_detected(e: CacheEntry, data: st.DataObject) -> None:
    raw = encode_entry(e)
    cut = data.draw(st.integers(0, len(raw) - 1))
    with pytest.raises(CorruptEntryError):
        decode_entry(raw[:cut])


@settings(max_examples=300)
@given(e=entries(), data=st.data())
def test_any_byte_change_is_detected(e: CacheEntry, data: st.DataObject) -> None:
    raw = bytearray(encode_entry(e))
    i = data.draw(st.integers(0, len(raw) - 1))
    raw[i] ^= data.draw(st.integers(1, 255))
    with pytest.raises(CorruptEntryError):
        decode_entry(bytes(raw))


@given(e=entries(), extra=st.binary(min_size=1, max_size=50))
def test_trailing_bytes_are_detected(e: CacheEntry, extra: bytes) -> None:
    with pytest.raises(CorruptEntryError):
        decode_entry(encode_entry(e) + extra)


# --- concurrency --------------------------------------------------------------------------------


def _worker(root: str, markers: str, q: mp.Queue[tuple[int, str]]) -> None:
    be = FakeBackend(marker_dir=Path(markers), build_seconds=0.05)
    c: KernelCache[FakeKernel] = KernelCache(be, KernelStore(root))
    k = c.get(BASE)
    q.put((be.builds, hashlib.sha256(k.cubin).hexdigest()))


@pytest.mark.parametrize("round_", range(20))
def test_four_processes_same_uncached_key_build_once(tmp_path: Path, round_: int) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue[tuple[int, str]] = ctx.Queue()
    root, markers = tmp_path / "cache", tmp_path / "markers"
    ps = [ctx.Process(target=_worker, args=(str(root), str(markers), q)) for _ in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
        assert p.exitcode == 0
    results = [q.get() for _ in ps]
    assert sum(b for b, _ in results) == 1
    assert len(list(markers.iterdir())) == 1
    assert len({h for _, h in results}) == 1
    store = KernelStore(root)
    (only,) = store.entries()
    decode_entry(only.path.read_bytes())  # one valid entry
    assert not list(store.dir.glob("??/.*.tmp"))


def test_threads_same_key_build_once(tmp_path: Path) -> None:
    c, be = make_cache(tmp_path, build_seconds=0.05)
    out: list[object] = []
    ts = [threading.Thread(target=lambda: out.append(c.get(BASE))) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert be.builds == 1
    assert all(o is out[0] for o in out)


def test_threads_different_keys_dont_serialise(tmp_path: Path) -> None:
    import time

    c, be = make_cache(tmp_path, build_seconds=0.3)
    specs = [spec(options={"max_registers": 32 + 8 * i}) for i in range(4)]
    t0 = time.perf_counter()
    ts = [threading.Thread(target=c.get, args=(s,)) for s in specs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert be.builds == 4
    # The cross-process lock is striped, so two keys could share a stripe; allow that once.
    assert time.perf_counter() - t0 < 0.3 * 2.5


# --- perf records ------------------------------------------------------------------------------

DEV_A, DEV_B = "sm_86 GPU A", "sm_86 GPU B"


def test_perf_record_round_trips_per_device(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    assert c.read_perf(BASE, DEV_A) is None
    c.write_perf(BASE, DEV_A, {"tau": 1e-6, "rate": 1e10})
    c.write_perf(BASE, DEV_B, {"tau": 2e-6, "rate": 5e9})
    c2, _ = make_cache(tmp_path)  # another process
    assert c2.read_perf(BASE, DEV_A) == {"tau": 1e-6, "rate": 1e10}
    assert c2.read_perf(BASE, DEV_B) == {"tau": 2e-6, "rate": 5e9}
    c2.write_perf(BASE, DEV_A, {"tau": 3e-6, "rate": 1e9})
    assert c.read_perf(BASE, DEV_A) == {"tau": 3e-6, "rate": 1e9}
    assert c.read_perf(BASE, DEV_B) == {"tau": 2e-6, "rate": 5e9}  # kept


def test_perf_record_is_per_key(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    c.write_perf(BASE, DEV_A, {"tau": 1e-6, "rate": 1e10})
    assert c.read_perf(VARIANTS["max_registers"], DEV_A) is None


def test_perf_record_leaves_the_entry_alone(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    c.get(BASE)
    d, _ = c.key(BASE)
    before = c.store.entry_path(d).read_bytes()
    c.write_perf(BASE, DEV_A, {"tau": 1e-6, "rate": 1e10})
    assert c.store.entry_path(d).read_bytes() == before
    assert [e.digest for e in c.store.entries()] == [d]  # not listed as an entry
    c2, be2 = make_cache(tmp_path)
    c2.get(BASE)
    assert be2.builds == 0


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{not json",
        b"\xff\xfe",
        b"[]",
        b'{"format": 2, "devices": {}}',
        b'{"format": 1}',
        b'{"format": 1, "devices": []}',
        b'{"format": 1, "devices": {"sm_86 GPU A": 3}}',
    ],
)
def test_damaged_perf_record_reads_as_none_and_is_replaced(tmp_path: Path, content: bytes) -> None:
    c, _ = make_cache(tmp_path)
    d, _ = c.key(BASE)
    path = c.store.perf_path(d)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    assert c.read_perf(BASE, DEV_A) is None
    c.write_perf(BASE, DEV_A, {"tau": 1e-6, "rate": 1e10})
    assert c.read_perf(BASE, DEV_A) == {"tau": 1e-6, "rate": 1e10}


def test_oversized_perf_record_is_ignored(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    d, _ = c.key(BASE)
    path = c.store.perf_path(d)
    path.parent.mkdir(parents=True)
    path.write_text('{"format": 1, "devices": {"x": "' + "a" * 70000 + '"}}')
    assert c.read_perf(BASE, "x") is None


def test_eviction_removes_the_perf_record(tmp_path: Path) -> None:
    store = KernelStore(tmp_path / "cache", max_bytes=1)
    c: KernelCache[FakeKernel] = KernelCache(FakeBackend(), store)
    old = spec(options={"max_registers": 40})
    c.get(old)
    c.write_perf(old, DEV_A, {"tau": 1e-6, "rate": 1e10})
    c.get(spec(options={"max_registers": 41}))  # evicts `old`
    assert not store.perf_path(c.key(old)[0]).exists()


# --- eviction -----------------------------------------------------------------------------------


def test_lru_eviction_keeps_newest_under_cap(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    specs = [spec(options={"max_registers": 32 + i}) for i in range(5)]
    for i, s in enumerate(specs):
        c.get(s)
        d, _ = c.key(s)
        os.utime(c.store.entry_path(d), (1000 + i, 1000 + i))
    size = c.store.entries()[0].size
    c.store.max_bytes = size * 3
    d_first, _ = c.key(specs[0])
    evicted = c.store.evict(keep=d_first)
    remaining = {e.digest for e in c.store.entries()}
    assert len(remaining) == 3
    assert d_first in remaining  # kept although oldest
    assert {c.key(specs[i])[0] for i in (1, 2)} == set(evicted)


def test_hit_refreshes_last_used(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    c.get(BASE)
    d, comps = c.key(BASE)
    p = c.store.entry_path(d)
    os.utime(p, (1000, 1000))
    assert c.store.read(d, comps) is not None
    assert p.stat().st_mtime > 1000


def test_eviction_runs_after_builds(tmp_path: Path) -> None:
    store = KernelStore(tmp_path / "cache", max_bytes=1)
    c: KernelCache[FakeKernel] = KernelCache(FakeBackend(), store)
    for i in range(3):
        c.get(spec(options={"max_registers": 40 + i}))
    assert len(store.entries()) == 1  # only the one just built survives a 1-byte cap


def test_stale_temp_files_are_cleaned(tmp_path: Path) -> None:
    c, _ = make_cache(tmp_path)
    c.get(BASE)
    d, _ = c.key(BASE)
    tmp = c.store.entry_path(d).with_name(f".{d[:16]}.1.dead.tmp")
    tmp.write_bytes(b"half")
    os.utime(tmp, (1000, 1000))
    c.store.evict()
    assert not tmp.exists()


# --- location -----------------------------------------------------------------------------------

MOUNTS = """\
/dev/sdc / ext4 rw,relatime 0 0
C:\\134 /mnt/c 9p rw,noatime,aname=drvfs 0 0
/dev/sdd /mnt/data ext4 rw 0 0
"""


def test_refuses_wsl_windows_drive() -> None:
    root = Path("/mnt/c/Users/someone/AppData/Local/brotkit")
    with pytest.raises(CacheLocationError) as ei:
        kc.check_location(root, MOUNTS)
    msg = str(ei.value)
    assert str(root) in msg
    assert "BROTKIT_CACHE_DIR" in msg
    assert "Linux-side" in msg


def test_accepts_linux_paths() -> None:
    kc.check_location(Path("/home/u/.cache/brotkit/kernels"), MOUNTS)
    kc.check_location(Path("/mnt/data/cache"), MOUNTS)
    assert kc.filesystem_type(Path("/mnt/cache"), MOUNTS) == "ext4"  # /mnt/c is not a prefix


def test_mount_point_escapes() -> None:
    mounts = "/dev/x / ext4 rw 0 0\nC: /mnt/c/Users/A\\040B 9p rw 0 0\n"
    assert kc.filesystem_type(Path("/mnt/c/Users/A B/x"), mounts) == "9p"


def test_store_refuses_on_construction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def nine_p(_p: Path, _m: str | None = None) -> str:
        return "9p"

    monkeypatch.setattr(kc, "filesystem_type", nine_p)
    with pytest.raises(CacheLocationError):
        KernelStore(tmp_path)


def test_default_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BROTKIT_CACHE_DIR", str(tmp_path / "x"))
    assert kc.default_root() == tmp_path / "x"
    monkeypatch.delenv("BROTKIT_CACHE_DIR")
    assert kc.default_root().parts[-2:] == ("brotkit", "kernels")

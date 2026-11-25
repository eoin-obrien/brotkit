"""On-disk, cross-process kernel cache (spike S2, ADR-0008). GPU-free.

brotkit keys and stores its own compiled kernels. It never uses Numba's `cache=True`, whose key
ignores `max_registers`, `ptxas_options`, `lineinfo`, `opt` and the function's constants
(spike S2; numba-cuda-mlir#493). A backend builds a kernel once, the cache stores the build
with everything it was keyed on, and every later process loads it without recompiling.

Pieces:

- `KernelSpec`: everything that determines the binary, as a frozen value object.
- `key_components` / `digest`: the cache key, a sha256 of canonical JSON.
- `KernelStore`: the directory of entries. One file per entry, written to a temp file and
  published with `os.replace`; a striped cross-process lock serialises builds of one key;
  corrupt entries are deleted, warned about once and rebuilt; least-recently-used eviction
  keeps it under a size cap.
- `KernelCache`: the process-wide front end. It checks memory, then disk, then builds under
  the key's lock, and hands entries to the backend's `load`.

Nothing here imports a GPU stack. The CUDA build and load adapter lives in
`brotkit.backends.cuda`.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import sysconfig
import threading
import time
import uuid
import warnings
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol

import filelock
import platformdirs

__all__ = [
    "CACHE_FORMAT",
    "CODEGEN_VERSION",
    "DEFAULT_MAX_BYTES",
    "ENV_DIR",
    "OPTION_DEFAULTS",
    "Build",
    "CacheBackend",
    "CacheEntry",
    "CacheLocationError",
    "CacheStats",
    "CorruptEntryError",
    "EntryInfo",
    "KernelCache",
    "KernelCacheWarning",
    "KernelSpec",
    "KernelStore",
    "canonical_json",
    "check_location",
    "decode_entry",
    "default_root",
    "digest",
    "encode_entry",
    "filesystem_type",
    "key_components",
    "python_tag",
    "replace_option",
]

type JSON = bool | int | float | str | list[JSON] | dict[str, JSON] | None

CACHE_FORMAT: Final = 1
"""On-disk entry format. Part of the key and of the directory name (`v1/`)."""

CODEGEN_VERSION: Final = 1
"""Bump when generated source or the kernel ABI changes in a way the source hash can't see
(for example, how a backend interprets a signature). Part of the key."""

ENV_DIR: Final = "BROTKIT_CACHE_DIR"
DEFAULT_MAX_BYTES: Final = 1 << 30

OPTION_DEFAULTS: Final[Mapping[str, JSON]] = {
    "fastmath": False,
    "max_registers": None,
    "launch_bounds": None,
    "lineinfo": False,
    "debug": False,
    "opt": True,
    "ptxas_options": [],
    "nvjitlink_options": [],
}
"""Every compile option a backend may read, with its default. A backend reads options only
from `KernelSpec.options`, which accepts only these names, so every option that reaches the
compiler is in the key. An option equal to its default is dropped, so `{}` and
`{"fastmath": False}` give the same key."""

REFUSED_FILESYSTEMS: Final = frozenset({"9p", "drvfs", "v9fs"})
"""Filesystems the cache refuses to live on. Under WSL2 a Windows drive (`/mnt/c`) is a 9p
mount: Linux file locks there serialise Linux processes but don't exclude Windows processes
(spike S2), so a Windows-side brotkit sharing the directory could race a build."""

_MAGIC: Final = b"BKKC"
_HEAD: Final = struct.Struct("<4sIQ32s")  # magic, format, header length, header sha256
_STRIPES: Final = 256  # lock files; never deleted, so the lock inode is stable
_MAX_HEADER: Final = 1 << 24


class KernelCacheWarning(UserWarning):
    """A cache entry was unusable (corrupt, truncated or mis-keyed) and was rebuilt."""


class CacheLocationError(RuntimeError):
    """The cache directory is on a filesystem where its locking isn't reliable."""


class CorruptEntryError(ValueError):
    """An entry file failed validation."""


def canonical_json(value: JSON) -> str:
    """Canonical JSON text: sorted keys, no whitespace, ASCII (what the key hashes)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalise(value: object) -> JSON:
    """Options as JSON: tuples become lists; anything else non-JSON is rejected."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"option value {value!r} isn't finite")
        return value
    if isinstance(value, list | tuple):
        items: list[object] | tuple[object, ...] = value  # pyright: ignore[reportUnknownVariableType]
        return [_normalise(v) for v in items]
    raise TypeError(f"option values must be JSON scalars or sequences, not {type(value).__name__}")


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelSpec:
    """Everything that determines a compiled kernel binary.

    `signature` is the kernel's argument types in brotkit's ABI notation: scalar type names
    (`float32`, `float64`, `int32`, `int64`, `uint32`, ...) and pointers as `float32*`.
    `options` holds `(name, canonical JSON)` pairs, sorted; build it with `KernelSpec.make`.
    """

    source: str
    entry: str
    signature: tuple[str, ...]
    backend: str
    target: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.entry.isidentifier():
            raise ValueError(f"entry {self.entry!r} isn't a Python identifier")
        names = [n for n, _ in self.options]
        if names != sorted(set(names)):
            raise ValueError("options must be sorted and unique; use KernelSpec.make")
        unknown = set(names) - OPTION_DEFAULTS.keys()
        if unknown:
            raise ValueError(f"unknown compile options {sorted(unknown)}")
        for n, v in self.options:
            if v == canonical_json(OPTION_DEFAULTS[n]):
                raise ValueError(f"option {n} equals its default; use KernelSpec.make")

    @classmethod
    def make(
        cls,
        *,
        source: str,
        entry: str,
        signature: tuple[str, ...] | list[str],
        backend: str,
        target: str,
        options: Mapping[str, object] | None = None,
    ) -> KernelSpec:
        opts: list[tuple[str, str]] = []
        for name, value in sorted((options or {}).items()):
            if name not in OPTION_DEFAULTS:
                raise ValueError(f"unknown compile option {name!r}")
            canon = canonical_json(_normalise(value))
            if canon != canonical_json(OPTION_DEFAULTS[name]):
                opts.append((name, canon))
        return cls(
            source=source,
            entry=entry,
            signature=tuple(signature),
            backend=backend,
            target=target,
            options=tuple(opts),
        )

    def option(self, name: str) -> JSON:
        """The value of option `name`, or its default."""
        if name not in OPTION_DEFAULTS:
            raise KeyError(name)
        for n, v in self.options:
            if n == name:
                return json.loads(v)
        return OPTION_DEFAULTS[name]

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()


def python_tag() -> str:
    """Interpreter, minor version and free-threading: `cpython-3.15`, `cpython-3.15t`."""
    ft = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    return f"{sys.implementation.name}-{sys.version_info[0]}.{sys.version_info[1]}{ft}"


def key_components(
    spec: KernelSpec, toolchain: Mapping[str, str], *, codegen_version: int = CODEGEN_VERSION
) -> dict[str, JSON]:
    """The cache key, as the JSON object that is hashed and stored in the entry's manifest."""
    return {
        "format": CACHE_FORMAT,
        "codegen": codegen_version,
        "python": python_tag(),
        "backend": spec.backend,
        "target": spec.target,
        "entry": spec.entry,
        "signature": list(spec.signature),
        "options": {n: json.loads(v) for n, v in spec.options},
        "source_sha256": spec.source_sha256,
        "toolchain": dict(sorted(toolchain.items())),
    }


def digest(components: Mapping[str, JSON]) -> str:
    return hashlib.sha256(canonical_json(dict(components)).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class Build:
    """What a backend's `build` returns: named binary blobs plus JSON metadata."""

    blobs: tuple[tuple[str, bytes], ...]
    meta_json: str = "{}"

    @property
    def meta(self) -> dict[str, JSON]:
        return json.loads(self.meta_json)


@dataclass(frozen=True, slots=True, kw_only=True)
class CacheEntry:
    """A validated entry: what was stored, under which key, and when."""

    digest: str
    key_json: str
    meta_json: str
    blobs: tuple[tuple[str, bytes], ...]
    created: str

    @property
    def key(self) -> dict[str, JSON]:
        return json.loads(self.key_json)

    @property
    def meta(self) -> dict[str, JSON]:
        return json.loads(self.meta_json)

    def blob(self, name: str) -> bytes:
        for n, b in self.blobs:
            if n == name:
                return b
        raise KeyError(name)


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryInfo:
    digest: str
    path: Path
    size: int
    last_used: float


def encode_entry(entry: CacheEntry) -> bytes:
    blobs_meta: list[JSON] = [
        {"name": n, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in entry.blobs
    ]
    header: dict[str, JSON] = {
        "digest": entry.digest,
        "key": json.loads(entry.key_json),
        "meta": json.loads(entry.meta_json),
        "blobs": blobs_meta,
        "created": entry.created,
    }
    hb = canonical_json(header).encode()
    head = _HEAD.pack(_MAGIC, CACHE_FORMAT, len(hb), hashlib.sha256(hb).digest())
    return b"".join([head, hb, *(b for _, b in entry.blobs)])


def decode_entry(data: bytes) -> CacheEntry:
    """Parse and verify an entry file. Raises `CorruptEntryError` on any inconsistency."""
    if len(data) < _HEAD.size:
        raise CorruptEntryError("shorter than the header")
    magic, fmt, hlen, hsha = _HEAD.unpack_from(data)
    if magic != _MAGIC:
        raise CorruptEntryError("bad magic")
    if fmt != CACHE_FORMAT:
        raise CorruptEntryError(f"format {fmt}, expected {CACHE_FORMAT}")
    if hlen > _MAX_HEADER or _HEAD.size + hlen > len(data):
        raise CorruptEntryError("header length out of range (truncated?)")
    hb = data[_HEAD.size : _HEAD.size + hlen]
    if hashlib.sha256(hb).digest() != hsha:
        raise CorruptEntryError("header fails its checksum")
    try:
        header = json.loads(hb)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CorruptEntryError(f"header isn't JSON: {e}") from None
    try:
        blobs_meta = header["blobs"]
        sizes = [int(b["size"]) for b in blobs_meta]
        off = _HEAD.size + hlen
        if off + sum(sizes) != len(data):
            raise CorruptEntryError(
                f"payload is {len(data) - off} bytes, manifest says {sum(sizes)} (truncated?)"
            )
        blobs: list[tuple[str, bytes]] = []
        for b, size in zip(blobs_meta, sizes, strict=True):
            payload = data[off : off + size]
            if hashlib.sha256(payload).hexdigest() != b["sha256"]:
                raise CorruptEntryError(f"blob {b['name']!r} fails its checksum")
            blobs.append((str(b["name"]), payload))
            off += size
        return CacheEntry(
            digest=str(header["digest"]),
            key_json=canonical_json(header["key"]),
            meta_json=canonical_json(header["meta"]),
            blobs=tuple(blobs),
            created=str(header["created"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, CorruptEntryError):
            raise
        raise CorruptEntryError(f"malformed manifest: {e!r}") from None


def default_root() -> Path:
    """`$BROTKIT_CACHE_DIR`, else the platform user cache dir + `kernels`
    (`~/.cache/brotkit/kernels` on Linux)."""
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env).expanduser()
    return platformdirs.user_cache_path("brotkit", appauthor=False) / "kernels"


def _unescape_mount(field_: str) -> str:
    # /proc/mounts escapes space, tab, newline and backslash as octal.
    return (
        field_.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def filesystem_type(path: Path, mounts: str | None = None) -> str | None:
    """Filesystem type of the mount holding `path` (Linux; None elsewhere or if unknown)."""
    if mounts is None:
        try:
            mounts = Path("/proc/self/mounts").read_text()
        except OSError:
            return None
    # Resolve symlinks in the part that exists, then put back the part that doesn't (the cache
    # root is often created later).
    p = path.expanduser().absolute()
    tail: list[str] = []
    while not p.exists() and p != p.parent:
        tail.append(p.name)
        p = p.parent
    target = str(p.resolve().joinpath(*reversed(tail)))
    best, fstype = "", None
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mnt = _unescape_mount(parts[1])
        inside = target == mnt or target.startswith(mnt.rstrip("/") + "/") or mnt == "/"
        if inside and len(mnt) >= len(best):
            best, fstype = mnt, parts[2]
    return fstype


def check_location(root: Path, mounts: str | None = None) -> None:
    """Refuse a cache root on a filesystem where cross-process locking isn't reliable."""
    fs = filesystem_type(root, mounts)
    if fs in REFUSED_FILESYSTEMS:
        raise CacheLocationError(
            f"brotkit's kernel cache can't live at {root}: it is on a {fs} mount (a Windows "
            "drive under WSL2). File locks there don't exclude Windows processes, so two "
            f"brotkits could race a build. Set {ENV_DIR} to a Linux-side path such as "
            "~/.cache/brotkit/kernels."
        )


class KernelStore:
    """The cache directory: `<root>/v1/<d[:2]>/<d>.bkk`, with locks in `<root>/v1/locks/`.

    One file per entry, published atomically. The file's mtime is its last-used time, bumped
    on every hit, so a hit never rewrites the entry.
    """

    __slots__ = ("_warned", "corrupt_count", "dir", "max_bytes", "root")

    def __init__(self, root: Path | str | None = None, *, max_bytes: int = DEFAULT_MAX_BYTES):
        self.root = Path(root).expanduser() if root is not None else default_root()
        check_location(self.root)
        self.dir = self.root / f"v{CACHE_FORMAT}"
        self.max_bytes = max_bytes
        self._warned: set[str] = set()
        self.corrupt_count = 0  # entries this process found unusable and deleted

    def entry_path(self, digest_: str) -> Path:
        return self.dir / digest_[:2] / f"{digest_}.bkk"

    def source_path(self, digest_: str) -> Path:
        """Where a backend writes the generated `.py` it compiles (for tracebacks)."""
        return self.dir / "src" / f"k_{digest_[:32]}.py"

    def lock(self, digest_: str, timeout: float = -1) -> filelock.BaseFileLock:
        """Cross-process lock for `digest_` (one of 256 stripes)."""
        d = self.dir / "locks"
        d.mkdir(parents=True, exist_ok=True)
        stripe = int(digest_[:2], 16) % _STRIPES
        return filelock.FileLock(str(d / f"{stripe:02x}.lock"), timeout=timeout)

    def _corrupt(self, digest_: str, path: Path, reason: str) -> None:
        self.corrupt_count += 1
        with suppress(FileNotFoundError):
            path.unlink()
        if digest_ not in self._warned:
            self._warned.add(digest_)
            warnings.warn(
                f"brotkit kernel cache: entry {digest_[:12]} at {path} was unusable ({reason}); "
                "deleted it and rebuilding",
                KernelCacheWarning,
                stacklevel=4,
            )

    def read(self, digest_: str, components: Mapping[str, JSON]) -> CacheEntry | None:
        """The entry for `digest_` if it exists, is intact and was stored under exactly
        `components`. A bad entry is deleted (with one warning) and treated as a miss."""
        path = self.entry_path(digest_)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            entry = decode_entry(data)
            if entry.digest != digest_:
                raise CorruptEntryError(f"manifest digest {entry.digest[:12]} != file name")
            if entry.key_json != canonical_json(dict(components)):
                raise CorruptEntryError("stored key components differ from the requested key")
        except CorruptEntryError as e:
            self._corrupt(digest_, path, str(e))
            return None
        with suppress(OSError):
            os.utime(path)
        return entry

    def write(self, entry: CacheEntry) -> Path:
        """Publish `entry` atomically: temp file in the same directory, fsync, os.replace."""
        path = self.entry_path(entry.digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{entry.digest[:16]}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(encode_entry(entry))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            with suppress(FileNotFoundError):
                tmp.unlink()
        return path

    def entries(self) -> list[EntryInfo]:
        out: list[EntryInfo] = []
        if not self.dir.exists():
            return out
        for p in self.dir.glob("??/*.bkk"):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            out.append(EntryInfo(digest=p.stem, path=p, size=st.st_size, last_used=st.st_mtime))
        return out

    def evict(self, *, keep: str | None = None, stale_tmp_seconds: float = 3600.0) -> list[str]:
        """Delete least-recently-used entries until the total is at most `max_bytes`, and
        temp files left by crashed writers. Never deletes `keep`. Returns evicted digests."""
        lock = filelock.FileLock(str(self.dir / "locks" / "evict.lock"), timeout=0)
        (self.dir / "locks").mkdir(parents=True, exist_ok=True)
        try:
            lock.acquire()
        except filelock.Timeout:
            return []  # another process is evicting
        try:
            now = time.time()
            for t in self.dir.glob("??/.*.tmp"):
                with suppress(FileNotFoundError):
                    if now - t.stat().st_mtime > stale_tmp_seconds:
                        t.unlink()
            infos = sorted(self.entries(), key=lambda e: e.last_used)
            total = sum(e.size for e in infos)
            evicted: list[str] = []
            for e in infos:
                if total <= self.max_bytes:
                    break
                if e.digest == keep:
                    continue
                with suppress(FileNotFoundError):
                    e.path.unlink()
                    src = self.source_path(e.digest)
                    src.unlink(missing_ok=True)
                total -= e.size
                evicted.append(e.digest)
            return evicted
        finally:
            lock.release()

    def size(self) -> int:
        return sum(e.size for e in self.entries())


class CacheBackend[K](Protocol):
    """What a backend gives the cache. `build` runs on a miss, under the key's lock; `load`
    turns a stored entry into a launchable kernel and must not compile."""

    @property
    def name(self) -> str: ...

    def toolchain(self) -> Mapping[str, str]:
        """Versions of everything that can change the binary for a given spec."""
        ...

    def build(self, spec: KernelSpec, source_path: Path) -> Build: ...

    def load(self, entry: CacheEntry) -> K: ...


@dataclass(slots=True)
class CacheStats:
    memory_hits: int = 0
    disk_hits: int = 0
    builds: int = 0
    waited_for_other_process: int = 0
    corrupt: int = 0
    last_lookup_seconds: float = 0.0
    events: list[str] = field(default_factory=list[str])


class KernelCache[K]:
    """Process-wide, lock-protected kernel cache (spec: `KernelCache`).

    `get(spec)` returns the loaded kernel: from memory, else from disk, else built by the
    backend under the key's cross-process lock and stored. Threads asking for different keys
    don't wait for each other; threads asking for the same key build once.
    """

    __slots__ = (
        "_key_locks",
        "_lock",
        "_memory",
        "_toolchain",
        "backend",
        "codegen_version",
        "stats",
        "store",
    )

    def __init__(
        self,
        backend: CacheBackend[K],
        store: KernelStore | None = None,
        *,
        codegen_version: int = CODEGEN_VERSION,
    ) -> None:
        self.backend = backend
        self.store = store if store is not None else KernelStore()
        self.codegen_version = codegen_version
        self.stats = CacheStats()
        self._memory: dict[str, K] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._toolchain: Mapping[str, str] | None = None

    def key(self, spec: KernelSpec) -> tuple[str, dict[str, JSON]]:
        if spec.backend != self.backend.name:
            raise ValueError(
                f"spec is for backend {spec.backend!r}, cache is {self.backend.name!r}"
            )
        if self._toolchain is None:
            self._toolchain = dict(self.backend.toolchain())
        comps = key_components(spec, self._toolchain, codegen_version=self.codegen_version)
        return digest(comps), comps

    def _key_lock(self, d: str) -> threading.Lock:
        with self._lock:
            lk = self._key_locks.get(d)
            if lk is None:
                lk = self._key_locks[d] = threading.Lock()
            return lk

    def get(self, spec: KernelSpec) -> K:
        t0 = time.perf_counter()
        d, comps = self.key(spec)
        hit = self._memory.get(d)
        if hit is not None:
            self.stats.memory_hits += 1
            return hit
        with self._key_lock(d):
            hit = self._memory.get(d)
            if hit is not None:
                self.stats.memory_hits += 1
                return hit
            entry = self._read(d, comps)
            if entry is not None:
                self.stats.disk_hits += 1
            else:
                entry = self._build_locked(spec, d, comps)
            kernel = self.backend.load(entry)
            self._memory[d] = kernel
        self.stats.last_lookup_seconds = time.perf_counter() - t0
        return kernel

    def _read(self, d: str, comps: Mapping[str, JSON]) -> CacheEntry | None:
        n = self.store.corrupt_count
        entry = self.store.read(d, comps)
        self.stats.corrupt += self.store.corrupt_count - n
        return entry

    def _build_locked(self, spec: KernelSpec, d: str, comps: dict[str, JSON]) -> CacheEntry:
        lock = self.store.lock(d)
        t_wait = time.perf_counter()
        with lock:
            waited = time.perf_counter() - t_wait
            entry = self._read(d, comps)  # another process may have built it while we waited
            if entry is not None:
                self.stats.waited_for_other_process += 1
                self.stats.events.append(f"{d[:12]}: built by another process ({waited:.2f} s)")
                return entry
            src = self.store.source_path(d)
            src.parent.mkdir(parents=True, exist_ok=True)
            t_build = time.perf_counter()
            build = self.backend.build(spec, src)
            build_s = time.perf_counter() - t_build
            meta = build.meta
            meta["build_seconds"] = round(build_s, 4)
            entry = CacheEntry(
                digest=d,
                key_json=canonical_json(comps),
                meta_json=canonical_json(meta),
                blobs=build.blobs,
                created=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            self.store.write(entry)
            self.stats.builds += 1
        self.store.evict(keep=d)
        return entry

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()


def replace_option(spec: KernelSpec, **options: object) -> KernelSpec:
    """`spec` with some options changed (a convenience over `dataclasses.replace`)."""
    current: dict[str, object] = {n: json.loads(v) for n, v in spec.options}
    current.update(options)
    return KernelSpec.make(
        source=spec.source,
        entry=spec.entry,
        signature=spec.signature,
        backend=spec.backend,
        target=spec.target,
        options=current,
    )

"""A GPU-free backend for kernel cache tests.

`FakeBackend.build` "compiles" by hashing the spec into a fake cubin and reports a register
count that depends on `max_registers`, the way the real build does (S4). Every build also drops
a marker file in `marker_dir`, so tests can count builds across processes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from brotkit.kernels.cache import Build, CacheEntry, KernelSpec

SOURCE = "def k(p, n):\n    pass\n"
UNCAPPED_REGISTERS = 60


def spec(**overrides: object) -> KernelSpec:
    fields: dict[str, object] = {
        "source": SOURCE,
        "entry": "k",
        "signature": ("float32*", "int64"),
        "backend": "fake",
        "target": "sm_86",
        "options": {},
    }
    fields.update(overrides)
    return KernelSpec.make(**fields)  # pyright: ignore[reportArgumentType]


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeKernel:
    digest: str
    cubin: bytes
    registers: int


class FakeBackend:
    def __init__(
        self,
        *,
        versions: Mapping[str, str] | None = None,
        marker_dir: Path | None = None,
        build_seconds: float = 0.0,
    ) -> None:
        self.versions = dict(versions or {"fakecc": "1.0", "fakelink": "1.0"})
        self.marker_dir = marker_dir
        self.build_seconds = build_seconds
        self.builds = 0
        self.loads = 0

    @property
    def name(self) -> str:
        return "fake"

    def toolchain(self) -> Mapping[str, str]:
        return self.versions

    def build(self, spec: KernelSpec, source_path: Path) -> Build:
        source_path.write_text(spec.source)
        if self.build_seconds:
            time.sleep(self.build_seconds)
        self.builds += 1
        if self.marker_dir is not None:
            self.marker_dir.mkdir(parents=True, exist_ok=True)
            (self.marker_dir / f"{os.getpid()}.{uuid.uuid4().hex}").touch()
        cap = spec.option("max_registers")
        registers = min(UNCAPPED_REGISTERS, cap) if isinstance(cap, int) else UNCAPPED_REGISTERS
        ident = json.dumps(
            [spec.source, spec.signature, spec.options, sorted(self.versions.items())]
        )
        cubin = b"CUBIN" + hashlib.sha256(ident.encode()).digest() * 64
        meta = {"symbol": f"_Z1k_{spec.entry}", "resources": {"registers": registers}}
        return Build(blobs=(("cubin", cubin), ("ptx", b"// ptx")), meta_json=json.dumps(meta))

    def load(self, entry: CacheEntry) -> FakeKernel:
        self.loads += 1
        res = entry.meta["resources"]
        assert isinstance(res, dict)
        regs = res["registers"]
        assert isinstance(regs, int)
        return FakeKernel(digest=entry.digest, cubin=entry.blob("cubin"), registers=regs)

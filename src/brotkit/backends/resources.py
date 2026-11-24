"""Kernel resource usage and theoretical occupancy, in pure Python.

Nothing here imports the GPU stack, so CPU CI can test it. `brotkit.backends.cuda.resource_usage`
fills a `ResourceUsage` from nvJitLink's verbose log; `occupancy` turns it into resident
warps per SM with the same arithmetic as CUDA's `cuda_occupancy.h` (registers are allocated per
warp in 256-register units, from four SM sub-partitions). Spike S4 checked this against
`cuOccupancyMaxActiveBlocksPerMultiprocessor` on sm_86 (docs/design/spikes/S4.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "ARCH_LIMITS",
    "ArchLimits",
    "Limiter",
    "Occupancy",
    "ResourceUsage",
    "arch_limits",
    "max_threads_per_block",
    "occupancy",
    "parse_nvjitlink_log",
]

FROZEN = {"frozen": True, "slots": True, "kw_only": True}


@dataclass(**FROZEN)
class ArchLimits:
    """Per-SM limits for one compute capability (CUDA Programming Guide, Compute Capabilities)."""

    cc: tuple[int, int]
    max_warps_per_sm: int
    max_blocks_per_sm: int
    shared_per_sm: int  # bytes, largest carveout
    max_shared_per_block: int  # bytes, with opt-in
    regs_per_sm: int = 65536
    max_regs_per_block: int = 65536
    max_regs_per_thread: int = 255
    reg_alloc_unit: int = 256  # registers, allocated per warp
    sub_partitions: int = 4
    warp_size: int = 32
    max_threads_per_block: int = 1024
    shared_alloc_unit: int = 128
    reserved_shared_per_block: int = 1024  # sm_80+: runtime reserves 1 KB per block


_KB = 1024
ARCH_LIMITS: MappingProxyType[tuple[int, int], ArchLimits] = MappingProxyType(
    {
        (8, 0): ArchLimits(cc=(8, 0), max_warps_per_sm=64, max_blocks_per_sm=32,
                           shared_per_sm=164 * _KB, max_shared_per_block=163 * _KB),
        (8, 6): ArchLimits(cc=(8, 6), max_warps_per_sm=48, max_blocks_per_sm=16,
                           shared_per_sm=100 * _KB, max_shared_per_block=99 * _KB),
        (8, 9): ArchLimits(cc=(8, 9), max_warps_per_sm=48, max_blocks_per_sm=24,
                           shared_per_sm=100 * _KB, max_shared_per_block=99 * _KB),
        (9, 0): ArchLimits(cc=(9, 0), max_warps_per_sm=64, max_blocks_per_sm=32,
                           shared_per_sm=228 * _KB, max_shared_per_block=227 * _KB),
        (10, 0): ArchLimits(cc=(10, 0), max_warps_per_sm=64, max_blocks_per_sm=32,
                            shared_per_sm=228 * _KB, max_shared_per_block=227 * _KB),
        (12, 0): ArchLimits(cc=(12, 0), max_warps_per_sm=48, max_blocks_per_sm=24,
                            shared_per_sm=100 * _KB, max_shared_per_block=99 * _KB),
    }
)  # fmt: skip


def arch_limits(cc: tuple[int, int]) -> ArchLimits:
    """Limits for `cc`; raises `KeyError` naming the known architectures if it isn't tabled."""
    try:
        return ARCH_LIMITS[cc]
    except KeyError:
        known = ", ".join(f"sm_{a}{b}" for a, b in ARCH_LIMITS)
        raise KeyError(f"no SM limits for sm_{cc[0]}{cc[1]} (known: {known})") from None


@dataclass(**FROZEN)
class ResourceUsage:
    """Per-kernel resources of one compiled kernel for one compute capability.

    `registers`, `stack_bytes` and `static_shared_bytes` match the driver's
    `cuFuncGetAttribute` values (NUM_REGS, LOCAL_SIZE_BYTES, SHARED_SIZE_BYTES). Spill bytes
    are summed over the entry function and any callee subroutines ptxas reports.
    """

    cc: tuple[int, int]
    registers: int
    spill_store_bytes: int
    spill_load_bytes: int
    stack_bytes: int  # local memory per thread (stack frame: spills plus local arrays)
    static_shared_bytes: int
    max_threads_per_block: int
    max_registers: int | None = None  # the cap it was compiled under, if any

    @property
    def spills(self) -> bool:
        return self.spill_store_bytes > 0 or self.spill_load_bytes > 0


class Limiter(StrEnum):
    WARPS = "warps"
    BLOCKS = "blocks"
    REGISTERS = "registers"
    SHARED = "shared"


@dataclass(**FROZEN)
class Occupancy:
    block_size: int
    blocks_per_sm: int
    warps_per_sm: int
    max_warps_per_sm: int
    limiter: Limiter

    @property
    def fraction(self) -> float:
        return self.warps_per_sm / self.max_warps_per_sm


def _round_up(x: int, unit: int) -> int:
    return -(-x // unit) * unit


def _warps_by_registers(registers: int, lim: ArchLimits) -> int:
    """Resident warps per SM that `registers`/thread allows (per sub-partition, as the driver)."""
    if registers <= 0:
        return lim.max_warps_per_sm
    per_warp = _round_up(registers * lim.warp_size, lim.reg_alloc_unit)
    per_partition = (lim.regs_per_sm // lim.sub_partitions) // per_warp
    return per_partition * lim.sub_partitions


def max_threads_per_block(registers: int, cc: tuple[int, int], maxntid: int | None = None) -> int:
    """The driver's MAX_THREADS_PER_BLOCK for a kernel using `registers` per thread."""
    lim = arch_limits(cc)
    # A block must fit on one SM; the per-sub-partition register split is the binding limit
    # (S4: 77 registers -> 768 threads, not 800).
    threads = min(lim.max_threads_per_block, _warps_by_registers(registers, lim) * lim.warp_size)
    return min(threads, maxntid) if maxntid else threads


def occupancy(usage: ResourceUsage, block_size: int, *, dynamic_shared_bytes: int = 0) -> Occupancy:
    """Theoretical occupancy of `usage` at `block_size` threads per block."""
    lim = arch_limits(usage.cc)
    if not 0 < block_size <= usage.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} outside 1..{usage.max_threads_per_block} for this kernel"
        )
    warps_per_block = -(-block_size // lim.warp_size)
    by_warps = lim.max_warps_per_sm // warps_per_block
    by_regs = _warps_by_registers(usage.registers, lim) // warps_per_block
    smem = _round_up(
        usage.static_shared_bytes + dynamic_shared_bytes + lim.reserved_shared_per_block,
        lim.shared_alloc_unit,
    )
    by_smem = lim.shared_per_sm // smem
    candidates = [
        (by_warps, Limiter.WARPS),
        (lim.max_blocks_per_sm, Limiter.BLOCKS),
        (by_regs, Limiter.REGISTERS),
        (by_smem, Limiter.SHARED),
    ]
    blocks, limiter = min(candidates, key=lambda c: c[0])
    return Occupancy(
        block_size=block_size,
        blocks_per_sm=blocks,
        warps_per_sm=blocks * warps_per_block,
        max_warps_per_sm=lim.max_warps_per_sm,
        limiter=limiter,
    )


# nvJitLink `-verbose -Xptxas=-v` log. The final "info : Function properties for '<entry>':"
# summary is whole-program (it equals the driver's attributes); the "ptxas" blocks carry the
# spill counts, one block per function (entry and callee subroutines).
_SUMMARY = re.compile(
    r"info\s*:\s*used (?P<regs>\d+) registers,.*?(?P<stack>\d+) stack, (?P<smem>\d+) bytes smem"
)
_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_MAXNTID = re.compile(r"^\s*\.maxntid\s+(\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?", re.MULTILINE)


def parse_nvjitlink_log(
    log: str,
    cc: tuple[int, int],
    *,
    ptx: str = "",
    max_registers: int | None = None,
) -> ResourceUsage:
    """Build a `ResourceUsage` from nvJitLink's info log (and the PTX, for `.maxntid`)."""
    summaries = list(_SUMMARY.finditer(log))
    if not summaries:
        raise ValueError("no 'used N registers' summary in nvJitLink log; was -verbose passed?")
    s = summaries[-1]
    spills = _SPILL.findall(log)
    if not spills:
        raise ValueError("no ptxas spill lines in nvJitLink log; was -Xptxas=-v passed?")
    maxntid = None
    if m := _MAXNTID.search(ptx):
        maxntid = 1
        for g in m.groups():
            maxntid *= int(g) if g else 1
    regs = int(s["regs"])
    return ResourceUsage(
        cc=cc,
        registers=regs,
        spill_store_bytes=sum(int(st) for _, st, _ in spills),
        spill_load_bytes=sum(int(ld) for _, _, ld in spills),
        stack_bytes=int(s["stack"]),
        static_shared_bytes=int(s["smem"]),
        max_threads_per_block=max_threads_per_block(regs, cc, maxntid),
        max_registers=max_registers,
    )

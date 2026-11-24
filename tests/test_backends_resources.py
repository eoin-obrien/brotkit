"""Resource parsing and the occupancy calculator (pure Python, CPU CI)."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from brotkit.backends.resources import (
    ARCH_LIMITS,
    Limiter,
    ResourceUsage,
    arch_limits,
    max_threads_per_block,
    occupancy,
    parse_nvjitlink_log,
)

LOGS = pathlib.Path(__file__).parent / "fixtures" / "nvjitlink"

# Measured on an RTX 3090 Ti (sm_86) in spike S4: registers -> (driver MAX_THREADS_PER_BLOCK,
# cuOccupancyMaxActiveBlocksPerMultiprocessor at 128 and at 256 threads). 175 of 175 kernel
# builds matched the calculator; these are the distinct register counts among them.
SM86_DRIVER = {
    20: (1024, 12, 6), 32: (1024, 12, 6), 40: (1024, 12, 6), 42: (1024, 10, 5),
    47: (1024, 10, 5), 52: (1024, 9, 4), 56: (1024, 9, 4), 58: (1024, 8, 4),
    64: (1024, 8, 4), 66: (896, 7, 3), 72: (896, 7, 3), 77: (768, 6, 3),
    83: (640, 5, 2), 96: (640, 5, 2), 111: (512, 4, 2), 118: (512, 4, 2),
}  # fmt: skip


def usage(registers: int, cc: tuple[int, int] = (8, 6), shared: int = 0) -> ResourceUsage:
    return ResourceUsage(
        cc=cc,
        registers=registers,
        spill_store_bytes=0,
        spill_load_bytes=0,
        stack_bytes=0,
        static_shared_bytes=shared,
        max_threads_per_block=max_threads_per_block(registers, cc),
    )


@pytest.mark.parametrize(("regs", "expected"), SM86_DRIVER.items())
def test_matches_sm86_driver(regs: int, expected: tuple[int, int, int]) -> None:
    maxtpb, b128, b256 = expected
    assert max_threads_per_block(regs, (8, 6)) == maxtpb
    assert occupancy(usage(regs), 128).blocks_per_sm == b128
    assert occupancy(usage(regs), 256).blocks_per_sm == b256


def test_register_split_is_per_sub_partition() -> None:
    # 77 regs -> 2560 per warp; 65536 / 2560 = 25.6 warps, but per sub-partition
    # 16384 / 2560 = 6, so 24 warps = 768 threads (the driver agrees; a flat formula says 800).
    assert max_threads_per_block(77, (8, 6)) == 768


def test_limiters() -> None:
    assert occupancy(usage(20), 256).limiter is Limiter.WARPS
    assert occupancy(usage(72), 256).limiter is Limiter.REGISTERS
    assert occupancy(usage(20, shared=48 * 1024), 128).limiter is Limiter.SHARED
    assert occupancy(usage(16, cc=(8, 6)), 32).limiter is Limiter.BLOCKS


def test_fraction_and_block_size_bounds() -> None:
    occ = occupancy(usage(64), 256)
    assert occ.fraction == pytest.approx(32 / 48)
    with pytest.raises(ValueError, match="block size"):
        occupancy(usage(77), 1024)
    with pytest.raises(KeyError, match="sm_75"):
        arch_limits((7, 5))


def test_launch_bounds_cap_max_threads() -> None:
    assert max_threads_per_block(20, (8, 6), maxntid=256) == 256


@given(
    regs=st.integers(1, 255),
    cc=st.sampled_from(sorted(ARCH_LIMITS)),
    block=st.sampled_from([32, 64, 128, 256, 512]),
)
def test_occupancy_properties(regs: int, cc: tuple[int, int], block: int) -> None:
    u = usage(regs, cc)
    lim = arch_limits(cc)
    mtpb = u.max_threads_per_block
    assert mtpb % 32 == 0
    assert 0 < mtpb <= 1024
    if block > mtpb:
        return
    occ = occupancy(u, block)
    assert 0 < occ.warps_per_sm <= lim.max_warps_per_sm
    assert 0 < occ.fraction <= 1
    # more registers never raise occupancy
    if regs < 255:
        more = usage(regs + 1, cc)
        if block <= more.max_threads_per_block:
            assert occupancy(more, block).warps_per_sm <= occ.warps_per_sm


@given(regs=st.integers(1, 255), stores=st.integers(0, 4096), loads=st.integers(0, 4096))
def test_resource_usage_is_a_frozen_value(regs: int, stores: int, loads: int) -> None:
    u = dataclasses.replace(usage(regs), spill_store_bytes=stores, spill_load_bytes=loads)
    assert u.spills == (stores > 0 or loads > 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.registers = 1  # type: ignore[misc]
    assert u == dataclasses.replace(u)
    assert hash(u) == hash(dataclasses.replace(u))


def test_parse_fresh_log() -> None:
    u = parse_nvjitlink_log((LOGS / "dd_C_sm86_fresh.log").read_text(), (8, 6))
    assert (u.registers, u.stack_bytes, u.static_shared_bytes) == (59, 0, 0)
    assert not u.spills
    assert u.max_threads_per_block == 1024


def test_parse_capped_log_reports_spills() -> None:
    log = (LOGS / "dd_C_sm86_maxrreg32.log").read_text()
    u = parse_nvjitlink_log(log, (8, 6), max_registers=32)
    assert u.registers == 32
    assert (u.spill_store_bytes, u.spill_load_bytes, u.stack_bytes) == (196, 212, 152)
    assert u.spills
    assert u.max_registers == 32


def test_parse_reads_maxntid_from_ptx() -> None:
    log = (LOGS / "f32_A_sm86_fresh.log").read_text()
    ptx = ".entry k(\n)\n.maxntid 128, 2, 1\n{\n}\n"
    assert parse_nvjitlink_log(log, (8, 6), ptx=ptx).max_threads_per_block == 256


def test_parse_rejects_cache_hit_log_without_spill_lines() -> None:
    # nvJitLink omits the ptxas block on a compute-cache hit; resource_usage salts the PTX to
    # avoid this, and the parser refuses rather than reporting "no spills" by omission.
    with pytest.raises(ValueError, match="spill"):
        parse_nvjitlink_log((LOGS / "f32_A_sm86_cache_hit.log").read_text(), (8, 6))


def test_parse_rejects_non_verbose_log() -> None:
    with pytest.raises(ValueError, match="-verbose"):
        parse_nvjitlink_log("", (8, 6))

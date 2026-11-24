"""Q2 validation: brotkit.backends.cuda.resource_usage + resources.occupancy vs the driver.

For every (tier, kernel) x cap in (None, 32, 64, 96, 128) on sm_86: registers, local memory and
max threads per block vs cuFuncGetAttribute, and occupancy at 128/256 vs
cuOccupancyMaxActiveBlocksPerMultiprocessor. Also records sm_100/sm_120 resource_usage (offline).
"""

import json
import pathlib

import gen
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.cudadrv import devices

from brotkit.backends.cuda import resource_usage
from brotkit.backends.resources import occupancy

ctx = devices.get_context()
rows, bad = [], 0
for tier in gen.TIERS:
    for kern in gen.KERNELS:
        sig = gen.signature(tier)
        py = gen.load(tier, kern).py_func
        for cap in (None, 32, 64, 96, 128):
            d = cuda.jit(max_registers=cap)(py) if cap else cuda.jit(py)
            d.compile(sig)
            ru = resource_usage(py, sig, (8, 6), max_registers=cap)
            cufunc = d.overloads[tuple(sig)]._codelibrary.get_cufunc()
            drv = dict(
                regs=d.get_regs_per_thread(sig),
                local=d.get_local_mem_per_thread(sig),
                shared=d.get_shared_mem_per_block(sig),
                maxtpb=d.get_max_threads_per_block(sig),
            )
            occ = {}
            for b in (128, 256):
                occ[b] = (
                    occupancy(ru, b).blocks_per_sm,
                    ctx.get_active_blocks_per_multiprocessor(cufunc, b, 0),
                )
            ok = (
                ru.registers == drv["regs"]
                and ru.stack_bytes == drv["local"]
                and ru.static_shared_bytes == drv["shared"]
                and ru.max_threads_per_block == drv["maxtpb"]
                and all(a == b for a, b in occ.values())
            )
            bad += not ok
            row = dict(
                tier=tier,
                kernel=kern,
                cap=cap,
                ours=dict(
                    regs=ru.registers,
                    local=ru.stack_bytes,
                    maxtpb=ru.max_threads_per_block,
                    spill_st=ru.spill_store_bytes,
                    spill_ld=ru.spill_load_bytes,
                ),
                driver=drv,
                blocks_per_sm={b: dict(ours=o, driver=v) for b, (o, v) in occ.items()},
                match=ok,
            )
            rows.append(row)
            print(
                ("OK  " if ok else "BAD ")
                + f"{tier:5} {kern:3} cap={cap!s:4} regs {ru.registers}/{drv['regs']} "
                f"local {ru.stack_bytes}/{drv['local']} "
                f"maxtpb {ru.max_threads_per_block}/{drv['maxtpb']} "
                f"spill {ru.spill_store_bytes}/{ru.spill_load_bytes} "
                f"occ128 {occ[128]} occ256 {occ[256]}",
                flush=True,
            )
print(f"{len(rows) - bad}/{len(rows)} match")
pathlib.Path("results/q2_validate.json").write_text(json.dumps(rows, indent=1))

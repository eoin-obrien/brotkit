"""Compile-only resource table: every tier x kernel at sm_86, sm_100, sm_120, plus occupancy.

Uses brotkit's resource_usage (compile_ptx + nvJitLink -verbose -Xptxas=-v), which S4 Q2
validated against the driver on sm_86. sm_100/sm_120 have no device here, so those numbers are
compile-only. compile_ptx still needs a CUDA context on numba-cuda-mlir 0.6.0 (Q1), so run this
on the GPU host; it launches nothing.

usage: uv run python q3_resources.py  -> results/q3_resources.json
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
CCS = ((8, 6), (10, 0), (12, 0))
TIERS = ("f32", "f32rn", "f64", "df64", "dd", "qf", "qfb")
KERNELS = ("Z", "A", "B", "C", "F6", "S2")


def main() -> None:
    sys.path.insert(0, str(HERE))
    import gen

    from brotkit.backends.cuda import resource_usage
    from brotkit.backends.resources import occupancy

    res: dict[str, dict] = {}
    for cc in CCS:
        for tier in TIERS:
            for kern in KERNELS:
                key = f"sm_{cc[0]}{cc[1]}/{tier}/{kern}"
                try:
                    ru = resource_usage(gen.load(tier, kern), gen.signature(tier), cc)
                except Exception as e:
                    res[key] = dict(error=f"{type(e).__name__}: {e}"[:300])
                    print(f"{key:24} ERROR {res[key]['error'][:120]}", flush=True)
                    continue
                row = dict(
                    cc=cc,
                    tier=tier,
                    kernel=kern,
                    features=gen.FEATURE_COUNT[kern],
                    regs=ru.registers,
                    spill_st=ru.spill_store_bytes,
                    spill_ld=ru.spill_load_bytes,
                    stack=ru.stack_bytes,
                    smem=ru.static_shared_bytes,
                    maxtpb=ru.max_threads_per_block,
                )
                for bs in (128, 256):
                    o = occupancy(ru, bs)
                    row[f"occ{bs}"] = round(o.fraction, 4)
                    row[f"blocks{bs}"] = o.blocks_per_sm
                    row[f"lim{bs}"] = str(o.limiter)
                res[key] = row
                print(
                    f"{key:24} regs={ru.registers:3} spill={ru.spill_store_bytes}/"
                    f"{ru.spill_load_bytes} stack={ru.stack_bytes:3} occ128={row['occ128']:.3f} "
                    f"occ256={row['occ256']:.3f}",
                    flush=True,
                )
    (HERE / "results" / "q3_resources.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

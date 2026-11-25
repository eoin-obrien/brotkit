"""ABI cost, part 1: registers, spills and the SASS escape loop, struct vs ptr ABI (throwaway).

usage: .venv/bin/python abi_resources.py [kernels...] -> results/abi_resources.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import gen_s2
import s2lib

from brotkit.backends.cuda import compile_ptx
from brotkit.backends.resources import parse_nvjitlink_log

TIERS = ("f32", "df64", "f64", "dd", "qf")
PTR = __import__("os").environ.get("S2_PTR", "ptr64")  # the ABI under test (ADR-0008: int64)
ALLOC = 256  # register allocation unit per warp (sm_80+)
CC = (8, 6)


def main() -> None:
    kernels = sys.argv[1:] or ["C"]
    out: dict = {}
    tmp = pathlib.Path(tempfile.mkdtemp()) / "k.cubin"
    for kern in kernels:
        for tier in TIERS:
            row = {}
            for abi in ("struct", PTR):
                k = gen_s2.load(tier, kern, abi)
                ptx = compile_ptx(k, gen_s2.signature(tier, abi), CC)
                cubin, log = s2lib.link(ptx, CC)
                ru = parse_nvjitlink_log(log, CC, ptx=ptx, max_registers=None)
                text = s2lib.sass(cubin, tmp)
                import re

                entry = re.search(r"\.visible \.entry (\S+)\(", ptx).group(1)
                ins = s2lib.entry_sass(text, entry)
                loop = s2lib.hot_loop(ins)
                row[abi] = dict(
                    registers=ru.registers,
                    spill_store=ru.spill_store_bytes,
                    spill_load=ru.spill_load_bytes,
                    stack=ru.stack_bytes,
                    entry_insns=len(ins),
                    loop_insns=loop.get("n_insns"),
                    loop_addr_ops=loop.get("addr_ops"),
                    loop_mem_ops=loop.get("mem_ops"),
                    loop_opcodes=loop.get("opcodes"),
                    entry=entry,
                )
            s, p = row["struct"], row[PTR]
            row["loop_identical"] = s["loop_opcodes"] == p["loop_opcodes"]
            row["same_loop_multiset"] = sorted(s["loop_opcodes"].split()) == sorted(
                p["loop_opcodes"].split()
            )
            alloc = lambda r: -(-r * 32 // ALLOC)  # noqa: E731
            row["same_warp_alloc"] = alloc(s["registers"]) == alloc(p["registers"])
            row["work_equal"] = (
                s["loop_insns"] == p["loop_insns"]
                and s["loop_addr_ops"] == p["loop_addr_ops"]
                and s["loop_mem_ops"] == p["loop_mem_ops"]
                and p["spill_store"] == p["spill_load"] == 0
                and row["same_warp_alloc"]
            )
            out[f"{tier}/{kern}"] = row
            print(
                f"{tier:5}/{kern:3} regs {s['registers']:3} -> {p['registers']:3}  "
                f"spill {s['spill_store']}/{s['spill_load']} -> "
                f"{p['spill_store']}/{p['spill_load']}  "
                f"entry {s['entry_insns']} -> {p['entry_insns']}  "
                f"loop {s['loop_insns']} -> {p['loop_insns']} (addr {s['loop_addr_ops']} -> "
                f"{p['loop_addr_ops']}, mem {s['loop_mem_ops']} -> {p['loop_mem_ops']}) "
                f"multiset={row['same_loop_multiset']} warpalloc={row['same_warp_alloc']} "
                f"WORK_EQUAL={row['work_equal']}",
                flush=True,
            )
    (s2lib.RESULTS / f"abi_resources_{PTR}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

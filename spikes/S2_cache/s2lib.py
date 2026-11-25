"""Shared spike helpers (throwaway): link PTX to cubin + log, SASS loop analysis, GPU guard."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import uuid

HERE = pathlib.Path(__file__).parent
S4 = HERE.parent / "S4_registers"
BIN = next((HERE / ".venv").glob("lib/python3.*/site-packages/nvidia/cu13/bin"))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def s4_on_path() -> None:
    if str(S4) not in sys.path:
        sys.path.insert(0, str(S4))


def link(ptx: str, cc: tuple[int, int], *, max_registers: int | None = None) -> tuple[bytes, str]:
    """nvJitLink PTX -> (cubin, verbose log). Salted so the log keeps the spill block (S4)."""
    from cuda.bindings import nvjitlink

    opts = [f"-arch=sm_{cc[0]}{cc[1]}", "-verbose", "-Xptxas=-v"]
    if max_registers is not None:
        opts.append(f"-maxrregcount={max_registers}")
    ptx = f"{ptx}\n// s2 {uuid.uuid4().hex}\n"
    h = nvjitlink.create(len(opts), opts)
    try:
        d = ptx.encode()
        nvjitlink.add_data(h, nvjitlink.InputType.PTX, d, len(d), "k.ptx")
        nvjitlink.complete(h)
        cub = bytearray(nvjitlink.get_linked_cubin_size(h))
        nvjitlink.get_linked_cubin(h, cub)
        log = bytearray(nvjitlink.get_info_log_size(h))
        nvjitlink.get_info_log(h, log)
    finally:
        nvjitlink.destroy(h)
    return bytes(cub), log.decode(errors="replace").rstrip("\x00")


def sass(cubin: bytes, tmp: pathlib.Path) -> str:
    tmp.write_bytes(cubin)
    return subprocess.check_output([str(BIN / "cuobjdump"), "-sass", str(tmp)], text=True)


_INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")
ADDR_OPS = ("LEA", "IMAD.WIDE", "IADD3.X", "LEA.HI", "SHF")


def entry_sass(text: str, entry: str) -> list[tuple[int, str, str]]:
    """(address, opcode, operands) for the entry function only."""
    out, on = [], False
    for line in text.splitlines():
        if "Function :" in line:
            on = entry in line
            continue
        if on:
            m = _INSN.search(line)
            if m:
                out.append((int(m.group(1), 16), m.group(3), m.group(4)))
    return out


def hot_loop(insns: list[tuple[int, str, str]]) -> dict:
    """The largest backward branch in the entry = the escape loop. Count its instructions."""
    best = None
    for addr, op, args in insns:
        if op.startswith("BRA"):
            m = re.search(r"0x([0-9a-f]+)", args)
            if m:
                t = int(m.group(1), 16)
                if t < addr and (best is None or addr - t > best[1] - best[0]):
                    best = (t, addr)
    if best is None:
        return {"found": False}
    body = [(a, op, args) for a, op, args in insns if best[0] <= a <= best[1]]
    ops = [op for _, op, _ in body]
    return {
        "found": True,
        "n_insns": len(body),
        "addr_ops": sum(1 for op in ops if op.startswith(ADDR_OPS)),
        "mem_ops": sum(
            1 for op in ops if op.startswith(("LDG", "STG", "LDL", "STL", "LD.", "ST."))
        ),
        "opcodes": " ".join(ops),
    }

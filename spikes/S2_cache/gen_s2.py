"""Spike S2 kernel generator (throwaway): S4's kernels in two ABIs.

- abi="struct": S4's kernels unchanged (Numba array arguments, Numba's array struct ABI).
- abi="ptr": the raw-pointer ABI brotkit adopts (ADR-0008). Output layers arrive as
  CPointer(float32) / CPointer(uint32) plus int32 h, w and a row pitch in elements; the kernel
  rebuilds C-contiguous views with the public `cuda.carray`. Layer stride is h * pitch, so a
  padded (pitched) allocation works and an unpadded one passes pitch = w.

Everything else (tiers, features, libraries) is S4's `gen.py`, imported by path.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
S4 = HERE.parent / "S4_registers"
GEN_DIR = HERE / "gen"


def _s4gen():
    mod = sys.modules.get("gen")
    if mod is None:
        spec = importlib.util.spec_from_file_location("gen", S4 / "gen.py")
        assert spec
        assert spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen"] = mod
        spec.loader.exec_module(mod)
    return mod


s4 = _s4gen()
TIERS, KERNELS, NLF, NLU = s4.TIERS, s4.KERNELS, s4.NLF, s4.NLU

_STRUCT_HEAD = """def kern(c0r_, c0i_, dx, max_iter, row0, out_f, out_u):
    x, y = cuda.grid(2)
    h = out_u.shape[1]
    w = out_u.shape[2]
"""
_PTR_HEAD = """def kern(c0r_, c0i_, dx, max_iter, row0, p_f, p_u, h, w, pitch):
    out_f = cuda.carray(p_f, (NLF, h, pitch), F32)
    out_u = cuda.carray(p_u, (NLU, h, pitch), U32)
    x, y = cuda.grid(2)
"""
assert _STRUCT_HEAD in s4.KERNEL


def emit(tier: str, kernel: str, abi: str = "ptr", *, fastmath: bool = False) -> str:
    """abi: "struct" (S4), "ptr" (int32 h, w, pitch) or "ptr64" (int64 h, w, pitch; same source)."""
    if abi == "struct":
        return s4.emit(tier, kernel, fastmath=fastmath)
    assert abi in ("ptr", "ptr64")
    orig = s4.KERNEL
    try:
        s4.KERNEL = orig.replace(_STRUCT_HEAD, _PTR_HEAD)
        src = s4.emit(tier, kernel, fastmath=fastmath)
    finally:
        s4.KERNEL = orig
    return src.replace("TRAPY = F32(0.1)\n", f"TRAPY = F32(0.1)\nNLF = {NLF}\nNLU = {NLU}\n", 1)


def entry_name(tier: str, kernel: str) -> str:
    return f"k_{tier}_{kernel}"


def signature(tier: str, abi: str = "ptr"):
    from numba_cuda_mlir import types

    if abi == "struct":
        return s4.signature(tier)
    word, nw, _ = TIERS[tier]
    ft = getattr(types, word)
    return (
        (ft,) * (2 * nw)
        + (ft, types.int32, types.int32)
        + (types.CPointer(types.float32), types.CPointer(types.uint32))
        + ((types.int64,) * 3 if abi == "ptr64" else (types.int32,) * 3)
    )


def write(
    tier: str, kernel: str, abi: str = "ptr", out_dir: pathlib.Path = GEN_DIR
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # ptr64 is the same source as ptr (only the signature differs) but a separate module, so
    # the two dispatchers don't share overloads.
    path = out_dir / f"s2_{abi}_{tier}_{kernel}.py"
    src = emit(tier, kernel, abi)
    if not path.exists() or path.read_text() != src:
        path.write_text(src)
    return path


def load(tier: str, kernel: str, abi: str = "ptr", *, jit_options: dict | None = None):
    path = write(tier, kernel, abi)
    name = path.stem
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec
        assert spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    k = getattr(mod, entry_name(tier, kernel))
    if jit_options:
        from numba_cuda_mlir import cuda

        k = cuda.jit(**jit_options)(k.py_func)
    return k


class PtrAdapter:
    """Lets S4's timing code (which calls k[grid, block](..., out_f, out_u)) drive a ptr kernel."""

    def __init__(self, k, tier: str, abi: str = "ptr") -> None:
        # Explicit signature, as brotkit always compiles: a lazily typed dispatcher would type
        # an int pointer as int64 and fail to resolve cuda.carray.
        k.compile(signature(tier, abi))
        self.k = k
        self.it = "int64" if abi == "ptr64" else "int32"

    def __getitem__(self, cfg):
        import numpy as np

        launch = self.k[cfg]

        def call(*args):
            *head, out_f, out_u = args
            _, h, w = out_u.shape
            pf = out_f.__cuda_array_interface__["data"][0]
            pu = out_u.__cuda_array_interface__["data"][0]
            it = getattr(np, self.it)
            return launch(*head, pf, pu, it(h), it(w), it(w))

        return call


if __name__ == "__main__":
    print(
        emit(sys.argv[1] if len(sys.argv) > 1 else "dd", sys.argv[2] if len(sys.argv) > 2 else "C")
    )

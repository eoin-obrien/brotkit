"""Spike S5 kernel generator (throwaway): S4/S2 kernels with a tile-shaped launch.

A chunk is a tile (row0, col0, th, tw) of the h x w image; a band is a full-width tile
(col0 = 0, tw = w). The kernel's grid covers the tile only. All integers are int64 and arrays
are raw pointers (ADR-0008):

    kern(c0r_, c0i_, dx, max_iter, row0, col0, th, tw, p_f, p_u, h, w, pitch)

Also `tile_sum`: sums the escape-count layer (out_u[0]) over a tile into a uint64 counter, so
the chunk planner can measure iterations per second for every finished chunk.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
S2 = HERE.parent / "S2_cache"


def _gen_s2():
    mod = sys.modules.get("gen_s2")
    if mod is None:
        sys.path.insert(0, str(S2))
        spec = importlib.util.spec_from_file_location("gen_s2", S2 / "gen_s2.py")
        assert spec
        assert spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_s2"] = mod
        spec.loader.exec_module(mod)
    return mod


g2 = _gen_s2()
s4 = g2.s4
TIERS, NLF, NLU = g2.TIERS, g2.NLF, g2.NLU

_STRUCT_HEAD = """def kern(c0r_, c0i_, dx, max_iter, row0, out_f, out_u):
    x, y = cuda.grid(2)
    h = out_u.shape[1]
    w = out_u.shape[2]
    yy = y + row0
    if x < w and yy < h:
"""
_TILE_HEAD = """def kern(c0r_, c0i_, dx, max_iter, row0, col0, th, tw, p_f, p_u, h, w, pitch):
    out_f = cuda.carray(p_f, (NLF, h, pitch), F32)
    out_u = cuda.carray(p_u, (NLU, h, pitch), U32)
    tx, y = cuda.grid(2)
    x = tx + col0
    yy = y + row0
    if tx < tw and y < th and x < w and yy < h:
"""
assert _STRUCT_HEAD in s4.KERNEL


def emit(tier: str, kernel: str) -> str:
    orig = s4.KERNEL
    try:
        s4.KERNEL = orig.replace(_STRUCT_HEAD, _TILE_HEAD)
        src = s4.emit(tier, kernel)
    finally:
        s4.KERNEL = orig
    return src.replace("TRAPY = F32(0.1)\n", f"TRAPY = F32(0.1)\nNLF = {NLF}\nNLU = {NLU}\n", 1)


def entry_name(tier: str, kernel: str) -> str:
    return f"k_{tier}_{kernel}"


def signature(tier: str) -> tuple[str, ...]:
    """brotkit ABI notation (KernelSpec.signature)."""
    word, nw, _ = TIERS[tier]
    return (
        (word,) * (2 * nw)
        + (word,)
        + ("int64",) * 5  # max_iter, row0, col0, th, tw
        + ("float32*", "uint32*")
        + ("int64",) * 3  # h, w, pitch
    )


TILE_SUM_SOURCE = """
import numpy as np
from numba_cuda_mlir import cuda

BLOCK = 256


def tile_sum(p_n, p_acc, row0, col0, th, tw, h, pitch):
    # acc[0] += sum of escape counts over the tile; acc[1] = max(acc[1], largest count)
    n = cuda.carray(p_n, (h, pitch), np.uint32)
    acc = cuda.carray(p_acc, (2,), np.uint64)
    sm = cuda.shared.array(BLOCK, np.uint64)
    mx = cuda.shared.array(BLOCK, np.uint64)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * BLOCK + tid
    stride = cuda.gridDim.x * BLOCK
    total = th * tw
    s = np.uint64(0)
    m = np.uint64(0)
    while i < total:
        r = i // tw
        v = np.uint64(n[row0 + r, col0 + (i - r * tw)])
        s += v
        if v > m:
            m = v
        i += stride
    sm[tid] = s
    mx[tid] = m
    cuda.syncthreads()
    step = BLOCK // 2
    while step > 0:
        if tid < step:
            sm[tid] += sm[tid + step]
            if mx[tid + step] > mx[tid]:
                mx[tid] = mx[tid + step]
        cuda.syncthreads()
        step //= 2
    if tid == 0:
        cuda.atomic.add(acc, 0, sm[0])
        cuda.atomic.max(acc, 1, mx[0])
"""
TILE_SUM_SIGNATURE = ("uint32*", "uint64*", "int64", "int64", "int64", "int64", "int64", "int64")


# A clock warm-up kernel: one warp per block spinning on a dependent FMA chain, so it occupies
# almost nothing and keeps the GPU busy (and its clock rising) while the host prepares a render.
WARM_SOURCE = """
import numpy as np
from numba_cuda_mlir import cuda


def warm(n, p_out):
    out = cuda.carray(p_out, (1,), np.float32)
    x = np.float32(cuda.threadIdx.x)
    for _ in range(n):
        x = cuda.fma(x, np.float32(0.9999999), np.float32(1e-7))
    if x == np.float32(-1.0):
        out[0] = x
"""
WARM_SIGNATURE = ("int64", "float32*")

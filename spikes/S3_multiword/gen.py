"""Spike S3 code generator (throwaway).

Multi-word primitives are written once below with plain `+`, `-`, `*`. `build()` emits a real
`.py` module per (precision, variant) into `gen/` and imports it:

- variant "rn":    an AST pass rewrites every float `+`/`-`/`*` inside the device functions into
                   libdevice round-to-nearest intrinsics (`dadd_rn`/`dmul_rn` for f64,
                   `fadd_rn`/`fsub_rn`/`fmul_rn` for f32; f64 subtraction is `dadd_rn(x, -y)`,
                   negation being exact). Explicit FMAs are `cuda.fma`. This is what the
                   multi-word printers would emit (ADR-0006).
- variant "naive": the same source left as bare operators, so libnvvm may contract mul+add.

All kernels compile with fastmath=False unless `fastmath=True` is passed (step 5c / probes).
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import textwrap

GEN_DIR = pathlib.Path(__file__).parent / "gen"

# Device functions. Every name listed in DEVICE gets @cuda.jit(device=True). Constants SPLIT,
# TWO, ZERO, FOUR are module globals of the right numpy dtype so f32 code stays f32.
DEVICE_SRC = """
def two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e

def fast_two_sum(a, b):
    s = a + b
    e = b - (s - a)
    return s, e

def two_prod(a, b):
    p = a * b
    e = cuda.fma(a, b, -p)
    return p, e

def split(a):
    t = SPLIT * a
    hi = t - (t - a)
    lo = a - hi
    return hi, lo

def two_prod_dekker(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e

def w_add(ah, al, bh, bl):
    s1, s2 = two_sum(ah, bh)
    t1, t2 = two_sum(al, bl)
    s2 = s2 + t1
    s1, s2 = fast_two_sum(s1, s2)
    s2 = s2 + t2
    return fast_two_sum(s1, s2)

def w_sub(ah, al, bh, bl):
    return w_add(ah, al, -bh, -bl)

def w_mul(ah, al, bh, bl):
    p1, p2 = two_prod(ah, bh)
    p2 = p2 + (ah * bl + al * bh)
    return fast_two_sum(p1, p2)

def w_sqr(ah, al):
    p1, p2 = two_prod(ah, ah)
    p2 = p2 + TWO * ah * al
    p2 = p2 + al * al
    return fast_two_sum(p1, p2)

def w_mul_dekker(ah, al, bh, bl):
    p1, p2 = two_prod_dekker(ah, bh)
    p2 = p2 + (ah * bl + al * bh)
    return fast_two_sum(p1, p2)

def c_mul(ar, arl, ai, ail, br, brl, bi, bil):
    xr, xrl = w_mul(ar, arl, br, brl)
    yr, yrl = w_mul(ai, ail, bi, bil)
    rr, rrl = w_sub(xr, xrl, yr, yrl)
    xi, xil = w_mul(ar, arl, bi, bil)
    yi, yil = w_mul(ai, ail, br, brl)
    ri, ril = w_add(xi, xil, yi, yil)
    return rr, rrl, ri, ril

def w_mul2(h, l):
    return TWO * h, TWO * l

def offs(k, dx):
    return k * dx

def c_sqr(ar, arl, ai, ail):
    xr, xrl = w_sqr(ar, arl)
    yr, yrl = w_sqr(ai, ail)
    rr, rrl = w_sub(xr, xrl, yr, yrl)
    m, ml = w_mul(ar, arl, ai, ail)
    return rr, rrl, TWO * m, TWO * ml
"""

DEVICE = [
    "two_sum",
    "fast_two_sum",
    "two_prod",
    "split",
    "two_prod_dekker",
    "w_add",
    "w_sub",
    "w_mul",
    "w_sqr",
    "w_mul_dekker",
    "c_mul",
    "w_mul2",
    "offs",
    "c_sqr",
]

# Kernels: plain index arithmetic only, never rewritten.
KERNEL_SRC = """
def k_eft2(fn_id, a, b, s, e):
    i = cuda.grid(1)
    if i < a.size:
        if fn_id == 0:
            x, y = two_sum(a[i], b[i])
        elif fn_id == 1:
            x, y = fast_two_sum(a[i], b[i])
        elif fn_id == 2:
            x, y = two_prod(a[i], b[i])
        else:
            x, y = two_prod_dekker(a[i], b[i])
        s[i] = x
        e[i] = y

def k_wop(fn_id, a, b, out):
    # a, b, out: (n, 4) = (re_hi, re_lo, im_hi, im_lo); real ops use columns 0..1
    i = cuda.grid(1)
    if i < a.shape[0]:
        if fn_id == 0:
            x, y = w_add(a[i, 0], a[i, 1], b[i, 0], b[i, 1])
            out[i, 0] = x
            out[i, 1] = y
        elif fn_id == 1:
            x, y = w_mul(a[i, 0], a[i, 1], b[i, 0], b[i, 1])
            out[i, 0] = x
            out[i, 1] = y
        elif fn_id == 2:
            x, y = w_sqr(a[i, 0], a[i, 1])
            out[i, 0] = x
            out[i, 1] = y
        elif fn_id == 3:
            r0, r1, r2, r3 = c_mul(a[i, 0], a[i, 1], a[i, 2], a[i, 3],
                                   b[i, 0], b[i, 1], b[i, 2], b[i, 3])
            out[i, 0] = r0
            out[i, 1] = r1
            out[i, 2] = r2
            out[i, 3] = r3
        elif fn_id == 4:
            r0, r1, r2, r3 = c_sqr(a[i, 0], a[i, 1], a[i, 2], a[i, 3])
            out[i, 0] = r0
            out[i, 1] = r1
            out[i, 2] = r2
            out[i, 3] = r3
        else:
            x, y = w_mul_dekker(a[i, 0], a[i, 1], b[i, 0], b[i, 1])
            out[i, 0] = x
            out[i, 1] = y

def k_mandel_w(cr, crl, ci, cil, max_iter, row0, counts):
    # Multi-word z^2 + c. c given per pixel as (hi, lo) arrays shaped like counts.
    x, y = cuda.grid(2)
    h, w = counts.shape
    yy = y + row0
    if x < w and yy < h:
        ar = cr[yy, x]
        arl = crl[yy, x]
        ai = ci[yy, x]
        ail = cil[yy, x]
        zr = ZERO
        zrl = ZERO
        zi = ZERO
        zil = ZERO
        n = 0
        while n < max_iter:
            xr, xrl = w_sqr(zr, zrl)
            yi, yil = w_sqr(zi, zil)
            if xr + yi > FOUR:
                break
            m, ml = w_mul(zr, zrl, zi, zil)
            tr, trl = w_sub(xr, xrl, yi, yil)
            zr, zrl = w_add(tr, trl, ar, arl)
            m, ml = w_mul2(m, ml)
            zi, zil = w_add(m, ml, ai, ail)
            n += 1
        counts[yy, x] = n

def k_mandel_w_view(c0r, c0rl, c0i, c0il, dx, max_iter, row0, counts):
    # Throughput variant: c = centre + ((x - w/2) dx, (h/2 - y) dx) computed on device.
    x, y = cuda.grid(2)
    h, w = counts.shape
    yy = y + row0
    if x < w and yy < h:
        ox = offs(FT(x - w // 2), dx)
        oy = offs(FT(h // 2 - yy), dx)
        ar, arl = w_add(c0r, c0rl, ox, ZERO)
        ai, ail = w_add(c0i, c0il, oy, ZERO)
        zr = ZERO
        zrl = ZERO
        zi = ZERO
        zil = ZERO
        n = 0
        while n < max_iter:
            xr, xrl = w_sqr(zr, zrl)
            yi, yil = w_sqr(zi, zil)
            if xr + yi > FOUR:
                break
            m, ml = w_mul(zr, zrl, zi, zil)
            tr, trl = w_sub(xr, xrl, yi, yil)
            zr, zrl = w_add(tr, trl, ar, arl)
            m, ml = w_mul2(m, ml)
            zi, zil = w_add(m, ml, ai, ail)
            n += 1
        counts[yy, x] = n

def k_mandel_plain(cr, ci, max_iter, row0, counts):
    x, y = cuda.grid(2)
    h, w = counts.shape
    yy = y + row0
    if x < w and yy < h:
        a = cr[yy, x]
        b = ci[yy, x]
        zr = ZERO
        zi = ZERO
        n = 0
        while n < max_iter:
            x2 = zr * zr
            y2 = zi * zi
            if x2 + y2 > FOUR:
                break
            zi = TWO * zr * zi + b
            zr = x2 - y2 + a
            n += 1
        counts[yy, x] = n

def k_mandel_plain_view(c0r, c0i, dx, max_iter, row0, counts):
    x, y = cuda.grid(2)
    h, w = counts.shape
    yy = y + row0
    if x < w and yy < h:
        a = c0r + FT(x - w // 2) * dx
        b = c0i + FT(h // 2 - yy) * dx
        zr = ZERO
        zi = ZERO
        n = 0
        while n < max_iter:
            x2 = zr * zr
            y2 = zi * zi
            if x2 + y2 > FOUR:
                break
            zi = TWO * zr * zi + b
            zr = x2 - y2 + a
            n += 1
        counts[yy, x] = n

def k_one_mul_dekker(a, b, out):
    i = cuda.grid(1)
    if i < a.shape[0]:
        x, y = w_mul_dekker(a[i, 0], a[i, 1], b[i, 0], b[i, 1])
        out[i, 0] = x
        out[i, 1] = y

def k_one_mul(a, b, out):
    i = cuda.grid(1)
    if i < a.shape[0]:
        x, y = w_mul(a[i, 0], a[i, 1], b[i, 0], b[i, 1])
        out[i, 0] = x
        out[i, 1] = y

def k_one_two_sum(a, b, out):
    i = cuda.grid(1)
    if i < a.shape[0]:
        x, y = two_sum(a[i, 0], b[i, 0])
        out[i, 0] = x
        out[i, 1] = y
"""

KERNELS = [
    "k_eft2",
    "k_wop",
    "k_mandel_w",
    "k_mandel_w_view",
    "k_mandel_plain",
    "k_mandel_plain_view",
    "k_one_mul_dekker",
    "k_one_mul",
    "k_one_two_sum",
]

_RN = {
    "f64": {
        "add": "ld.dadd_rn({0}, {1})",
        "sub": "ld.dadd_rn({0}, -({1}))",
        "mul": "ld.dmul_rn({0}, {1})",
    },
    "f32": {
        "add": "ld.fadd_rn({0}, {1})",
        "sub": "ld.fsub_rn({0}, {1})",
        "mul": "ld.fmul_rn({0}, {1})",
    },
}


class _ToRN(ast.NodeTransformer):
    def __init__(self, prec: str) -> None:
        self.t = _RN[prec]

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}.get(type(node.op))
        if op is None:
            raise ValueError(f"unexpected operator {ast.dump(node.op)}")
        src = self.t[op].format(ast.unparse(node.left), ast.unparse(node.right))
        return ast.parse(src, mode="eval").body


def emit(prec: str, variant: str, fastmath: bool = False) -> str:
    dev = ast.parse(textwrap.dedent(DEVICE_SRC))
    if variant == "rn":
        dev = ast.fix_missing_locations(_ToRN(prec).visit(dev))
    elif variant != "naive":
        raise ValueError(variant)
    dt = "float64" if prec == "f64" else "float32"
    split = "134217729.0" if prec == "f64" else "4097.0"  # 2^27+1 / 2^12+1
    head = f"""# GENERATED by spikes/S3_multiword/gen.py
# prec={prec} variant={variant} fastmath={fastmath}
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import libdevice as ld

SPLIT = np.{dt}({split})
TWO = np.{dt}(2.0)
FOUR = np.{dt}(4.0)
ZERO = np.{dt}(0.0)
FT = np.{dt}
"""
    body = []
    for fn in dev.body:
        assert isinstance(fn, ast.FunctionDef)
        body.append(f"@cuda.jit(device=True, fastmath={fastmath})\n{ast.unparse(fn)}\n")
    kern = ast.parse(textwrap.dedent(KERNEL_SRC))
    for fn in kern.body:
        assert isinstance(fn, ast.FunctionDef)
        body.append(f"@cuda.jit(fastmath={fastmath})\n{ast.unparse(fn)}\n")
    return head + "\n\n" + "\n\n".join(body)


def build(prec: str, variant: str, fastmath: bool = False):
    GEN_DIR.mkdir(exist_ok=True)
    name = f"eft_{prec}_{variant}{'_fm' if fastmath else ''}"
    path = GEN_DIR / f"{name}.py"
    path.write_text(emit(prec, variant, fastmath))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec
    assert spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    print(
        emit(
            sys.argv[1] if len(sys.argv) > 1 else "f64", sys.argv[2] if len(sys.argv) > 2 else "rn"
        )
    )

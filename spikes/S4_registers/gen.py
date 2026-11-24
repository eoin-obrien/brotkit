"""Spike S4 kernel generator (throwaway).

Emits one escape-time z^2 + c kernel per (tier, feature set) into `gen/` as a real `.py` file.

Tiers (word type x words):
  f32, f64   plain, bare operators (contraction on, as production f32/f64 would run; S3)
  f32rn      plain f32 with every add/sub/mul an `_rn` intrinsic (cost of reproducible f32)
  df64       2 x f32 words, `_rn` + explicit `cuda.fma` (ADR-0006)
  dd         2 x f64 words, same
  qf         4 x f32 words, same. MINIMAL prototype: QD-library sloppy add / sloppy mul / sqr
             with the branchy 5-term renormalisation (Hida, Li, Bailey), on f32 units.

Features (one colouring accumulator fused into the iteration loop each):
  smooth   final |z|^2 -> mu                      (kernel A = {smooth})
  de       running dz/dc in tier precision         (kernel B = A + de)
  trap     point-trap min distance + iteration     (kernel C = B + trap)
  stripe   stripe average sum + previous sum       (6-feature = C + stripe + tia + binary)
  tia      triangle-inequality average + previous
  binary   sign(Im z_k) bit history (uint32), the binary-decomposition / external-angle bits

Accumulators are float32 computed from the hi word (spec layer table). Escape radius follows
the spec's Smooth.min_escape_radius: R = 1e3 for f32 tiers' word type f32 plain, 1e6 otherwise.

Template conventions
- Names ending in `_` are multi-word values and expand to their words: `zr_` -> `zr_0, zr_1`.
  `zr_0` is the hi word. W-returning calls are never nested.
- Scalar float ops in kernel bodies are written as helper calls: `fa fs fm` (float32
  accumulator precision) and `ha hs hm` (hi-word precision). For plain tiers an AST pass turns
  them into bare operators; for rn tiers into libdevice `_rn` intrinsics. Integer index
  arithmetic is written with bare operators on an explicit allowlist of int names.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys
import textwrap

GEN_DIR = pathlib.Path(__file__).parent / "gen"

TIERS = {
    #        word dtype, words, rn rewrite
    "f32": ("float32", 1, False),
    "f32rn": ("float32", 1, True),
    "f64": ("float64", 1, False),
    "df64": ("float32", 2, True),
    "dd": ("float64", 2, True),
    "qf": ("float32", 4, True),  # flat (branch-free) renorm
    "qfb": ("float32", 4, True),  # QD's branchy zero-skipping renorm, for comparison
}
MULTIWORD = ("df64", "dd", "qf", "qfb")

FEATURE_ORDER = ("smooth", "de", "trap", "stripe", "tia", "binary")
KERNELS = {
    "Z": (),  # escape loop only, no finaliser: the strict PTX test target (no libdevice maths)
    "A": ("smooth",),
    "B": ("smooth", "de"),
    "C": ("smooth", "de", "trap"),  # = the 3-feature kernel
    "F6": FEATURE_ORDER,
    "S2": ("stripe", "tia", "binary"),  # second pass of the split variant (C + S2 vs F6)
}
FEATURE_COUNT = {"Z": 0, "A": 1, "B": 2, "C": 3, "F6": 6, "S2": 3}

# Integer names the kernel template uses with bare operators (passed to the checker).
INT_NAMES = frozenset({"x", "y", "yy", "w", "h", "n", "row0", "max_iter", "trap_n", "bits"})

# Output layer slots.
LF = {"mu": 0, "de": 1, "trap_d": 2, "stripe": 3, "stripe_prev": 4, "tia": 5, "tia_prev": 6}
LU = {"n": 0, "trap_n": 1, "bits": 2}
NLF, NLU = len(LF), len(LU)

# --------------------------------------------------------------------------------------------
# Tier libraries: w_add, w_sub, w_mul, w_sqr, w_mul2, w_from over flat word arguments.
# Written with bare operators; rn tiers get them rewritten to `_rn` intrinsics.

LIB_1 = """
def w_add(a, b):
    return a + b

def w_sub(a, b):
    return a - b

def w_mul(a, b):
    return a * b

def w_sqr(a):
    return a * a

def w_mul2(a):
    return TWO * a

def w_from(a):
    return a
"""

EFT = """
def two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e

def quick_two_sum(a, b):
    s = a + b
    e = b - (s - a)
    return s, e

def two_prod(a, b):
    p = a * b
    e = cuda.fma(a, b, -p)
    return p, e
"""

LIB_2 = (
    EFT
    + """
def w_add(ah, al, bh, bl):
    s1, s2 = two_sum(ah, bh)
    t1, t2 = two_sum(al, bl)
    s2 = s2 + t1
    s1, s2 = quick_two_sum(s1, s2)
    s2 = s2 + t2
    return quick_two_sum(s1, s2)

def w_sub(ah, al, bh, bl):
    return w_add(ah, al, -bh, -bl)

def w_mul(ah, al, bh, bl):
    p1, p2 = two_prod(ah, bh)
    p2 = p2 + (ah * bl + al * bh)
    return quick_two_sum(p1, p2)

def w_sqr(ah, al):
    p1, p2 = two_prod(ah, ah)
    p2 = p2 + TWO * ah * al
    p2 = p2 + al * al
    return quick_two_sum(p1, p2)

def w_mul2(h, l):
    return TWO * h, TWO * l

def w_from(a):
    return a, ZERO
"""
)

# QD library (Hida, Li, Bailey) quad-double algorithms, on f32 words here.
LIB_4 = (
    EFT
    + """
def three_sum(a, b, c):
    t1, t2 = two_sum(a, b)
    a, t3 = two_sum(c, t1)
    b, c = two_sum(t2, t3)
    return a, b, c

def three_sum2(a, b, c):
    t1, t2 = two_sum(a, b)
    a, t3 = two_sum(c, t1)
    b = t2 + t3
    return a, b

def renorm_branchy(c0, c1, c2, c3, c4):
    s0, c4 = quick_two_sum(c3, c4)
    s0, c3 = quick_two_sum(c2, s0)
    s0, c2 = quick_two_sum(c1, s0)
    c0, c1 = quick_two_sum(c0, s0)
    s0 = c0
    s1 = c1
    s2 = ZERO
    s3 = ZERO
    if s1 != ZERO:
        s1, s2 = quick_two_sum(s1, c2)
        if s2 != ZERO:
            s2, s3 = quick_two_sum(s2, c3)
            if s3 != ZERO:
                s3 = s3 + c4
            else:
                s2, s3 = quick_two_sum(s2, c4)
        else:
            s1, s2 = quick_two_sum(s1, c3)
            if s2 != ZERO:
                s2, s3 = quick_two_sum(s2, c4)
            else:
                s1, s2 = quick_two_sum(s1, c4)
    else:
        s0, s1 = quick_two_sum(s0, c2)
        if s1 != ZERO:
            s1, s2 = quick_two_sum(s1, c3)
            if s2 != ZERO:
                s2, s3 = quick_two_sum(s2, c4)
            else:
                s1, s2 = quick_two_sum(s1, c4)
        else:
            s0, s1 = quick_two_sum(s0, c3)
            if s1 != ZERO:
                s1, s2 = quick_two_sum(s1, c4)
            else:
                s0, s1 = quick_two_sum(s0, c4)
    return s0, s1, s2, s3

RENORM_FLAT

def w_add(a0, a1, a2, a3, b0, b1, b2, b3):
    s0, t0 = two_sum(a0, b0)
    s1, t1 = two_sum(a1, b1)
    s2, t2 = two_sum(a2, b2)
    s3, t3 = two_sum(a3, b3)
    s1, t0 = two_sum(s1, t0)
    s2, t0, t1 = three_sum(s2, t0, t1)
    s3, t0 = three_sum2(s3, t0, t2)
    t0 = t0 + t1 + t3
    return renorm(s0, s1, s2, s3, t0)

def w_sub(a0, a1, a2, a3, b0, b1, b2, b3):
    return w_add(a0, a1, a2, a3, -b0, -b1, -b2, -b3)

def w_mul(a0, a1, a2, a3, b0, b1, b2, b3):
    p0, q0 = two_prod(a0, b0)
    p1, q1 = two_prod(a0, b1)
    p2, q2 = two_prod(a1, b0)
    p3, q3 = two_prod(a0, b2)
    p4, q4 = two_prod(a1, b1)
    p5, q5 = two_prod(a2, b0)
    p1, p2, q0 = three_sum(p1, p2, q0)
    p2, q1, q2 = three_sum(p2, q1, q2)
    p3, p4, p5 = three_sum(p3, p4, p5)
    s0, t0 = two_sum(p2, p3)
    s1, t1 = two_sum(q1, p4)
    s2 = q2 + p5
    s1, t0 = two_sum(s1, t0)
    s2 = s2 + (t0 + t1)
    s1 = s1 + (a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + q0 + q3 + q4 + q5)
    return renorm(p0, p1, s0, s1, s2)

def w_sqr(a0, a1, a2, a3):
    p0, q0 = two_prod(a0, a0)
    p1, q1 = two_prod(TWO * a0, a1)
    p2, q2 = two_prod(TWO * a0, a2)
    p3, q3 = two_prod(a1, a1)
    p1, q0 = two_sum(q0, p1)
    q0, q1 = two_sum(q0, q1)
    p2, p3 = two_sum(p2, p3)
    s0, t0 = two_sum(q0, p2)
    s1, t1 = two_sum(q1, p3)
    s1, t0 = two_sum(s1, t0)
    t0 = t0 + t1
    s1, t0 = quick_two_sum(s1, t0)
    p2, t1 = quick_two_sum(s0, s1)
    p3, q0 = quick_two_sum(t1, t0)
    p4 = TWO * a0 * a3
    p5 = TWO * a1 * a2
    p4, p5 = two_sum(p4, p5)
    q2, q3 = two_sum(q2, q3)
    t0, t1 = two_sum(p4, q2)
    t1 = t1 + p5 + q3
    p3, p4 = two_sum(p3, t0)
    p4 = p4 + q0 + t1
    return renorm(p0, p1, p2, p3, p4)

def w_mul2(a0, a1, a2, a3):
    return TWO * a0, TWO * a1, TWO * a2, TWO * a3

def w_from(a):
    return a, ZERO, ZERO, ZERO
"""
)

# QD's renorm when no intermediate word is zero. A zero word is moved up exactly by
# quick_two_sum(0, b) = (b, 0), so no zero-skipping branches are needed.
RENORM_FLAT = """
def renorm(c0, c1, c2, c3, c4):
    s0, c4 = quick_two_sum(c3, c4)
    s0, c3 = quick_two_sum(c2, s0)
    s0, c2 = quick_two_sum(c1, s0)
    c0, c1 = quick_two_sum(c0, s0)
    s1, s2 = quick_two_sum(c1, c2)
    s2, s3 = quick_two_sum(s2, c3)
    s3 = s3 + c4
    return c0, s1, s2, s3
"""

RENORM_BRANCHY_ALIAS = """
def renorm(c0, c1, c2, c3, c4):
    return renorm_branchy(c0, c1, c2, c3, c4)
"""

# Complex helpers in the multi-word DSL (names ending `_` expand to words).
CPLX = """
def c_mul(ar_, ai_, br_, bi_):
    xr_ = w_mul(ar_, br_)
    yr_ = w_mul(ai_, bi_)
    rr_ = w_sub(xr_, yr_)
    xi_ = w_mul(ar_, bi_)
    yi_ = w_mul(ai_, br_)
    ri_ = w_add(xi_, yi_)
    return rr_, ri_
"""

# --------------------------------------------------------------------------------------------
# Kernel template. Lines tagged `#@feat` are kept only when that feature is enabled.

KERNEL = """
def kern(c0r_, c0i_, dx, max_iter, row0, out_f, out_u):
    x, y = cuda.grid(2)
    h = out_u.shape[1]
    w = out_u.shape[2]
    yy = y + row0
    if x < w and yy < h:
        ox = FT(x - w // 2)
        oy = FT(h // 2 - yy)
        pr_ = w_from(hm(ox, dx))
        pi_ = w_from(hm(oy, dx))
        cr_ = w_add(c0r_, pr_)
        ci_ = w_add(c0i_, pi_)
        zr_ = w_from(ZERO)
        zi_ = w_from(ZERO)
        dr_ = w_from(ZERO)  #@de
        di_ = w_from(ZERO)  #@de
        one_ = w_from(ONE)  #@de
        trap_d = BIG32  #@trap
        trap_n = 0  #@trap
        st = Z32  #@stripe
        st_prev = Z32  #@stripe
        ti = Z32  #@tia
        ti_prev = Z32  #@tia
        r2_prev = Z32  #@tia
        ccr = F32(cr_0)  #@tia
        cci = F32(ci_0)  #@tia
        cabs = math.sqrt(fa(fm(ccr, ccr), fm(cci, cci)))  #@tia
        bits = U32(0)  #@binary
        r2 = ZERO
        n = 0
        while n < max_iter:
            x2_ = w_sqr(zr_)
            y2_ = w_sqr(zi_)
            r2 = ha(x2_0, y2_0)
            if r2 > BAIL:
                break
            fzr = F32(zr_0)  #@trap #@stripe
            fzi = F32(zi_0)  #@trap #@stripe
            tdx = fs(fzr, TRAPX)  #@trap
            tdy = fs(fzi, TRAPY)  #@trap
            td = fa(fm(tdx, tdx), fm(tdy, tdy))  #@trap
            if td < trap_d:  #@trap
                trap_d = td  #@trap
                trap_n = n  #@trap
            if n > 0:  #@stripe #@tia
                st_prev = st  #@stripe
                ang = math.atan2(fzi, fzr)  #@stripe
                st = fa(st, fa(fm(HALF32, math.sin(fm(DENS32, ang))), HALF32))  #@stripe
                fr2 = F32(r2)  #@tia
                tlo = abs(fs(r2_prev, cabs))  #@tia
                thi = fa(r2_prev, cabs)  #@tia
                ti_prev = ti  #@tia
                ti = fa(ti, fs(math.sqrt(fr2), tlo) / fs(thi, tlo))  #@tia
            r2_prev = F32(r2)  #@tia
            bits = U32(bits << U32(1)) | (U32(1) if zi_0 < ZERO else U32(0))  #@binary
            er_, ei_ = c_mul(zr_, zi_, dr_, di_)  #@de
            t2_ = w_mul2(er_)  #@de
            dr_ = w_add(t2_, one_)  #@de
            di_ = w_mul2(ei_)  #@de
            m_ = w_mul(zr_, zi_)
            m2_ = w_mul2(m_)
            q_ = w_sub(x2_, y2_)
            zr_ = w_add(q_, cr_)
            zi_ = w_add(m2_, ci_)
            n += 1
        out_u[0, yy, x] = n
        if n < max_iter:  #@smooth #@de
            fr = F32(r2)  #@smooth #@de
            lz = fm(HALF32, math.log(fr))  #@smooth #@de
            out_f[0, yy, x] = fs(fa(F32(n), ONE32), math.log2(lz))  #@smooth
            dz2 = F32(ha(hm(dr_0, dr_0), hm(di_0, di_0)))  #@de
            out_f[1, yy, x] = fm(math.sqrt(fr), lz) / math.sqrt(dz2)  #@de
        out_f[2, yy, x] = trap_d  #@trap
        out_u[1, yy, x] = trap_n  #@trap
        out_f[3, yy, x] = st  #@stripe
        out_f[4, yy, x] = st_prev  #@stripe
        out_f[5, yy, x] = ti  #@tia
        out_f[6, yy, x] = ti_prev  #@tia
        out_u[2, yy, x] = bits  #@binary
"""

# --------------------------------------------------------------------------------------------

_RN = {
    "float64": {
        "add": "ld.dadd_rn({0}, {1})",
        "sub": "ld.dadd_rn({0}, -({1}))",
        "mul": "ld.dmul_rn({0}, {1})",
    },
    "float32": {
        "add": "ld.fadd_rn({0}, {1})",
        "sub": "ld.fsub_rn({0}, {1})",
        "mul": "ld.fmul_rn({0}, {1})",
    },
}
_BARE = {"add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})"}
_OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}


class _LibToRN(ast.NodeTransformer):
    """Every float BinOp in a tier library -> `_rn` intrinsic (S3's rewrite)."""

    def __init__(self, dtype: str) -> None:
        self.t = _RN[dtype]

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unexpected operator {ast.dump(node.op)}")
        src = self.t[op].format(ast.unparse(node.left), ast.unparse(node.right))
        return ast.parse(src, mode="eval").body


class _Helpers(ast.NodeTransformer):
    """fa/fs/fm (float32) and ha/hs/hm (hi-word dtype) -> bare ops or `_rn` intrinsics."""

    def __init__(self, word: str, rn: bool) -> None:
        self.map = {
            "f": _RN["float32"] if rn else _BARE,
            "h": _RN[word] if rn else _BARE,
        }

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and re.fullmatch(r"[fh][asm]", node.func.id):
            kind, op = node.func.id
            tmpl = self.map[kind][{"a": "add", "s": "sub", "m": "mul"}[op]]
            src = tmpl.format(*(ast.unparse(a) for a in node.args))
            return ast.parse(src, mode="eval").body
        return node


_WVAR = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)_(?!\w)")


def _expand(src: str, nw: int) -> str:
    return _WVAR.sub(lambda m: ", ".join(f"{m.group(1)}_{i}" for i in range(nw)), src)


def _select(src: str, features: tuple[str, ...]) -> str:
    out = []
    for line in src.splitlines():
        tags = re.findall(r"#@(\w+)", line)
        if tags and not any(t in features for t in tags):
            continue
        out.append(re.sub(r"\s*#@.*$", "", line))
    return "\n".join(out)


def emit(tier: str, kernel: str, *, fastmath: bool = False) -> str:
    word, nw, rn = TIERS[tier]
    features = KERNELS[kernel]
    lib_src = {1: LIB_1, 2: LIB_2, 4: LIB_4}[nw]
    if nw == 4:
        lib_src = lib_src.replace(
            "RENORM_FLAT", RENORM_BRANCHY_ALIAS if tier == "qfb" else RENORM_FLAT
        )
    lib = ast.parse(textwrap.dedent(lib_src))
    if rn:
        lib = ast.fix_missing_locations(_LibToRN(word).visit(lib))
    cplx = ast.parse(_expand(textwrap.dedent(CPLX), nw))
    ksrc = _expand(_select(textwrap.dedent(KERNEL), features), nw)
    kern = ast.fix_missing_locations(_Helpers(word, rn).visit(ast.parse(ksrc)))
    plain_f32 = word == "float32" and nw == 1
    bail = "1e6" if plain_f32 else "1e12"  # R^2; R = 1e3 (f32) / 1e6 (spec Smooth)
    head = f"""# GENERATED by spikes/S4_registers/gen.py -- tier={tier} kernel={kernel}
# features={",".join(features)} words={nw} word={word} rn={rn} fastmath={fastmath}
import math

import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import libdevice as ld

FT = np.{word}
F32 = np.float32
U32 = np.uint32
TWO = FT(2.0)
ONE = FT(1.0)
ZERO = FT(0.0)
BAIL = FT({bail})
Z32 = F32(0.0)
ONE32 = F32(1.0)
HALF32 = F32(0.5)
BIG32 = F32(3.0e38)
DENS32 = F32(5.0)
TRAPX = F32(0.25)
TRAPY = F32(0.1)
"""
    parts = []
    for fn in [*lib.body, *cplx.body]:
        assert isinstance(fn, ast.FunctionDef)
        parts.append(f"@cuda.jit(device=True, fastmath={fastmath})\n{ast.unparse(fn)}\n")
    (kfn,) = kern.body
    assert isinstance(kfn, ast.FunctionDef)
    kfn.name = f"k_{tier}_{kernel}"
    parts.append(f"@cuda.jit(fastmath={fastmath})\n{ast.unparse(kfn)}\n")
    return head + "\n\n" + "\n\n".join(parts)


def kernel_body(src: str) -> str:
    """Source of the `@cuda.jit` kernel(s) only (no device library) -- for the checker."""
    tree = ast.parse(src)
    return "\n\n".join(
        ast.unparse(f)
        for f in tree.body
        if isinstance(f, ast.FunctionDef)
        and not any("device=True" in ast.unparse(d) for d in f.decorator_list)
    )


def module_name(tier: str, kernel: str, tag: str = "") -> str:
    return f"s4_{tier}_{kernel}{('_' + tag) if tag else ''}"


def write(tier: str, kernel: str, *, fastmath: bool = False) -> pathlib.Path:
    GEN_DIR.mkdir(exist_ok=True)
    path = GEN_DIR / f"{module_name(tier, kernel)}.py"
    src = emit(tier, kernel, fastmath=fastmath)
    if not path.exists() or path.read_text() != src:
        path.write_text(src)
    return path


def load(tier: str, kernel: str, *, jit_options: dict | None = None):
    """Import the generated module and return the kernel dispatcher.

    `jit_options` (e.g. {"max_registers": 64}) re-wraps the kernel's py_func in a fresh
    `cuda.jit` with those options, so one source gives several builds.
    """
    path = write(tier, kernel)
    name = path.stem
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec
        assert spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    k = getattr(mod, f"k_{tier}_{kernel}")
    if jit_options:
        from numba_cuda_mlir import cuda

        k = cuda.jit(**jit_options)(k.py_func)
    return k


def signature(tier: str):
    from numba_cuda_mlir import types

    word, nw, _ = TIERS[tier]
    ft = getattr(types, word)
    return (
        (ft,) * (2 * nw)
        + (ft, types.int32, types.int32)
        + (types.float32[:, :, ::1], types.uint32[:, :, ::1])
    )


if __name__ == "__main__":
    print(
        emit(sys.argv[1] if len(sys.argv) > 1 else "dd", sys.argv[2] if len(sys.argv) > 2 else "C")
    )

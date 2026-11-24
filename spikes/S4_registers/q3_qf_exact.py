"""qf prototype checks, reusing S3's methods (docs/design/spikes/S3.md steps 2-4).

1. EFT exactness: the qf module's f32 two_sum / quick_two_sum / two_prod over S3's input
   groups (eftcheck.inputs), checked in exact rational arithmetic (eftcheck.verify).
2. Composite accuracy vs mpmath (60 digits): qf add, add under cancellation, mul, sqr and
   complex mul, in units of u = 2^-96 (4 x 24 bits, S3's convention of p x words). Add is
   reported both relative to |a + b| and normwise (relative to |a| + |b|).
3. Deep check: qf vs dd escape counts, 64x64 at pixel spacing 1e-20 around S3's M(4,1),
   max_iter 5000 (dd matched mpmath on 4096/4096 there in S3).
Both renormalisations: qf (flat) and qfb (QD's branchy one).

usage: uv run q3_qf_exact.py [n_eft=1000000] [n_comp=200000]
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "S3_multiword"))  # eftcheck (S3's exactness harness)

U = 2.0**-96

TEST_KERNELS = """

@cuda.jit
def k_eft2(fn_id, a, b, s, e):
    i = cuda.grid(1)
    if i < a.size:
        if fn_id == 0:
            x, y = two_sum(a[i], b[i])
        elif fn_id == 1:
            x, y = quick_two_sum(a[i], b[i])
        else:
            x, y = two_prod(a[i], b[i])
        s[i] = x
        e[i] = y


@cuda.jit
def k_qop(fn_id, a, b, out):
    i = cuda.grid(1)
    if i < a.shape[0]:
        if fn_id == 0:
            r0, r1, r2, r3 = w_add(
                a[i, 0], a[i, 1], a[i, 2], a[i, 3], b[i, 0], b[i, 1], b[i, 2], b[i, 3]
            )
        elif fn_id == 1:
            r0, r1, r2, r3 = w_mul(
                a[i, 0], a[i, 1], a[i, 2], a[i, 3], b[i, 0], b[i, 1], b[i, 2], b[i, 3]
            )
        else:
            r0, r1, r2, r3 = w_sqr(a[i, 0], a[i, 1], a[i, 2], a[i, 3])
        out[i, 0] = r0
        out[i, 1] = r1
        out[i, 2] = r2
        out[i, 3] = r3


@cuda.jit
def k_qcmul(a, b, out):
    i = cuda.grid(1)
    if i < a.shape[0]:
        r0, r1, r2, r3, s0, s1, s2, s3 = c_mul(
            a[i, 0], a[i, 1], a[i, 2], a[i, 3], a[i, 4], a[i, 5], a[i, 6], a[i, 7],
            b[i, 0], b[i, 1], b[i, 2], b[i, 3], b[i, 4], b[i, 5], b[i, 6], b[i, 7],
        )
        out[i, 0] = r0
        out[i, 1] = r1
        out[i, 2] = r2
        out[i, 3] = r3
        out[i, 4] = s0
        out[i, 5] = s1
        out[i, 6] = s2
        out[i, 7] = s3
"""


def build(tier: str):
    import gen

    src = gen.emit(tier, "A") + TEST_KERNELS
    path = gen.GEN_DIR / f"s4_{tier}_tests.py"
    gen.GEN_DIR.mkdir(exist_ok=True)
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec
    assert spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def split4(x) -> list[float]:
    """mpf -> 4 f32 words by successive round-to-nearest."""
    import mpmath as mp

    words = []
    r = x
    for _ in range(4):
        w = float(np.float32(float(r)))
        words.append(w)
        r = r - mp.mpf(w)
    return words


def make_qf(n: int, seed: int, like: np.ndarray | None = None) -> np.ndarray:
    import mpmath as mp

    mp.mp.prec = 200
    rng = np.random.default_rng(seed)
    out = np.empty((n, 8), np.float32)
    for i in range(n):
        for c in (0, 4):
            if like is not None and c == 0 and i % 2 == 0:
                # cancellation partner: -a * (1 + d), d ~ 2^-k with k up to 90
                a = sum(mp.mpf(float(v)) for v in like[i, 0:4])
                d = mp.mpf(2) ** -int(rng.integers(1, 90)) * (1 + mp.mpf(rng.random()))
                x = -a * (1 + d)
            else:
                e = int(rng.integers(-4, 2))
                frac = (
                    mp.mpf(int(rng.integers(0, 2**62))) / 2**62
                    + mp.mpf(int(rng.integers(0, 2**62))) / mp.mpf(2) ** 124
                )
                x = (1 if rng.random() < 0.5 else -1) * mp.ldexp(1 + frac, e)
            out[i, c : c + 4] = split4(x)
    return out


def _ref_chunk(args):
    import mpmath as mp

    op, a, b, r = args
    mp.mp.dps = 60

    def val(row, o=0):
        return sum((mp.mpf(float(v)) for v in row[o : o + 4]), mp.mpf(0))

    rel, norm = [], []
    for x, y, z in zip(a, b, r, strict=True):
        if op == "cmul":
            ref = mp.mpc(val(x), val(x, 4)) * mp.mpc(val(y), val(y, 4))
            got = mp.mpc(val(z), val(z, 4))
            rel.append(float(abs(got - ref) / abs(ref)))
            continue
        ax, by = val(x), val(y)
        ref = {"add": ax + by, "add_cancel": ax + by, "mul": ax * by, "sqr": ax * ax}[op]
        got = val(z)
        err = abs(got - ref)
        rel.append(float(err / abs(ref)) if ref != 0 else float(err))
        norm.append(float(err / (abs(ax) + abs(by))))
    return rel, norm


def main():
    import eftcheck as ec
    import guard
    from numba_cuda_mlir import cuda

    guard.check("q3")
    n_eft = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    n_comp = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    res: dict = {
        "u": "2^-96",
        "n_eft_base": n_eft,
        "n_comp": n_comp,
        "eft": [],
        "composite": [],
        "deep": {},
    }
    with ProcessPoolExecutor(24) as pool:
        # 1. EFT exactness (shared by qf and qfb: same primitives)
        mod = build("qf")
        for op in ("two_sum", "fast_two_sum", "two_prod"):
            groups = ec.inputs("f32", op, n_eft, seed=zlib.crc32(f"s4{op}".encode()))
            for gname, (a, b) in groups.items():
                keep = ec.finite_inputs(a, b, op)
                a, b = a[keep], b[keep]
                s, e = ec.run_gpu(mod, op, a, b)
                bad = ec.verify(pool, op, a, b, s, e)
                if op == "two_prod":
                    dom = ec.in_domain_two_prod(a, b, "f32")
                    row = dict(
                        op=op,
                        group=gname,
                        n=int(a.size),
                        in_domain=int(dom.sum()),
                        failures_in_domain=int((bad & dom).sum()),
                        inexact_out_of_domain=int((bad & ~dom).sum()),
                    )
                else:
                    row = dict(op=op, group=gname, n=int(a.size), failures=int(bad.sum()))
                print(json.dumps(row), flush=True)
                res["eft"].append(row)

        # 2. composite accuracy
        a = make_qf(n_comp, 41)
        b = make_qf(n_comp, 42)
        bc = make_qf(n_comp, 43, like=a)
        for tier in ("qf", "qfb"):
            mod = build(tier)
            da = cuda.to_device(a)
            for op, fn, bb in (
                ("add", 0, b),
                ("add_cancel", 0, bc),
                ("mul", 1, b),
                ("sqr", 2, b),
                ("cmul", None, b),
            ):
                db = cuda.to_device(bb)
                dout = cuda.device_array_like(a)
                if fn is None:
                    mod.k_qcmul[(n_comp + 255) // 256, 256](da, db, dout)
                else:
                    mod.k_qop[(n_comp + 255) // 256, 256](fn, da, db, dout)
                cuda.synchronize()
                out = dout.copy_to_host()
                idx = np.array_split(np.arange(n_comp), 96)
                parts = list(pool.map(_ref_chunk, [(op, a[i], bb[i], out[i]) for i in idx]))
                rel = np.concatenate([np.asarray(p[0]) for p in parts]) / U
                row = dict(
                    tier=tier,
                    op=op,
                    n=n_comp,
                    max_u=float(rel.max()),
                    median_u=float(np.median(rel)),
                    p99_u=float(np.percentile(rel, 99)),
                    over_16u=int((rel > 16).sum()),
                )
                if op.startswith("add"):
                    norm = np.concatenate([np.asarray(p[1]) for p in parts]) / U
                    row |= dict(
                        normwise_max_u=float(norm.max()),
                        normwise_p99_u=float(np.percentile(norm, 99)),
                    )
                print(json.dumps(row), flush=True)
                res["composite"].append(row)

    # 3. deep: qf vs dd counts at spacing 1e-20 around M(4,1)
    import gen
    import mpmath as mp

    mp.mp.dps = 60
    cr = mp.mpf("-0.101096363845622161025785445738622565463805442826253483876931")
    ci = mp.mpf("0.956286510809141500771096057729977435809833336510529170034314")
    n_px, spacing, max_iter = 64, 1e-20, 5000
    counts = {}
    for tier in ("dd", "qf", "qfb"):
        word, nw, _ = gen.TIERS[tier]
        dt = np.dtype(word).type

        def words(x, dt=dt, nw=nw):
            ws, r = [], x
            for _ in range(nw):
                w = dt(float(r))
                ws.append(w)
                r = r - mp.mpf(float(w))
            return ws

        k = gen.load(tier, "A")
        out_f = cuda.device_array((gen.NLF, n_px, n_px), np.float32)
        out_u = cuda.device_array((gen.NLU, n_px, n_px), np.uint32)
        k[(n_px // 16, n_px // 16), (16, 16)](
            *words(cr), *words(ci), dt(spacing), np.int32(max_iter), np.int32(0), out_f, out_u
        )
        cuda.synchronize()
        counts[tier] = out_u.copy_to_host()[0]
    for tier in ("qf", "qfb"):
        same = int((counts[tier] == counts["dd"]).sum())
        res["deep"][f"{tier}_vs_dd"] = dict(
            identical=same,
            of=n_px * n_px,
            count_range=[int(counts[tier].min()), int(counts[tier].max())],
            distinct=int(np.unique(counts[tier]).size),
        )
    res["deep"]["dd_count_range"] = [int(counts["dd"].min()), int(counts["dd"].max())]
    print(json.dumps(res["deep"]), flush=True)
    (HERE / "results" / "q3_qf_exact.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

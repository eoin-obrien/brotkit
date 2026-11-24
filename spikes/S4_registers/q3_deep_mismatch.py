"""Resolve q3 deep-check mismatches (qf vs dd) against mpmath at 80 digits, same c and bailout."""

import json
import pathlib

import gen
import mpmath as mp
import numpy as np
from numba_cuda_mlir import cuda

mp.mp.dps = 80
cr = mp.mpf("-0.101096363845622161025785445738622565463805442826253483876931")
ci = mp.mpf("0.956286510809141500771096057729977435809833336510529170034314")
N, SP, MAXIT = 64, 1e-20, 5000


def run(tier):
    word, nw, _ = gen.TIERS[tier]
    dt = np.dtype(word).type

    def words(x):
        ws, r = [], x
        for _ in range(nw):
            w = dt(float(r))
            ws.append(w)
            r = r - mp.mpf(float(w))
        return ws

    k = gen.load(tier, "A")
    of = cuda.device_array((gen.NLF, N, N), np.float32)
    ou = cuda.device_array((gen.NLU, N, N), np.uint32)
    k[(N // 16, N // 16), (16, 16)](
        *words(cr), *words(ci), dt(SP), np.int32(MAXIT), np.int32(0), of, ou
    )
    cuda.synchronize()
    return ou.copy_to_host()[0], of.copy_to_host()[1]


dd, de = run("dd")
qf, _ = run("qf")
rows = []
for yy, x in zip(*np.nonzero(dd != qf), strict=True):
    c = mp.mpc(cr + (x - N // 2) * mp.mpf(SP), ci + (N // 2 - yy) * mp.mpf(SP))
    z, n = mp.mpc(0), 0
    while n < MAXIT:
        if z.real**2 + z.imag**2 > 1e12:
            break
        z = z * z + c
        n += 1
    rows.append(
        dict(
            x=int(x),
            y=int(yy),
            dd=int(dd[yy, x]),
            qf=int(qf[yy, x]),
            mpmath=n,
            de_px=float(de[yy, x] / SP),
        )
    )
print(json.dumps(rows, indent=1))
pathlib.Path("results/q3_deep_mismatch.json").write_text(json.dumps(rows, indent=1))


# Re-run the reference from the c each tier actually forms: c0 words + fl(FT(ox) * FT(dx)).
def tier_c(tier, x, yy):
    word, nw, _ = gen.TIERS[tier]
    dt = np.dtype(word).type

    def wsum(v):
        ws, r = [], v
        for _ in range(nw):
            w = dt(float(r))
            ws.append(w)
            r = r - mp.mpf(float(w))
        return sum(mp.mpf(float(w)) for w in ws)

    def off(k):
        return mp.mpf(float(dt(dt(k) * dt(SP))))

    return mp.mpc(wsum(cr) + off(x - N // 2), wsum(ci) + off(N // 2 - yy))


def ref(c):
    z, n = mp.mpc(0), 0
    while n < MAXIT:
        if z.real**2 + z.imag**2 > 1e12:
            break
        z = z * z + c
        n += 1
    return n


for r in rows:
    r["mpmath_at_dd_c"] = ref(tier_c("dd", r["x"], r["y"]))
    r["mpmath_at_qf_c"] = ref(tier_c("qf", r["x"], r["y"]))
    r.pop("de_px")
print(json.dumps(rows, indent=1))
pathlib.Path("results/q3_deep_mismatch.json").write_text(json.dumps(rows, indent=1))

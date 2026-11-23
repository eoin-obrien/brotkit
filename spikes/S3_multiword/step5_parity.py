"""Step 5a/5b, numeric side: does contraction change results?

1. Bit-for-bit parity of GPU multi-word ops against a CPU emulation that runs the *same*
   template source (gen.DEVICE_SRC) in pure Python, where nothing is contracted:
   dd uses Python floats + math.fma; df64 uses numpy float32 scalars + a correctly rounded
   f32 fma. GPU "rn" should match the CPU bit for bit; GPU "naive" may not.
2. Exact identities that contraction breaks: a*b - a*b == 0 and Im(z * conj(z)) == 0.

usage: uv run step5_parity.py
"""

import json
import math
import textwrap
import types
from fractions import Fraction

import guard
import numpy as np

N = 200_000


def fma32(a, b, c):
    q = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    r = np.float32(float(q))
    cands = [np.nextafter(r, np.float32(-np.inf)), r, np.nextafter(r, np.float32(np.inf))]
    best = min(
        cands,
        key=lambda x: (
            abs(Fraction(float(x)) - q),
            int(np.array(x, np.float32).view(np.uint32)) & 1,
        ),
    )
    return np.float32(best)


def cpu_module(prec: str):
    import gen

    ns = {
        "cuda": types.SimpleNamespace(fma=math.fma if prec == "f64" else fma32),
        "SPLIT": 134217729.0 if prec == "f64" else np.float32(4097.0),
        "TWO": 2.0 if prec == "f64" else np.float32(2.0),
    }
    exec(compile(textwrap.dedent(gen.DEVICE_SRC), "<template>", "exec"), ns)
    return ns


def cpu_apply(ns, fn, a, b, dt):
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        x = [dt(v) for v in a[i]] if dt is np.float32 else a[i].tolist()
        y = [dt(v) for v in b[i]] if dt is np.float32 else b[i].tolist()
        if fn == 0:
            r = ns["w_add"](x[0], x[1], y[0], y[1])
        elif fn == 1:
            r = ns["w_mul"](x[0], x[1], y[0], y[1])
        elif fn == 2:
            r = ns["w_sqr"](x[0], x[1])
        elif fn == 3:
            r = ns["c_mul"](*x, *y)
        elif fn == 4:
            r = ns["c_sqr"](*x)
        else:
            r = ns["w_mul_dekker"](x[0], x[1], y[0], y[1])
        out[i, : len(r)] = r
    return out


def main():
    guard.check("step5 parity")
    import gen
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.cuda import libdevice as ld
    from step3_composite import OPS, make_inputs

    res = {"n": N, "parity": [], "identities": []}
    for prec, dt in (("f64", np.float64), ("f32", np.float32)):
        a, b = make_inputs(prec, N, seed=11)
        ns = cpu_module(prec)
        da, db = cuda.to_device(a), cuda.to_device(b)
        mods = {v: gen.build(prec, v) for v in ("rn", "naive")}
        for fn in (0, 1, 2, 3, 4, 5):
            cpu = cpu_apply(ns, fn, a, b, dt)
            row = dict(fmt="dd" if prec == "f64" else "df64", op=OPS[fn], n=N)
            gpu = {}
            for v, mod in mods.items():
                d = cuda.device_array_like(a)
                mod.k_wop[(N + 255) // 256, 256](fn, da, db, d)
                cuda.synchronize()
                gpu[v] = d.copy_to_host()
                width = 4 if fn in (3, 4) else 2
                row[f"{v}_differs_from_cpu"] = int(
                    (
                        gpu[v][:, :width].view(np.uint64 if prec == "f64" else np.uint32)
                        != cpu[:, :width].view(np.uint64 if prec == "f64" else np.uint32)
                    )
                    .any(axis=1)
                    .sum()
                )
            print(json.dumps(row), flush=True)
            res["parity"].append(row)

    # exact identities, bare vs rn, f64 and f32
    def k_ident_bare(a, b, out):
        i = cuda.grid(1)
        if i < a.size:
            x = a[i]
            y = b[i]
            out[i, 0] = x * y - x * y  # == 0 exactly without contraction
            out[i, 1] = x * (-y) + y * x  # Im(z * conj(z)), z = x + iy

    def k_ident_rn64(a, b, out):
        i = cuda.grid(1)
        if i < a.size:
            x = a[i]
            y = b[i]
            out[i, 0] = ld.dadd_rn(ld.dmul_rn(x, y), -ld.dmul_rn(x, y))
            out[i, 1] = ld.dadd_rn(ld.dmul_rn(x, -y), ld.dmul_rn(y, x))

    def k_ident_rn32(a, b, out):
        i = cuda.grid(1)
        if i < a.size:
            x = a[i]
            y = b[i]
            out[i, 0] = ld.fsub_rn(ld.fmul_rn(x, y), ld.fmul_rn(x, y))
            out[i, 1] = ld.fadd_rn(ld.fmul_rn(x, -y), ld.fmul_rn(y, x))

    rng = np.random.default_rng(12)
    for prec, dt, krn in (("f64", np.float64, k_ident_rn64), ("f32", np.float32, k_ident_rn32)):
        x = rng.uniform(-2, 2, N).astype(dt)
        y = rng.uniform(-2, 2, N).astype(dt)
        for tag, kf in (("bare", k_ident_bare), ("rn", krn)):
            k = cuda.jit(kf)
            out = cuda.device_array((N, 2), dt)
            k[(N + 255) // 256, 256](cuda.to_device(x), cuda.to_device(y), out)
            cuda.synchronize()
            o = out.copy_to_host()
            row = dict(
                prec=prec,
                kernel=tag,
                n=N,
                ab_minus_ab_nonzero=int((o[:, 0] != 0).sum()),
                im_z_conjz_nonzero=int((o[:, 1] != 0).sum()),
            )
            print(json.dumps(row), flush=True)
            res["identities"].append(row)
    with open("results/step5_parity.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

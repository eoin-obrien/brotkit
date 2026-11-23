"""Step 1: numba-cuda-mlir compiler controls (original gate) and _rn intrinsics (revised gate).

usage: uv run step1_api.py > results/step1.txt
"""

import re

import guard


def main():
    guard.check("step1")
    import numpy as np
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.cuda import libdevice as ld

    def k_mix(a, b, c, out):
        out[0] = a[0] * b[0] + c[0]  # contractible
        out[1] = a[0] / b[0]  # prec-div
        out[2] = a[0] + b[0]  # ftz candidate
        out[3] = cuda.fma(a[0], b[0], c[0])  # explicit FMA

    def k_rn(a, b, c, out):
        out[0] = ld.fadd_rn(ld.fmul_rn(a[0], b[0]), c[0])  # must not fuse

    print("== fastmath=False vs True on this GPU (sm_86): IR, PTX, SASS, runtime")
    for fm in (False, True):
        k = cuda.jit(fastmath=fm)(k_mix)
        a = cuda.to_device(np.array([1e-40], np.float32))  # f32 subnormal
        b = cuda.to_device(np.array([1e-40], np.float32))
        c = cuda.to_device(np.array([0.0], np.float32))
        o = cuda.device_array(4, np.float32)
        k[1, 1](a, b, c, o)
        key = next(iter(k.overloads.keys()))
        mlir = k.inspect_mlir(key)
        llvm = k.inspect_llvm(key)
        ptx = k.inspect_asm(key)
        sass = k.inspect_sass(key)
        print(f"fastmath={fm}")
        print("  MLIR target :", re.findall(r"#nvvm\.target<[^>]*>", mlir))
        print(
            "  LLVM fp ops :",
            sorted(set(re.findall(r"= (f(?:add|mul|div)(?: [a-z]+)* float)", llvm))),
        )
        print("  PTX fp ops  :", re.findall(r"^\s*((?:fma|mul|add|div)\.[a-z0-9.]*f32)", ptx, re.M))
        print(
            "  SASS fp ops :", re.findall(r"\b(F(?:FMA|MUL|ADD)(?:\.[A-Z0-9]+)*|MUFU\.RCP)\b", sass)
        )
        print("  1e-40 + 1e-40 =", o.copy_to_host()[2], "(0.0 would mean flush-to-zero)")

    print("\n== _rn intrinsics survive (no fusion)")
    k = cuda.jit(k_rn)
    k[1, 1](*[cuda.to_device(np.ones(1, np.float32)) for _ in range(4)])
    key = next(iter(k.overloads.keys()))
    print("  PTX :", re.findall(r"^\s*((?:fma|mul|add)\.[a-z0-9.]*f32)", k.inspect_asm(key), re.M))
    print("  SASS:", re.findall(r"\b(F(?:FMA|MUL|ADD)(?:\.[A-Z0-9]+)*)\b", k.inspect_sass(key)))

    print("\n== API used")
    api = [
        "fastmath     : cuda.jit(fastmath=bool)",
        "               cuda.compile_ptx(fn, sig, fastmath=bool, cc=(M, m))",
        "explicit FMA : cuda.fma(a, b, c)  (also cuda.libdevice.fma / fmaf)",
        "rn add/mul   : cuda.libdevice.dadd_rn, dmul_rn, fadd_rn, fsub_rn, fmul_rn",
        "               (no dsub_rn: use dadd_rn(x, -y))",
        "inspection   : k.inspect_asm(sig) / inspect_ptx / inspect_lto_ptx / inspect_sass",
        "               / inspect_llvm / inspect_mlir",
    ]
    for line in api:
        print("  " + line)


if __name__ == "__main__":
    main()

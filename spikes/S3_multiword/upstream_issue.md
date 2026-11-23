# DRAFT: not filed. For NVIDIA/numba-cuda-mlir; review before posting.

**Title:** `fastmath=True` has no effect below sm_100, and FMA contraction can't be disabled

## Summary

Two related problems with floating-point compiler options in numba-cuda-mlir 0.6.0:

1. **`fastmath=True` is ignored for every target below sm_100.** The PTX is identical with it
   on or off: no `.ftz`, no approximate division, no fast-math flags. On sm_100 and sm_120 it
   works as documented.
2. **There's no way to disable FMA contraction.** With `fastmath=False`, `a * b + c` still
   compiles to `fma.rn`, on every architecture. There's no option equivalent to nvcc
   `--fmad=false` or libnvvm `-fma=0`.

## Environment

- numba-cuda-mlir 0.6.0 (`numba-cuda-mlir[cu13]`), CPython 3.15.2, Linux (WSL2)
- CUDA wheels 13.5.37 (`nvidia-nvvm`, `nvidia-cuda-nvcc`, ...), driver 610.88 (Windows host; CUDA 13.3)
- GPU: RTX 3090 Ti (sm_86). The repro below is compile-only (`compile_ptx` with `cc=`).

## Repro

```python
import re
from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import float32, void


def kernel(a, b, c, out):
    out[0] = a[0] * b[0] + c[0]  # contractible mul + add
    out[1] = a[0] / b[0]  # prec-div
    out[2] = a[0] + b[0]  # ftz candidate


sig = void(float32[:], float32[:], float32[:], float32[:])
for cc in ((8, 6), (9, 0), (10, 0), (12, 0)):
    for fastmath in (False, True):
        ptx, _ = cuda.compile_ptx(kernel, sig, fastmath=fastmath, cc=cc)
        ops = re.findall(r"^\s*((?:fma|mul|add|div)\.[a-z0-9.]*f32)", ptx, re.M)
        print(f"sm_{cc[0]}{cc[1]:<2} fastmath={fastmath!s:<5} {ops}")
```

Output:

```
sm_86  fastmath=False ['fma.rn.f32', 'div.rn.f32', 'add.f32']
sm_86  fastmath=True  ['fma.rn.f32', 'div.rn.f32', 'add.f32']        <- identical to False
sm_90  fastmath=False ['fma.rn.f32', 'div.rn.f32', 'add.f32']
sm_90  fastmath=True  ['fma.rn.f32', 'div.rn.f32', 'add.f32']        <- identical to False
sm_100 fastmath=False ['fma.rn.f32', 'div.rn.f32', 'add.f32']
sm_100 fastmath=True  ['fma.rn.ftz.f32', 'div.full.ftz.f32', 'add.ftz.f32']
sm_120 fastmath=False ['fma.rn.f32', 'div.rn.f32', 'add.f32']
sm_120 fastmath=True  ['fma.rn.ftz.f32', 'div.full.ftz.f32', 'add.ftz.f32']
```

The first line of every pair also shows problem 2: `a*b + c` became `fma.rn.f32` with
`fastmath=False`.

The runtime agrees on sm_86. With `fastmath=True`, f32 `1e-40 + 1e-40` returns
`1.99999e-40` (not flushed). For a kernel with no division (so no libdevice `div.rn` slow
path in the SASS), the `fastmath=True` SASS for `s = a + b` is plain `FADD`, not `FADD.FTZ`,
and the floating-point instruction sequence is the same as in the `fastmath=False` build.

## Where it happens (as far as we can tell)

**Problem 1.** `fastmath=True` does reach the compiler:
- The MLIR target attribute becomes `#nvvm.target<chip = "sm_86", flags = {fast}, ...>`.
- `mlir_optimization._nvvm_options()` builds `{"ftz": True, "fma": True, "prec_div": False,
  "prec_sqrt": False}`.

But for `sm < 100`, `_needs_llvm70_path()` is true and the module is compiled by
`_call_llvm70_capi()` through `libMLIRToLLVM70.so`. That call takes no option list. The bridge
only passes `-arch=` and `-opt=` to libnvvm (those are the only libnvvm option strings in the
`.so`), and the `{fast}` target flag doesn't turn into LLVM fast-math flags either
(`inspect_llvm()` shows plain `fadd float`). So `nvvm_opts` is computed on that path but never
used.

**Problem 2.**
- `_nvvm_options()` only adds `fma` when fastmath is on. Otherwise libnvvm's default
  (`-fma=1`, contraction enabled) applies.
- The Linker passes `fma=None` in that case too.
- The vendored `numba_cuda.cudadrv.nvvm.compile_ir` has the same shape.
- There's no target option to request `-fma=0`, and ptxas then fuses un-rounded `mul`/`add`
  further. In one kernel, 4 `fma.rn.f64` in the PTX became 9 `DFMA` in the SASS.

## Why it matters

- **Problem 1:** users who ask for fastmath on Ampere, Ada or Hopper silently don't get it. The
  result is correct but slower than requested, and the PTX gives no hint.
- **Problem 2:** reproducibility and error-free transformations. Double-double, double-float
  and compensated summation code written with plain operators changes results under
  contraction. For example, Veltkamp/Dekker splitting `t = s*a; hi = t - (t - a)` contracts to
  `fma(s, a, -a)` and the split is no longer exact.
- **Problem 2:** CPU/GPU parity. CPU Numba with `fastmath=False` doesn't contract, so the GPU
  differs in the last bits.
- **Current workaround:** route every add and mul through libdevice `__nv_dadd_rn` /
  `__nv_dmul_rn` / `__nv_fadd_rn` / ... (`cuda.libdevice.dadd_rn`, ...). These lower to
  `add.rn` / `mul.rn`, which the PTX ISA says are "treated conservatively by the code
  optimizer" and aren't fused. It works, but it's verbose, and only codegen-based projects can
  do it reasonably.

## Requests

1. Pass the libnvvm options (`-ftz`, `-prec-div`, `-prec-sqrt`, `-fma`) through the LLVM 7
   bridge, so `fastmath` behaves the same on all architectures. At minimum, warn when
   `fastmath=True` can't be honoured.
2. Expose FMA contraction as its own option, independent of `fastmath`, for example
   `cuda.jit(fma=False)` or a `contract=` flag mapping to libnvvm `-fma=0` (and ptxas
   `--fmad=false` where ptxas is involved), on both compile paths.
3. Document the current behaviour in the fastmath section.

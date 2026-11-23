# AGENTS.md: codegen

Rules for `src/brotkit/codegen/` (SymPy → real arithmetic → Numba kernel source). The root
`AGENTS.md` still applies; these add to it. Background: "Formula pipeline" and "Architecture
review" in `docs/design/spec.md`.

- **fastmath only for f32.** Only `Precision.F32` kernels may compile with `fastmath=True`. It's
  a performance option, never a correctness control: it does nothing below sm_100 on
  numba-cuda-mlir 0.6.0 (ADR-0006).
- **Multi-word arithmetic is rounding-explicit.** df64, dd and qf rely on error-free
  transformations (two-sum, two-product) that break under FMA contraction, reassociation or
  flush-to-zero. libnvvm contracts bare `a*b + c` even with `fastmath=False`, and nothing turns
  that off. So the multi-word printers emit every float add/sub/mul as a libdevice intrinsic
  (`dadd_rn`/`dmul_rn`, `fadd_rn`/`fsub_rn`/`fmul_rn`; f64 subtraction is `dadd_rn(x, -y)`) and
  every FMA as an explicit `cuda.fma`, never bare `+ - *`. That includes float ops in the
  template next to multi-word code. These kernels still compile with `fastmath=False`, because
  on sm_100+ fastmath adds `.ftz` even to `_rn` ops. Tests: no bare float operators in
  generated multi-word source, PTX FMA count equals the explicit `cuda.fma` count, and results
  are bit-identical to a CPU emulation of the same source (see `docs/design/spikes/S3.md`).
- **Cast integers to the working dtype.** Numba types `int64 * float32` as f64, which silently
  promotes an f32 kernel. Emit `np.float32(i)` (or the precision's dtype) for index arithmetic
  that feeds float maths.
- **Generated source must be inspectable.** Kernels are written to real `.py` files in the
  cache (never `exec`). `.source`, `.ptx` and `.explain()` must keep working. The cache key
  covers source, backend, precision and Numba/numba-cuda-mlir versions.
- **Codegen never imports the GPU JIT.** Generated CUDA source is compiled by
  `src/brotkit/backends/cuda.py`, the only module allowed to import `numba_cuda_mlir`
  (root AGENTS.md invariant 8). Don't import `numba_cuda_mlir` or `numba.cuda` here, even
  for types.
- Features compose at codegen time through template slots (`init`, `step`, `escape_test`,
  `per_iter`, `finalise`), never through runtime dispatch.

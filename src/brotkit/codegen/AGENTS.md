# AGENTS.md: codegen

Rules for `src/brotkit/codegen/` (SymPy → real arithmetic → Numba kernel source). The root
`AGENTS.md` still applies; these add to it. Background: "Formula pipeline" and "Architecture
review" in `docs/design/spec.md`.

- **fastmath only for f32.** Only `Precision.F32` kernels may compile with `fastmath=True`.
- **Multi-word arithmetic compiles with fastmath off.** df64, dd and qf rely on error-free
  transformations (two-sum, two-product) that break under reassociation or FMA contraction.
  Their kernels use `fastmath=False` and emit FMAs explicitly. Parity tests must cover this.
- **Generated source must be inspectable.** Kernels are written to real `.py` files in the
  cache (never `exec`). `.source`, `.ptx` and `.explain()` must keep working. The cache key
  covers source, backend, precision and Numba/numba-cuda versions.
- Features compose at codegen time through template slots (`init`, `step`, `escape_test`,
  `per_iter`, `finalise`), never through runtime dispatch.

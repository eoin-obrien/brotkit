---
name: run-gpu-suite
description: Run brotkit's GPU test suite (pytest -m gpu): CPU/GPU parity, fastmath and multi-word checks, golden images. Use when GPU tests need to run locally or a change touches kernels or backends.
---

# Run the GPU suite

1. **Check the GPU is free.** Run `nvidia-smi`. If LLM inference or another job is using the
   card, don't run; ask the maintainer instead.
2. Install with CUDA: `uv sync --group dev --extra cuda`.
3. Run `uv run pytest -m gpu` (add `-m "gpu and not slow"` for a quick pass).
4. Parity is precision-bound: compare at equal precision on pixels away from the boundary,
   never bit-for-bit across backends.
5. Golden-image failures: report them; don't regenerate (see `update-golden-images`).

---
name: run-gpu-suite
description: Run brotkit's GPU test suite (pytest -m gpu): CPU/GPU parity, fastmath and multi-word checks, golden images. Use when GPU tests need to run locally or a change touches kernels or backends.
---

# Run the GPU suite

1. **Check the GPU is free.** Run `nvidia-smi`. If LLM inference or another compute job is
   using the card, don't run; ask the maintainer instead. Ordinary desktop load (browsers,
   overlays) is fine for correctness tests. Timing tests (`perf`) need an idle GPU, which
   `tools/gpu_check.py --perf` waits for.
2. Install with CUDA: `uv sync --group dev --extra cuda`.
3. Run `uv run pytest -m "gpu and not perf"` (add `and not slow` for a quick pass), or
   `uv run pytest -m gpu` on an idle GPU to include the timing tests.
4. Parity is precision-bound: compare at equal precision on pixels away from the boundary,
   never bit-for-bit across backends.
5. Golden-image failures: report them; don't regenerate (see `update-golden-images`).
6. For a PR, the result that counts is the `gpu/xenon` status from `tools/gpu_check.py`
   (AGENTS.md, "GPU check"): only on the maintainer's own PRs, only when asked, `--dry-run`
   first, and never with `--reviewed`.

---
name: update-golden-images
description: Regenerate or add golden reference images under tests/golden (Git LFS). Requires explicit maintainer approval before any golden file is changed. Use when a golden-image test fails because of an intended rendering change, or when adding a new golden.
---

# Update golden images

**Ask for explicit approval first, every time.** Never update goldens just to make a failing
test pass.

1. Show the perceptual diff and metric for each failing golden, and explain why the change
   is intended (which commit or spec decision changed the output).
2. Wait for the maintainer to approve the specific images.
3. Regenerate on the GPU machine with the documented recipe (see `run-gpu-suite` for GPU
   etiquette). Confirm the files are stored as Git LFS pointers (`git lfs ls-files`).
4. Commit as `test: update golden images for <reason>` with the approval noted in the PR.

---
name: update-golden-images
description: Regenerate or add golden images and the mpmath golden references under tests/golden (Git LFS). Requires explicit maintainer approval before any golden file is changed. Use when a golden-image test fails because of an intended rendering change, or when adding a new golden.
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

## Golden references (mpmath)

`tests/golden/<view>.npz` and `manifest.json` are exact mpmath references, not renders
(`docs/design/goldens.md`, ADR-0011). The same approval rule applies, every time.

1. Explain what changed: the pixel convention (a new ADR), the views, or the layers. A failing
   comparison is a renderer bug until shown otherwise; never regenerate to make it pass.
2. After approval, commit the generator change first (the run refuses uncommitted generator
   changes and records the git SHA), then run `uv run tools/make_goldens.py [view ...]` on
   Xenon (5–7 minutes for all views on 24 cores; CPU only).
3. Check the output: zero P vs 2P mismatches, and the decidable and sensitive fractions in
   the manifest against the table in `docs/design/goldens.md`.
4. `git lfs ls-files` lists the NPZs; `manifest.json` stays a normal file. Commit as
   `test: update golden references for <reason>`.

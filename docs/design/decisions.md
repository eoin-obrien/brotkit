# Decision log (ADRs)

Architecture decision records for changes made **after** the design spec
([`spec.md`](spec.md)) was settled. The spec is authoritative. Any deviation from it is recorded
here as a numbered ADR, and the spec snapshot is edited to match in the same PR.

ADRs are append-only. To reverse a decision, add a new ADR that supersedes it, and mark the
old one `Superseded by ADR-NNNN`.

## Template

```markdown
## ADR-NNNN: Title

- **Status:** Proposed | Accepted | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD
- **Backlog:** task ID(s)

### Context
What forced the decision, with the spec section it touches.

### Decision
What we do now.

### Consequences
What changes, what it costs, and the exit condition if it's temporary.
```

<!-- ADRs below, newest last. -->

## ADR-0001: Dev and docs tooling as dependency groups; default branch `main`

- **Status:** Accepted
- **Date:** 2026-11-23
- **Backlog:** 0.2, 0.6

### Context
The spec's Toolchain table listed `[dev]` as an optional extra beside `[cuda]` and `[video]`,
and the AGENTS.md contents gave the setup command as `uv sync --extra cuda --group dev`.
Development and docs tooling aren't features a user installs with the package, so publishing
them as extras would expose them on PyPI. The repository was also created with a default
branch of `master`, while the Repository conventions (squash-merge onto `main`) assume `main`.

### Decision
- `dev` and `docs` are PEP 735 dependency groups (`[dependency-groups]` in `pyproject.toml`),
  not extras. The only extras are `[cuda]` (empty until spike S1 pins the wheels) and
  `[video]` (`av`).
- The setup command is `uv sync --group dev`, plus `--extra cuda` on the GPU machine.
- The default branch is renamed from `master` to `main` on GitHub before the first PR.

### Consequences
- Groups aren't published to PyPI metadata; contributors use `uv sync --group dev`, and docs
  builds use `uv run --group docs ...`.
- Spec edits: the Toolchain "Packaging" row and the AGENTS.md setup bullet.
- Existing clones must follow the rename (`git branch -m master main`,
  `git fetch origin`, `git branch -u origin/main main`, `git remote set-head origin -a`).

## ADR-0002: Cap Sphinx below 9 until pytest-markdown-docs supports markdown-it-py 4

- **Status:** Accepted
- **Date:** 2026-11-23
- **Backlog:** 0.2 (revisit at 1.23)

### Context
uv locks every dependency group into one `uv.lock`. `pytest-markdown-docs` 0.9.2 (dev group,
the latest release) requires `markdown-it-py<4`. On Sphinx 9, `myst-nb` 1.4 resolves to
`myst-parser>=5.1`, which requires `markdown-it-py>=4.2`. With `sphinx>=9.2` the `dev` and
`docs` groups can't be resolved together. Without the floor, the lock resolves to Sphinx 8.2.3,
myst-nb 1.4 and markdown-it-py 3.

### Decision
The `docs` group pins `sphinx>=8.2,<9`, with a comment in `pyproject.toml` naming the blocker.
Both groups stay installable together from a single lock.

### Consequences
- Docs build on Sphinx 8.2.x instead of 9.x.
- **Exit condition:** lift the cap once `pytest-markdown-docs` supports `markdown-it-py` 4.
  If that still hasn't happened by backlog 1.23 (when README code testing is switched on),
  switch README testing to another tool rather than keep Sphinx capped.
- Spec edit: the Repository conventions "Docs" row notes the cap.

## ADR-0003: Hero images are ordinary git files, not Git LFS

- **Status:** Accepted
- **Date:** 2026-11-23
- **Backlog:** 0.2, 1.23

### Context
The scaffold put both `tests/golden/**` and `docs/hero/**` in Git LFS. The README hero strip
(spec: "README, community files and agent conventions") must use absolute
`raw.githubusercontent.com` URLs, because PyPI renders the same README. For LFS-tracked files,
`raw.githubusercontent.com` serves the LFS pointer text, not the image, so the hero strip
would be broken on GitHub and PyPI.

### Decision
- `docs/hero/**` is removed from LFS tracking. Hero PNGs are committed as ordinary files,
  optimised to about 300 KB each, and served from `raw.githubusercontent.com`.
- `tests/golden/**` stays in Git LFS.

### Consequences
- Each hero regeneration adds about 1.2 MB of history for the four images, so regenerate them
  only when the output visibly changes.
- The images must be optimised (for example, `oxipng`) before committing. The pre-commit
  large-file guard (1024 KB) catches anything far over budget.
- Hero renders are still produced from `docs/hero/*.recipe.json` in CI and still act as a
  golden test (1.23).
- Spec edit: the README hero-strip item notes the files aren't in LFS.

## ADR-0004: Create the PyPI project through the release workflow, from version 0.0.0

- **Status:** Accepted
- **Date:** 2026-11-23
- **Backlog:** 0.1, 0.5

### Context
Backlog 0.1 allowed a one-off manual `uv publish` to reserve `brotkit`, with the Trusted
Publisher registered afterwards. That needs an API token, even if only once. PyPI's pending
Trusted Publishers make the token unnecessary: the first OIDC publish from the configured
workflow creates the project. Three practical problems came up while building the workflow
(spec: Repository conventions, "Versioning", "Release manager" and "Publishing"):

- `pyproject.toml` already said 0.0.1 with no tag. `cz bump` can't release the current
  version as-is, and the scaffold's `docs:`/`chore:` commits aren't eligible for a bump.
- `uv.lock` records the project's own version. With the `pep621` provider, `cz bump` changes
  `pyproject.toml` only and leaves `uv.lock` stale.
- `cz bump --increment PATCH` with no commits since the last tag exits 0 without bumping
  anything, unless `--allow-no-commit` is passed.

### Decision
- No manual upload. Pending Trusted Publishers on PyPI and TestPyPI (workflow `release.yml`,
  environments `pypi` and `testpypi`) let the release workflow create the project.
- The version in `pyproject.toml` is reset to 0.0.0 before the first release. Run 1 forces
  `PATCH` and publishes 0.0.1, the name reservation. Run 2 forces `PATCH` and publishes 0.0.2,
  the end-to-end check.
- The release workflow takes an `increment` input: `auto` (from Conventional Commits) or a
  forced `PATCH`, `MINOR` or `MAJOR`, which adds `--allow-no-commit`. The workflow fails if
  the version didn't change.
- Commitizen uses the `uv` version provider, which writes the version to both
  `pyproject.toml` and `uv.lock`.
- TestPyPI publishes from its own `testpypi` environment, and an install-from-TestPyPI check
  must pass before the protected `pypi` job runs.

### Consequences
- The name isn't held until run 1 completes. Pending publishers don't reserve names, so
  run 1 should happen right after the publishers are registered.
- Changelog entries for 0.0.1 and 0.0.2 are empty headings, because Commitizen's default
  changelog lists only `feat`, `fix`, `refactor`, `perf` and breaking changes.
- The bump commit and tag are pushed to `main` with the workflow's `GITHUB_TOKEN`. If `main`
  is ever protected, the workflow (or a GitHub App) needs a bypass.
- Spec edits: the Repository conventions "Versioning" and "Publishing" rows.

**Update (2026-11-23):** Release run 1
([45215851937](https://github.com/eoin-obrien/brotkit/actions/runs/45215851937)) tagged
v0.0.1 and published it to TestPyPI. The TestPyPI install check passed, but `setup-uv`'s
post step failed the job: caching is on by default and the step runs with `--no-cache`, so
there was nothing to save. The PyPI job was skipped. "Re-run failed jobs" uses the workflow
file from the original commit, so it couldn't recover the run, and the workflow can't
publish an existing tag. The release workflow now disables the uv cache and also installs
from PyPI after publishing. **0.0.1 exists only as a git tag and on TestPyPI. 0.0.2 is the
first PyPI release, and it creates the project and holds the name.**
Run 2 ([45219689842](https://github.com/eoin-obrien/brotkit/actions/runs/45219689842))
published 0.0.2 to TestPyPI and PyPI, with attestations. The PyPI install check was added
afterwards: its commit was pushed to #63's branch after that PR had merged, so it missed
run 2. It was checked by hand instead (`uv run --with brotkit==0.0.2` from PyPI imports and
reports 0.0.2) and lands with the next release.

## ADR-0005: GPU JIT is numba-cuda-mlir

- **Status:** Accepted
- **Date:** 2026-11-23
- **Backlog:** S1 (gates S3 and S4; affects 0.3, 1.4, 1.20)

### Context
The spec chose `numba-cuda` as the GPU JIT, with Python 3.15 as the target and an open
question about cp315 wheels (spec: Toolchain decisions). Spike S1
([`spikes/S1.md`](spikes/S1.md)) found that `numba-cuda` will never support 3.15:

- NVIDIA put `numba-cuda` in maintenance mode on 2026-06-25
  ([NVIDIA/numba-cuda#902](https://github.com/NVIDIA/numba-cuda/issues/902)): security and
  critical fixes only, through the life of CUDA 13, with `numba-cuda-mlir` as the successor.
- The Python 3.15 request was closed as won't-do
  ([#926](https://github.com/NVIDIA/numba-cuda/issues/926)): "Users who need to use Python
  3.15 should switch to numba-cuda-mlir instead."
- NumPy 2.5 support was closed the same way
  ([#907](https://github.com/NVIDIA/numba-cuda/issues/907)). `numba-cuda` 0.30.4 calls
  `np.row_stack`, which NumPy 2.5 removed, so it fails to compile any kernel unless NumPy is
  held below 2.5.
- 0.30.4, the last release, has cp310–cp314 wheels and no sdist, so uv can't install it on
  3.15 at all.

On Xenon (WSL2, RTX 3090 Ti, CC 8.6, driver 610.57 / CUDA 13.3), `numba-cuda-mlir[cu13]`
0.6.0 passed the S1 smoke test (f32 elementwise, f64 `cuda.atomic.add`, `cache=True`
written and then loaded in a fresh process) on both CPython 3.14.0 and 3.15.2, with NumPy
2.5.5. It has no `numba` or `llvmlite` dependency and runs alongside CPU `numba` 0.68.1 in the
same process. `uv sync --group dev --extra cuda` from the brotkit project worked on both
Pythons.

### Decision
- The `[cuda]` extra is `numba-cuda-mlir[cu13]==0.6.0`, pinned exactly, for 3.14 and 3.15. No
  `python_version` marker and no 3.15 error path.
- Only `src/brotkit/backends/cuda.py` imports `numba_cuda_mlir`. Nothing in `src/brotkit`
  imports `numba.cuda`. This is AGENTS.md invariant 8, enforced by ruff `TID251`.
- S3 and S4 gate this decision. They must show that fastmath on/off behaves as the spec's
  fastmath policy needs, and that PTX inspection and register-count reporting work on
  `numba-cuda-mlir`. If either fails, stop and report; don't work around it.
- A Dependabot PR that bumps `numba-cuda-mlir` doesn't auto-merge. It needs the nightly GPU
  suite green first (backlog 0.3).

### Consequences
- The spec's GPU JIT row, open questions, toolkit-install row, and other mentions of
  `numba-cuda` now name `numba-cuda-mlir`.
- `numba-cuda-mlir` is pre-1.0. An exact pin plus the nightly GPU gate is how we absorb its
  churn. Its extension API differs from `numba-cuda` (MLIR, not LLVM IR). brotkit generates
  Python source and shouldn't need it, but any use of extension APIs has to target MLIR.
- `numba-cuda-mlir` has no CUDA simulator (`NUMBA_ENABLE_CUDASIM` is ignored, and the package
  has no simulator module). The spec's "CPU tests + `numba-cuda` simulator on every push" CI
  plan is open and must be settled in 0.3 or 1.4.
- **Documented fallback, no code:** if `numba-cuda-mlir` fails the S3/S4 gate, the revival
  path is `numba-cuda[cu13]==0.30.4` plus `numpy<2.5`, on Python 3.14 only (marker
  `python_version < "3.15"`), import `from numba import cuda`, which S1 verified works on
  Xenon. It's a dead end: no 3.15, no NumPy ≥ 2.5, and fixes only for security issues until
  CUDA 13 ends. Reviving it would need a new ADR that supersedes this one.

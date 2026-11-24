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
  has no simulator module), so the spec's simulator CI job is dropped. CPU CI runs codegen
  snapshot tests (generated kernel source against reviewed snapshots) and CPU-backend parity
  tests. GPU coverage comes from the self-hosted runner: nightly, before releases, and on a PR
  only when the maintainer adds the `gpu` label. Untrusted PR code never runs on the
  self-hosted runner. If S4 shows `numba-cuda-mlir` can compile to PTX with no GPU present,
  CPU CI adds a compile-only check.
- **Documented fallback, no code:** if `numba-cuda-mlir` fails the S3/S4 gate, the revival
  path is `numba-cuda[cu13]==0.30.4` plus `numpy<2.5`, on Python 3.14 only (marker
  `python_version < "3.15"`), import `from numba import cuda`, which S1 verified works on
  Xenon. It's a dead end: no 3.15, no NumPy ≥ 2.5, and fixes only for security issues until
  CUDA 13 ends. Reviving it would need a new ADR that supersedes this one.

## ADR-0006: Multi-word correctness comes from rounding-explicit intrinsics and explicit FMA

- **Status:** Accepted
- **Date:** 2026-11-23
- **Backlog:** S3 (revises the S3 half of the ADR-0005 gate; affects 1.3, 2.3)

### Context
Spec review item 1 ("Fastmath breaks multi-word arithmetic") made `fastmath=False` the thing
that protects error-free transformations. It assumed `fastmath=False` means no FMA
contraction, no reassociation and no flush-to-zero, and that `fastmath=True` turns those on.
ADR-0005 made S3 confirm that on `numba-cuda-mlir`. Spike S3 ([`spikes/S3.md`](spikes/S3.md))
found on `numba-cuda-mlir` 0.6.0:

- **`fastmath=True` has no effect below sm_100.** Pre-Blackwell targets compile through a
  bundled LLVM 7 bridge (`libMLIRToLLVM70.so`) that passes libnvvm only `-arch=` and `-opt=`.
  The PTX, the SASS and the runtime behaviour are the same with the flag on or off. On sm_100+
  it works (`.ftz`, approximate division).
- **FMA contraction is on by default on every architecture, and there's no way to turn it
  off.** With `fastmath=False`, `a*b + c` compiles to `fma.rn`, and ptxas fuses un-rounded
  `mul`/`add` further. The vendored numba-cuda NVVM code has the same default, so the
  ADR-0005 fallback is unlikely to differ.

So the original gate (fastmath control) failed. The maintainer revised it. Fastmath control
was the wrong requirement, because correctness shouldn't depend on compiler flags. The
revised gate needs explicit FMA, round-to-nearest non-contractible add and mul, no FTZ or
reassociation by default, and PTX/SASS inspection. All four pass.

### Decision
- **Multi-word printers (df64, dd, qf) never emit bare `+`, `-`, `*` on floats.** Every add,
  subtract and multiply in multi-word code is a libdevice round-to-nearest intrinsic:
  - `cuda.libdevice.dadd_rn` / `dmul_rn` for f64 words, with subtraction as `dadd_rn(x, -y)`
    (libdevice has no `dsub_rn`; negation is exact)
  - `fadd_rn` / `fsub_rn` / `fmul_rn` for f32 words
  - every FMA is an explicit `cuda.fma`

  These lower to `add.rn` / `mul.rn` / `sub.rn`, which the PTX ISA protects from
  optimization, and S3 confirmed they reach SASS unfused on sm_86 and sm_100.
- **Bare float operations next to multi-word code go through the same rewrite.** That
  includes scaling by 2 and pixel offsets. Contraction works on inlined code, so it can fuse
  across a device-function boundary.
- **fastmath is a performance option only**, allowed for `f32` kernels as before, and
  currently inert below sm_100. Multi-word kernels still compile with `fastmath=False`: on
  sm_100+, fastmath adds `.ftz` even to the `_rn` intrinsics, and flush-to-zero breaks
  `two_sum` on subnormals.
- **Codegen emits integer-to-float casts explicitly** (`np.float32(i)`) in generated f32 and
  df64 code. Numba types `int64 * float32` as f64, which silently promotes a whole "f32"
  kernel to f64.
- **Multi-word tests (backlog 1.3, 2.3) include:**
  - a snapshot check that generated multi-word source contains no bare float `BinOp`
  - a PTX check that a multi-word kernel's FMA count equals the number of `cuda.fma` calls in
    its source and that it has no un-rounded `add`/`mul`/`sub` outside comparisons
  - a bit-exact CPU/GPU parity test against a CPU emulation of the same generated source
    (S3: 0 differing results out of 200,000 per op)

### Consequences
- Multi-word correctness no longer depends on numba-cuda-mlir's fastmath handling or its
  contraction default. If upstream fixes either ([numba-cuda-mlir#486](https://github.com/NVIDIA/numba-cuda-mlir/issues/486),
  [#220](https://github.com/NVIDIA/numba-cuda-mlir/issues/220)), nothing here changes.
- **Plain `f32`/`f64` kernels still contract**, since nothing turns it off. Their CPU/GPU
  parity stays tolerance-based (review item 9). Multi-word paths get exact parity.
- **`_rn` intrinsics rule out fusions the compiler might otherwise pick.** S3 measured the
  resulting kernels anyway: on the RTX 3090 Ti, df64 runs at 178 Giter/s (5.5× native f64)
  and dd at 2.53 Giter/s (f64 / 12.9).
- The fastmath policy in the spec (f32 only) is unchanged. Only the stated reason and
  mechanism change.
- **Spec edits:**
  - Architecture review items 1 and 9, and the multi-word row of "What holds up"
  - the Toolchain open-questions bullet on the numba-cuda-mlir gate
  - the precision-tier note on AUTO and dd-vs-qf
  - the `Precision.fastmath_allowed` comment
- **Agent rules:** the codegen `AGENTS.md` and root `AGENTS.md` invariant 4.

## ADR-0007: Resource numbers come from nvJitLink; fuse features, split only on spills

- **Status:** Accepted
- **Date:** 2026-11-24
- **Backlog:** S4 (completes the ADR-0005 gate; affects 1.3, 1.4, 0.3 and the phase 2
  precision work)

### Context
Spec architecture-review item 3 ("Register pressure from fused features") says to report
registers per thread from the compiled kernel and, above a threshold, split a render into two
launches. The CI row says CPU CI adds a compile-only check if `numba-cuda-mlir` can compile to
PTX with no GPU. ADR-0005 made S4 confirm on `numba-cuda-mlir` that register reporting works.
Spike S4 ([`spikes/S4.md`](spikes/S4.md)) found on `numba-cuda-mlir` 0.6.0:

- **Reporting works, through nvJitLink.** Compiling to PTX for a target cc and linking it with
  nvJitLink (`-verbose -Xptxas=-v`, plus `-maxrregcount` for a cap) gives the driver's exact
  registers, local memory, shared memory and max threads per block: 175 of 175
  configurations on sm_86. Standalone `ptxas -v` is wrong in 18 of 35 kernels because it
  ignores called subroutines. The driver's getters don't report spills and need a device.
- **No tier spills.** f32, f64, df64, dd and a qf prototype, with 1 to 6 fused features, use
  20–72 registers with zero spills on sm_86, sm_100 and sm_120. The fused 6-feature kernel is
  13–71% faster than the same work split into two 3-feature passes, because a split re-runs
  the escape loop.
- **Caps don't pay below the default allocation.** Forcing occupancy up with
  `max_registers` costs 3–70% once it spills. A loose cap (128) lets ptxas use more registers
  than its default and speeds qf up 12%. `max_registers` applies per kernel and works on both
  of numba-cuda-mlir's code paths. Its `cache=True` key ignores it
  ([numba-cuda-mlir#493](https://github.com/NVIDIA/numba-cuda-mlir/issues/493)).
- **No PTX without a device.** `compile_ptx(..., cc=...)` creates a CUDA context, so hosted CI
  can't compile kernels. Filed upstream as [numba-cuda-mlir#492](https://github.com/NVIDIA/numba-cuda-mlir/issues/492).

### Decision
- **`brotkit.backends.cuda.resource_usage(kernel, signature, cc)` is the one source of
  resource numbers.** It returns a frozen `ResourceUsage` (registers, spill bytes, stack,
  static shared memory, max threads per block). `brotkit.backends.resources` turns that into
  occupancy in pure Python, with per-architecture limit tables, and imports nothing from the
  GPU stack. `CompiledKernel` gets its resource fields from here, not from the driver getters.
- **Kernels are fused by default; the automatic split triggers on spills, not on a threshold.**
  A render splits into core + orbit-statistics launches only when `resource_usage` reports
  spills at the default build, and only if the split timing is faster. S4 found no such kernel
  up to six features, so the split path isn't needed for phase 1.
- **No default register cap.** Never cap a kernel below its default register count. qf starts
  at `max_registers=128`, to be re-measured when qf lands (phase 2) and on another
  architecture.
- **brotkit's kernel cache keys on every compile option and doesn't use Numba's
  `cache=True`.** The `KernelCache` sha256 covers the source and every option that changes the
  binary: backend, precision, target cc, fastmath, `max_registers`, launch bounds, ptxas
  options and tool versions. numba-cuda-mlir's `cache=True` key ignores `max_registers` and
  `ptxas_options`, so it silently serves a capped build to an uncapped dispatcher (#493).
  brotkit doesn't rely on it at all, on either backend. S2 settles how warm starts store the
  compiled artefact under brotkit's key, and adds a test that changing any one option misses
  the cache.
- **CI:** PTX snapshot and resource tests are marked `gpu` until `compile_ptx` works without a
  device. CPU CI runs `brotkit.codegen.checks.find_bare_float_ops` over generated multi-word
  source. It's an AST heuristic for ADR-0006, and the PTX check remains authoritative.

### Consequences
- **ADR-0005's gate is met:** PTX inspection (S3) and register reporting (S4) both work on
  `numba-cuda-mlir`.
- Resource numbers need only `compile_ptx` + nvJitLink, so when upstream drops the host-cc
  lookup they work on hosted CI unchanged. Exit condition for the `gpu` marker on
  `tests/test_cuda_ptx.py`: a `numba-cuda-mlir` release where `compile_ptx(cc=...)` works with
  `CUDA_VISIBLE_DEVICES=""`. The dispatcher-comparison test stays GPU-only.
- **Every resource query costs one more compile and link per kernel.** It happens only when the
  recipe changes, since results are cached with the kernel.
- `resource_usage` salts the PTX with a comment, because nvJitLink's compute cache otherwise
  drops the spill lines. The parser refuses a log without them rather than report zero.
- **Benchmarks have to live with ±10% run-to-run drift on Xenon** (S4: identical SASS timed
  13–22% apart in separate runs). A/B comparisons alternate the variants in one process,
  every result records SM clock and temperature, and a regression check compares against a
  baseline timed in the same run rather than a fixed threshold on stored numbers (backlog
  1.24).
- **Spec updated in the same PR:**
  - architecture-review item 3: fuse by default, split only on spills
  - the CI row: no compile-only CPU check yet
  - the Toolchain bullet: the S4 half of the ADR-0005 gate passed
  - the precision-tier note: qf measured at 9–12× dd on an RTX 3090 Ti
  - pipeline step 5, the pipeline diagram, the cross-session caching row and `KernelSource`:
    the cache key covers every compile option and doesn't rely on `cache=True`
  - the Benchmarks row: alternating timing, clocks recorded, same-run baselines

# brotkit backlog

59 tasks in six stages, matching the Roadmap in `spec.md`. Phases 0 and 1 are issue-sized tasks; phases 2–4 are epics to split when you reach them. Acceptance criteria come from the spec's gates and decisions. Don't start phase 1a until every phase 0 spike has a written result.

Status is tracked in GitHub issues (one per row), not in this file.

## Phase 0: spikes and scaffolding

| ID | Task | Acceptance criteria | Depends on |
| --- | --- | --- | --- |
| 0.1 | Reserve `brotkit` on PyPI | Pending Trusted Publishers registered on PyPI and TestPyPI for the release workflow; its first successful run creates the project (0.0.2, since run 1 reached only TestPyPI), with no token ever uploaded (ADR-0004) | 0.5 |
| 0.2 | Repo scaffold | uv project, src layout, ruff, pyright strict, pytest + Hypothesis, pre-commit with Commitizen `commit-msg` hook, BSD-3 `LICENSE`, GitHub Actions CPU workflow plus PR-title Conventional Commits check, all green on an empty package | — |
| 0.3 | Self-hosted GPU runner on Xenon | Nightly job runs `nvidia-smi` and a numba-cuda-mlir smoke kernel; doesn't clash with LLM inference (scheduled off-hours or free-VRAM check); Dependabot PRs bumping `numba-cuda-mlir` can't auto-merge and need the nightly GPU suite green first (ADR-0005); GPU jobs run nightly, before releases and on PRs labelled `gpu` by the maintainer, never on untrusted PR code (ADR-0005) | 0.2, S1 |
| 0.5 | Release workflow | Manually triggered: `cz bump` (version, `CHANGELOG.md`, tag) → `uv build` → TestPyPI → PyPI via Trusted Publishing and `pypa/gh-action-pypi-publish` with attestations, from a protected `pypi` environment; a 0.0.2 dry run succeeds end to end | 0.2 |
| 0.6 | README, community files and agent instructions (skeleton) | README skeleton with badge rows and install section; AGENTS.md with setup, commands and architecture invariants; CLAUDE.md as `@AGENTS.md` bridge; `.agents/skills/` with `.claude/skills` symlink; CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION.cff, issue and PR templates, CODEOWNERS, Dependabot. Hero images, quickstart output and llms.txt come in 1.23 | 0.2, 0.5 |
| S1 | Install spike | `uv sync --extra cuda` on WSL2 gives a working kernel on Python 3.14 and 3.15 (or 3.15 blocker recorded); exact NVIDIA wheel names pinned in the spec. Result: `docs/design/spikes/S1.md`, ADR-0005 | 0.2 |
| S2 | Kernel cache spike | Generated `.py` loads a compiled kernel in a fresh notebook session without recompiling; cold and warm times recorded. The cache key is a sha256 over the source and every compile option, and doesn't rely on Numba's `cache=True`, whose key ignores `max_registers` and `ptxas_options` (ADR-0007, numba-cuda-mlir#493). A cache-key test shows that changing any one option (backend, precision, target cc, fastmath, `max_registers`, launch bounds, ptxas options) misses the cache and repeating the same options hits it | S1 |
| S3 | Multi-word arithmetic spike | df64 and dd two-sum/two-product with `fastmath=False` on GPU match an mpmath reference at a 1e-20 coordinate within expected error; `fastmath=True` shown to break it. Run on numba-cuda-mlir; this gates ADR-0005, so if fastmath on/off doesn't behave as the policy needs, stop and report rather than work around it | S1 |
| S4 | Register-pressure spike | Table of registers/thread and occupancy for f32 and dd with 1, 3 and 6 fused features; split threshold decided and written into the spec. Run on numba-cuda-mlir: PTX inspection and register-count reporting must work there, and record whether it can compile a kernel to PTX with no GPU present (if so, CPU CI gets a compile-only check). This gates ADR-0005, so if either fails, stop and report rather than work around it | S1 |
| S5 | Chunked-launch spike | Chunk overhead under 1% at 4K; Ctrl-C in Jupyter stops within one chunk with no corrupted state | S1 |
| 0.4 | Baseline renders from the old framework | 10 known coordinates rendered with the previous Numba version, stored in Git LFS with their parameters | 0.2 |

## Phase 1a: walking skeleton (z² + c end to end)

| ID | Task | Acceptance criteria | Depends on |
| --- | --- | --- | --- |
| 1.1 | Core value objects | `Frame`, `Embedding` (Mandelbrot, Julia), `View`, `Precision`, `Recipe`, `ResolvedRecipe`: frozen, slotted, hashable; Hypothesis JSON round-trip tests pass | 0.2 |
| 1.2 | `Formula` + safe JSON tree | z² + c analysed (degree, critical point); `to_json`/`from_json` fuzz-tested; no `eval`/`sympify` path anywhere | 1.1 |
| 1.3 | Printer + escape-time template | `RealPairPrinter` (f32, f64), template slots, `Smooth` and `DistanceEstimate` features; generated source inspectable and matches a hand-written reference kernel | 1.2, S3, S4 |
| 1.4 | CPU and CUDA backends | njit/prange and numba-cuda-mlir backends (`backends/cuda.py` is the only `numba_cuda_mlir` importer), `KernelCache` (store and CUDA adapter landed in S2, ADR-0008) wired into both backends, chunked launches through `ChunkPlanner` (landed in S5, ADR-0009: hybrid poll wait, SIGINT deferred to the chunk end, `RenderInterrupted` with the partial result); f64 CPU/GPU parity on pixels with DE > 2 px | 1.3, S2, S5 |
| 1.5 | `Layers` (GPU-first) | Device arrays held; host copy only on first NumPy access (tested); lazy `mu`, `de` | 1.4 |
| 1.6 | Minimal colouring | field → linear/log scale → `FromMPL` ramp (linearised) → sRGB → 16-bit PNG | 1.5 |
| 1.7 | NPZ persistence | `__meta__` JSON, naming scheme, registry, `allow_pickle=False`; `load(save(x))` round-trips; renamed file warns | 1.5 |
| 1.8 | `render()` facade | Progress bar, `_repr_png_` preview; 4K Mandelbrot in a notebook on the 3090 Ti, time recorded | 1.4–1.7 |
| 1.9 | First golden test in nightly | One golden image compared perceptually, running nightly on Xenon; matches the 0.4 baseline within tolerance | 1.8, 0.3, 0.4 |

## Phase 1b: core complete

| ID | Task | Acceptance criteria | Depends on |
| --- | --- | --- | --- |
| 1.10 | Parameter defaults and validation | Per-feature escape radius, f32 overflow budget, `bailout` alias requires `bailout_kind`, `de_convention` recorded | 1.9 |
| 1.11 | Pluggable `EscapeTest` | Modulus, RealPart, ImagPart, Custom; Smooth/DE refuse invalid tests with a clear error; λeᶻ Julia renders correctly | 1.10 |
| 1.12 | More features | `FinalZ`/angle, n-ary `BinaryDecomp`, `StripeAverage`, `potential` + equipotentials | 1.10 |
| 1.13 | Full ramp and scale algebra | Concat, reverse, repeat (wrap/mirror, fractional), shift, sub-range, mix; OKLab/OKLCH; all Scales; JSON tree + baked LUT | 1.6 |
| 1.14 | Typed compositing and masks | Layer refs, operator expression tree, blend modes, JSON serialisation; no string parsing | 1.13 |
| 1.15 | Supersampling | `PixelSampling` strategies, adaptive scrambled Sobol (4–64), Philox device code, Mitchell filter with aprons, `SampleStore`, `aa="field"` preview | 1.12, 1.14 |
| 1.16 | Colour management and output | Linear Rec.2020 working space, sRGB/P3 output with cICP + ICC, OKLCH gamut mapping, blue-noise dither for 8-bit | 1.13 |
| 1.17 | Overflow guards | log\|dz\| rescaling in f32 DE, `Outcome.NUMERIC_FAIL`, count in metadata, warning above 0.1% | 1.10 |
| 1.18 | `render_async` | Future with cancel and progress callbacks | 1.8 |
| 1.19 | Memory planning and tiling | `estimate()`, auto-tiling from free VRAM, square tiles + queue (single GPU), OOM halve-and-retry to 64 px | 1.15, 1.18 |
| 1.20 | Typer CLI | `render`, `colour`, `estimate`, `doctor`; JSON to stdout, progress to stderr; `doctor` reports driver vs runtime CUDA version (flag runtime > driver), CUDA library paths from `cuda-pathfinder`, compute capability, FP64:FP32 ratio and the kernel cache directory, under WSL2 the Windows `TdrDelay` and `TdrLevel` (warn below 2 s, which the 0.5 s chunk ceiling assumes; ADR-0009), and runs a smoke kernel (see S1.md) | 1.19 |
| 1.21 | Spawn worker pool | `get_context("spawn")`, per-process GPU context; recipes only across processes | 1.19 |
| 1.22 | Recipe files, PNG metadata, Zarr | `.recipe.json`, PNG iTXt with recipe + palette + layer hash, Zarr above 4 GB, semver schema + migrations | 1.7, 1.16 |
| 1.24 | Benchmark harness | Giter/s per device logged in metadata; A/B comparisons alternate the variants in one process (S4's `q4_repeat.py` method): warm-up, median of at least five runs per variant, SM clock and temperature recorded with every result, clock variation above 5% flagged; regressions judged against a baseline timed in the same run with a threshold above the measured noise (about ±10% on Xenon), never a fixed threshold on stored numbers; runs nightly on Xenon behind the GPU busy check | 1.8, 0.3 |
| 1.23 | Docs site and gallery-as-tests | Sphinx + PyData theme + MyST-NB with executed notebooks; sphinx-llms-txt generates llms.txt; gallery built from recipes and checked against golden images; README finished with CI-rendered hero strip and quickstart, README code tested by pytest-markdown-docs; phase 1 gate passes | 1.20, 1.22 |
| 1.25 | Kernel cache: driver-update check | On the first NVIDIA driver update on Xenon, a fresh process either loads every stored cubin or rebuilds those the driver rejects, one warning each and no error (the load-failure-is-a-miss path is implemented and tested with an invalid cubin, ADR-0008); result recorded in `docs/design/spikes/S2.md` | 1.4 |
| 1.26 | Kernel cache: refuse network filesystems | NFS (`nfs`, `nfs4`) and SMB (`cifs`, `smb3`) cache roots are refused like 9p/drvfs, with an error naming the path and suggesting a local one. Refusal is the default because S2's `/mnt/c` test showed locks on a shared mount needn't exclude every client; fall back to a one-time warning only with the reason recorded in ADR-0008 | 1.4 |
| 1.27 | Kernel cache on free-threaded Python and native Windows | Once CI has 3.15t and Windows targets: the cache's thread and process tests, including the 4-process race, pass on free-threaded CPython (the key already carries the `t` flag) and on Windows (`filelock`'s `msvcrt` path) | 1.4, 0.2 |
| 1.28 | `brotkit cache prewarm` | Builds a `FeatureSet.standard()` preset set for the current GPU in parallel processes, skips cached entries and reports cold time per kernel; for the first session after an install or a toolchain upgrade, which invalidates every entry (S2: about 21 s serial per formula for 25 kernels) | 1.20, 1.4 |
| 1.29 | Iteration-sliced launches | Resumable kernels: each pixel's state (z, derivatives, feature accumulators, escape flag) persists in device buffers between launches of at most k iterations, so a render whose `max_iter·τ` breaks the 0.5 s chunk ceiling runs instead of raising `CeilingInfeasibleError` (S5: dd past about 27,000 iterations, f64 past about 320,000 on an RTX 3090 Ti); results bit-identical to one launch; `ChunkPlanner` plans (tile, iteration range) chunks | 1.4 |
| 1.30 | Pipelined chunk launches | Only if the per-chunk cost matters (S5: about 0.6 ms of host time per chunk, plus each chunk's latency floor tail): keep two chunks in flight, which hides both but makes the worst-case cancel latency two chunks; or add a host-mapped cancel flag the kernel checks on entry so it stays one; measured against S5's sweep | 1.4 |

## Phase 2: more families and precision (epics)

| ID | Epic | Acceptance criteria | Depends on |
| --- | --- | --- | --- |
| 2.1 | Convergent family | Newton/Halley/Nova/Magnet, root identification, smooth convergence, outcome layer, multiple-root handling | 1.23 |
| 2.2 | Interior layers | Brent period detection, multiplier, atom domains, interior DE, `Lyapunov` | 1.23 |
| 2.3 | Multi-word precision | df64, dd, qf printers on GPU; per-device dd vs qf benchmark cached; AUTO tier selection; dd still matches mpmath reference; qf forms the pixel offset in qf precision (S4: the f32 offset made 3 of 4096 pixels differ from dd at 1e-20 spacing) and handles the DE derivative's f32 exponent range | 1.23, S3 |
| 2.4 | Density family: Buddhabrot | Escape windows, `SampleDomain`, uniform/importance/MH samplers, resumable `Histograms`; Nebulabrot resume gate | 1.23 |
| 2.5 | MIIM and strange attractors | Inverse iteration with visit caps; Clifford, de Jong, Hénon point maps | 2.4 |
| 2.6 | Sequence iteration | `FormulaSequence` periodic and random; composite compile; degree-product mu; holomorphic check and `rendering_only` flag | 1.23 |
| 2.7 | New colourings | Trap shapes, `ExponentialSmoothing`, `NewtonApproach` | 2.1 |
| 2.8 | Lighting and height | DE normals, emboss from `arg_dz`, slope shading, 16-bit height maps | 1.23 |
| 2.9 | Formula zoo and algebras | McMullen, multicorns, λz(1−z) and 1/c planes, Burning Ship, Phoenix, Collatz; bicomplex and split-complex printers | 2.6 |
| 2.10 | Jupyter explorer | anywidget live f32 preview, pan/zoom, bookmarks to recipes | 1.18 |
| 2.11 | Public `@feature` API | Snippet API documented and marked experimental; built-in features ported onto it | 2.7 |
| 2.12 | Multi-GPU work queue | One worker per GPU, hardest tiles first from a cost pass | 1.19 |

## Phase 3: animation (epics)

| ID | Epic | Acceptance criteria | Depends on |
| --- | --- | --- | --- |
| 3.1 | Tracks and paths | Keyframes with easing, log-width zoom, van Wijk–Nuij pan-zoom, circle/cardioid/Bézier paths with arc-length timing | 1.23 |
| 3.2 | 4D slices and rotations | `Rotation4` (quaternion pairs, slerp), oblique-slice DE seeding, Buddhabrot projection from the `Embedding` | 3.1, 2.4 |
| 3.3 | Frame pipeline | Threaded render/colour/encode stages, layer retention policy, resume after crash, pinned colour stats | 3.1 |
| 3.4 | Encoders and video colour | PyAV and ffmpeg-pipe encoders; BT.709 and BT.2020 SDR 10-bit with correct tags; FFV1/ProRes/PNG masters | 3.3 |
| 3.5 | Exponential-map zooms | `ExpMapRender` and `frames_from_expmap()` | 3.3 |
| 3.6 | Iteration folding views | Pull-back with mip-level AA, push-forward with multiplicity, staged homotopy from the SymPy tree, textures | 3.1, 2.4 |
| 3.7 | Feature numerics | Period in a box, nucleus, Misiurewicz points, size estimates, external rays | 2.2 |

## Phase 4: deep zoom and scale-out (epics)

| ID | Epic | Acceptance criteria | Depends on |
| --- | --- | --- | --- |
| 4.1 | Reference orbits | gmpy2 (MPFR) orbit at the precision the zoom needs, exported as f64 or extended-exponent arrays | 2.3 |
| 4.2 | Perturbation kernels | Tiered f32/f64/extended-exponent deltas, BLA tables on GPU, Zhuoran rebasing; 1e-100 zoom gate | 4.1 |
| 4.3 | Glitch fallback | Glitch detector triggering multi-reference correction | 4.2 |
| 4.4 | Scale-out | Tiles across machines (Dask or Ray) merged by the layer schema | 2.12 |
| 4.5 | Print output | Poster tiling with overlap, 16-bit TIFF with DPI and ICC | 1.22 |

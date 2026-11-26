# AGENTS.md

Instructions for AI coding agents (and humans) working on brotkit. This file is canonical;
`CLAUDE.md` imports it. A nested `AGENTS.md` overrides this one for files under its directory
(see `src/brotkit/codegen/AGENTS.md`).

## Project

brotkit is a GPU-accelerated, SymPy-driven fractal rendering framework for Jupyter. A formula
goes in as SymPy, is differentiated and code-generated into one fused kernel, and is compiled
by Numba for the CPU or by `numba-cuda-mlir` for the GPU (ADR-0005). A render writes a typed bundle of data
layers that can be recoloured, saved and animated without re-iterating.

- **The design spec in `docs/design/spec.md` is authoritative.** Record any deviation from it
  as an ADR in `docs/design/decisions.md` and update the spec to match in the same PR.
- Task plan: `docs/design/backlog.md`. Status lives in GitHub issues (one per backlog row).
- Layout: `src/brotkit/` (package, src layout), `tests/`, `docs/`.

## Setup

```bash
uv sync --group dev                  # laptop / CPU only
uv sync --group dev --extra cuda     # GPU machine
uv run pre-commit install            # ruff, pyright, large-file guard, commit-msg check
```

Python `>=3.14` (3.15 is the target). `dev` and `docs` are PEP 735 dependency groups, not extras.

## Commands

| Task | Command |
| --- | --- |
| Lint | `uv run ruff check` (autofix: `uv run ruff check --fix`) |
| Format | `uv run ruff format` (check only: `uv run ruff format --check`) |
| Typecheck | `uv run pyright` (strict) |
| CPU tests | `uv run pytest -m "not gpu"` |
| GPU tests | `uv run pytest -m gpu` (needs a CUDA GPU and `--extra cuda`) |
| Single test | `uv run pytest tests/test_version.py::test_version_is_pep440` |
| All hooks | `uv run pre-commit run --all-files` |
| Docs build | `uv run --group docs sphinx-build -W -b html docs docs/_build/html` (from backlog 1.23) |

Lint, typecheck and tests always run through `uv run` (`uv run pre-commit run -a`,
`uv run pytest`): ruff, pyright and pytest live only in the uv-managed venv, not on `PATH`, so a
bare `ruff` or `pytest` (including gstack's `/build-check`) fails with a missing-binary error.

Test markers: `gpu` (needs a GPU; excluded from CPU CI) and `slow`. Markers are strict, so
register new ones in `pyproject.toml`.

## Architecture invariants

Don't break these. If a change seems to need it, stop and raise it instead.

1. **Frozen value objects only.** Configs, recipes, ramps, scales, features and views are
   `@dataclass(frozen=True, slots=True, kw_only=True)`. Use `dataclasses.replace` for variants.
2. **No Python in the hot loop.** No per-iteration or per-pixel Python calls, no callbacks
   from kernels, no first-class functions passed into `cuda.jit`. Features are spliced into
   generated source at codegen time.
3. **Never `eval` / `sympify` / `parse_expr` on loaded data.** Formulas load only through the
   whitelisted JSON expression tree (`Formula.from_json`). NPZ files load with
   `allow_pickle=False`, and no class name from a file is ever imported dynamically.
4. **fastmath only for f32.** df64, dd and qf kernels compile with `fastmath=False`, and their
   correctness comes from codegen, not flags: every float add/sub/mul is a libdevice `_rn`
   intrinsic and every FMA an explicit `cuda.fma`, never bare `+ - *` (ADR-0006).
5. **No arrays in frozen dataclasses.** Use `tuple`, `frozendict` and `frozenset`. Arrays
   belong to resource holders (`Layers`, `Histograms`, ...).
6. **Golden images are updated only with explicit approval** from the maintainer (see the
   `update-golden-images` skill). They live in Git LFS and are compared perceptually.
7. **The generated kernel cache is never committed.** It lives in `~/.cache/brotkit/kernels/`.
   Render outputs aren't committed either; `.gitignore` covers both.
8. **Only the CUDA backend imports the GPU JIT.** `numba_cuda_mlir` is imported in
   `src/brotkit/backends/cuda.py` and nowhere else in `src/brotkit`. Codegen emits source; it
   doesn't import the compiler. The legacy `numba.cuda` target isn't used at all (ADR-0005).
   Ruff enforces both (`TID251`).

See also "Rules that keep it fast" and "Conventions" in the spec.

## GPU check

There is no self-hosted runner. Nothing runs on Xenon, the maintainer's GPU machine, unless the
maintainer starts it (ADR-0010). `uv run tools/gpu_check.py <PR>` runs `pytest -m gpu` on the
PR's exact head SHA, on Python 3.14 and 3.15, and posts the `gpu/xenon` commit status. The
required `gpu-gate` check enforces that status whenever a PR touches a GPU-relevant path
(`tools/gpu_gate_paths.toml`). How to run it: [`docs/gpu-check.md`](docs/gpu-check.md).

- gpu_check refuses PRs whose author isn't a maintainer unless `--reviewed` is given.
- `--reviewed` means a human has read the full diff, including the lockfile and the tests.
  Reviewed PRs from outside contributors run only in the sandbox (`--sandbox docker`), which
  holds none of the maintainer's credentials. The status is posted from outside it.
- **Agents never pass `--reviewed`, and never run gpu_check on PRs from other people.** On the
  maintainer's own PRs, and only when asked, an agent may run it, `--dry-run` first, after
  the `nvidia-smi` check in the `run-gpu-suite` skill.
- Never change the maintainer list, the gate, its path config or `gpu-gate.yml` to get a
  check to pass. Those files are themselves GPU-relevant, and edits to them go through the
  check like any other.

## Commits and PRs

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `perf:`,
  `refactor:`, `docs:`, `test:`, `chore:`, `ci:`; `!` or `BREAKING CHANGE:` for breaking
  changes. Commitizen checks messages in a `commit-msg` hook and PR titles in CI.
- PRs are squash-merged into `main`, so **the PR title becomes the commit**. Make it a valid
  Conventional Commit.
- Versioning is semver, 0.x until the API settles. The version lives only in
  `pyproject.toml` (Commitizen's `uv` provider mirrors it into `uv.lock`). Releases use
  `cz bump` via the release workflow (`cut-release` skill); never bump or tag by hand.

### PR checklist

- [ ] PR title is a Conventional Commit
- [ ] `uv run pre-commit run --all-files` passes (ruff, format, pyright strict)
- [ ] `uv run pytest -m "not gpu"` passes; new behaviour has tests (Hypothesis for value
      objects and serialisation)
- [ ] GPU-relevant changes: `gpu-gate` is green, i.e. the maintainer ran
      `uv run tools/gpu_check.py <PR>` on the head SHA (see "GPU check" below)
- [ ] Architecture invariants respected
- [ ] Any deviation from the spec has an ADR in `docs/design/decisions.md`, and the spec is
      updated to match
- [ ] Golden images unchanged, or the update was explicitly approved
- [ ] User-facing docs and README updated where behaviour changed
- [ ] Issue links: `Closes #n` only when merging this PR by itself meets the issue's
      acceptance criteria. When acceptance depends on something that happens after the merge
      (a release, a published package, a nightly run), use `Refs #n` and close the issue by
      hand with a link to the evidence
- [ ] Workflow changes: `uv run zizmor .github/workflows` is clean (CI runs it on any change
      under `.github/workflows/`)

# AGENTS.md

Instructions for AI coding agents (and humans) working on brotkit. This file is canonical;
`CLAUDE.md` imports it. A nested `AGENTS.md` overrides this one for files under its directory
(see `src/brotkit/codegen/AGENTS.md`).

## Project

brotkit is a GPU-accelerated, SymPy-driven fractal rendering framework for Jupyter. A formula
goes in as SymPy, is differentiated and code-generated into one fused kernel, and is compiled
by Numba for the CPU or by `numba-cuda` for the GPU. A render writes a typed bundle of data
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
4. **fastmath only for f32.** df64, dd and qf kernels compile with `fastmath=False` and emit
   their FMAs explicitly.
5. **No arrays in frozen dataclasses.** Use `tuple`, `frozendict` and `frozenset`. Arrays
   belong to resource holders (`Layers`, `Histograms`, ...).
6. **Golden images are updated only with explicit approval** from the maintainer (see the
   `update-golden-images` skill). They live in Git LFS and are compared perceptually.
7. **The generated kernel cache is never committed.** It lives in `~/.cache/brotkit/kernels/`.
   Render outputs aren't committed either; `.gitignore` covers both.

See also "Rules that keep it fast" and "Conventions" in the spec.

## Commits and PRs

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `perf:`,
  `refactor:`, `docs:`, `test:`, `chore:`, `ci:`; `!` or `BREAKING CHANGE:` for breaking
  changes. Commitizen checks messages in a `commit-msg` hook and PR titles in CI.
- PRs are squash-merged into `main`, so **the PR title becomes the commit**. Make it a valid
  Conventional Commit.
- Versioning is semver, 0.x until the API settles. The version lives only in
  `pyproject.toml`. Releases use `cz bump` via the release workflow; never bump by hand.

### PR checklist

- [ ] PR title is a Conventional Commit
- [ ] `uv run pre-commit run --all-files` passes (ruff, format, pyright strict)
- [ ] `uv run pytest -m "not gpu"` passes; new behaviour has tests (Hypothesis for value
      objects and serialisation)
- [ ] GPU-affecting changes: `uv run pytest -m gpu` run on a GPU machine, or flagged for the
      nightly suite
- [ ] Architecture invariants respected
- [ ] Any deviation from the spec has an ADR in `docs/design/decisions.md`, and the spec is
      updated to match
- [ ] Golden images unchanged, or the update was explicitly approved
- [ ] User-facing docs and README updated where behaviour changed

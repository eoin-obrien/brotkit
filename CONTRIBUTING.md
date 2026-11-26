# Contributing to brotkit

Thanks for your interest. brotkit is in early development, so the design is settled but most
of the code isn't written yet. Please open an issue to discuss anything non-trivial before
sending a PR.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Where things are

- **Design spec:** [`docs/design/spec.md`](docs/design/spec.md) is authoritative. A change that
  departs from it needs an ADR in [`docs/design/decisions.md`](docs/design/decisions.md) and a
  matching spec edit in the same PR.
- **Task plan:** [`docs/design/backlog.md`](docs/design/backlog.md). Each row is a GitHub issue,
  grouped by milestone.
- **Commands, invariants and the PR checklist:** [`AGENTS.md`](AGENTS.md). It's written for
  coding agents but is the best reference for humans too.

## Development setup

You need [uv](https://docs.astral.sh/uv/) and, for GPU work, an NVIDIA driver.

```bash
git clone https://github.com/eoin-obrien/brotkit.git
cd brotkit
uv sync --group dev              # add --extra cuda on a GPU machine
uv run pre-commit install        # installs pre-commit and commit-msg hooks
```

Golden images live in Git LFS, so install [git-lfs](https://git-lfs.com/) and run
`git lfs install` once.

## Before you push

```bash
uv run pre-commit run --all-files   # ruff, ruff format, pyright strict, hygiene
uv run pytest -m "not gpu"          # CPU tests (what CI runs)
uv run pytest -m gpu                # GPU tests, if you touched kernels and have a GPU
```

## GPU tests and the `gpu-gate` check

Hosted CI has no GPU, so GPU tests (`pytest -m gpu`) run on the maintainer's machine, and only
when the maintainer starts them. There's no self-hosted runner, so nothing in a PR can trigger
code on that machine. When your PR touches a GPU-relevant path (backends, codegen, kernels,
GPU tests, GPU-stack dependencies; the full list is in `tools/gpu_gate_paths.toml`), the
`gpu-gate` check shows a warning until the maintainer has run `tools/gpu_check.py` on your
PR's latest commit. A new push needs a new run. The check is advisory for now: it doesn't
block merging, but the maintainer won't merge a GPU-relevant PR without a passing
`gpu/xenon` status.

For a PR from anyone who isn't a maintainer, the maintainer first reads the full diff,
including the lockfile and the tests, and only then runs the check with `--reviewed`. The
result is posted as a `gpu/xenon` commit status and one PR comment listing the environment
and any failing tests. So expect a GPU-relevant PR to wait for review before its GPU result
appears.

Mark any GPU test that asserts on timings `@pytest.mark.perf`. Those run only with `--perf`,
on an idle GPU; every other `gpu` test must pass while other programs use the card. You can
run `uv run pytest -m gpu` on your own GPU first. Details are in
[`docs/gpu-check.md`](docs/gpu-check.md), and the reasoning is in ADR-0010.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`,
  `refactor:`, `docs:`, `test:`, `chore:`, `ci:`). The `commit-msg` hook checks this.
- PRs are squash-merged, so **the PR title becomes the commit message on `main`**. CI checks
  that it's a valid Conventional Commit.
- Fill in the PR template checklist.
- Don't update golden images without the maintainer's explicit approval.
- Don't bump the version. Releases are cut with Commitizen from the release workflow.

## Reporting bugs and security issues

- Bugs: use the bug report template and include the output of `brotkit doctor`.
- Security issues: please don't open a public issue. See [SECURITY.md](SECURITY.md).

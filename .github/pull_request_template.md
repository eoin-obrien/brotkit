<!-- The PR title becomes the squash-merge commit on main: make it a Conventional Commit (feat:, fix:, docs:, ...). -->

## What and why

<!-- What this changes, and the backlog item / issue it closes. -->

Closes #

## Checklist

- [ ] PR title is a Conventional Commit
- [ ] `uv run pre-commit run --all-files` passes (ruff, format, pyright strict)
- [ ] `uv run pytest -m "not gpu"` passes; new behaviour has tests (Hypothesis for value objects and serialisation)
- [ ] GPU-affecting changes: `uv run pytest -m gpu` run on a GPU machine, or flagged for the nightly suite
- [ ] Architecture invariants in `AGENTS.md` respected
- [ ] Any deviation from the spec has an ADR in `docs/design/decisions.md`, and the spec is updated to match
- [ ] Golden images unchanged, or the update was explicitly approved
- [ ] User-facing docs and README updated where behaviour changed

<!-- The PR title becomes the squash-merge commit on main: make it a Conventional Commit (feat:, fix:, docs:, ...). -->

## What and why

<!-- What this changes, and the backlog item / issue it addresses.
     "Closes #n" only if merging this PR by itself meets the issue's acceptance criteria.
     If acceptance needs something after the merge (a release, a published package, a
     nightly run), write "Refs #n" and close the issue by hand with a link to the evidence. -->

Closes #

## Checklist

- [ ] PR title is a Conventional Commit
- [ ] `uv run pre-commit run --all-files` passes (ruff, format, pyright strict)
- [ ] `uv run pytest -m "not gpu"` passes; new behaviour has tests (Hypothesis for value objects and serialisation)
- [ ] GPU-relevant changes: the maintainer has run `uv run tools/gpu_check.py <PR>` (the `gpu-gate` check says whether it's needed)
- [ ] Architecture invariants in `AGENTS.md` respected
- [ ] Any deviation from the spec has an ADR in `docs/design/decisions.md`, and the spec is updated to match
- [ ] Golden images unchanged, or the update was explicitly approved
- [ ] User-facing docs and README updated where behaviour changed
- [ ] `Closes #n` only if merging alone meets the acceptance criteria; otherwise `Refs #n`
- [ ] Workflow changes: `uv run zizmor .github/workflows` is clean

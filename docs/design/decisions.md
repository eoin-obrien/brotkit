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

---
name: cut-release
description: Cut a brotkit release with Commitizen through the manually triggered release workflow (cz bump, changelog, tag, uv build, TestPyPI, PyPI via Trusted Publishing). Use when asked to release, bump the version or publish.
---

# Cut a release

Releases run only from `.github/workflows/release.yml` on `main`. Never bump `pyproject.toml`,
tag or publish by hand, and never publish with a token. A release is outward-facing and can't
be undone (PyPI never accepts the same version twice), so **confirm with the maintainer
before triggering a run**.

## Before the run

1. `main` is green (CI, and the nightly GPU suite once it exists) and the phase gate, if any,
   passes.
2. Preview the bump locally: `uv run cz bump --dry-run --yes`. `NO_COMMITS_TO_BUMP` means
   nothing since the last tag is a `feat`, `fix`, `perf`, `refactor` or breaking change.
   In that case, either don't release or force an increment (below). 0.x:
   `major_version_zero` is on, so a breaking change bumps MINOR.

## Run

```bash
gh workflow run release.yml --ref main -f increment=auto   # or PATCH | MINOR | MAJOR
gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

`increment=auto` derives the bump from the commits. A forced increment also passes
`--allow-no-commit`. The workflow fails if the version didn't change.

Job order: **bump** (CPU tests, `cz bump`, `uv build --no-sources`, wheel smoke test, atomic
push of the bump commit and `vX.Y.Z` tag) → **testpypi** → **testpypi-install** (installs
the exact version from TestPyPI) → **pypi**, which waits for the maintainer to approve the
protected `pypi` environment.

## After

- Check `https://pypi.org/project/brotkit/` shows the version with attestations ("Verified
  details" / provenance), and that `uv pip install brotkit==X.Y.Z` works.
- If a publish job failed after the tag was pushed, fix the cause and use **Re-run failed
  jobs**. The `dist` artifact is reused, so the same files are uploaded. Don't re-run the
  whole workflow, because that would bump again.

## Trusted Publisher settings

These are bound to the workflow file name and environment names. Renaming either breaks
publishing. See ADR-0004.

| Index | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- |
| PyPI | `eoin-obrien` | `brotkit` | `release.yml` | `pypi` |
| TestPyPI | `eoin-obrien` | `brotkit` | `release.yml` | `testpypi` |

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

1. `main` is green (CI), the release commit has a `gpu/xenon` success status (the maintainer
   runs `uv run tools/gpu_check.py <sha>`; agents don't), and the phase gate, if any, passes.
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
protected `pypi` environment → **pypi-install** (installs the exact version from PyPI). A
green run therefore proves the release installs from PyPI.

## After

- Check `https://pypi.org/project/brotkit/` shows the version with attestations ("Verified
  details" / provenance). The `pypi-install` job has already checked that it installs.
- If a publish job failed after the tag was pushed because of a transient problem (an index
  outage, a rejected approval), use **Re-run failed jobs**. The `dist` artifact is reused,
  so the same files are uploaded. A re-run uses the workflow file **from the original
  commit**, so it can't pick up a fix to the workflow. If the workflow itself was broken,
  fix it on `main` and cut the next version; the stranded tag stays TestPyPI-only
  (ADR-0004 update). Don't re-run the whole workflow, because that would bump again.

## Trusted Publisher settings

These are bound to the workflow file name and environment names. Renaming either breaks
publishing. See ADR-0004.

| Index | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- |
| PyPI | `eoin-obrien` | `brotkit` | `release.yml` | `pypi` |
| TestPyPI | `eoin-obrien` | `brotkit` | `release.yml` | `testpypi` |

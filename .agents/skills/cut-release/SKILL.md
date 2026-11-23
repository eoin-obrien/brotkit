---
name: cut-release
description: Cut a brotkit release with Commitizen through the manually triggered release workflow (cz bump, changelog, tag, uv build, TestPyPI, PyPI via Trusted Publishing). Use when asked to release, bump the version or publish.
---

# Cut a release

Stub. The release workflow arrives in backlog 0.5.

1. Confirm `main` is green (CI and the nightly GPU suite) and the phase gate, if any, passes.
2. Preview the bump: `uv run cz bump --dry-run` (0.x: `major_version_zero` is on).
3. Trigger the release workflow from GitHub Actions. Never bump `pyproject.toml` or tag by
   hand, and never publish with a local token.
4. Check the TestPyPI upload installs, then approve the protected `pypi` environment.

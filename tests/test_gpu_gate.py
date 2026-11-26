"""The gpu-gate classifier and decision (tools/gpu_gate.py, ADR-0010)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import gpu_gate as g

CONFIG = g.load_config()
HEAD = "b" * 40
OLD = "a" * 40
LISTED = [
    "numba-cuda-mlir", "numba", "llvmlite", "cuda-core", "cuda-bindings", "cuda-python",
    "cuda-pathfinder", "nvidia-nvjitlink", "nvidia-nvvm", "numpy", "sympy", "mpmath", "gmpy2",
]  # fmt: skip


def files(mapping: Mapping[str, str | None]) -> g.ReadFile:
    def read(path: str) -> bytes | None:
        value = mapping.get(path)
        return None if value is None else value.encode()

    return read


def classify(
    changed: list[str], base: Mapping[str, str | None], head: Mapping[str, str | None]
) -> g.Classification:
    return g.classify(changed, files(base), files(head), CONFIG)


# --- config -------------------------------------------------------------------------------------


def test_repo_config_loads() -> None:
    assert CONFIG.context == "gpu/xenon"
    assert "eoin-obrien" in CONFIG.maintainers
    assert all(g.is_gpu_package(name, CONFIG) for name in LISTED)
    assert g.is_gpu_package("Numba_CUDA.MLIR", CONFIG)  # PEP 503 normalised
    assert not any(g.is_gpu_package(n, CONFIG) for n in ("uv-build", "pytest", "numbafoo"))


@pytest.mark.parametrize(
    "data",
    [{}, {"context": "x", "maintainers": "eoin", "paths": {}, "dependencies": {}}],
)
def test_bad_config_is_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(g.ConfigError):
        g.parse_config(data)


# --- paths --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "src/brotkit/backends/cuda.py",
        "src/brotkit/backends/chunks.py",  # the planner
        "src/brotkit/codegen/checks.py",
        "src/brotkit/kernels/cache.py",
        "src/brotkit/codegen/new/deep/module.py",
        ".github/workflows/gpu-gate.yml",
        "tools/gpu_gate.py",
        "tools/gpu_gate_paths.toml",
        "tools/gpu_check.py",
        "tools/sandbox/Dockerfile",
        "tests/conftest.py",
        "tests/fixtures/codegen/s4_f32_A_plain.py",
    ],
)
def test_gpu_paths_need_the_check(path: str) -> None:
    c = classify([path], {}, {})
    assert c.relevant
    assert path in c.reasons[0]


@pytest.mark.parametrize(
    "path",
    ["README.md", "docs/gpu-check.md", "src/brotkit/__init__.py", ".github/workflows/ci.yml"],
)
def test_other_paths_dont(path: str) -> None:
    c = classify([path], {path: "x"}, {path: "y"})
    assert c == g.Classification(relevant=False, reasons=())


def test_test_marked_gpu_at_head_needs_the_check() -> None:
    p = "tests/test_new.py"
    assert classify([p], {}, {p: "pytestmark = pytest.mark.gpu\n"}).relevant


def test_removing_the_gpu_marker_needs_the_check() -> None:
    p = "tests/test_cuda_ptx.py"
    assert classify([p], {p: "pytestmark = pytest.mark.gpu\n"}, {p: "pass\n"}).relevant


def test_unmarked_test_doesnt() -> None:
    p = "tests/test_chunks.py"
    assert not classify([p], {p: "def test_a(): ...\n"}, {p: "def test_b(): ...\n"}).relevant


# --- uv.lock --------------------------------------------------------------------------------------

ROOT = """\
[[package]]
name = "brotkit"
version = "0.0.2"
source = {{ editable = "." }}
dependencies = [{{ name = "filelock" }}]

[package.optional-dependencies]
cuda = [{{ name = "numba-cuda-mlir", extra = ["cu13"] }}]

[package.dev-dependencies]
dev = [{dev}]

[package.metadata]
requires-dist = [
    {{ name = "filelock", specifier = ">=3.20" }},
{{name = "numba-cuda-mlir", extras = ["cu13"], marker = "extra == 'cuda'", specifier = "{pin}"}},
]
provides-extras = ["cuda"]
"""


def pkg(name: str, version: str) -> str:
    return (
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )


def lock(
    packages: Mapping[str, str],
    *,
    requires_python: str = ">=3.14",
    dev: str = '{ name = "pytest" }',
    pin: str = "==0.6.0",
    extra_top: str = "",
) -> str:
    body = "\n".join(pkg(n, v) for n, v in sorted(packages.items()))
    return (
        f'version = 1\nrevision = 5\nrequires-python = "{requires_python}"\n{extra_top}\n'
        + ROOT.format(dev=dev, pin=pin)
        + "\n"
        + body
    )


BASE_PKGS = {"filelock": "3.20.0", "numba-cuda-mlir": "0.6.0", "numpy": "2.5.0", "pytest": "9.2"}


def lock_change(base: str, head: str) -> g.Classification:
    return classify(["uv.lock"], {"uv.lock": base}, {"uv.lock": head})


@pytest.mark.parametrize("name", LISTED)
def test_bumping_a_gpu_stack_package_needs_the_check(name: str) -> None:
    base = dict(BASE_PKGS) | {name: "1.0.0"}
    c = lock_change(lock(base), lock(base | {name: "1.0.1"}))
    assert c.relevant, name
    assert name in c.reasons[0]


@pytest.mark.parametrize("name", LISTED)
def test_adding_a_gpu_stack_package_needs_the_check(name: str) -> None:
    base = {k: v for k, v in BASE_PKGS.items() if k != name}
    assert lock_change(lock(base), lock(base | {name: "1.0.0"})).relevant


def test_dev_tool_bump_doesnt() -> None:
    c = lock_change(lock(BASE_PKGS), lock(BASE_PKGS | {"pytest": "9.3", "ruff": "0.18"}))
    assert c == g.Classification(relevant=False, reasons=())


def test_adding_a_dev_tool_to_the_project_doesnt() -> None:
    head = lock(
        BASE_PKGS | {"pytest-randomly": "5.1.0"},
        dev='{ name = "pytest" }, { name = "pytest-randomly" }',
    )
    assert not lock_change(lock(BASE_PKGS), head).relevant


def test_the_real_lockfile_against_itself_is_clean() -> None:
    text = (Path(__file__).parents[1] / "uv.lock").read_text()
    assert not lock_change(text, text).relevant


@pytest.mark.parametrize(
    ("base", "head", "why"),
    [
        (lock(BASE_PKGS), lock(BASE_PKGS, requires_python=">=3.15"), "requires-python"),
        (lock(BASE_PKGS), lock(BASE_PKGS, pin="==0.7.0"), "cuda"),
        (lock(BASE_PKGS), "not = [toml", "could not parse"),
        (lock(BASE_PKGS), lock(BASE_PKGS, extra_top='surprise = "x"'), "unrecognised"),
        (lock(BASE_PKGS), lock(BASE_PKGS) + '[[package]]\nversion = "1"\n', "unrecognised"),
        (lock(BASE_PKGS), lock(BASE_PKGS).replace("revision = 5", "revision = 6"), "revision"),
        (
            lock(BASE_PKGS),
            lock(BASE_PKGS) + pkg("zz", "1").replace("source", "novel = 1\nsource"),
            "unrecognised",
        ),
    ],
)
def test_lockfile_fails_safe(base: str, head: str, why: str) -> None:
    c = lock_change(base, head)
    assert c.relevant
    assert why in " ".join(c.reasons)


def test_lockfile_removed_fails_safe() -> None:
    assert classify(["uv.lock"], {"uv.lock": lock(BASE_PKGS)}, {}).relevant


# --- pyproject.toml -------------------------------------------------------------------------------

PYPROJECT = """\
[build-system]
requires = ["{uv_build}"]
build-backend = "uv_build"

[project]
name = "brotkit"
version = "0.0.2"
requires-python = "{rp}"
dependencies = ["filelock>=3.20"]

[project.optional-dependencies]
cuda = ["numba-cuda-mlir[cu13]=={pin}"]
video = ["av"]

[dependency-groups]
dev = [{dev}]

[tool.ruff]
line-length = {ll}
{extra}
"""


def pyproject(
    *,
    uv_build: str = "uv_build>=0.9.10,<0.10.0",
    rp: str = ">=3.14",
    pin: str = "0.6.0",
    dev: str = '"pytest>=9.2", "numpy>=2.5"',
    ll: int = 100,
    extra: str = "",
) -> str:
    return PYPROJECT.format(uv_build=uv_build, rp=rp, pin=pin, dev=dev, ll=ll, extra=extra)


def py_change(head: str, base: str | None = None) -> g.Classification:
    b = pyproject() if base is None else base
    return classify(["pyproject.toml"], {"pyproject.toml": b}, {"pyproject.toml": head})


def test_uv_build_bump_like_pr_61_doesnt() -> None:
    c = py_change(pyproject(uv_build="uv_build>=0.9.10,<0.14.0"))
    assert c == g.Classification(relevant=False, reasons=())


@pytest.mark.parametrize(
    "head",
    [
        pyproject(ll=120),
        pyproject(dev='"pytest>=9.3", "numpy>=2.5"'),
        pyproject(extra='[tool.commitizen]\nname = "cz"\n'),
    ],
)
def test_safe_pyproject_changes_dont(head: str) -> None:
    assert not py_change(head).relevant


@pytest.mark.parametrize(
    ("head", "why"),
    [
        (pyproject(rp=">=3.15"), "requires-python"),
        (pyproject(pin="0.7.0"), "[cuda] extra"),
        (pyproject(dev='"pytest>=9.2", "numpy>=2.6"'), "numpy"),
        (pyproject(dev='"pytest>=9.2", "numpy>=2.5", "sympy>=1.14"'), "sympy"),
        (pyproject(extra='[tool.uv]\nindex-url = "https://example.com"\n'), "tool.uv"),
        (pyproject(extra='[tool.pytest.ini_options]\naddopts = ["-x"]\n'), "tool.pytest"),
        (pyproject(dev='"pytest>=9.2", "numpy>=2.5", { include-group = "x" }'), "not understood"),
        (pyproject(extra="[something-new]\nx = 1\n"), "something-new"),
        ("[project\n", "could not parse"),
    ],
)
def test_gpu_relevant_pyproject_changes_do(head: str, why: str) -> None:
    c = py_change(head)
    assert c.relevant
    assert why in " ".join(c.reasons)


def test_the_real_pyproject_against_itself_is_clean() -> None:
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert not py_change(text, text).relevant


# --- decision -------------------------------------------------------------------------------------

RELEVANT = g.Classification(relevant=True, reasons=("`src/brotkit/backends/cuda.py`: GPU path",))


def status(
    state: str,
    *,
    sha: str = HEAD,
    login: str = "eoin-obrien",
    at: str = "2026-11-26T10:00:00Z",
    context: str = "gpu/xenon",
    id_: int = 1,
) -> dict[str, Any]:
    return {
        "url": f"https://api.github.com/repos/eoin-obrien/brotkit/statuses/{sha}",
        "id": id_,
        "state": state,
        "context": context,
        "description": f"{state} desc",
        "created_at": at,
        "creator": {"login": login},
    }


def decide(statuses: list[dict[str, Any]], *, head: str = HEAD, c: g.Classification = RELEVANT):
    return g.decide(c, statuses, head_sha=head, pr=12, config=CONFIG)


def test_not_relevant_passes_without_a_status() -> None:
    d = decide([], c=g.Classification(relevant=False, reasons=()))
    assert d.passed
    assert "GPU check not required" in d.summary


def test_missing_status_fails_and_says_what_to_run() -> None:
    d = decide([])
    assert not d.passed
    assert "uv run tools/gpu_check.py 12" in d.summary
    assert "backends/cuda.py" in d.summary


@pytest.mark.parametrize("state", ["pending", "failure", "error"])
def test_non_success_fails(state: str) -> None:
    assert not decide([status(state)]).passed


def test_success_on_the_head_passes() -> None:
    d = decide([status("success")])
    assert d.passed
    assert "success desc" in d.summary


def test_success_on_an_old_sha_fails() -> None:
    assert not decide([status("success", sha=OLD)]).passed


def test_new_push_invalidates_the_old_status() -> None:
    statuses = [status("success", sha=OLD)]
    assert decide(statuses, head=OLD).passed
    assert not decide(statuses, head=HEAD).passed


def test_success_from_a_non_maintainer_is_ignored() -> None:
    assert not decide([status("success", login="mallory")]).passed


def test_other_contexts_are_ignored() -> None:
    assert not decide([status("success", context="gpu/elsewhere")]).passed


def test_newest_status_wins() -> None:
    older_ok = status("success", at="2026-11-26T10:00:00Z", id_=1)
    newer_bad = status("failure", at="2026-11-26T11:00:00Z", id_=2)
    assert not decide([older_ok, newer_bad]).passed
    older_bad = status("failure", at="2026-11-26T10:00:00Z", id_=1)
    newer_ok = status("success", at="2026-11-26T11:00:00Z", id_=2)
    assert decide([newer_ok, older_bad]).passed


def test_statuses_load_from_jsonl_and_arrays() -> None:
    items = [status("success"), status("failure", id_=2)]
    assert g.load_statuses("\n".join(json.dumps(s) for s in items)) == items
    assert g.load_statuses(json.dumps(items)) == items
    assert g.load_statuses("") == []


# --- CLI, against a real git repo -----------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


def commit(repo: Path, files_: Mapping[str, str]) -> str:
    for name, text in files_.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "x")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    return r


def run_cli(
    repo: Path, base: str, head: str, statuses: list[dict[str, Any]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, str]:  # fmt: skip
    sfile = tmp_path / "statuses.jsonl"
    sfile.write_text("\n".join(json.dumps(s) for s in statuses))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    rc = g.main(
        ["--repo-dir", str(repo), "--base", base, "--head", head, "--pr", "5",
         "--statuses", str(sfile)]
    )  # fmt: skip
    return rc, summary.read_text()


def test_cli_docs_only_change_passes(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = commit(repo, {"README.md": "a\n", "src/brotkit/backends/cuda.py": "x = 1\n"})
    head = commit(repo, {"README.md": "b\n"})
    rc, summary = run_cli(repo, base, head, [], tmp_path, monkeypatch)
    assert rc == 0
    assert "GPU check not required" in summary


def test_cli_gpu_change_needs_the_status_on_its_head(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = commit(repo, {"README.md": "a\n", "src/brotkit/backends/cuda.py": "x = 1\n"})
    first = commit(repo, {"src/brotkit/backends/cuda.py": "x = 2\n"})
    assert run_cli(repo, base, first, [], tmp_path, monkeypatch)[0] == 1
    ok = [status("success", sha=first)]
    assert run_cli(repo, base, first, ok, tmp_path, monkeypatch)[0] == 0
    second = commit(repo, {"src/brotkit/backends/cuda.py": "x = 3\n"})  # a new push
    rc, summary = run_cli(repo, base, second, ok, tmp_path, monkeypatch)
    assert rc == 1
    assert "uv run tools/gpu_check.py 5" in summary


def test_cli_uses_the_merge_base(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = commit(repo, {"README.md": "a\n", "src/brotkit/backends/cuda.py": "x = 1\n"})
    git(repo, "switch", "-q", "-c", "pr")
    head = commit(repo, {"README.md": "b\n"})
    git(repo, "switch", "-q", "main")
    base = commit(repo, {"src/brotkit/backends/cuda.py": "x = 2\n"})  # main moved on
    assert root != base
    assert run_cli(repo, base, head, [], tmp_path, monkeypatch)[0] == 0

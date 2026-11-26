"""The local GPU check (tools/gpu_check.py, ADR-0010): payloads, refusals, and --dry-run end to
end with gh, uv, nvidia-smi and the test venv faked. git is real."""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import gpu_check as c

REPO = "eoin-obrien/brotkit"
GPU = c.Gpu(name="NVIDIA GeForce RTX 3090 Ti", cc="8.6", driver="610.57")


def junit_xml(passed: int = 3, failed: Sequence[str] = (), skipped: int = 1) -> bytes:
    cases = [f'<testcase classname="tests.test_a" name="test_ok_{i}"/>' for i in range(passed)]
    cases += [
        f'<testcase classname="tests.test_a" name="{n}"><failure message="boom"/></testcase>'
        for n in failed
    ]
    cases += [
        f'<testcase classname="tests.test_a" name="test_skip_{i}"><skipped/></testcase>'
        for i in range(skipped)
    ]
    return (
        '<testsuites><testsuite name="pytest">' + "".join(cases) + "</testsuite></testsuites>"
    ).encode()


def result(python: str, *, passed: int = 40, failed: Sequence[str] = ()) -> c.PyResult:
    return c.PyResult(
        python=python, version=f"{python}.2", packages={p: "1.0" for p in c.PACKAGES}, seed=7,
        junit=c.parse_junit(junit_xml(passed, failed)), returncode=1 if failed else 0,
    )  # fmt: skip


# --- JUnit ----------------------------------------------------------------------------------------


def test_junit_counts_and_failed_names() -> None:
    j = c.parse_junit(junit_xml(passed=3, failed=["test_bad"], skipped=2))
    assert (j.passed, j.failed, j.skipped) == (3, 1, 2)
    assert j.failed_names == ("tests.test_a::test_bad",)
    assert len(j.sha256) == 64


@pytest.mark.parametrize(
    "data",
    [
        b'<!DOCTYPE x [<!ENTITY a "aaaa">]><testsuite>&a;</testsuite>',
        b'<?xml version="1.0"?><!doctype x><testsuite/>',
    ],
)
def test_junit_with_a_dtd_is_rejected(data: bytes) -> None:
    with pytest.raises(ValueError, match="DTD"):
        c.parse_junit(data)


def test_junit_too_large_is_rejected() -> None:
    with pytest.raises(ValueError, match="too large"):
        c.parse_junit(b" " * (c.MAX_JUNIT_BYTES + 1))


def test_a_run_with_no_passing_tests_is_not_ok() -> None:
    r = c.PyResult(python="3.14", junit=c.parse_junit(junit_xml(passed=0, skipped=5)))
    assert not r.ok


# --- description ----------------------------------------------------------------------------------


def test_description_shows_both_versions_and_the_gpu() -> None:
    d = c.build_description([result("3.14"), result("3.15")], GPU, partial=False)
    assert d == (
        "3.14: 40 passed, 0 failed · 3.15: 40 passed, 0 failed · RTX 3090 Ti sm_86 · driver 610.57"
    )


def test_partial_description_says_so() -> None:
    d = c.build_description([result("3.15")], GPU, partial=True)
    assert d.startswith("partial: 3.15 only · 3.15: 40 passed")


def test_setup_failure_is_described() -> None:
    r = c.PyResult(python="3.15", setup_error="uv sync failed", returncode=2)
    assert "3.15: setup failed" in c.build_description([result("3.14"), r], GPU, partial=False)


@given(
    name=st.text(min_size=1, max_size=300),
    driver=st.text(min_size=1, max_size=80),
    counts=st.lists(st.integers(0, 10**9), min_size=2, max_size=2),
    partial=st.booleans(),
)
def test_description_never_exceeds_140_chars(
    name: str, driver: str, counts: list[int], partial: bool
) -> None:
    gpu = c.Gpu(name=name, cc="12.0", driver=driver)
    rs = [
        c.PyResult(
            python=py,
            junit=c.JUnit(passed=n, failed=n // 3, skipped=0, failed_names=(), sha256="0"),
        )
        for py, n in zip(c.PYTHONS, counts, strict=True)
    ]
    assert len(c.build_description(rs, gpu, partial=partial)) <= c.DESCRIPTION_LIMIT


# --- scrubbing ------------------------------------------------------------------------------------

SCRUB = c.Scrubber(
    literals=(("/tmp/brotkit-gpu-check-abc", "<tmp>"), ("/home/qqzz", "~")), user="qqzz"
)
SECRETS = [
    "/home/qqzz/.config/gh/hosts.yml",
    "/home/someoneelse/x",
    "/Users/qqzz/x",
    "/mnt/c/Users/QQZZ/AppData",
    "C:\\Users\\qqzz\\AppData",
    "ghp_" + "A" * 36,
    "github_pat_" + "B" * 40,
    "GH_TOKEN=abc123",
    "BROTKIT_CACHE_DIR=/x",
    "/tmp/brotkit-gpu-check-abc/wt/tests",
    "qqzz",
]


@pytest.mark.parametrize("secret", SECRETS)
def test_scrubber_removes(secret: str) -> None:
    out = SCRUB(f"failed at {secret} here")
    assert secret not in out
    assert "qqzz" not in out.lower()
    assert not SCRUB.leaks(out)


def test_scrubber_keeps_ordinary_text() -> None:
    text = "tests/test_cuda_ptx.py::test_fma[dd-sm_86] · driver 610.57 · eoin-obrien/brotkit"
    assert SCRUB(text) == text


@given(
    prefix=st.text(alphabet="abcdefghij /:._-\n", max_size=40),
    suffix=st.text(alphabet="abcdefghij /:._-\n", max_size=40),
    secret=st.sampled_from(SECRETS),
)
def test_scrubbed_text_never_contains_a_secret(prefix: str, suffix: str, secret: str) -> None:
    out = SCRUB(f"{prefix} {secret} {suffix}")
    assert "qqzz" not in out.lower()
    assert "ghp_" not in out
    assert "github_pat_" not in out
    assert "=abc123" not in out


def test_comment_has_the_marker_environment_and_failures() -> None:
    target = c.Target(sha="c" * 40, pr=9, author="eoin-obrien", fetch_ref="refs/pull/9/head")
    rs = [result("3.14", failed=["test_fma"]), result("3.15")]
    body = c.build_comment(
        target=target, results=rs, gpu=GPU, state="failure", description="d",
        env_names=["HOME", "PATH", "UV_CACHE_DIR"], reviewed=False, perf=False,
        context="gpu/xenon",
    )  # fmt: skip
    assert body.startswith(c.MARKER)
    assert "tests.test_a::test_fma" in body
    assert rs[0].junit is not None
    assert rs[0].junit.sha256 in body
    assert "| numba-cuda-mlir | 1.0 | 1.0 |" in body
    assert "`HOME`, `PATH`, `UV_CACHE_DIR`" in body
    assert "author is a maintainer" in body
    assert "correctness tests only" in body


# --- environment ----------------------------------------------------------------------------------


def test_scrubbed_env_is_an_allowlist(tmp_path: Path) -> None:
    shell = {
        "PATH": "/usr/local/cuda-12.9/bin:/home/x/.local/bin:/usr/bin",
        "HOME": "/home/x",
        "GH_TOKEN": "ghp_x",
        "BROTKIT_CACHE_DIR": "/x",
        "NUMBA_OPT": "0",
        "NUMBA_CUDA_DEFAULT_PTX_CC": "8.0",
        "CUDA_HOME": "/usr/local/cuda-12.9",
        "CUDA_VISIBLE_DEVICES": "0",
        "SSH_AUTH_SOCK": "/tmp/ssh",
        "UV_INDEX_STRATEGY": "first-index",
    }
    env = c.scrubbed_env(
        shell, home=tmp_path / "home", tmp=tmp_path / "tmp", extra_path=[Path("/opt/uv")],
        uv_cache="/cache/uv", uv_python="/py",
    )  # fmt: skip
    assert set(env) == {
        "PATH", "HOME", "TMPDIR", "LANG", "CUDA_VISIBLE_DEVICES", "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR", "UV_NO_CONFIG", "UV_INDEX_STRATEGY",
    }  # fmt: skip
    assert env["HOME"] == str(tmp_path / "home")
    assert "cuda" not in env["PATH"]
    assert env["PATH"].startswith("/opt/uv:")


def test_ld_library_path_is_only_the_wsl_driver_dir(tmp_path: Path) -> None:
    shell = {
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib/",
        "NUMBA_CUDA_DRIVER": "x",
    }
    kw: dict[str, Any] = dict(
        home=tmp_path, tmp=tmp_path, extra_path=[], uv_cache="/c", uv_python="/p"
    )
    assert "LD_LIBRARY_PATH" not in c.scrubbed_env(shell, **kw)
    env = c.scrubbed_env(shell, **kw, driver_dir=Path("/usr/lib/wsl/lib"))
    assert env["LD_LIBRARY_PATH"] == "/usr/lib/wsl/lib"
    assert "NUMBA_CUDA_DRIVER" not in env


# --- busy check -----------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.t += s
        self.sleeps.append(s)

    def now(self) -> float:
        return self.t


def samples(values: Sequence[tuple[float, bool]]) -> Iterator[tuple[float, bool]]:
    return itertools.chain(values, itertools.repeat(values[-1]))


def test_idle_gpu_starts_at_once() -> None:
    clock, it = Clock(), samples([(0.0, False)])
    c.wait_for_idle(lambda: next(it), sleep=clock.sleep, clock=clock.now, max_wait_s=60, out=print)
    assert c.BUSY_RETRY_S not in clock.sleeps


def test_a_single_burst_doesnt_count() -> None:
    clock, it = Clock(), samples([(40.0, False), *[(1.0, False)] * 7])
    c.wait_for_idle(lambda: next(it), sleep=clock.sleep, clock=clock.now, max_wait_s=60, out=print)
    assert c.BUSY_RETRY_S not in clock.sleeps


def test_a_busy_gpu_is_waited_out() -> None:
    clock, it = Clock(), samples([*[(50.0, False)] * 8, (0.0, False)])
    msgs: list[str] = []
    c.wait_for_idle(
        lambda: next(it), sleep=clock.sleep, clock=clock.now, max_wait_s=60, out=msgs.append
    )
    assert clock.sleeps.count(c.BUSY_RETRY_S) == 1
    assert "median 50%" in msgs[0]


def test_a_compute_process_counts_as_busy() -> None:
    clock, it = Clock(), samples([(0.0, True)])
    with pytest.raises(c.Refusal, match="still busy"):
        c.wait_for_idle(
            lambda: next(it), sleep=clock.sleep, clock=clock.now, max_wait_s=60, out=print
        )


# --- authors --------------------------------------------------------------------------------------


def test_maintainer_runs_without_review() -> None:
    c.check_author(True, author="eoin-obrien", reviewed=False)


def test_non_maintainer_without_reviewed_is_refused() -> None:
    with pytest.raises(c.Refusal, match="--reviewed"):
        c.check_author(False, author="mallory", reviewed=False)


def test_reviewed_non_maintainer_runs() -> None:
    c.check_author(False, author="dependabot[bot]", reviewed=True)


# --- fake processes -------------------------------------------------------------------------------


def done(args: Sequence[str], out: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(args), rc, out, "")


class Fake:
    """Stands in for gh, uv, nvidia-smi and the venv's python; passes git through."""

    def __init__(self, sha: str, *, author: str = "eoin-obrien", role: str = "admin") -> None:
        self.heads: Iterator[str] = itertools.repeat(sha)
        self.author = author
        self.role = role
        self.calls: list[list[str]] = []
        self.envs: list[Mapping[str, str]] = []
        self.inputs: list[str] = []
        self.comments: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.junit = junit_xml()
        self.util = 1.0

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        input: str | None = None,
        check: bool = True,
        echo: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        a = list(args)
        if a[0] == "git":
            return c.run(a, cwd=cwd, input=input, check=check)
        self.calls.append(a)
        if env is not None:
            self.envs.append(env)
        if input is not None:
            self.inputs.append(input)
        r = self.respond(a)
        if check and r.returncode:
            raise subprocess.CalledProcessError(r.returncode, a, r.stdout)
        return r

    def respond(self, a: list[str]) -> subprocess.CompletedProcess[str]:
        joined = " ".join(a)
        name = Path(a[0]).name
        if a[:3] == ["gh", "api", "user"]:
            return done(a, "eoin-obrien\n")
        if a[:3] == ["gh", "repo", "view"]:
            return done(a, json.dumps({"nameWithOwner": REPO}))
        if m := re.fullmatch(rf"gh api repos/{REPO}/pulls/(\d+)", joined):
            pr = {"state": "open", "number": int(m[1]), "user": {"login": self.author},
                  "head": {"sha": next(self.heads)}}  # fmt: skip
            return done(a, json.dumps(pr))
        if "/permission" in joined:
            return done(a, json.dumps({"role_name": self.role}))
        if "/comments --jq" in joined:
            return done(a, "\n".join(json.dumps(x) for x in self.comments))
        if a[:4] in (["gh", "api", "-X", "POST"], ["gh", "api", "-X", "PATCH"]):
            return done(a, json.dumps({"html_url": "https://github.com/x/y/pull/7#issuecomment-1"}))
        if a[:3] == ["gh", "run", "list"]:
            return done(a, json.dumps(self.runs))
        if a[:3] == ["gh", "run", "rerun"]:
            return done(a)
        if name == "uv" and a[1:3] == ["cache", "dir"]:
            return done(a, "/fake/uv-cache\n")
        if name == "uv" and a[1:3] == ["python", "dir"]:
            return done(a, "/fake/uv-python\n")
        if name == "uv" and a[1] == "sync":
            return done(a)
        if name == "nvidia-smi":
            if "--query-gpu=name,compute_cap,driver_version" in a:
                return done(a, f"{GPU.name}, {GPU.cc}, {GPU.driver}\n")
            if "--query-gpu=utilization.gpu" in a:
                return done(a, f"{self.util:g}\n")
            return done(a, "")
        if name == "python" and "-c" in a:
            pk = {p: "1.0" for p in c.PACKAGES} | {c.RANDOMLY: "5.1.0"}
            return done(a, json.dumps({"python": "3.14.0", "packages": pk}) + "\n")
        if name == "python" and "pytest" in a:
            junit = next(x for x in a if x.startswith("--junitxml="))
            Path(junit.removeprefix("--junitxml=")).write_bytes(self.junit)
            return done(a, "Using --randomly-seed=7\n")
        raise AssertionError(f"unexpected command {a}")

    def posted(self) -> list[list[str]]:
        return [
            x
            for x in self.calls
            if x[:4] in (["gh", "api", "-X", "POST"], ["gh", "api", "-X", "PATCH"])
        ]


# --- git fixtures ---------------------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


@pytest.fixture
def origin(tmp_path: Path) -> tuple[Path, Path, str]:
    """(origin repo with refs/pull/7/head, local clone acting as the repo root, PR head SHA)."""
    src = tmp_path / "origin"
    src.mkdir()
    git(src, "init", "-q", "-b", "main")
    (src / "tools").mkdir()
    shutil.copy(Path(c.__file__).with_name("gpu_gate_paths.toml"), src / "tools")
    (src / "README.md").write_text("x\n")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "base")
    (src / "README.md").write_text("y\n")
    git(src, "commit", "-q", "-am", "pr")
    sha = git(src, "rev-parse", "HEAD")
    git(src, "update-ref", "refs/pull/7/head", sha)
    git(src, "reset", "-q", "--hard", "HEAD~1")
    root = tmp_path / "root"
    subprocess.run(["git", "clone", "-q", str(src), str(root)], check=True)
    return src, root, sha


def run_main(
    fake: Fake, root: Path, src: Path, tmp_path: Path, *extra: str, clock: Clock | None = None
) -> tuple[int, list[str]]:
    msgs: list[str] = []
    shell = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "fakehome"),
        "BROTKIT_CACHE_DIR": "/nope",
        "NUMBA_OPT": "0",
        "GH_TOKEN": "ghp_" + "x" * 36,
    }
    rc = c.main(
        ["7", "--repo", REPO, "--remote-url", str(src), "--seed", "7", *extra],
        runner=fake, sleep=clock.sleep if clock else lambda _: None,
        clock=clock.now if clock else lambda: 0.0, shell_env=shell, repo_root=root,
        out=msgs.append,
    )  # fmt: skip
    return rc, msgs


# --- worktree refusals ----------------------------------------------------------------------------


def target_for(sha: str) -> c.Target:
    return c.Target(sha=sha, pr=7, author="eoin-obrien", fetch_ref="refs/pull/7/head")


def test_reused_worktree_path_is_refused(origin: tuple[Path, Path, str], tmp_path: Path) -> None:
    src, root, sha = origin
    path = tmp_path / "wt"
    path.mkdir()
    with pytest.raises(c.Refusal, match="already exists"):
        c.prepare_worktree(
            c.run, repo_root=root, remote_url=str(src), target=target_for(sha), path=path
        )


def test_moved_ref_is_refused_at_fetch(origin: tuple[Path, Path, str], tmp_path: Path) -> None:
    src, root, _ = origin
    stale = target_for("d" * 40)
    with pytest.raises(c.Refusal, match="not the resolved"):
        c.prepare_worktree(
            c.run, repo_root=root, remote_url=str(src), target=stale, path=tmp_path / "wt"
        )


def test_dirty_worktree_is_refused(origin: tuple[Path, Path, str], tmp_path: Path) -> None:
    src, root, sha = origin
    path = tmp_path / "wt"
    c.prepare_worktree(
        c.run, repo_root=root, remote_url=str(src), target=target_for(sha), path=path
    )
    c.check_worktree(c.run, path, sha)
    (path / "stray.py").write_text("x = 1\n")
    with pytest.raises(c.Refusal, match="dirty"):
        c.check_worktree(c.run, path, sha)
    with pytest.raises(c.Refusal, match="already"):  # and the path can't be reused
        c.prepare_worktree(
            c.run, repo_root=root, remote_url=str(src), target=target_for(sha), path=path
        )


def test_moved_pr_head_is_refused(origin: tuple[Path, Path, str]) -> None:
    _, _, sha = origin
    fake = Fake(sha)
    fake.heads = iter(["e" * 40])
    with pytest.raises(c.Refusal, match="moved"):
        c.ensure_unmoved(fake, REPO, target_for(sha))


def test_bots_are_never_maintainers() -> None:
    fake = Fake("a" * 40)
    assert not c.is_maintainer(fake, REPO, "dependabot[bot]")
    assert fake.calls == []
    assert c.is_maintainer(fake, REPO, "eoin-obrien")
    fake.role = "write"
    assert not c.is_maintainer(fake, REPO, "someone")


# --- end to end -----------------------------------------------------------------------------------


def test_dry_run_end_to_end(origin: tuple[Path, Path, str], tmp_path: Path) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    rc, msgs = run_main(fake, root, src, tmp_path, "--dry-run")
    assert rc == 0, msgs
    msgs = [m for m in msgs if not m.startswith("GPU utilisation")]
    assert fake.posted() == []
    assert not any(x[:3] == ["gh", "run", "rerun"] for x in fake.calls)
    status = json.loads(msgs[0].split(":\n", 1)[1])
    assert status == {
        "state": "success",
        "context": "gpu/xenon",
        "description": "3.14: 3 passed, 0 failed · 3.15: 3 passed, 0 failed · RTX 3090 Ti sm_86 · "
        "driver 610.57",
    }
    comment = msgs[1]
    assert c.MARKER in comment
    assert sha in comment
    for leak in (str(tmp_path), "ghp_", "/nope", "fakehome", "NUMBA_OPT"):
        assert leak not in comment
    # Both versions ran pytest with the shared flags, under the scrubbed env.
    pytest_calls = [x for x in fake.calls if "pytest" in x]
    assert len(pytest_calls) == 2
    assert all("--randomly-seed=7" in x and "gpu and not perf" in x for x in pytest_calls)
    assert all(not any(k.startswith(("BROTKIT_", "NUMBA_", "GH_")) for k in e) for e in fake.envs)
    # The throwaway worktree is gone.
    assert "brotkit-gpu-check-" not in git(root, "worktree", "list")


def test_narrowed_run_doesnt_post_without_post_partial(
    origin: tuple[Path, Path, str], tmp_path: Path
) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    rc, msgs = run_main(fake, root, src, tmp_path, "--python", "3.15")
    assert rc == 0
    assert fake.posted() == []
    assert "narrowed run" in msgs[-2]
    assert '"partial: 3.15 only' in msgs[-2]


def test_real_run_posts_comment_status_and_reruns_the_gate(
    origin: tuple[Path, Path, str], tmp_path: Path
) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    fake.comments = [
        {"id": 5, "html_url": "u", "body": "unrelated", "login": "eoin-obrien"},
        {"id": 6, "html_url": "u", "body": f"{c.MARKER}\nold", "login": "eoin-obrien"},
    ]
    fake.runs = [{"databaseId": 99, "status": "completed", "event": "pull_request",
                  "createdAt": "2026-11-26T10:00:00Z"}]  # fmt: skip
    rc, _ = run_main(fake, root, src, tmp_path)
    assert rc == 0
    posted = fake.posted()
    assert posted[0][3:5] == ["PATCH", f"repos/{REPO}/issues/comments/6"]
    assert posted[1][4] == f"repos/{REPO}/statuses/{sha}"
    status = json.loads(fake.inputs[-1])
    assert status["state"] == "success"
    assert status["target_url"].endswith("#issuecomment-1")
    assert ["gh", "run", "rerun", "99", "-R", REPO] in fake.calls


def test_failing_tests_post_failure(origin: tuple[Path, Path, str], tmp_path: Path) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    fake.junit = junit_xml(failed=["test_fma"])
    rc, _ = run_main(fake, root, src, tmp_path)
    assert rc == 1
    assert json.loads(fake.inputs[-1])["state"] == "failure"
    assert "tests.test_a::test_fma" in json.loads(fake.inputs[0])["body"]


def test_non_maintainer_is_refused_before_anything_runs(
    origin: tuple[Path, Path, str], tmp_path: Path
) -> None:
    src, root, sha = origin
    fake = Fake(sha, author="mallory", role="read")
    rc, _ = run_main(fake, root, src, tmp_path)
    assert rc == 2
    assert not any(Path(x[0]).name in ("uv", "python", "nvidia-smi") for x in fake.calls)


def test_sha_moving_during_the_run_posts_nothing(
    origin: tuple[Path, Path, str], tmp_path: Path
) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    # resolve, check after the worktree, then the pre-post check sees a new push.
    fake.heads = itertools.chain([sha, sha], itertools.repeat("f" * 40))
    rc, _ = run_main(fake, root, src, tmp_path)
    assert rc == 2
    assert fake.posted() == []


# --- correctness vs perf --------------------------------------------------------------------------


def test_correctness_runs_on_a_busy_gpu_and_records_the_load(
    origin: tuple[Path, Path, str], tmp_path: Path
) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    fake.util = 27.0
    clock = Clock()
    rc, msgs = run_main(fake, root, src, tmp_path, "--dry-run", clock=clock)
    assert rc == 0, msgs
    assert c.BUSY_RETRY_S not in clock.sleeps  # never waited
    pytest_calls = [x for x in fake.calls if "pytest" in x]
    assert all("gpu and not perf" in x for x in pytest_calls)
    comment = msgs[-1]
    assert "| GPU utilisation before tests | 27% | 27% |" in comment
    assert "correctness tests only" in comment


def test_perf_waits_for_an_idle_gpu_and_runs_the_whole_suite(
    origin: tuple[Path, Path, str], tmp_path: Path
) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    rc, msgs = run_main(fake, root, src, tmp_path, "--dry-run", "--perf", clock=Clock())
    assert rc == 0, msgs
    pytest_calls = [x for x in fake.calls if "pytest" in x]
    assert all(x[x.index("-m", 2) + 1] == "gpu" for x in pytest_calls)
    assert "the whole GPU suite (`--perf`" in msgs[-1]


def test_perf_on_a_busy_gpu_gives_up_and_posts_nothing(
    origin: tuple[Path, Path, str], tmp_path: Path
) -> None:
    src, root, sha = origin
    fake = Fake(sha)
    fake.util = 27.0
    clock = Clock()
    rc, msgs = run_main(fake, root, src, tmp_path, "--perf", "--max-wait", "60", clock=clock)
    assert rc == 2
    assert fake.posted() == []
    assert not any("pytest" in x for x in fake.calls)
    assert any("GPU busy (median 27%)" in m for m in msgs)

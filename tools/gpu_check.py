"""Run brotkit's GPU suite on Xenon against a PR's head and record the result as a commit status.

Nothing runs on the GPU machine unless the maintainer starts this script. It checks the exact
head commit out into a throwaway worktree, runs `pytest -m gpu` on Python 3.14 and 3.15 under a
scrubbed environment, posts a `gpu/xenon` commit status on that SHA, upserts one PR comment with
the environment and results, and re-runs the PR's gpu-gate job so the gate sees the
new status. See ADR-0010 and docs/gpu-check.md.

    uv run tools/gpu_check.py <PR number | 40-char SHA> [--python 3.15] [--perf] [--dry-run]

Correctness tests (`gpu and not perf`) run straight away, whatever else is using the GPU; the
utilisation seen just before them is recorded in the comment. Timing tests (`perf`) run only with
--perf, and only once the GPU is idle (spike S4's busy check).

PRs by anyone who isn't a maintainer are refused unless --reviewed is given, meaning a human has
read the full diff (lockfile and tests included). Agents never pass --reviewed.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_gate  # sibling module; tools/ isn't a package

PYTHONS = ("3.14", "3.15")
MARKER = "<!-- brotkit:gpu-check -->"
DESCRIPTION_LIMIT = 140
GATE_WORKFLOW = "gpu-gate.yml"
PACKAGES = ("numba-cuda-mlir", "nvidia-nvjitlink", "nvidia-nvvm", "cuda-core", "cuda-bindings")
RANDOMLY = "pytest-randomly"
MAX_JUNIT_BYTES = 32 * 1024 * 1024
MAX_FAILED_LISTED = 30
# Busy check from spike S4, for perf tests only: median of 8 utilisation samples over 1.2 s must
# be <= 5%, and no compute process may hold the GPU; otherwise wait 20 s and look again.
BUSY_SAMPLES, BUSY_INTERVAL_S, BUSY_THRESHOLD_PCT, BUSY_RETRY_S = 8, 0.15, 5.0, 20.0
SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin", "/usr/lib/wsl/lib")


class Refusal(Exception):
    """The check won't run (or won't post): wrong author, moved SHA, reused worktree, ..."""


# --- processes ------------------------------------------------------------------------------------


class Runner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        input: str | None = None,
        check: bool = True,
        echo: bool = False,
    ) -> subprocess.CompletedProcess[str]: ...


def run(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input: str | None = None,
    check: bool = True,
    echo: bool = False,
) -> subprocess.CompletedProcess[str]:
    """subprocess.run with text output. `echo` streams output to the terminal as it arrives."""
    if not echo:
        r = subprocess.run(
            list(args), cwd=cwd, env=dict(env) if env is not None else None, input=input,
            text=True, capture_output=True, check=False,
        )  # fmt: skip
    else:
        p = subprocess.Popen(
            list(args), cwd=cwd, env=dict(env) if env is not None else None, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )  # fmt: skip
        lines: list[str] = []
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            lines.append(line)
        r = subprocess.CompletedProcess(list(args), p.wait(), "".join(lines), "")
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, list(args), r.stdout, r.stderr)
    return r


def gh_json(runner: Runner, *args: str) -> Any:
    return json.loads(runner(["gh", *args]).stdout)


# --- target ---------------------------------------------------------------------------------------

_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Target:
    sha: str
    pr: int | None
    author: str
    fetch_ref: str


def resolve(runner: Runner, repo: str, target: str) -> Target:
    if target.isdigit():
        pr = gh_json(runner, "api", f"repos/{repo}/pulls/{target}")
        if pr.get("state") != "open":
            raise Refusal(f"PR #{target} is {pr.get('state')}, not open")
        return Target(
            sha=pr["head"]["sha"], pr=int(target), author=pr["user"]["login"],
            fetch_ref=f"refs/pull/{target}/head",
        )  # fmt: skip
    if not _SHA.fullmatch(target):
        raise Refusal(f"{target!r} is neither a PR number nor a full 40-character SHA")
    pulls = gh_json(runner, "api", f"repos/{repo}/commits/{target}/pulls")
    for pr in cast(list[dict[str, Any]], pulls):
        if pr.get("state") == "open" and pr["head"]["sha"] == target:
            return Target(
                sha=target, pr=int(pr["number"]), author=pr["user"]["login"],
                fetch_ref=f"refs/pull/{pr['number']}/head",
            )  # fmt: skip
    commit = cast(dict[str, Any], gh_json(runner, "api", f"repos/{repo}/commits/{target}"))
    who = cast(dict[str, Any], commit.get("author") or {})
    author = str(who.get("login") or "(unknown)")
    return Target(sha=target, pr=None, author=author, fetch_ref=target)


def current_sha(runner: Runner, repo: str, target: Target) -> str:
    if target.pr is None:
        return target.sha
    return gh_json(runner, "api", f"repos/{repo}/pulls/{target.pr}")["head"]["sha"]


def ensure_unmoved(runner: Runner, repo: str, target: Target) -> None:
    now = current_sha(runner, repo, target)
    if now != target.sha:
        raise Refusal(
            f"PR #{target.pr} head moved from {target.sha[:12]} to {now[:12]} since it was "
            "resolved; nothing was posted. Run the check again."
        )


def is_maintainer(runner: Runner, repo: str, login: str) -> bool:
    """Admin or maintain role on the repo, from the collaborator permission API. Bots never are."""
    if login.endswith("[bot]") or login == "(unknown)":
        return False
    r = runner(["gh", "api", f"repos/{repo}/collaborators/{login}/permission"], check=False)
    if r.returncode != 0:
        return False
    return json.loads(r.stdout).get("role_name") in ("admin", "maintain")


def check_author(maintainer: bool, *, author: str, reviewed: bool) -> None:
    if maintainer or reviewed:
        return
    raise Refusal(
        f"{author} isn't a maintainer. Read the full diff (lockfile and tests included), "
        "then re-run with --reviewed. Agents never pass --reviewed."
    )


# --- worktree -------------------------------------------------------------------------------------


def _git(runner: Runner, repo_root: Path, *args: str) -> str:
    return runner(["git", "-C", str(repo_root), *args]).stdout.strip()


def prepare_worktree(
    runner: Runner, *, repo_root: Path, remote_url: str, target: Target, path: Path
) -> None:
    """Check `target.sha` out at a fresh path. Refuse a reused path, a moved ref or a dirty tree."""
    if path.exists():
        raise Refusal(f"worktree path {path} already exists; refusing to reuse it")
    listed = _git(runner, repo_root, "worktree", "list", "--porcelain")
    if any(line == f"worktree {path}" for line in listed.splitlines()):
        raise Refusal(f"{path} is already a registered worktree; refusing to reuse it")
    _git(runner, repo_root, "fetch", "--quiet", "--no-tags", remote_url, target.fetch_ref)
    fetched = _git(runner, repo_root, "rev-parse", "FETCH_HEAD")
    if fetched != target.sha:
        raise Refusal(
            f"{target.fetch_ref} is now {fetched[:12]}, not the resolved {target.sha[:12]}"
        )
    _git(runner, repo_root, "worktree", "add", "--quiet", "--detach", str(path), target.sha)
    check_worktree(runner, path, target.sha)


def check_worktree(runner: Runner, path: Path, sha: str) -> None:
    head = runner(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()
    if head != sha:
        raise Refusal(f"worktree is at {head[:12]}, expected {sha[:12]}")
    dirty = runner(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"])
    if dirty.stdout.strip():
        raise Refusal("worktree is dirty; refusing to test it")


def remove_worktree(runner: Runner, repo_root: Path, path: Path) -> None:
    runner(["git", "-C", str(repo_root), "worktree", "remove", "--force", str(path)], check=False)
    runner(["git", "-C", str(repo_root), "worktree", "prune"], check=False)


# --- GPU ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Gpu:
    name: str
    cc: str
    driver: str

    @property
    def sm(self) -> str:
        return "sm_" + self.cc.replace(".", "")

    @property
    def short_name(self) -> str:
        return re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", self.name)


def gpu_info(runner: Runner) -> Gpu:
    out = runner(
        ["nvidia-smi", "-i", "0", "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader"]
    ).stdout  # fmt: skip
    name, cc, driver = (s.strip() for s in out.strip().splitlines()[0].split(","))
    return Gpu(name=name, cc=cc, driver=driver)


def gpu_sample(runner: Runner) -> tuple[float, bool]:
    """(utilisation %, whether any compute process holds the GPU)."""
    util = runner(
        ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    ).stdout
    apps = runner(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], check=False
    ).stdout
    return float(util.strip().splitlines()[0]), bool(apps.strip())


def gpu_idle(samples: Sequence[float]) -> bool:
    return statistics.median(samples) <= BUSY_THRESHOLD_PCT


def observe_util(
    sample: Callable[[], tuple[float, bool]], *, sleep: Callable[[float], None]
) -> tuple[float, bool]:
    """(median utilisation % of 8 samples over 1.2 s, whether a compute process held the GPU)."""
    utils: list[float] = []
    holders = False
    for i in range(BUSY_SAMPLES):
        u, h = sample()
        utils.append(u)
        holders = holders or h
        if i < BUSY_SAMPLES - 1:
            sleep(BUSY_INTERVAL_S)
    return statistics.median(utils), holders


def wait_for_idle(
    sample: Callable[[], tuple[float, bool]],
    *,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    max_wait_s: float,
    out: Callable[[str], None],
) -> float:
    """S4's busy check, before perf tests: wait out bursts, give up (posting nothing) after
    `max_wait_s`. Returns the idle median it saw."""
    start = clock()
    while True:
        median, holders = observe_util(sample, sleep=sleep)
        if gpu_idle([median]) and not holders:
            return median
        if clock() - start > max_wait_s:
            raise Refusal(f"GPU still busy after {max_wait_s:.0f} s; nothing was posted")
        why = "a compute process holds it" if holders else f"median {median:.0f}%"
        out(f"GPU busy ({why}); waiting {BUSY_RETRY_S:.0f} s")
        sleep(BUSY_RETRY_S)


# --- environment ----------------------------------------------------------------------------------


def scrubbed_env(
    shell_env: Mapping[str, str],
    *,
    home: Path,
    tmp: Path,
    extra_path: Sequence[Path],
    uv_cache: str,
    uv_python: str,
) -> dict[str, str]:
    """The only environment the suite sees: an allowlist, nothing else from the shell.

    PATH is rebuilt rather than passed through, so a system CUDA toolkit on the shell's PATH
    can't be picked up. HOME and TMPDIR point into the run's temp dir (so the kernel cache
    starts empty). uv's own UV_* variables pass through; BROTKIT_*, NUMBA_* and CUDA_* don't,
    apart from CUDA_VISIBLE_DEVICES.
    """
    env = {k: v for k, v in shell_env.items() if k.startswith("UV_")}
    env |= {
        "PATH": os.pathsep.join([*map(str, extra_path), *SYSTEM_PATH]),
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "LANG": "C.UTF-8",
        "UV_CACHE_DIR": uv_cache,
        "UV_PYTHON_INSTALL_DIR": uv_python,
        "UV_NO_CONFIG": "1",
    }
    if "CUDA_VISIBLE_DEVICES" in shell_env:
        env["CUDA_VISIBLE_DEVICES"] = shell_env["CUDA_VISIBLE_DEVICES"]
    return env


# --- JUnit ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class JUnit:
    passed: int
    failed: int
    skipped: int
    failed_names: tuple[str, ...]
    sha256: str


def parse_junit(data: bytes) -> JUnit:
    """Counts from a JUnit file. The file is written by the code under test, so it's treated
    as untrusted: size-capped, and any DTD or entity declaration is rejected before parsing."""
    if len(data) > MAX_JUNIT_BYTES:
        raise ValueError("JUnit file too large")
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("JUnit file declares a DTD or entities")
    root = ElementTree.fromstring(data)
    passed = failed = skipped = 0
    names: list[str] = []
    for case in root.iter("testcase"):
        tags = {child.tag for child in case}
        if tags & {"failure", "error"}:
            failed += 1
            names.append(f"{case.get('classname', '')}::{case.get('name', '')}")
        elif "skipped" in tags:
            skipped += 1
        else:
            passed += 1
    return JUnit(
        passed=passed, failed=failed, skipped=skipped, failed_names=tuple(names),
        sha256=hashlib.sha256(data).hexdigest(),
    )  # fmt: skip


# --- one Python version ---------------------------------------------------------------------------

PROBE = """\
import importlib.metadata as m, json, sys
def v(n):
    try:
        return m.version(n)
    except m.PackageNotFoundError:
        return None
print(json.dumps({"python": sys.version.split()[0], "packages": {n: v(n) for n in sys.argv[1:]}}))
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class PyResult:
    python: str
    version: str = "?"
    packages: Mapping[str, str | None] = field(default_factory=lambda: {})
    seed: int | None = None
    junit: JUnit | None = None
    returncode: int = 0
    setup_error: str | None = None
    util_pct: float | None = None

    @property
    def ok(self) -> bool:
        return (
            self.setup_error is None
            and self.junit is not None
            and self.returncode == 0
            and self.junit.failed == 0
            and self.junit.passed > 0
        )

    def summary(self) -> str:
        if self.setup_error is not None or self.junit is None:
            return f"{self.python}: setup failed"
        return f"{self.python}: {self.junit.passed} passed, {self.junit.failed} failed"


def pytest_args(junit: str, seed: int | None, *, perf: bool) -> list[str]:
    """The flags hosted CI uses too (JUnit XML, pytest-randomly seed), plus the GPU marker.
    Without --perf, timing tests are left out; with it, the whole GPU suite runs."""
    marker = "gpu" if perf else "gpu and not perf"
    args = ["-m", "pytest", "-m", marker, "-p", "no:cacheprovider", f"--junitxml={junit}"]
    if seed is not None:
        args.append(f"--randomly-seed={seed}")
    return args


@dataclass(frozen=True, slots=True, kw_only=True)
class Plan:
    """Where one run lives."""

    worktree: Path
    tmp: Path
    uv: str
    env: Mapping[str, str]


def env_names(plan: Plan) -> list[str]:
    """The names (never values) of every variable the suite's process sees."""
    return sorted({*plan.env, "UV_PROJECT_ENVIRONMENT"})


def run_python(
    runner: Runner,
    plan: Plan,
    python: str,
    *,
    before_tests: Callable[[], float],
    seed: int,
    perf: bool,
) -> PyResult:
    """Sync, probe versions and run the suite for one Python. `before_tests` returns the GPU
    utilisation to record: an observation, or (with --perf) the busy check's idle reading."""
    env = dict(plan.env) | {"UV_PROJECT_ENVIRONMENT": str(plan.tmp / f"venv-{python}")}
    cwd = plan.worktree
    venv_python = str(plan.tmp / f"venv-{python}" / "bin" / "python")
    sync = runner(
        [plan.uv, "sync", "--locked", "--extra", "cuda", "--python", python],
        cwd=cwd, env=env, check=False, echo=True,
    )  # fmt: skip
    if sync.returncode != 0:
        return PyResult(python=python, setup_error="uv sync failed", returncode=sync.returncode)
    probe = runner([venv_python, "-c", PROBE, *PACKAGES, RANDOMLY], cwd=cwd, env=env, check=False)
    if probe.returncode != 0:
        return PyResult(python=python, setup_error="probe failed", returncode=probe.returncode)
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    packages = cast(dict[str, str | None], info["packages"])
    use_seed = seed if packages.pop(RANDOMLY, None) else None
    util = before_tests()
    junit_path = plan.tmp / "out" / f"junit-{python}.xml"
    tests = runner(
        [venv_python, *pytest_args(str(junit_path), use_seed, perf=perf)],
        cwd=cwd, env=env, check=False, echo=True,
    )  # fmt: skip
    common: dict[str, Any] = dict(
        python=python, version=info["python"], packages=packages, seed=use_seed,
        returncode=tests.returncode, util_pct=util,
    )  # fmt: skip
    try:
        junit = parse_junit(junit_path.read_bytes())
    except (OSError, ValueError, ElementTree.ParseError) as e:
        return PyResult(**common, setup_error=f"no usable JUnit file ({type(e).__name__})")
    return PyResult(**common, junit=junit)


# --- payloads -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Scrubber:
    """Removes home paths, usernames, tokens and env assignments from anything posted."""

    literals: tuple[tuple[str, str], ...] = ()
    user: str = ""

    _PATTERNS = (
        (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})"), "<token>"),
        (re.compile(r"(?:/home|/Users)/[^/\s`|<>]+"), "~"),
        (re.compile(r"/mnt/[a-z]/Users/[^/\s`|<>]+", re.IGNORECASE), "~"),
        (re.compile(r"[A-Za-z]:\\Users\\[^\\\s`|<>]+", re.IGNORECASE), "~"),
        (re.compile(r"(?<![\w/])/root\b"), "~"),
        (re.compile(r"\b[A-Z][A-Z0-9_]{2,}=[^\s`|]*"), "<env>"),
    )

    def __call__(self, text: str) -> str:
        for literal, token in sorted(self.literals, key=lambda p: -len(p[0])):
            if literal:
                text = text.replace(literal, token)
        for pattern, token in self._PATTERNS:
            text = pattern.sub(token, text)
        if len(self.user) >= 3:
            text = re.sub(rf"(?<![\w-]){re.escape(self.user)}(?![\w-])", "<user>", text)
        return text

    def leaks(self, text: str) -> list[str]:
        found = [lit for lit, _ in self.literals if lit and lit in text]
        found += [p.pattern for p, _ in self._PATTERNS if p.search(text)]
        return found


def build_description(results: Sequence[PyResult], gpu: Gpu, *, partial: bool) -> str:
    """At most 140 characters, e.g. "3.14: 40 passed, 0 failed · 3.15: ... · RTX 3090 Ti sm_86 ·
    driver 610.57". Detail is dropped from the right until it fits."""
    head = [f"partial: {'+'.join(r.python for r in results)} only"] if partial else []
    head += [r.summary() for r in results]
    gpu_part, driver = f"{gpu.short_name} {gpu.sm}", f"driver {gpu.driver}"
    for parts in ([*head, gpu_part, driver], [*head, gpu_part], head):
        text = " · ".join(parts)
        if len(text) <= DESCRIPTION_LIMIT:
            return text
    return " · ".join(head)[: DESCRIPTION_LIMIT - 1] + "…"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("`", "'").replace("\n", " ")


def build_comment(
    *,
    target: Target,
    results: Sequence[PyResult],
    gpu: Gpu,
    state: str,
    description: str,
    env_names: Sequence[str],
    reviewed: bool,
    perf: bool,
    context: str,
) -> str:
    cols = [r.python for r in results]
    rows: list[tuple[str, list[str]]] = [("Python", [r.version for r in results])]
    rows += [(p, [str(r.packages.get(p) or "-") for r in results]) for p in PACKAGES]
    rows += [
        ("passed / failed / skipped", [
            f"{r.junit.passed} / {r.junit.failed} / {r.junit.skipped}" if r.junit else "-"
            for r in results
        ]),
        ("pytest exit code", [str(r.returncode) for r in results]),
        ("pytest-randomly seed", [str(r.seed) if r.seed is not None else "-" for r in results]),
        ("GPU utilisation before tests", [
            f"{r.util_pct:.0f}%" if r.util_pct is not None else "-" for r in results
        ]),
        ("JUnit sha256", [f"`{r.junit.sha256}`" if r.junit else "-" for r in results]),
    ]  # fmt: skip
    lines = [
        MARKER,
        f"### GPU check: {state}",
        "",
        f"`{context}` on `{target.sha}`: {_cell(description)}",
        "",
        f"GPU: {_cell(gpu.name)} · {gpu.sm} · driver {_cell(gpu.driver)} · brotkit commit "
        f"`{target.sha[:12]}`",
        "",
        "| | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
        *(f"| {name} | " + " | ".join(_cell(v) for v in vals) + " |" for name, vals in rows),
    ]
    for r in results:
        if r.setup_error:
            lines += ["", f"**{r.python}:** {_cell(r.setup_error)}"]
        if r.junit and r.junit.failed_names:
            shown = r.junit.failed_names[:MAX_FAILED_LISTED]
            lines += ["", f"**Failed on {r.python}:**", *(f"- `{_cell(n)}`" for n in shown)]
            if len(r.junit.failed_names) > len(shown):
                lines.append(f"- ... and {len(r.junit.failed_names) - len(shown)} more")
    who = "reviewed by a human (`--reviewed`)" if reviewed else "author is a maintainer"
    scope = (
        "the whole GPU suite (`--perf`: timing tests ran once the GPU was idle)"
        if perf
        else "correctness tests only (`gpu and not perf`; timing tests need `--perf`)"
    )
    lines += [
        "",
        f"Ran {scope}. Utilisation is the median of 8 `nvidia-smi` samples.",
        "",
        f"Run: {who}. Test environment allowlist (names only): "
        + ", ".join(f"`{n}`" for n in sorted(env_names))
        + ". PATH is rebuilt (venv, uv, system dirs), HOME and TMPDIR point into a fresh temp dir.",
    ]
    return "\n".join(lines) + "\n"


# --- GitHub ---------------------------------------------------------------------------------------


def upsert_comment(runner: Runner, repo: str, pr: int, me: str, body: str) -> str:
    raw = runner(
        ["gh", "api", "--paginate", f"repos/{repo}/issues/{pr}/comments",
         "--jq", ".[] | {id, html_url, body, login: .user.login}"]
    ).stdout  # fmt: skip
    for line in raw.splitlines():
        c = json.loads(line)
        if c["login"] == me and MARKER in (c["body"] or ""):
            r = runner(
                ["gh", "api", "-X", "PATCH", f"repos/{repo}/issues/comments/{c['id']}",
                 "--input", "-"],
                input=json.dumps({"body": body}),
            )  # fmt: skip
            return json.loads(r.stdout)["html_url"]
    r = runner(
        ["gh", "api", "-X", "POST", f"repos/{repo}/issues/{pr}/comments", "--input", "-"],
        input=json.dumps({"body": body}),
    )
    return json.loads(r.stdout)["html_url"]


def post_status(runner: Runner, repo: str, sha: str, payload: Mapping[str, str]) -> None:
    runner(
        ["gh", "api", "-X", "POST", f"repos/{repo}/statuses/{sha}", "--input", "-"],
        input=json.dumps(dict(payload)),
    )


def rerun_gate(
    runner: Runner,
    repo: str,
    sha: str,
    *,
    sleep: Callable[[float], None],
    out: Callable[[str], None],
) -> None:
    """Re-run the newest gpu-gate run for `sha`, waiting for it to finish first if needed."""
    for _ in range(30):
        runs = gh_json(
            runner, "run", "list", "-R", repo, "--workflow", GATE_WORKFLOW, "--commit", sha,
            "--json", "databaseId,status,event,createdAt", "--limit", "20",
        )  # fmt: skip
        prs = sorted(
            (r for r in cast(list[dict[str, Any]], runs) if r.get("event") == "pull_request"),
            key=lambda r: str(r.get("createdAt", "")),
        )
        if not prs:
            out(f"No gpu-gate run for {sha[:12]} yet; it will read the new status when it runs.")
            return
        latest = prs[-1]
        if latest.get("status") == "completed":
            runner(["gh", "run", "rerun", str(latest["databaseId"]), "-R", repo])
            out(f"Re-ran gpu-gate run {latest['databaseId']}.")
            return
        sleep(10)
    out("gpu-gate is still running after 5 minutes; re-run it by hand.")


# --- main -----------------------------------------------------------------------------------------


@contextmanager
def temp_root() -> Generator[Path]:
    path = Path(tempfile.mkdtemp(prefix="brotkit-gpu-check-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("target", help="PR number or full head SHA")
    ap.add_argument(
        "--python", action="append", choices=PYTHONS,
        help="run only this version (repeatable); narrowed runs post only with --post-partial",
    )  # fmt: skip
    ap.add_argument("--post-partial", action="store_true", help="post a narrowed run's status")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the status and comment; post nothing"
    )
    ap.add_argument(
        "--reviewed", action="store_true",
        help="a human has read the full diff, lockfile and tests included. Agents never pass this.",
    )  # fmt: skip
    ap.add_argument(
        "--perf", action="store_true",
        help="also run the timing (perf) tests, after waiting for an idle GPU",
    )  # fmt: skip
    ap.add_argument(
        "--max-wait", type=float, default=1800.0, help="--perf: GPU busy wait limit, seconds"
    )
    ap.add_argument("--seed", type=int, default=None, help="pytest-randomly seed (default: random)")
    ap.add_argument("--repo", default=None, help="OWNER/NAME (default: this checkout's repo)")
    ap.add_argument("--remote-url", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = run,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    shell_env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
    out: Callable[[str], None] = print,
) -> int:
    args = parse_args(argv)
    shell_env = dict(os.environ) if shell_env is None else shell_env
    root = repo_root or Path(__file__).resolve().parent.parent
    config = gpu_gate.load_config(root / "tools" / "gpu_gate_paths.toml")
    try:
        repo = (
            args.repo or gh_json(runner, "repo", "view", "--json", "nameWithOwner")["nameWithOwner"]
        )
        target = resolve(runner, repo, args.target)
        me = runner(["gh", "api", "user", "--jq", ".login"]).stdout.strip()
        check_author(
            is_maintainer(runner, repo, target.author),
            author=target.author, reviewed=args.reviewed,
        )  # fmt: skip
        pythons = tuple(p for p in PYTHONS if p in (args.python or PYTHONS))
        partial = pythons != PYTHONS
        uv = shutil.which("uv", path=shell_env.get("PATH")) or "uv"
        uv_cache = runner([uv, "cache", "dir"]).stdout.strip()
        uv_python = runner([uv, "python", "dir"]).stdout.strip()
        seed = args.seed if args.seed is not None else random.randrange(2**32)
        with temp_root() as tmp:
            for d in ("home", "tmp", "out"):
                (tmp / d).mkdir()
            worktree = tmp / "wt"
            prepare_worktree(
                runner, repo_root=root, target=target, path=worktree,
                remote_url=args.remote_url or f"https://github.com/{repo}.git",
            )  # fmt: skip
            try:
                ensure_unmoved(runner, repo, target)
                gpu = gpu_info(runner)
                env = scrubbed_env(
                    shell_env, home=tmp / "home", tmp=tmp / "tmp", uv_cache=uv_cache,
                    uv_python=uv_python, extra_path=[Path(uv).parent],
                )  # fmt: skip
                plan = Plan(worktree=worktree, tmp=tmp, uv=uv, env=env)

                def before_tests() -> float:
                    sample = lambda: gpu_sample(runner)  # noqa: E731
                    if args.perf:
                        return wait_for_idle(
                            sample, sleep=sleep, clock=clock, max_wait_s=args.max_wait, out=out
                        )
                    util, _ = observe_util(sample, sleep=sleep)
                    out(f"GPU utilisation {util:.0f}%; running correctness tests (no --perf).")
                    return util

                results = [
                    run_python(
                        runner, plan, py, before_tests=before_tests, seed=seed, perf=args.perf
                    )
                    for py in pythons
                ]
            finally:
                remove_worktree(runner, root, worktree)
            home = shell_env.get("HOME", str(Path.home()))
            scrub = Scrubber(
                literals=((str(tmp), "<tmp>"), (home, "~"), (str(root), "<repo>")),
                user=getpass.getuser(),
            )
        ensure_unmoved(runner, repo, target)
        state = "success" if all(r.ok for r in results) else "failure"
        description = scrub(build_description(results, gpu, partial=partial))
        body = scrub(
            build_comment(
                target=target, results=results, gpu=gpu, state=state, description=description,
                env_names=env_names(plan), reviewed=args.reviewed,
                perf=args.perf, context=config.context,
            )
        )  # fmt: skip
        leaks = scrub.leaks(description) + scrub.leaks(body)
        if leaks:
            raise Refusal(f"payload still contains private data ({leaks}); nothing was posted")
        payload = {"state": state, "context": config.context, "description": description}
        if args.dry_run or (partial and not args.post_partial):
            why = "--dry-run" if args.dry_run else "narrowed run without --post-partial"
            out(f"Not posting ({why}). Status for {target.sha}:\n{json.dumps(payload, indent=2)}")
            out(f"Comment:\n{body}")
            return 0 if state == "success" else 1
        if target.pr is not None:
            payload["target_url"] = upsert_comment(runner, repo, target.pr, me, body)
        post_status(runner, repo, target.sha, payload)
        out(f"Posted {config.context}={state} on {target.sha[:12]}: {description}")
        if target.pr is not None:
            rerun_gate(runner, repo, target.sha, sleep=sleep, out=out)
        return 0 if state == "success" else 1
    except Refusal as e:
        print(f"gpu_check: refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""The gpu-gate decision: does this PR need the local GPU check, and has it passed?

Run by `.github/workflows/gpu-gate.yml` on hosted CI, with no GPU and a read-only token. The
workflow checks this file and `gpu_gate_paths.toml` out from the PR's *base* commit and treats
the PR's head only as data (changed paths, file contents, the lockfile diff), so a PR can't
change the rules it is judged by. See ADR-0010 and docs/gpu-check.md.

Standard library only: the workflow runs it with `uv run --no-project`.

    python tools/gpu_gate.py --repo-dir pr --base <sha> --head <sha> --pr <n> \
        --statuses statuses.jsonl [--config tools/gpu_gate_paths.toml] [--force-required]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

DEFAULT_CONFIG = Path(__file__).with_name("gpu_gate_paths.toml")

type ReadFile = Callable[[str], bytes | None]
"""Return a file's contents at one commit, or None if it doesn't exist there."""


class ConfigError(ValueError):
    """gpu_gate_paths.toml is missing a key or has the wrong type."""


class _Unrecognised(Exception):
    """A lockfile or pyproject structure the classifier doesn't know: fail safe."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GateConfig:
    context: str
    maintainers: frozenset[str]
    gpu_paths: tuple[str, ...]
    gpu_test_marker: str
    lockfile: str
    pyproject: str
    gpu_packages: tuple[str, ...]
    gpu_extras: frozenset[str]
    pyproject_safe_keys: frozenset[str]


def _strs(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in cast(list[Any], value)):
        raise ConfigError(f"{key} must be a list of strings")
    return tuple(cast(list[str], value))


def _str(table: Mapping[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _table(table: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] table is missing")
    return cast(dict[str, Any], value)


def parse_config(data: Mapping[str, Any]) -> GateConfig:
    paths, deps = _table(data, "paths"), _table(data, "dependencies")
    return GateConfig(
        context=_str(data, "context"),
        maintainers=frozenset(_strs(data, "maintainers")),
        gpu_paths=_strs(paths, "gpu"),
        gpu_test_marker=_str(paths, "gpu_test_marker"),
        lockfile=_str(paths, "lockfile"),
        pyproject=_str(paths, "pyproject"),
        gpu_packages=tuple(normalise(p) for p in _strs(deps, "gpu_packages")),
        gpu_extras=frozenset(_strs(deps, "gpu_extras")),
        pyproject_safe_keys=frozenset(_strs(deps, "pyproject_safe_keys")),
    )


def load_config(path: Path = DEFAULT_CONFIG) -> GateConfig:
    with path.open("rb") as f:
        return parse_config(tomllib.load(f))


def normalise(name: str) -> str:
    """PEP 503 name normalisation."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_gpu_package(name: str, config: GateConfig) -> bool:
    n = normalise(name)
    return any(fnmatch.fnmatchcase(n, pattern) for pattern in config.gpu_packages)


# --- classification -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Classification:
    relevant: bool
    reasons: tuple[str, ...]


def classify(
    changed: Iterable[str], read_base: ReadFile, read_head: ReadFile, config: GateConfig
) -> Classification:
    """Decide whether the changed paths need the GPU check, and say why."""
    reasons: list[str] = []
    for path in sorted(set(changed)):
        p = PurePosixPath(path)
        if any(p.full_match(glob) for glob in config.gpu_paths):
            reasons.append(f"`{path}`: GPU path")
        elif path == config.lockfile:
            reasons += [
                f"`{path}`: {r}" for r in lock_reasons(read_base(path), read_head(path), config)
            ]
        elif path == config.pyproject:
            reasons += [
                f"`{path}`: {r}"
                for r in pyproject_reasons(read_base(path), read_head(path), config)
            ]
        elif p.full_match("tests/**/*.py"):
            marker = config.gpu_test_marker.encode()
            if any(marker in (read(path) or b"") for read in (read_base, read_head)):
                reasons.append(f"`{path}`: test marked gpu")
    return Classification(relevant=bool(reasons), reasons=tuple(reasons))


def _load_toml(data: bytes | None) -> dict[str, Any]:
    if data is None:
        raise _Unrecognised("file added or removed")
    return tomllib.loads(data.decode("utf-8"))


def lock_reasons(base: bytes | None, head: bytes | None, config: GateConfig) -> list[str]:
    """Why a uv.lock change is GPU-relevant (empty if it isn't). Fails safe."""
    try:
        return _lock_diff(_load_toml(base), _load_toml(head), config)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return [f"could not parse ({type(e).__name__}), so it counts as GPU-relevant"]
    except _Unrecognised as e:
        return [f"unrecognised structure ({e}), so it counts as GPU-relevant"]


_LOCK_TOP = frozenset(
    {"version", "revision", "requires-python", "resolution-markers", "supported-markers",
     "options", "manifest", "package"}
)  # fmt: skip
_LOCK_PKG = frozenset(
    {"name", "version", "source", "dependencies", "sdist", "wheels", "optional-dependencies",
     "dev-dependencies", "metadata", "resolution-markers"}
)  # fmt: skip
_ROOT_METADATA = frozenset({"requires-dist", "provides-extras", "requires-dev", "requires-python"})


def _dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _lock_packages(lock: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    unknown = set(lock) - _LOCK_TOP
    if unknown:
        raise _Unrecognised(f"top-level keys {sorted(unknown)}")
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        raise _Unrecognised("`package` is not an array")
    out: dict[str, list[dict[str, Any]]] = {}
    for pkg in cast(list[Any], packages):
        if not isinstance(pkg, dict) or not isinstance(cast(dict[str, Any], pkg).get("name"), str):
            raise _Unrecognised("package entry without a name")
        entry = cast(dict[str, Any], pkg)
        if set(entry) - _LOCK_PKG:
            raise _Unrecognised(f"package keys {sorted(set(entry) - _LOCK_PKG)}")
        out.setdefault(normalise(entry["name"]), []).append(entry)
    return out


def _is_root(entries: Sequence[Mapping[str, Any]]) -> bool:
    for e in entries:
        source = e.get("source")
        if isinstance(source, dict) and (
            {"editable", "virtual"} & set(cast(dict[str, Any], source))
        ):
            return True
    return False


def _cuda_marker(marker: object, extras: frozenset[str]) -> bool:
    if not isinstance(marker, str):
        return False
    return any(re.search(rf"extra\s*==\s*['\"]{re.escape(x)}['\"]", marker) for x in extras)


def _root_items(entry: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The project's own lock entry, flattened into (section, requirement) pairs."""
    items: list[tuple[str, dict[str, Any]]] = []

    def add(section: str, reqs: object) -> None:
        if not isinstance(reqs, list):
            raise _Unrecognised(f"{section} is not an array")
        for r in cast(list[Any], reqs):
            if not isinstance(r, dict) or not isinstance(cast(dict[str, Any], r).get("name"), str):
                raise _Unrecognised(f"{section} entry without a name")
            items.append((section, cast(dict[str, Any], r)))

    def add_groups(section: str, groups: object) -> None:
        if not isinstance(groups, dict):
            raise _Unrecognised(f"{section} is not a table")
        for name, reqs in cast(dict[str, Any], groups).items():
            add(f"{section}.{name}", reqs)

    add("dependencies", entry.get("dependencies", []))
    add_groups("optional-dependencies", entry.get("optional-dependencies", {}))
    add_groups("dev-dependencies", entry.get("dev-dependencies", {}))
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise _Unrecognised("root metadata is not a table")
    meta = cast(dict[str, Any], metadata)
    if set(meta) - _ROOT_METADATA:
        raise _Unrecognised(f"root metadata keys {sorted(set(meta) - _ROOT_METADATA)}")
    add("requires-dist", meta.get("requires-dist", []))
    add_groups("requires-dev", meta.get("requires-dev", {}))
    return items


def _root_reasons(
    base: Sequence[Mapping[str, Any]], head: Sequence[Mapping[str, Any]], config: GateConfig
) -> list[str]:
    if len(base) != 1 or len(head) != 1:
        raise _Unrecognised("project entry appears more than once")
    b, h = base[0], head[0]
    reasons: list[str] = []
    meta_b = cast(dict[str, Any], b.get("metadata", {}))
    meta_h = cast(dict[str, Any], h.get("metadata", {}))
    if meta_b.get("requires-python") != meta_h.get("requires-python"):
        reasons.append("project requires-python changed")
    extras_b = set(cast(list[str], meta_b.get("provides-extras", [])))
    extras_h = set(cast(list[str], meta_h.get("provides-extras", [])))
    if (extras_b ^ extras_h) & config.gpu_extras:
        reasons.append("GPU extra added or removed")
    items_b = Counter((s, _dumps(r)) for s, r in _root_items(b))
    items_h = Counter((s, _dumps(r)) for s, r in _root_items(h))
    for section, raw in sorted((items_b - items_h) + (items_h - items_b)):
        req = cast(dict[str, Any], json.loads(raw))
        extra = section.partition(".")[2]
        if section.startswith("optional-dependencies.") and extra in config.gpu_extras:
            reasons.append(f"[{extra}] extra changed ({req['name']})")
        elif _cuda_marker(req.get("marker"), config.gpu_extras):
            reasons.append(f"[cuda] extra requirement changed ({req['name']})")
        elif is_gpu_package(req["name"], config):
            reasons.append(f"requirement on GPU-stack package {req['name']} changed")
    return reasons


def _versions(entries: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(sorted(str(e.get("version", "?")) for e in entries)) or "none"


def _lock_diff(base: dict[str, Any], head: dict[str, Any], config: GateConfig) -> list[str]:
    bp, hp = _lock_packages(base), _lock_packages(head)
    reasons: list[str] = []
    for key in sorted(_LOCK_TOP - {"package"}):
        if base.get(key) != head.get(key):
            reasons.append(f"lock `{key}` changed")
    for name in sorted(set(bp) | set(hp)):
        b, h = bp.get(name, []), hp.get(name, [])
        if sorted(map(_dumps, b)) == sorted(map(_dumps, h)):
            continue
        if _is_root(b) or _is_root(h):
            reasons += _root_reasons(b, h, config)
        elif is_gpu_package(name, config):
            reasons.append(f"GPU-stack package {name}: {_versions(b)} -> {_versions(h)}")
    return reasons


# --- pyproject ------------------------------------------------------------------------------------

_REQ_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_HANDLED = frozenset(
    {"project.dependencies", "project.optional-dependencies", "project.requires-python",
     "dependency-groups"}
)  # fmt: skip


def pyproject_reasons(base: bytes | None, head: bytes | None, config: GateConfig) -> list[str]:
    """Why a pyproject.toml change is GPU-relevant (empty if it isn't). Fails safe."""
    try:
        return _pyproject_diff(_load_toml(base), _load_toml(head), config)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return [f"could not parse ({type(e).__name__}), so it counts as GPU-relevant"]
    except _Unrecognised as e:
        return [f"unrecognised structure ({e}), so it counts as GPU-relevant"]


def _sub(table: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = table.get(key, {})
    if not isinstance(value, dict):
        raise _Unrecognised(f"`{key}` is not a table")
    return cast(dict[str, Any], value)


def _req_list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise _Unrecognised(f"`{where}` is not an array")
    return cast(list[object], value)


def _requirement_reasons(
    base: list[object], head: list[object], where: str, config: GateConfig
) -> list[str]:
    reasons: list[str] = []
    cb, ch = Counter(map(_dumps, base)), Counter(map(_dumps, head))
    for raw in sorted((cb - ch) + (ch - cb)):
        entry = json.loads(raw)
        match = _REQ_NAME.match(entry) if isinstance(entry, str) else None
        if match is None:
            reasons.append(f"`{where}` entry {raw} not understood, so it counts as GPU-relevant")
        elif is_gpu_package(match.group(1), config):
            reasons.append(f"`{where}`: GPU-stack requirement changed ({entry})")
    return reasons


def _changed_keys(base: Mapping[str, Any], head: Mapping[str, Any]) -> list[str]:
    """Dotted keys that differ: top-level tables, one level deeper under project and tool."""
    out: list[str] = []
    for top in sorted(set(base) | set(head)):
        b, h = base.get(top), head.get(top)
        if b == h:
            continue
        if top in ("project", "tool") and isinstance(b, dict) and isinstance(h, dict):
            bt, ht = cast(dict[str, Any], b), cast(dict[str, Any], h)
            out += [f"{top}.{k}" for k in sorted(set(bt) | set(ht)) if bt.get(k) != ht.get(k)]
        else:
            out.append(top)
    return out


def _pyproject_diff(base: dict[str, Any], head: dict[str, Any], config: GateConfig) -> list[str]:
    reasons: list[str] = []
    pb, ph = _sub(base, "project"), _sub(head, "project")
    if pb.get("requires-python") != ph.get("requires-python"):
        reasons.append("requires-python changed")
    reasons += _requirement_reasons(
        _req_list(pb.get("dependencies", []), "project.dependencies"),
        _req_list(ph.get("dependencies", []), "project.dependencies"),
        "project.dependencies",
        config,
    )
    ob, oh = _sub(pb, "optional-dependencies"), _sub(ph, "optional-dependencies")
    for extra in sorted(set(ob) | set(oh)):
        b, h = ob.get(extra, []), oh.get(extra, [])
        if b == h:
            continue
        if extra in config.gpu_extras:
            reasons.append(f"[{extra}] extra changed")
        else:
            where = f"project.optional-dependencies.{extra}"
            reasons += _requirement_reasons(_req_list(b, where), _req_list(h, where), where, config)
    gb, gh = _sub(base, "dependency-groups"), _sub(head, "dependency-groups")
    for group in sorted(set(gb) | set(gh)):
        where = f"dependency-groups.{group}"
        reasons += _requirement_reasons(
            _req_list(gb.get(group, []), where), _req_list(gh.get(group, []), where), where, config
        )
    for key in _changed_keys(base, head):
        if key in _HANDLED:
            continue
        parts = key.split(".")
        if not any(
            ".".join(parts[:i]) in config.pyproject_safe_keys for i in range(1, len(parts) + 1)
        ):
            reasons.append(f"`{key}` changed")
    return reasons


# --- decision -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    passed: bool
    summary: str


def _status_sha(status: Mapping[str, Any]) -> str | None:
    """The commit a status belongs to, from its API url (`.../statuses/<sha>`)."""
    url = status.get("url")
    if not isinstance(url, str):
        return None
    return url.rstrip("/").rpartition("/statuses/")[2] or None


def _creator(status: Mapping[str, Any]) -> str | None:
    creator = status.get("creator")
    if not isinstance(creator, dict):
        return None
    login = cast(dict[str, Any], creator).get("login")
    return login if isinstance(login, str) else None


def decide(
    classification: Classification,
    statuses: Sequence[Mapping[str, Any]],
    *,
    head_sha: str,
    pr: int | None,
    config: GateConfig,
) -> Decision:
    """Pass if no GPU-relevant path changed, or if the newest maintainer-posted `context`
    status on exactly `head_sha` is `success`."""
    if not classification.relevant:
        return Decision(passed=True, summary="GPU check not required: no GPU-relevant changes.")
    why = "\n".join(f"- {r}" for r in classification.reasons)
    run = f"uv run tools/gpu_check.py {pr if pr is not None else head_sha}"
    ours = [
        s
        for s in statuses
        if s.get("context") == config.context
        and _status_sha(s) == head_sha
        and _creator(s) in config.maintainers
    ]
    if not ours:
        verdict = f"No `{config.context}` status from a maintainer on {head_sha[:12]}."
        return Decision(passed=False, summary=_fail(verdict, why, run))
    latest = max(ours, key=lambda s: (str(s.get("created_at", "")), int(s.get("id", 0) or 0)))
    state, desc = latest.get("state"), str(latest.get("description") or "")
    if state == "success":
        return Decision(
            passed=True,
            summary=f"GPU check passed on {head_sha[:12]}: {desc}\n\nRequired because:\n{why}",
        )
    verdict = f"`{config.context}` is `{state}` on {head_sha[:12]}: {desc}"
    return Decision(passed=False, summary=_fail(verdict, why, run))


def _fail(verdict: str, why: str, run: str) -> str:
    return (
        f"{verdict}\n\nThis PR needs the local GPU check because:\n{why}\n\n"
        f"Maintainer: run `{run}` on Xenon (see docs/gpu-check.md). It posts the status and "
        "re-runs this job."
    )


# --- CLI ------------------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _reader(repo: Path, rev: str) -> ReadFile:
    def read(path: str) -> bytes | None:
        r = subprocess.run(
            ["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, check=False
        )
        return r.stdout if r.returncode == 0 else None

    return read


def load_statuses(text: str) -> list[dict[str, Any]]:
    """Statuses as a JSON array, or one JSON object per line (`gh api --paginate --jq '.[]'`)."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError("statuses must be a list")
    return [cast(dict[str, Any], s) for s in cast(list[Any], data) if isinstance(s, dict)]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo-dir", type=Path, required=True, help="checkout holding both commits")
    ap.add_argument("--base", required=True, help="the PR's base commit")
    ap.add_argument("--head", required=True, help="the PR's head commit")
    ap.add_argument("--pr", type=int, default=None)
    ap.add_argument("--statuses", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--force-required",
        action="store_true",
        help="treat the PR as GPU-relevant (used when the base has no gate yet)",
    )
    args = ap.parse_args(argv)
    config = load_config(args.config)
    if args.force_required:
        classification = Classification(
            relevant=True, reasons=("the base commit has no gate yet (bootstrap)",)
        )
    else:
        merge_base = _git(args.repo_dir, "merge-base", args.base, args.head).decode().strip()
        raw = _git(
            args.repo_dir, "diff", "--name-only", "--no-renames", "-z", merge_base, args.head
        )
        changed = [p for p in raw.decode().split("\0") if p]
        classification = classify(
            changed, _reader(args.repo_dir, merge_base), _reader(args.repo_dir, args.head), config
        )
    statuses = load_statuses(args.statuses.read_text())
    decision = decide(classification, statuses, head_sha=args.head, pr=args.pr, config=config)
    print(decision.summary)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        mark = "passed" if decision.passed else "failed"
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"## gpu-gate {mark}\n\n{decision.summary}\n")
    if not decision.passed:
        first = decision.summary.splitlines()[0]
        print(f"::error title=gpu-gate::{first}")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Cross-process locking and atomic replace on a given directory (throwaway, GPU-free).

Tests, each with N worker processes (spawn):
  flock     fcntl.flock(LOCK_EX) around a read-modify-write counter
  lockf     fcntl.lockf (POSIX record lock) around the same
  filelock  filelock.FileLock around the same
  none      no lock (control: should lose updates)
  replace   one writer os.replace()s a 1 MiB file (temp + fsync + replace) 300x while readers
            read it whole and verify its embedded sha256; counts torn / missing reads
  dirswap   os.rename(tmpdir, existing_dir) behaviour (publish a directory entry)
  unlink_open  delete a file another process holds open, then recreate (eviction race)

usage: .venv/bin/python lock_probe.py <dir> [label] -> results/lock_<label>.json
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

N = 4
ITERS = 300


def _rmw(path: pathlib.Path) -> None:
    v = int(path.read_text() or "0") if path.exists() else 0
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(str(v + 1))
    os.replace(tmp, path)


def worker(kind: str, d: str, q) -> None:
    root = pathlib.Path(d)
    counter = root / f"counter_{kind}"
    lockp = root / f"lock_{kind}"
    t0 = time.perf_counter()
    errs = []
    try:
        if kind == "filelock":
            import filelock

            fl = filelock.FileLock(str(lockp))
        for _ in range(ITERS):
            if kind == "flock":
                with open(lockp, "a+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    _rmw(counter)
                    fcntl.flock(f, fcntl.LOCK_UN)
            elif kind == "lockf":
                with open(lockp, "a+") as f:
                    fcntl.lockf(f, fcntl.LOCK_EX)
                    _rmw(counter)
                    fcntl.lockf(f, fcntl.LOCK_UN)
            elif kind == "filelock":
                with fl:
                    _rmw(counter)
            else:
                _rmw(counter)
    except Exception as e:
        errs.append(f"{type(e).__name__}: {e}")
    q.put((kind, time.perf_counter() - t0, errs))


def counter_test(d: pathlib.Path, kind: str) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    (d / f"counter_{kind}").write_text("0")
    ps = [ctx.Process(target=worker, args=(kind, str(d), q)) for _ in range(N)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    outs = [q.get() for _ in ps]
    final = int((d / f"counter_{kind}").read_text())
    return {
        "expected": N * ITERS,
        "final": final,
        "ok": final == N * ITERS,
        "seconds": round(max(o[1] for o in outs), 3),
        "errors": sorted({e for o in outs for e in o[2]})[:5],
    }


def _blob(i: int) -> bytes:
    body = (str(i).encode() * (1 << 20))[: 1 << 20]
    return hashlib.sha256(body).hexdigest().encode() + body


def writer(d: str, n: int) -> None:
    p = pathlib.Path(d) / "atomic.bin"
    for i in range(n):
        tmp = p.with_name(f".atomic.{os.getpid()}.{i}.tmp")
        with open(tmp, "wb") as f:
            f.write(_blob(i))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)


def reader(d: str, until: float, q) -> None:
    p = pathlib.Path(d) / "atomic.bin"
    good = torn = missing = 0
    while time.time() < until:
        try:
            b = p.read_bytes()
        except FileNotFoundError:
            missing += 1
            continue
        if hashlib.sha256(b[64:]).hexdigest().encode() == b[:64]:
            good += 1
        else:
            torn += 1
    q.put((good, torn, missing))


def replace_test(d: pathlib.Path) -> dict:
    ctx = mp.get_context("spawn")
    writer(str(d), 1)
    q = ctx.Queue()
    t0 = time.perf_counter()
    w = ctx.Process(target=writer, args=(str(d), 300))
    w.start()
    rs = [ctx.Process(target=reader, args=(str(d), time.time() + 6, q)) for _ in range(3)]
    for r in rs:
        r.start()
    w.join()
    wt = time.perf_counter() - t0
    for r in rs:
        r.join()
    res = [q.get() for _ in rs]
    return {
        "reads_good": sum(r[0] for r in res),
        "reads_torn": sum(r[1] for r in res),
        "reads_missing": sum(r[2] for r in res),
        "writer_seconds_300": round(wt, 2),
        "ok": all(r[1] == 0 and r[2] == 0 for r in res),
    }


def dirswap_test(d: pathlib.Path) -> dict:
    a, b = d / "entry_tmp", d / "entry"
    for x in (a, b):
        x.mkdir(exist_ok=True)
        (x / "f").write_text(x.name)
    try:
        os.rename(a, b)
        r = "replaced non-empty dir (unexpected)"
    except OSError as e:
        r = f"{type(e).__name__} errno={e.errno}"
    return {"rename_onto_nonempty_dir": r}


def unlink_open_test(d: pathlib.Path) -> dict:
    p = d / "held.bin"
    p.write_bytes(b"x" * 4096)
    f = open(p, "rb")  # noqa: SIM115
    try:
        os.unlink(p)
        unlink = "ok"
    except OSError as e:
        unlink = f"{type(e).__name__} errno={e.errno}"
    try:
        p.write_bytes(b"y")
        recreate = "ok"
    except OSError as e:
        recreate = f"{type(e).__name__} errno={e.errno}"
    readback = f.read(1)
    f.close()
    return {
        "unlink_while_open": unlink,
        "recreate": recreate,
        "old_handle_reads": readback.decode(),
    }


def fs_type(d: pathlib.Path) -> str:
    best, typ = "", "?"
    for line in pathlib.Path("/proc/mounts").read_text().splitlines():
        dev, mnt, t, *_ = line.split()
        if str(d.resolve()).startswith(mnt) and len(mnt) > len(best):
            best, typ = mnt, f"{t} on {mnt} ({dev})"
    return typ


def main() -> None:
    d = pathlib.Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else "x"
    d.mkdir(parents=True, exist_ok=True)
    out = {"dir": str(d), "fs": fs_type(d)}
    for kind in ("none", "flock", "lockf", "filelock"):
        out[kind] = counter_test(d, kind)
        print(kind, out[kind], flush=True)
    out["replace"] = replace_test(d)
    print("replace", out["replace"], flush=True)
    out["dirswap"] = dirswap_test(d)
    out["unlink_open"] = unlink_open_test(d)
    print(out["dirswap"], out["unlink_open"], flush=True)
    here = pathlib.Path(__file__).parent / "results"
    here.mkdir(exist_ok=True)
    (here / f"lock_{label}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Warm start: fresh-process load time, mechanism (b) brotkit store vs (a) Numba cache=True.

Setup builds every (tier, kernel) once into a scratch root for both mechanisms. Then each
sample is a fresh interpreter (warm_child.py); samples round-robin across configs. Wall time
of the subprocess (interpreter start to exit) is recorded too.

usage: .venv/bin/python warm.py <root> [N] [render 0|1] -> results/warm[_render].json
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import gen_s2
import s2lib

HERE = pathlib.Path(__file__).parent
TIERS = ("f32", "df64", "f64", "dd", "qf")
KERNELS = ("A", "C", "F6")


def sig(tier: str) -> tuple[str, ...]:
    """The ptr64 signature in brotkit's ABI notation (same as gen_s2.signature(tier, 'ptr64'))."""
    word, nw, _ = gen_s2.TIERS[tier]
    return (word,) * (2 * nw) + (word, "int32", "int32", "float32*", "uint32*") + ("int64",) * 3


def setup(root: pathlib.Path) -> None:
    from brotkit.backends.cuda import CudaCacheBackend
    from brotkit.kernels.cache import KernelCache, KernelStore

    be = CudaCacheBackend()
    cache = KernelCache(be, KernelStore(root / "store"))
    (root / "src").mkdir(parents=True, exist_ok=True)
    for kern in KERNELS:
        for tier in TIERS:
            src = gen_s2.emit(tier, kern, "ptr64")
            (root / "src" / f"{tier}_{kern}.py").write_text(src)
            k = cache.get(
                be.spec(source=src, entry=gen_s2.entry_name(tier, kern), signature=sig(tier))
            )
            # (a): own directory per key so Numba's key bug can't cross keys
            d = root / "a" / f"{tier}_{kern}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "kmod.py").write_text(src)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            code = (
                "import sys, importlib.util; sys.path.insert(0, {!r}); import gen_s2;"
                "from numba_cuda_mlir import cuda;"
                "s=importlib.util.spec_from_file_location('kmod', {!r});"
                "m=importlib.util.module_from_spec(s);"
                "s.loader.exec_module(m); d=cuda.jit(cache=True)(getattr(m, {!r}).py_func);"
                "d.compile(gen_s2.signature({!r}, 'ptr64'))"
            ).format(str(HERE), str(d / "kmod.py"), gen_s2.entry_name(tier, kern), tier)
            subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True)
            print(f"setup {tier}/{kern}: regs {k.resources.registers}", flush=True)


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    render = sys.argv[3] if len(sys.argv) > 3 else "0"
    if not (root / "store").exists():
        setup(root)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    kernels = ("C",) if render == "1" else KERNELS
    samples: dict[str, list[dict]] = {}
    for _ in range(n):
        for kern in kernels:
            for tier in TIERS:
                for mech in ("b", "a"):
                    t0 = time.perf_counter()
                    r = subprocess.run(
                        [
                            sys.executable,
                            str(HERE / "warm_child.py"),
                            mech,
                            tier,
                            kern,
                            str(root),
                            render,
                        ],
                        capture_output=True,
                        text=True,
                        env=env,
                        cwd=HERE,
                    )
                    wall = time.perf_counter() - t0
                    if r.returncode:
                        print(r.stderr[-3000:])
                        raise SystemExit(f"{mech} {tier}/{kern} failed")
                    rec = json.loads(r.stdout.strip().splitlines()[-1])
                    rec["wall"] = round(wall, 4)
                    samples.setdefault(f"{mech} {tier}/{kern}", []).append(rec)
    med = {}
    for key, recs in samples.items():
        fields = [k for k, v in recs[0].items() if isinstance(v, float)]
        med[key] = {f: round(statistics.median(r[f] for r in recs), 4) for f in fields}
        med[key]["get_max"] = max(r["get"] for r in recs)
        med[key]["numba_imported"] = recs[0]["numba_imported"]
        med[key]["n"] = len(recs)
        print(key, med[key], flush=True)
    out = "warm_render.json" if render == "1" else "warm.json"
    (s2lib.RESULTS / out).write_text(json.dumps({"median": med, "samples": samples}, indent=1))


if __name__ == "__main__":
    main()

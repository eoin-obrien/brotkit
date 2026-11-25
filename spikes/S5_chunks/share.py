"""Responsiveness of a shared GPU while brotkit renders in chunks (throwaway).

A probe process (its own CUDA context, like LLM inference or another app) launches a one-thread
kernel every 10 ms and records submit-to-complete latency. Meanwhile this process renders a
worst-case view (qf, "whole", 4K, max_iter 10000: about 5 s) with chunk targets from 10 to
400 ms. A baseline with no render gives the floor.

usage: .venv/bin/python share.py <cache root> -> results/share.json
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent

PROBE = r"""
import json, sys, time
from cuda.core import Device, EventOptions
from brotkit.backends.cuda import CudaCacheBackend
from brotkit.kernels.cache import KernelCache, KernelStore
be = CudaCacheBackend()
c = KernelCache(be, KernelStore(sys.argv[1]))
src = (
    "import numpy as np\nfrom numba_cuda_mlir import cuda\n\n"
    "def tick(p):\n    a = cuda.carray(p, (1,), np.float32)\n    a[0] += np.float32(1.0)\n"
)
k = c.get(be.spec(source=src, entry="tick", signature=("float32*",)))
buf = be.device.memory_resource.allocate(4, stream=be.stream)
for _ in range(20):
    k.launch(1, 1, buf); be.stream.sync()
print("READY", flush=True)
stop = time.perf_counter() + float(sys.argv[2])
lat = []
while time.perf_counter() < stop:
    t0 = time.perf_counter()
    k.launch(1, 1, buf)
    be.stream.sync()
    t1 = time.perf_counter()
    lat.append((t0, t1 - t0))
    time.sleep(max(0.0, 0.01 - (time.perf_counter() - t0)))
print("LAT " + json.dumps(lat), flush=True)
"""


def probe(root: str, seconds: float) -> subprocess.Popen[str]:
    p = subprocess.Popen(
        [sys.executable, "-c", PROBE, root, str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=HERE,
    )
    assert p.stdout is not None
    for line in p.stdout:
        if line.startswith("READY"):
            return p
    raise RuntimeError("probe failed to start")


def collect(p: subprocess.Popen[str]) -> list[tuple[float, float]]:
    out, _ = p.communicate(timeout=600)
    line = next(x for x in out.splitlines() if x.startswith("LAT "))
    return json.loads(line[4:])


def stats(lat_s: list[float]) -> dict[str, float]:
    a = np.array(lat_s) * 1e3
    return {
        "n": int(a.size),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p90_ms": round(float(np.percentile(a, 90)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "max_ms": round(float(a.max()), 3),
    }


def main() -> None:
    sys.path.insert(0, str(HERE))
    import s5lib

    from brotkit.backends.chunks import ChunkPolicy

    root = sys.argv[1]
    gpu = s5lib.Gpu.open(pathlib.Path(root))
    gpu.kernel("qf", "C")
    s5lib.s4_on_path()
    import q3_throughput as q

    baseline_mem = int(q.smi("memory.used")[0])
    res: dict[str, dict] = {}
    s5lib.wait_idle(q, baseline_mem, "share baseline")
    p = probe(root, 4.0)
    res["idle"] = stats([d for _, d in collect(p)])
    print("idle", res["idle"], flush=True)
    for target in (0.01, 0.025, 0.05, 0.1, 0.2, 0.4):
        s5lib.wait_idle(q, baseline_mem, f"share {target}")
        pol = ChunkPolicy(target_s=target, ceiling_s=max(0.5, target))
        p = probe(root, 12.0)
        t0 = time.perf_counter()
        r = s5lib.render(gpu, "qf", "C", "whole", max_iter=10000, policy=pol)
        t1 = time.perf_counter()
        lat = [d for t, d in collect(p) if t0 <= t <= t1]
        s = r.summary()
        res[f"target_{target}"] = {
            "probe": stats(lat),
            "chunks": s["chunks"],
            "chunk_s_median": s["chunk_s_median"],
            "chunk_s_max": s["chunk_s_max"],
            "render_wall_s": s["wall_s"],
            "overhead": s["overhead"],
            "actual_over_pred_max": s["actual_over_pred_max"],
            "over_ceiling": s["over_ceiling"],
        }
        print(target, res[f"target_{target}"], flush=True)
    (HERE / "results" / "share.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

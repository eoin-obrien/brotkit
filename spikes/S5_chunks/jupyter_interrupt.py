"""Ctrl-C in a real (headless) Jupyter kernel during a chunked render (throwaway).

Starts an ipykernel from this venv with jupyter_client and, for each wait mode and round:
  1. runs a long render (dd, "interior", 4K, max_iter 5000: about 40 s) in the kernel;
  2. after a few seconds sends interrupt_kernel() (SIGINT, as JupyterLab's stop button does);
  3. records the time from interrupt to the cell's error reply, and the in-kernel latency
     from SIGINT (the runner defers it to the end of the chunk in flight) to the runner
     returning;
  4. in the kernel, checks the partial result: complete is False, finished tiles are exactly the
     reference render's pixels, untouched pixels still hold the sentinel, and no tile is half
     written;
  5. re-runs a short render in the same kernel (re-run after interrupt) and checks the kernel
     wasn't rebuilt (KernelCache memory hit).

usage: .venv/bin/python jupyter_interrupt.py <cache root> [rounds] -> results/jupyter_interrupt.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap
import time

from jupyter_client import KernelManager

HERE = pathlib.Path(__file__).parent

SETUP = """
import sys, time, json
sys.path.insert(0, {here!r})
import numpy as np
import s5lib
from brotkit.backends.chunks import ChunkPolicy
gpu = s5lib.Gpu.open(__import__("pathlib").Path({root!r}))
gpu.kernel("dd", "C"); gpu.kernel("f32", "C")
# Reference for the long render's view, computed once in full (also warms everything).
REF = s5lib.render(gpu, "dd", "C", "interior", max_iter=5000, sentinel=True)
ref_f, ref_u = s5lib.host_layers(gpu, REF)
lookups0 = gpu.lookups
print("setup ok", REF.complete, len(REF.records))
"""

LONG = """
t_start = time.perf_counter()
R = None
try:
    s5lib.render(gpu, "dd", "C", "interior", max_iter=5000, sentinel=True,
                 wait={wait!r}, reraise=True)
except s5lib.RenderInterrupted as e:
    R = e.result
    t_caught = time.perf_counter()
    raise
"""

CHECK = """
f, u = s5lib.host_layers(gpu, R)
cov = s5lib.coverage(R)
untouched = (u[0] == 0xFFFFFFFF)
last = R.records[-1].seconds if R.records else None
esc = cov & (ref_u[0] < 5000)
out = dict(
    complete=R.complete,
    chunks_done=len(R.done),
    done_fraction=float(cov.mean()),
    done_match_ref=bool(np.array_equal(u[:2][:, cov], ref_u[:2][:, cov])
                        and np.array_equal(f[2][cov], ref_f[2][cov])
                        and np.array_equal(f[:2][:, esc], ref_f[:2][:, esc])),
    undone_untouched=bool(untouched[~cov].all()),
    no_partial_tiles=bool((~untouched[cov]).all()),
    in_kernel_stop_s=R.stopped_at - R.interrupted_at,
    last_chunk_s=last,
    max_chunk_s=max(r.seconds for r in R.records),
    cpu_per_wall=R.cpu_s / R.wall_s,
)
print("CHECK " + json.dumps(out))
"""

RERUN = """
r2 = s5lib.render(gpu, "f32", "C", "seahorse")
print("RERUN " + json.dumps(dict(complete=r2.complete, chunks=len(r2.records),
                                 lookups_added=gpu.lookups - lookups0)))
"""


def run_cell(kc, code: str, timeout: float = 900) -> tuple[dict, list[str], float]:
    msg_id = kc.execute(code)
    out: list[str] = []
    t0 = time.perf_counter()
    while True:
        msg = kc.get_iopub_msg(timeout=timeout)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        mt, c = msg["msg_type"], msg["content"]
        if mt == "stream":
            out.append(c["text"])
        elif mt == "error":
            out.append(f"ERROR {c['ename']}: {c['evalue']}")
        elif mt == "status" and c["execution_state"] == "idle":
            break
    reply = kc.get_shell_msg(timeout=timeout)
    return reply["content"], out, time.perf_counter() - t0


def main() -> None:
    root = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    sys.path.insert(0, str(HERE))
    import s5lib

    s5lib.s4_on_path()
    import q3_throughput as q

    s5lib.wait_idle(q, int(q.smi("memory.used")[0]), "jupyter")
    km = KernelManager(
        kernel_cmd=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    )
    km.start_kernel(cwd=str(HERE))
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=120)
    results: list[dict] = []
    try:
        rep, out, dt = run_cell(kc, SETUP.format(here=str(HERE), root=root))
        print("setup:", rep["status"], "".join(out).strip()[-200:], f"{dt:.1f}s", flush=True)
        assert rep["status"] == "ok"
        for wait in ("spin", "hybrid"):
            for i in range(rounds):
                msg_id = kc.execute(textwrap.dedent(LONG.format(wait=wait)))
                time.sleep(3.0 + 1.3 * i)  # interrupt at different points in the render
                t_int = time.perf_counter()
                km.interrupt_kernel()
                ename = None
                while True:
                    msg = kc.get_iopub_msg(timeout=300)
                    if msg["parent_header"].get("msg_id") != msg_id:
                        continue
                    if msg["msg_type"] == "error":
                        ename = msg["content"]["ename"]
                    if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                        break
                t_idle = time.perf_counter()
                kc.get_shell_msg(timeout=60)
                _, out, _ = run_cell(kc, CHECK)
                check = json.loads(
                    next(line for line in "".join(out).splitlines() if line.startswith("CHECK "))[
                        6:
                    ]
                )
                _, out, _ = run_cell(kc, RERUN)
                rerun = json.loads(
                    next(line for line in "".join(out).splitlines() if line.startswith("RERUN "))[
                        6:
                    ]
                )
                rec = dict(
                    wait=wait,
                    round=i,
                    error=ename,
                    interrupt_to_idle_s=round(t_idle - t_int, 4),
                    **check,
                    rerun=rerun,
                )
                results.append(rec)
                print(json.dumps(rec), flush=True)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    (HERE / "results" / "jupyter_interrupt.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

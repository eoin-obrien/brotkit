"""Q2: compare resource numbers from every source, for every (tier, kernel), on sm_86.

driver   : dispatcher.get_regs_per_thread / get_local_mem_per_thread (cuFuncGetAttribute; needs GPU)
ptxas    : compile_ptx(cc) -> wheel ptxas -v
njl      : compile_ptx(cc) -> nvJitLink -verbose (cuda.bindings.nvjitlink)
lto      : dispatcher.inspect_lto_ptx -> wheel ptxas -v  (the PTX the dispatcher actually links)
Also: CALL count in the dispatcher's SASS (inspect_sass) vs in the offline cubin.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import gen
from cuda.bindings import nvjitlink
from numba_cuda_mlir import cuda

BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".venv/lib/python3.15/site-packages/nvidia/cu13/bin"
)
os.environ["PATH"] = BIN + os.pathsep + os.environ["PATH"]  # inspect_sass needs nvdisasm
RE_REG = re.compile(r"[Uu]sed (\d+) registers")
RE_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


def ptxas(ptx, cc=86, extra=()):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        c = os.path.join(d, "k.cubin")
        pathlib.Path(p).write_text(ptx.rstrip("\x00"))
        r = subprocess.run(
            [f"{BIN}/ptxas", f"-arch=sm_{cc}", "-v", *extra, p, "-o", c],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        sass = subprocess.run([f"{BIN}/nvdisasm", "-c", c], capture_output=True, text=True).stdout
    entry = r.stderr.split("Compiling entry function")[-1]
    return dict(
        regs=int(RE_REG.search(entry)[1]),
        stack_spill=tuple(map(int, RE_SPILL.search(entry).groups())),
        calls=sass.count(" CALL"),
    )


def njl(ptx, cc=86, extra=()):
    o = [f"-arch=sm_{cc}", "-verbose", *extra]
    h = nvjitlink.create(len(o), o)
    data = ptx.rstrip("\x00").encode()
    nvjitlink.add_data(h, nvjitlink.InputType.PTX, data, len(data), "k.ptx")
    nvjitlink.complete(h)
    log = bytearray(nvjitlink.get_info_log_size(h))
    nvjitlink.get_info_log(h, log)
    nvjitlink.destroy(h)
    log = log.decode().rstrip("\x00")
    last = log.strip().splitlines()[-1]  # "info : used N registers, ... N stack, ... N bytes lmem"
    return dict(
        regs=int(RE_REG.search(last)[1]),
        stack=int(re.search(r"(\d+) stack", last)[1]),
        lmem=int(re.search(r"(\d+) bytes lmem", last)[1]),
    )


rows = {}
tiers = sys.argv[1].split(",") if len(sys.argv) > 1 else list(gen.TIERS)
for tier in tiers:
    for kern in gen.KERNELS:
        sig = gen.signature(tier)
        k = gen.load(tier, kern)
        d = cuda.jit(k.py_func)
        d.compile(sig)
        row = dict(
            driver=dict(
                regs=d.get_regs_per_thread(sig),
                local=d.get_local_mem_per_thread(sig),
                shared=d.get_shared_mem_per_block(sig),
                maxtpb=d.get_max_threads_per_block(sig),
                calls=d.inspect_sass(sig).count(" CALL"),
            ),
        )
        ptx, _ = cuda.compile_ptx(k.py_func, sig, cc=(8, 6))
        row["ptxas"] = ptxas(ptx)
        row["njl"] = njl(ptx)
        row["lto_ptxas"] = ptxas(d.inspect_lto_ptx(sig))
        rows[f"{tier}/{kern}"] = row
        drv, pt, njl, lto = row["driver"], row["ptxas"], row["njl"], row["lto_ptxas"]
        print(
            f"{tier:5} {kern:3} driver={drv['regs']:3} lmem={drv['local']:4} "
            f"calls={drv['calls']:2} | ptxas={pt['regs']:3} c={pt['calls']:2} "
            f"njl={njl['regs']:3} | njl_lmem={njl['lmem']} stack={njl['stack']} | "
            f"lto_ptxas={lto['regs']:3} c={lto['calls']}",
            flush=True,
        )
suffix = "_" + sys.argv[1].replace(",", "_") if len(sys.argv) > 1 else ""
pathlib.Path(f"results/q2_compare{suffix}.json").write_text(json.dumps(rows, indent=1))

"""nvJitLink: verbose info log loses the ptxas spill line on a compute-cache hit.

Links the same PTX twice in one process with -verbose -Xptxas=-v and prints the info logs.
The first link runs ptxas; the second is served from the CUDA compute cache (~/.nv/ComputeCache)
and its log lacks the "bytes spill stores, ... bytes spill loads" line.
Needs only cuda-bindings and the nvJitLink library (e.g. pip install cuda-bindings
nvidia-nvjitlink); no GPU is needed. Run it twice: on the second run both links are hits.
"""

import sys
import uuid

from cuda.bindings import nvjitlink

ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_86"
TAG = uuid.uuid4().hex  # makes the PTX new to the compute cache on every run of this script

PTX = f"""//
.version 8.0
.target {ARCH}
.address_size 64

// unique per run: {TAG}
.visible .entry k(.param .u64 out)
{{
    .reg .b64 %rd<3>;
    .reg .f32 %f<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f3F800000;
    add.f32 %f2, %f1, %f1;
    st.global.f32 [%rd2], %f2;
    ret;
}}
"""


def link_log(ptx: str) -> str:
    opts = [f"-arch={ARCH}", "-verbose", "-Xptxas=-v"]
    h = nvjitlink.create(len(opts), opts)
    try:
        data = ptx.encode()
        nvjitlink.add_data(h, nvjitlink.InputType.PTX, data, len(data), "k.ptx")
        nvjitlink.complete(h)
        log = bytearray(nvjitlink.get_info_log_size(h))
        nvjitlink.get_info_log(h, log)
    finally:
        nvjitlink.destroy(h)
    return log.decode(errors="replace").rstrip("\x00")


print("nvJitLink", nvjitlink.version(), "arch", ARCH)
for attempt in ("first link (cache miss)", "second link, same PTX (cache hit)"):
    log = link_log(PTX)
    print(f"--- {attempt}: spill line present = {'spill stores' in log}")
    print(log)

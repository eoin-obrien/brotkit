# Draft upstream issues for NVIDIA/numba-cuda-mlir

Filed: 1 as [#492](https://github.com/NVIDIA/numba-cuda-mlir/issues/492) (with the low-priority notes at the end), 2 as [#493](https://github.com/NVIDIA/numba-cuda-mlir/issues/493),
with fresh standalone repros. 2 was then rewritten with S2's confirmed cases (more options,
function constants) and posted as a comment on #493:
<https://github.com/NVIDIA/numba-cuda-mlir/issues/493#issuecomment-7160800477>. Compiling
without a GPU (S4's bug 1) is covered by #492, so it has no issue of its own. 3 is NVIDIA's bug,
not numba-cuda-mlir's: the report below is ready for NVIDIA's bug portal (filed by the
maintainer), with the standalone repro in `nvjitlink_cache_repro.py`.

Drafted in spike S4 (docs/design/spikes/S4.md). Three independent problems; they may be better
filed as three issues. Versions: numba-cuda-mlir 0.6.0, cuda-bindings 13.5.0,
nvidia-nvjitlink 13.5.37, CPython 3.15.2, Linux x86-64 (WSL2 host; Debian bookworm container).

---

## 1. `compile_ptx(..., cc=(M, m))` needs a CUDA driver and a device

**Expected.** Compiling to PTX for an explicitly given compute capability works on a machine
with no GPU and no driver (CI runners, build containers). numba-cuda's `compile_ptx` supports
this use.

**Actual.** It raises `CudaSupportError` at driver init:

```text
# container with no /dev/nvidia*, no /dev/dxg, no libcuda.so
CudaSupportError: Error at driver init:
CUDA driver library cannot be found.

# host with a driver but CUDA_VISIBLE_DEVICES=""
CudaSupportError: Error at driver init: Call to cuInit results in CUDA_ERROR_NO_DEVICE
```

**Repro.**

```python
from numba_cuda_mlir import cuda, types


def k(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] * 2.0


ptx, _ = cuda.compile_ptx(k, (types.float32[::1],), cc=(8, 6))  # fails without a driver
```

```text
compiler.py:429 compile -> _compile_and_optimize -> descriptor.py:3251 _compile_public
  -> descriptor.py:2277 _resolve_target_options -> tools.py:64 resolve_target_options
  -> tools.py:43 resolve_gpu_target:  host_cc = get_gpu_compute_capability(tuple)
  -> tools.py:160 get_context() -> cuCtxGetCurrent -> CudaSupportError
```

**Cause.** `tools.resolve_gpu_target` looks up the *host* device's cc unconditionally, even
when `chip` is set; it uses it only to choose `linker_cc`/`linker_arch` when the target is
older than the host.

**Second site (device functions).** With that lookup stubbed out, kernels whose device
functions are compiled as separate callees still fail:

```text
descriptor.py:3556 get_call_template -> descriptor.py:3381 _compile_as_device_callee
  -> descriptor.py:3346 _compile_device_callee -> mlir_compiler.py:315 compile_mlir
  -> tools.py:35 resolve_gpu_target(targetoptions)   # callee targetoptions have no 'chip'
```

So device callees re-resolve the target from the current device instead of inheriting the
caller's `chip`. On a GPU host this probably means callees of a cross-arch `compile_ptx` (for
example `cc=(10, 0)` on an sm_86 machine) are compiled for the host's arch. We didn't confirm
any wrong output from that; the PTX `.target` was correct in our tests.

**Suggested fix.** Resolve `host_cc` lazily, only when no `chip` is given or when a link
actually needs it, and treat "no device" as "no host constraint". Propagate the caller's `chip`
into device-callee target options.

**With both lookups stubbed** (diagnostic monkeypatch, not a proposal), the rest of the
toolchain runs with no GPU: `compile_ptx` for sm_86/100/120, nvJitLink linking with
`-maxrregcount`, and cuobjdump/nvdisasm from the `nvidia-cuda-*` wheels.

---

## 2. `cache=True` serves stale builds: the index key misses most target options and function constants

*Combined S4 + S2 write-up, replacing the original #493 text and the S2 follow-up. Posted as
[a comment on #493](https://github.com/NVIDIA/numba-cuda-mlir/issues/493#issuecomment-7160800477),
led by the standalone repro.*

**Versions.** numba-cuda-mlir 0.6.0 (`[cu13]`), cuda-bindings 13.5.0, nvidia-nvjitlink /
nvidia-nvvm 13.5.37, CPython 3.15.2, Linux x86-64 (WSL2), RTX 3090 Ti (sm_86).

**Expected.** With `cache=True`, a dispatcher whose target options or function differ from a
cached build misses the cache and compiles.

**Actual.** The index key (`caching.py: MLIRCache._index_key` plus `Cache._index_key`) is the
signature, `codegen.magic_tuple()` (CUDA runtime version and cc), `sha256(co_code)`, a hash of
the closure variables, and `option_key` = (`_lto_explicit`, `lto`, `chip`, `launch_bounds`).
Every other target option is left out, and so are the function's constants. A process that asks
for a different build gets the cached one.

**Repro.** For each row, three fresh interpreters share one kernel file (a 60-register f64
kernel):

1. build variant X with `cache=True`, which writes the cache;
2. build variant Y with `cache=True`, which should miss;
3. build Y with `cache=False`, which is the reference.

"Stale" means process 2 reported a cache hit and its build differs from process 3's.

| Changed between X and Y | Process 2 | Stale? |
| --- | --- | --- |
| `max_registers=32` | cache hit | **yes**: 60 registers, reference 32 |
| `ptxas_options="--maxrregcount=40"` | cache hit | **yes**: 60 registers, reference 40 |
| `lineinfo=True` | cache hit | **yes**: PTX without `.loc`, reference has line info |
| `opt=False` | cache hit | **yes**: 60 registers, reference 73 (unoptimised) |
| `fastmath=True` | cache hit | not on sm_86, where fastmath doesn't change the build (#486); the key ignores it, so on sm_100+, where fastmath does change codegen, it would be stale |
| `launch_bounds=(256, 4)` (control) | miss | no: in the key |
| a constant in the function body, file size and mtime unchanged | cache hit | **yes**: the kernel writes the old constant. `co_code` doesn't include `co_consts`, and the source stamp (mtime, size) didn't change |

S4's original two-dispatcher repro shows the same thing for `max_registers` within one file: a
cached `max_registers=24` build served to an undecorated dispatcher, 24 registers instead of 30.

The scripts and raw results are in brotkit's repo. The S2 scripts are
`spikes/S2_cache/a_keybugs.py` and `a_consts.py`, with results in
`spikes/S2_cache/results/a_keybugs.json` and `a_consts.json`; S4's is
`spikes/S4_registers/probe_caps.py`. A minimal standalone repro for the issue:

```python
# kmod.py
import numpy as np
from numba_cuda_mlir import cuda


def k(out, n):  # 8 live f64 accumulators: 33 registers uncapped on sm_86
    i = cuda.grid(1)
    a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = np.float64(i)
    for j in range(n):
        t = np.float64(j)
        a0 = a0 * 0.99 + t
        a1 = a1 * 0.98 + t
        a2 = a2 * 0.97 + t
        a3 = a3 * 0.96 + t
        a4 = a4 * 0.95 + t
        a5 = a5 * 0.94 + t
        a6 = a6 * 0.93 + t
        a7 = a7 * 0.92 + t
    out[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
```

```python
# run.py; run as `python run.py none`, then `python run.py capped`, in fresh processes
import sys
from numba_cuda_mlir import cuda, types
import kmod

sig = (types.float64[::1], types.int64)
opts = {"max_registers": 24} if sys.argv[1] == "capped" else {}
d = cuda.jit(cache=True, **opts)(kmod.k)
d.compile(sig)
print(sys.argv[1], "regs", d.get_regs_per_thread(sig), "cache hits", sum(d._cache_hits.values()))
# none   regs 33 cache hits 0
# capped regs 33 cache hits 1    <- the uncapped build; expected a miss and <= 24
# (the same capped dispatcher with cache=False: 24 registers)
```

**Checked and not a problem:** the index records `numba_cuda_mlir.__version__` and discards
itself on a version change. A changed constant imported from *another module* is also served
stale, but that is Numba's documented limitation (cache invalidation tracks only the function's
own file), so it isn't part of this report.

**Suggested fix.**

1. Build `option_key` from every target option that reaches `compile_mlir`, `optimize` or the
   `Linker`, as an allowlist that fails closed: an option the key doesn't know about disables
   caching rather than being ignored. Today that means at least `max_registers`,
   `ptxas_options`, `lineinfo`, `debug`, `opt`, `fastmath`, `forceinline`, `inline`,
   `opt_level`, `features` and `experimental_ast_transforms`.
2. Hash `co_consts` (recursively, for nested code objects) alongside `co_code`.

**Impact for us.** brotkit doesn't use `cache=True` at all (it stores and keys its own cubins),
so nothing here blocks us. The report is for other users, who get a silently wrong build: a
capped kernel where they asked for none, unoptimised code, or an old constant.

---

## 3. (FYI, NVIDIA nvJitLink) the verbose info log omits ptxas output on a compute-cache hit

Not numba-cuda-mlir's bug, but it affects anyone who reads resource usage from the link log.
nvJitLink caches ptxas results in the CUDA compute cache (`~/.nv/ComputeCache`, controlled by
`CUDA_CACHE_*`). On a hit, `-verbose -Xptxas=-v` still prints the final
`used N registers, ... N stack` summary but omits the ptxas block that carries
`N bytes spill stores, N bytes spill loads`. `CUDA_CACHE_DISABLE=1` restores the block, but only
if it's set before the library loads; setting it at runtime has no effect. brotkit works around
this by appending a unique comment to the PTX before linking.

Report for NVIDIA's bug portal (repro: `nvjitlink_cache_repro.py` in this directory, needs only
`cuda-bindings` and `nvidia-nvjitlink`, no GPU):

> **Title:** nvJitLink: `-verbose -Xptxas=-v` info log omits the ptxas function-properties block (spill stores/loads) on a compute-cache hit
>
> **Product / component:** CUDA Toolkit, nvJitLink (libnvJitLink)
>
> **Environment**
> - nvJitLink 13.5 (pip wheel `nvidia-nvjitlink` 13.5.37), via `cuda-bindings` 13.5.0
> - Linux x86-64 (Ubuntu under WSL2, kernel 6.18), driver 610.88; no GPU needed to reproduce
> - Target sm_86 (reproduces with any target)
>
> **Description**
>
> With `-verbose -Xptxas=-v`, nvJitLink's info log includes ptxas's per-function block:
>
> ```
> ptxas info    : Function properties for k
> ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
> ptxas info    : Used 8 registers, ...
> ```
>
> When the same input is linked again and the ptxas result is served from the CUDA compute cache
> (`~/.nv/ComputeCache`), the whole `ptxas info` block is missing. Only nvJitLink's own summary
> remains (`used N registers, ... N stack, ...`), and it has no spill counts. So a tool that reads
> spill counts from the link log gets nothing on a hit, and could easily report 0 by mistake.
>
> Setting `CUDA_CACHE_DISABLE=1` brings the block back, but only if it's set before the library
> loads. Changing it at runtime has no effect. The only workaround we found is to make every input
> unique (we append a random comment to the PTX), which defeats the cache.
>
> **Expected:** the info log is the same whether ptxas ran or its result came from the compute
> cache, including the spill stores and loads. Alternatively, the summary line could carry the
> spill counts.
>
> **Steps to reproduce:** `pip install cuda-bindings nvidia-nvjitlink`, then run the attached
> `nvjitlink_cache_repro.py`. It links the same small PTX kernel twice in one process. Output:
>
> ```
> nvJitLink (13, 5) arch sm_86
> --- first link (cache miss): spill line present = True
> ptxas info    : 0 bytes gmem
> ptxas info    : Compiling entry function 'k' for 'sm_86'
> ptxas info    : Function properties for k
> ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
> ptxas info    : Used 8 registers, used 0 barriers, 360 bytes cmem[0]
> ptxas info    : Compile time = 3.784 ms
> info    : 0 bytes gmem
> info    : Function properties for 'k':
> info    : used 8 registers, used 6 barriers, 0 stack, 0 bytes smem, 360 bytes cmem[0], 0 bytes lmem
>
> --- second link, same PTX (cache hit): spill line present = False
> info    : 0 bytes gmem
> info    : Function properties for 'k':
> info    : used 8 registers, used 6 barriers, 0 stack, 0 bytes smem, 360 bytes cmem[0], 0 bytes lmem
> ```
>
> **Impact:** register and spill reporting built on nvJitLink (for example, deciding whether to
> split a fused kernel when it spills) silently loses the spill numbers whenever the compute cache
> hits.

---

## Not bugs, noted for the docs

- `cuda.compile(..., output="cubin")` is unsupported and `output="ltoir"` rejects kernels ("The
  C ABI is not supported for kernels"), so cubin/SASS for another arch has to come from
  `compile_ptx` plus nvJitLink or ptxas.
- `forceinline=True` on `compile_ptx` raises "Can only designate forced inlining in LTO-IR".
- `dispatcher.get_regs_per_thread()` and friends query the driver (`cuFuncGetAttribute`) and so
  need a device.
- When device functions are compiled as separate callees (seen with the qf kernels), their
  `.func` bodies stay in the kernel's PTX although the entry has them inlined and nothing calls
  them. ptxas drops them, but anyone counting instructions in the PTX has to look at the
  `.entry` body only.

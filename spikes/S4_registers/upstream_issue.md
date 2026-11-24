# Draft upstream issue for NVIDIA/numba-cuda-mlir (NOT FILED)

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

## 2. `cache=True` ignores `max_registers` (and other link options) in the index key

**Expected.** Changing `max_registers` changes the cached artefact.

**Actual.** The cache key (`caching.py:MLIRCache._index_key`) includes `lto`, `chip` and
`launch_bounds` but not `max_registers` (nor `ptxas_options`). A dispatcher without a cap
loads a capped build from cache:

```python
# two_kernels.py
@cuda.jit(cache=True, max_registers=24)
def capped(x, out): ...  # body B
@cuda.jit(cache=True)
def uncapped(x, out): ...  # same function object as above, re-decorated
```

```text
process 1: capped.compile(sig)    -> regs 24, cache miss (written)
process 2: uncapped.compile(sig)  -> regs 24, cache HIT   # expected: 30, a miss
```

(Full script: `spikes/S4_registers/probe_caps.py cache_write` then `cache_read`.)

**Suggested fix.** Add `max_registers` and `ptxas_options` (and anything else passed to the
`Linker`) to `option_key`.

---

## 3. (FYI, NVIDIA nvJitLink) the verbose info log omits ptxas output on a compute-cache hit

Not numba-cuda-mlir's bug, but it affects anyone who reads resource usage from the link log.
nvJitLink caches ptxas results in the CUDA compute cache (`~/.nv/ComputeCache`, controlled by
`CUDA_CACHE_*`). On a hit, `-verbose -Xptxas=-v` still prints the final
`used N registers, ... N stack` summary but omits the ptxas block that carries
`N bytes spill stores, N bytes spill loads`. `CUDA_CACHE_DISABLE=1` restores the block, but only
if it's set before the library loads; setting it at runtime has no effect. brotkit works around
this by appending a unique comment to the PTX before linking.

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

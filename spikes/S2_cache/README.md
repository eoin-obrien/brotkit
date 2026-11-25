# Spike S2: kernel cache (throwaway)

Write-up: [`docs/design/spikes/S2.md`](../../docs/design/spikes/S2.md). Decision:
ADR-0008. Library code that came out of it: `brotkit.kernels.cache` (GPU-free store, key,
locking) and `brotkit.backends.cuda.CudaCacheBackend` (Numba + nvJitLink build, cuda.core
load). Tests: `tests/test_kernel_cache.py` (CPU, fake backend), `tests/test_cuda_cache.py` (gpu).

S4's kernels are reused through `gen_s2.py`, which imports `../S4_registers/gen.py` by path and
rewrites the kernel header to the raw-pointer ABI.

```bash
uv venv -p 3.15.2 && uv sync
uv pip install --no-deps -e ../..          # brotkit into this venv
```

| Script | What |
| --- | --- |
| `probe_carray.py` | Go/no-go: raw-pointer kernel via public `cuda.carray`; Numba launch and cuda.core launch of the nvJitLink cubin agree |
| `abi_resources.py [kernels]` | Struct vs pointer ABI: registers, spills, SASS escape-loop instruction / address / memory counts (`S2_PTR=ptr` or `ptr64`) |
| `abi_timing.py [rounds]` | Alternating throughput, struct vs pointer ABI (S4's timing and idle check; env `S2_KERNELS`, `S2_TIERS`, `S2_VARIANTS`, `S2_OUT`) |
| `lock_probe.py <dir> <label>` | flock / lockf / filelock counters, atomic replace under readers, dir rename, unlink-while-open |
| `a_keybugs.py` | Mechanism (a): which options Numba's `cache=True` key ignores (stale-build test per option, plus imported globals) |
| `a_consts.py <dir>` | Mechanism (a): constant changed with the file's size and mtime kept |
| `cold.py [N]` / `cold_child.py` | Cold cost, source string to launched kernel, stage breakdown, fresh processes |
| `warm.py <root> [N] [render]` / `warm_child.py` | Warm start in fresh processes, (b) brotkit store vs (a) Numba cache; `render=1` adds a full 3840x2160 render |

Results land in `results/`.

# Spike S4: register pressure, resource reporting, compiling without a GPU

Throwaway code for [docs/design/spikes/S4.md](../../docs/design/spikes/S4.md). Never library
code. The library part that came out of it is `brotkit.backends.resources` and
`brotkit.backends.cuda.resource_usage`.

## Setup

```bash
cd spikes/S4_registers
uv sync -p 3.15                       # own lock; same numba-cuda-mlir pin as the root [cuda] extra
uv pip install --no-deps -e ../..     # brotkit (resource_usage) into this venv
```

Before any timed step, check `nvidia-smi`. `q3_throughput.py` refuses to measure if anything
else is using the card (median utilisation > 5 % over 8 samples, or memory up > 1.5 GiB on the
baseline); it waits 20 s and re-checks, for up to 10 minutes per config. Every GPU
script calls `guard.check()` (aborts below 4 GB free VRAM or 5 GB free disk).

## Files

| File | What it does |
| --- | --- |
| `gen.py` | Kernel generator: tiers f32, f32rn, f64, df64, dd, qf, qfb x kernels Z (core), A (smooth), B (+DE), C (+trap), F6 (+stripe, TIA, binary), S2 (second split pass). `uv run python gen.py dd C` prints one |
| `probe_compile.py` | `compile_ptx` + `ptxas -v` for every (tier, kernel) at one cc: `uv run python probe_compile.py 120 qf,dd` |
| `q1_nogpu.py`, `Dockerfile.nogpu` | Q1: which compile/report calls work with no GPU and no driver |
| `q2_compare.py` | Q2: driver attributes vs ptxas vs nvJitLink vs LTO PTX, all kernels (GPU) |
| `q2_validate.py` | Q2: `resource_usage` + `occupancy` vs the driver, 35 kernels x 5 caps (GPU) |
| `probe_caps.py` | Q4 mechanics: per-kernel cap, both MLIR paths, launch bounds, cache-key bug |
| `q3_qf_exact.py`, `q3_deep_mismatch.py` | qf prototype: EFT exactness, accuracy vs mpmath, deep qf vs dd |
| `q3_throughput.py` | Q3/Q4 timings: `main` (all tiers/kernels, blocks 128/256) and `caps` |
| `q3_resources.py` | Compile-only resource table: every tier x kernel at sm_86/100/120 with occupancy (GPU host, launches nothing) |
| `q4_repeat.py` | Interleaved A/B repeats of the key cap pairs, to separate cap effects from run-to-run drift |
| `analyze.py` | Renders `results/q3_*.json` into the markdown tables in S4.md (`results/tables.md`) |
| `check_codegen.py` | Runs `brotkit.codegen.checks.find_bare_float_ops` over every generated module |
| `ptx_patterns.py` | Diagnostic: float instruction mix in the PTX (used to design `tests/test_cuda_ptx.py`) |
| `upstream_issue.md` | Upstream issue drafts (1 and 2 filed as numba-cuda-mlir#492 and #493) |
| `results/` | Raw JSON/logs behind every number in S4.md |

## Reproduce

```bash
# Q1 (no GPU needed for the container runs)
docker build -f Dockerfile.nogpu --build-context brotkit=../.. -t s4-nogpu .
docker run --rm -e CUDA_VISIBLE_DEVICES= -v "$PWD/results:/s4/results" s4-nogpu
docker run --rm -e S4_PATCH_HOSTCC=1 -v "$PWD/results:/s4/results" \
  -v "$PWD/q1_nogpu.py:/s4/q1_nogpu.py:ro" s4-nogpu /venv/bin/python q1_nogpu.py nogpu-container-patched
uv run python q1_nogpu.py host                          # GPU host, for comparison
CUDA_VISIBLE_DEVICES= uv run python q1_nogpu.py host-cvd-empty

# Q2
uv run python q2_compare.py && uv run python q2_validate.py

# Q4 mechanics (empty probe/cache before the cache pair)
uv run python probe_caps.py mech
uv run python probe_caps.py cache_write && uv run python probe_caps.py cache_read

# qf correctness
uv run python q3_qf_exact.py && uv run python q3_deep_mismatch.py

# throughput (resumes from results/q3_<set>.json if interrupted)
uv run python q3_throughput.py main
uv run python q3_throughput.py caps
uv run python q4_repeat.py
uv run python q3_resources.py
uv run python analyze.py > results/tables.md
uv run python check_codegen.py
```

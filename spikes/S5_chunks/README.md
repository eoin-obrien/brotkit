# Spike S5: chunked launches (throwaway)

Write-up: [`docs/design/spikes/S5.md`](../../docs/design/spikes/S5.md). Decision: ADR-0009.
Library code that came out of it: `brotkit.backends.chunks` (the chunk planner, GPU-free).
Tests: `tests/test_chunks.py` (CPU, simulated GPU built from this spike's measurements).

S4's kernels come through S2's `gen_s2.py`, and `gen_s5.py` rewrites their header to a
tile-shaped launch (`row0, col0, th, tw`). Kernels build and load through `brotkit`'s
`KernelCache`. `s5lib.py` holds the runner the scripts share: calibration on NaN coordinates in
scratch, the `tile_sum` readback, the wait modes, the deferred SIGINT and `RenderInterrupted`.

```bash
uv venv -p 3.15.2 && uv sync
uv pip install --no-deps -e ../..          # brotkit into this venv
```

| Script | What |
| --- | --- |
| `smoke.py <root>` | Chunked render against one launch: bit-identical layers, iteration sums against the host |
| `curve.py <root>` | Time against chunk size for all-interior tiles, per tier: latency floor, throughput, knee |
| `nan_tau.py <root>` | Latency from NaN probes against real interior pixels, kernels C and F6 |
| `sweep.py <root> [wait] [tiers] [views]` | 4K sweep, every tier and view, cold and warm start: chunks, overhead, longest chunk, bound check |
| `fixed_cost.py <root>` | Per-chunk pipeline cost and CPU time on 200 tiny chunks, by wait mode |
| `tiles.py <root>` | Bands against 256 and 512 px tiles, f32 and qf, worst-case views |
| `share.py <root>` | A second process's kernel latency while this one renders, chunk targets 10–400 ms |
| `tdr.py <root>` | Single long launches up to 75% of `TdrDelay`, each sized from launches run just before it |
| `jupyter_interrupt.py <root> [rounds]` | Headless ipykernel: interrupt a long render, check the partial result, re-run |
| `manual_check.ipynb` | The checklist in S5.md, for real JupyterLab: `uv run --with jupyterlab jupyter lab manual_check.ipynb` |

`<root>` is a kernel cache directory (`KernelCache` store). Every GPU script runs S4's idle
check before measuring. Results land in `results/`.

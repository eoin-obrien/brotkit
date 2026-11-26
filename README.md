<!--
Hero strip (backlog 1.23): four renders produced in CI from docs/hero/*.recipe.json, so the
front-page image is also a golden test. Absolute URLs, because PyPI renders this README too.
Hero PNGs are ordinary git files (not LFS), optimised to ~300 KB each: see ADR-0003.

<p align="center">
  <img src="https://raw.githubusercontent.com/eoin-obrien/brotkit/main/docs/hero/mandelbrot-de.png" width="24%" alt="Mandelbrot set with distance-estimate shading">
  <img src="https://raw.githubusercontent.com/eoin-obrien/brotkit/main/docs/hero/nebulabrot.png" width="24%" alt="Nebulabrot">
  <img src="https://raw.githubusercontent.com/eoin-obrien/brotkit/main/docs/hero/newton-basins.png" width="24%" alt="Newton basins">
  <img src="https://raw.githubusercontent.com/eoin-obrien/brotkit/main/docs/hero/oblique-4d-slice.png" width="24%" alt="Oblique 4D slice between the Mandelbrot and Julia planes">
</p>
-->

# brotkit

**GPU-accelerated, SymPy-driven fractal rendering for notebooks.**

<!-- Package -->
<!-- [![PyPI](https://img.shields.io/pypi/v/brotkit.svg)](https://pypi.org/project/brotkit/) -->
[![Python](https://img.shields.io/badge/python-3.14%20%7C%203.15-blue.svg)](https://github.com/eoin-obrien/brotkit/blob/main/pyproject.toml)
[![Licence: BSD-3-Clause](https://img.shields.io/badge/licence-BSD--3--Clause-blue.svg)](https://github.com/eoin-obrien/brotkit/blob/main/LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

<!-- Quality -->
[![CI](https://github.com/eoin-obrien/brotkit/actions/workflows/ci.yml/badge.svg)](https://github.com/eoin-obrien/brotkit/actions/workflows/ci.yml)
<!-- [![Coverage](https://codecov.io/gh/eoin-obrien/brotkit/graph/badge.svg)](https://codecov.io/gh/eoin-obrien/brotkit) -->
<!-- [![Docs](https://github.com/eoin-obrien/brotkit/actions/workflows/docs.yml/badge.svg)](https://github.com/eoin-obrien/brotkit/actions/workflows/docs.yml) -->

<!-- Tooling -->
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/)

<!-- Agent-ready -->
[![AGENTS.md](https://img.shields.io/badge/AGENTS.md-ready-black)](https://github.com/eoin-obrien/brotkit/blob/main/AGENTS.md)
<!-- [![llms.txt](https://img.shields.io/badge/llms.txt-available-black)](https://brotkit.readthedocs.io/llms.txt) -->

> [!NOTE]
> brotkit is in early development (phase 0). Nothing below is released yet; this README
> describes what's being built. See the [design spec](https://github.com/eoin-obrien/brotkit/blob/main/docs/design/spec.md)
> and the [roadmap](https://github.com/eoin-obrien/brotkit/milestones).

## Highlights

- **Any SymPy formula, with analytic derivatives.** Write f(z, c) in SymPy. brotkit
  differentiates it, lowers it to real arithmetic and compiles one fused kernel for the CPU
  (Numba) or GPU (numba-cuda).
- **Render once, colour many.** A render saves raw data layers (iterations, final z, log|dz|,
  orbit statistics), so you can try 50 palettes on a 16k image without re-iterating.
- **Correct anti-aliasing.** Adaptive scrambled-Sobol supersampling, with every sample
  coloured and then filtered in linear light.
- **Precision tiers for deep zooms.** f32, df64, f64, double-double and quad-float picked
  automatically from the zoom, with perturbation to come.
- **4D animation.** Parameter paths, SO(4) rotations through the (z₀, c) space between
  Mandelbrot and Julia slices, and a frame pipeline straight to video.

## Install

```bash
uv add brotkit             # CPU (Numba)
uv add "brotkit[cuda]"     # NVIDIA GPU (numba-cuda + CUDA runtime wheels)
uv add "brotkit[video]"    # video encoding (PyAV)
```

The GPU extra needs only an NVIDIA driver; the CUDA runtime and compiler come as wheels.
The minimum driver and compute capability will be pinned after spike S1. Check your setup
with `brotkit doctor` (backlog 1.20).

## Quickstart

Coming with the walking skeleton (backlog 1.8 and 1.23): at most 10 lines that render a
Mandelbrot set and then recolour it without re-rendering, each followed by its output image.
Every code block in this README will run in CI.

## Gallery and features

The gallery arrives with the docs site (backlog 1.23).

| Area | What's planned |
| --- | --- |
| Families | Escape time (Mandelbrot, Julia, Multibrot, Burning Ship, any SymPy f(z, c)); convergent (Newton, Halley, Nova, Magnet); density (Buddhabrot, Nebulabrot, inverse iteration, strange attractors) |
| Colourings | Smooth iteration, distance estimate, binary/n-ary decomposition, stripe/TIA/curvature averages, orbit traps, potential, interior period/multiplier, Lyapunov |
| Precision | `f32`, `df64`, `f64`, `dd`, `qf`, with `auto` selection; perturbation + BLA for unbounded zoom |
| Animation | Keyframe tracks, log-width zooms, complex-parameter paths, SO(4) rotations, exponential-map zooms, iteration-folding views |

## CLI

```bash
brotkit render recipe.json
```

(backlog 1.20)

## Docs

The documentation site and its [`llms.txt`](https://llmstxt.org/) arrive with backlog 1.23.

<!-- [Documentation](https://brotkit.readthedocs.io) · [llms.txt](https://brotkit.readthedocs.io/llms.txt) -->

## Contributing, citing, licence

- Contributions are welcome. See [CONTRIBUTING.md](https://github.com/eoin-obrien/brotkit/blob/main/CONTRIBUTING.md),
  and if you work with a coding agent, [AGENTS.md](https://github.com/eoin-obrien/brotkit/blob/main/AGENTS.md).
- If you use brotkit in research, please cite it using [CITATION.cff](https://github.com/eoin-obrien/brotkit/blob/main/CITATION.cff)
  (GitHub's "Cite this repository" button).
- Licensed under the [BSD-3-Clause licence](https://github.com/eoin-obrien/brotkit/blob/main/LICENSE).

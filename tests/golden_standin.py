"""Run S4's spike kernels on a golden view: stand-ins until the phase 1 renderer exists.

Replace this module (and tests/test_golden_gpu.py's use of it) when the phase 1 renderer
lands; every adjustment marked SPIKE below goes with it. The kernels are
tests/fixtures/codegen/s4_<tier>_B_r1e6.py.txt: S4's kernel B (smooth + DE), emitted by
`spikes/S4_registers/gen.py <tier> B 1e12`, so R = 1e6 on every tier.

    uv run python -m tests.golden_standin     # calibration table: errors with no budgets

The kernel module's own `cuda` keeps numba_cuda_mlir imports out of tests (TID251).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import mpmath
import numpy as np

from brotkit.testing import golden as g
from brotkit.testing.reference import RefFrame

FIXTURES = Path(__file__).parent / "fixtures" / "codegen"
WORDS = {
    "f32": ("f32", 1),
    "f64": ("f64", 1),
    "df64": ("f32", 2),
    "dd": ("f64", 2),
    "qf": ("f32", 4),
}
NLF, NLU = 7, 3  # S4's output layer counts (gen.py LF, LU)
_loaded: dict[str, Any] = {}


def load_kernel(tier: str, workdir: Path) -> Any:
    """Import a fixture kernel module from a real file (numba needs source on disk)."""
    if tier in _loaded:
        return _loaded[tier]
    name = f"s4_{tier}_B_r1e6"
    path = workdir / f"{name}.py"
    path.write_text((FIXTURES / f"{name}.py.txt").read_text())
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec
    assert spec.loader
    mod: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _loaded[tier] = (mod, getattr(mod, f"k_{tier}_B"))
    return _loaded[tier]


def split_words(x: Any, word: str, count: int) -> list[float]:
    """Split an mpf into `count` non-overlapping words of `word` precision (hi first)."""
    bits = 24 if word == "f32" else 53
    out: list[float] = []
    rest = x
    for _ in range(count):
        with mpmath.workprec(bits):
            w = +rest  # round to the word's precision
        out.append(float(w))
        rest = rest - w
    return out


def kernel_origin(meta: dict[str, Any]) -> tuple[Any, Any]:
    """SPIKE: S4's kernel maps pixel (x, y) to c0 + ((x - W//2) dx, (H//2 - y) dx), with no
    half-pixel offset. That equals ADR-0011's centre of pixel (W//2, H//2) as c0, so pass
    that point exactly (tests/test_golden.py checks the identity on every view)."""
    w, h = meta["pixels"]
    frame = RefFrame(center=tuple(meta["center"]), width=meta["width"], pixels=(w, h))
    return frame.pixel_centre(w // 2, h // 2)


def run_standin(
    tier: str, golden: g.Golden, workdir: Path, *, half_pixel: bool = True
) -> dict[str, np.ndarray]:
    """Render `golden`'s view with the tier's stand-in kernel; layers in golden units.
    `half_pixel=False` drops the SPIKE origin shift (the spike's own convention), as a
    deliberately wrong renderer for the negative test."""
    meta = dict(golden.meta)
    if not meta["standin"]:
        raise ValueError(f"{golden.name} has no stand-in (Julia or rotated view)")
    mod, kern = load_kernel(tier, workdir)
    word, count = WORDS[tier]
    ft = np.float32 if word == "f32" else np.float64
    w, h = meta["pixels"]
    with mpmath.workprec(int(meta["precision_bits"])):
        if half_pixel:
            c0r, c0i = kernel_origin(meta)
        else:
            c0r, c0i = mpmath.mpf(meta["center"][0]), mpmath.mpf(meta["center"][1])
        args = [*split_words(c0r, word, count), *split_words(c0i, word, count)]
        dx = float(mpmath.mpf(meta["width"]) / w)
    out_f = mod.cuda.device_array((NLF, h, w), dtype=np.float32)
    out_u = mod.cuda.device_array((NLU, h, w), dtype=np.uint32)
    block = (16, 16)
    grid = ((w + 15) // 16, (h + 15) // 16)
    kern[grid, block](
        *(ft(a) for a in args), ft(dx), np.int32(meta["max_iter"]), np.int32(0), out_f, out_u
    )
    f = out_f.copy_to_host()
    u = out_u.copy_to_host()
    n = u[0].astype(np.uint32)
    # SPIKE: S4's mu is n + 1 - log2(ln|z|), without the spec's / ln R.
    mu = f[0].astype(np.float64) + math.log2(math.log(float(meta["escape_radius"])))
    # SPIKE: S4's DE is |z| ln|z| / |dz|, half the spec's `estimate` convention.
    de = 2.0 * f[1].astype(np.float64)
    return {"n": n, "mu": mu, "de": de}


def main() -> None:
    manifest = g.load_manifest()
    rows: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, meta in manifest["views"].items():
            if not meta["standin"]:
                continue
            gold = g.load_golden(name)
            for tier in g.TIERS:
                if meta["tiers"][tier]["expect"] == "n/a":
                    continue
                layers = run_standin(tier, gold, Path(tmp))
                loose = g.Tolerance(mu_eps=math.inf, de_eps=math.inf, computed_in="f32")
                r = g.compare(layers, gold, tier, tolerance=loose)
                print(r.summary(), flush=True)
                rows.append(
                    f"| {name} | {tier} | {meta['tiers'][tier]['expect']} | "
                    f"{r.decidable_fraction:.3f} | {r.n_mismatches} | "
                    f"{'-' if r.mu_max_eps is None else f'{r.mu_max_eps:.3g}'} | "
                    f"{'-' if r.de_max_eps is None else f'{r.de_max_eps:.3g}'} ({r.de_status}) |"
                )
    print("\n| view | tier | expect | decidable | n mismatches | mu max (f32 eps) | DE max |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

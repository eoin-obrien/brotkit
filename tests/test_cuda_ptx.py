"""PTX snapshot checks and resource reporting on generated kernels (GPU suite).

These need only `compile_ptx`, but numba-cuda-mlir 0.6.0 creates a CUDA context even when the
target cc is explicit (spike S4, Q1), so they can't run on GPU-less hosted runners yet. They're
marked `gpu` and run by the local GPU check (tools/gpu_check.py). Once upstream fixes the
host-cc lookup, drop the marker (keep the dispatcher-comparison test GPU-only).

Fixtures are kernel modules emitted by spike S4's generator (spikes/S4_registers/gen.py):
- `*_Z_rn`: the multi-word escape loop with no finaliser, so the PTX holds no libdevice maths
  (log/sqrt/sin bring their own fma.rn and un-rounded ops) and must be fully rounding-explicit.
- `s4_f32_{A,B,C}_plain`: plain f32 kernels, which must contain no f64 instruction at all.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.gpu
pytest.importorskip("numba_cuda_mlir")

from brotkit.backends.cuda import compile_ptx, resource_usage, signature_args  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "codegen"
CCS = [(8, 6), (10, 0), (12, 0)]
OUT = "float32[:,:,::1], uint32[:,:,::1]"
SIGS = {
    "f32": f"void({', '.join(['float32'] * 2)}, float32, int32, int32, {OUT})",
    "df64": f"void({', '.join(['float32'] * 4)}, float32, int32, int32, {OUT})",
    "dd": f"void({', '.join(['float64'] * 4)}, float64, int32, int32, {OUT})",
    "qf": f"void({', '.join(['float32'] * 8)}, float32, int32, int32, {OUT})",
}
# Explicit cuda.fma sites after inlining the escape loop: z^2 needs w_sqr twice and w_mul once.
# One two_prod each for the 2-word tiers; the QD qf sqr has 4 and qf mul has 6 (2*4 + 6).
EXPECTED_FMA = {"df64": 3, "dd": 3, "qf": 14}

_FLOAT_OP = re.compile(
    r"^\s*(?:@!?%p\d+\s+)?((?:add|sub|mul|mad|fma)(?:\.\w+)*\.f(?:32|64))\b", re.M
)


def load(tmp_path: pathlib.Path, fixture: str) -> ModuleType:
    """Import a fixture kernel module from a real file (numba needs source on disk)."""
    path = tmp_path / fixture.removesuffix(".txt")
    path.write_text((FIXTURES / fixture).read_text())
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec
    assert spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def kernel(mod: ModuleType) -> Any:
    (name,) = [n for n in vars(mod) if n.startswith("k_")]
    return getattr(mod, name)


@pytest.mark.parametrize("cc", CCS, ids=lambda c: f"sm_{c[0]}{c[1]}")
@pytest.mark.parametrize("kern", ["A", "B", "C"])
def test_f32_kernels_have_no_f64_instructions(
    tmp_path: pathlib.Path, kern: str, cc: tuple[int, int]
) -> None:
    # Catches the int64 * float32 -> float64 promotion pitfall (S3). Kernels with stripe
    # average are excluded: libdevice sinf's large-argument reduction uses one mul.f64.
    ptx = compile_ptx(kernel(load(tmp_path, f"s4_f32_{kern}_plain.py.txt")), SIGS["f32"], cc)
    assert ".f64" not in ptx


@pytest.mark.parametrize("cc", CCS, ids=lambda c: f"sm_{c[0]}{c[1]}")
@pytest.mark.parametrize("tier", ["df64", "dd", "qf"])
def test_multiword_core_is_rounding_explicit(
    tmp_path: pathlib.Path, tier: str, cc: tuple[int, int]
) -> None:
    ptx = compile_ptx(kernel(load(tmp_path, f"s4_{tier}_Z_rn.py.txt")), SIGS[tier], cc)
    ops = _FLOAT_OP.findall(ptx)
    unrounded = [op for op in ops if op.split(".")[0] in ("add", "sub", "mul") and ".rn" not in op]
    assert unrounded == [], f"contractible float ops in multi-word PTX: {sorted(set(unrounded))}"
    assert not [op for op in ops if op.startswith("mad")]
    fmas = [op for op in ops if op.startswith("fma")]
    assert all(".rn." in op and ".ftz" not in op for op in fmas), sorted(set(fmas))
    # Count in the kernel entry only: for qf, numba-cuda-mlir also leaves uncalled `.func`
    # copies of the device callees (w_sqr, w_mul) in the PTX; the entry has them inlined.
    entry = re.split(r"^\.(?:visible\s+)?\.?func\b", ptx[ptx.index(".entry") :], flags=re.M)[0]
    entry_fmas = [op for op in _FLOAT_OP.findall(entry) if op.startswith("fma")]
    assert len(entry_fmas) == EXPECTED_FMA[tier]


@pytest.mark.parametrize("tier", ["dd", "qf"])
def test_resource_usage_matches_the_dispatcher(tmp_path: pathlib.Path, tier: str) -> None:
    """resource_usage (offline: compile_ptx + nvJitLink) equals the driver's attributes."""
    from brotkit.backends.resources import max_threads_per_block

    mod = load(tmp_path, f"s4_{tier}_Z_rn.py.txt")
    py = kernel(mod).py_func
    sig = signature_args(SIGS[tier])
    for cap in (None, 32, 48):
        # The fixture module's own `cuda` keeps numba_cuda_mlir imports out of tests (TID251).
        d = mod.cuda.jit(py) if cap is None else mod.cuda.jit(max_registers=cap)(py)
        d.compile(sig)
        target = re.search(r"^\.target sm_(\d+)(\d)\b", d.inspect_asm(sig), re.M)
        assert target
        cc = (int(target[1]), int(target[2]))
        u = resource_usage(py, sig, cc, max_registers=cap)
        assert u.registers == d.get_regs_per_thread(sig)
        assert u.stack_bytes == d.get_local_mem_per_thread(sig)
        assert u.static_shared_bytes == d.get_shared_mem_per_block(sig)
        assert u.max_threads_per_block == d.get_max_threads_per_block(sig)
        assert u.max_threads_per_block == max_threads_per_block(u.registers, cc)
        if cap:
            assert u.registers <= cap

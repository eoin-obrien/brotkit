"""The ADR-0006 bare-float-operator check on generated multi-word source (CPU only)."""

from __future__ import annotations

import pathlib
import textwrap

import pytest
from hypothesis import given
from hypothesis import strategies as st

from brotkit.codegen.checks import find_bare_float_ops

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "codegen"
# Integer names the spike S4 template uses with bare operators (spikes/S4_registers/gen.py).
S4_INT_NAMES = frozenset({"x", "y", "yy", "w", "h", "n", "row0", "max_iter", "trap_n", "bits"})


def check(src: str, *ints: str):
    return find_bare_float_ops(textwrap.dedent(src), int_names=ints)


def test_bare_float_ops_are_reported() -> None:
    found = check(
        """
        def k(a, b, c):
            return a * b + c
        """
    )
    assert [f.op for f in found] == ["+", "*"]
    assert all(f.function == "k" for f in found)
    assert "a * b + c" in found[0].text


def test_augmented_float_assignment_is_reported() -> None:
    found = check(
        """
        def k(s, x):
            s += x
            return s
        """
    )
    assert [f.op for f in found] == ["+"]


def test_rn_intrinsics_and_fma_pass() -> None:
    assert not check(
        """
        def two_prod(a, b):
            p = ld.fmul_rn(a, b)
            e = cuda.fma(a, b, -p)
            return p, ld.dadd_rn(e, -(p))
        """
    )


def test_index_arithmetic_passes() -> None:
    assert not check(
        """
        def k(cr, out, max_iter, row0):
            x, y = cuda.grid(2)
            h = out.shape[1]
            w = out.shape[2]
            yy = y + row0
            ox = FT(x - w // 2)
            out[0, yy + 1, x * 2 - 1] = ox
            for j in range(max_iter - 1):
                pass
            n = 0
            n += 1
        """,
        "x", "y", "yy", "w", "h", "row0", "max_iter", "n",
    )  # fmt: skip


def test_names_outside_the_allowlist_are_floats() -> None:
    found = check("def k(i, dx):\n    return FT(i) * dx\n", "i")
    assert [f.op for f in found] == ["*"]


def test_mixed_int_and_float_is_reported() -> None:
    # The S3 pitfall: int64 * float32 is typed float64 by Numba; the printer must cast first.
    assert len(check("def k(x, dx):\n    return (x - 1) * dx\n", "x")) == 1


def test_division_is_not_integer_arithmetic() -> None:
    assert len(check("def k(x):\n    return x / 2 + 1\n", "x")) == 1


def test_functions_filter() -> None:
    src = "def lib(a, b):\n    return a + b\n\ndef kern(i):\n    return i + 1\n"
    assert find_bare_float_ops(src, int_names={"i"}, functions={"kern"}) == ()
    assert len(find_bare_float_ops(src, int_names={"i"}, functions={"lib"})) == 1


_FLOAT_NAMES = st.sampled_from(["a", "b", "zr_0", "zi_1", "dx"])
_INT_NAMES = st.sampled_from(["x", "y", "n", "w"])
_OPS = st.sampled_from(["+", "-", "*"])


def _expr(leaves: st.SearchStrategy[str]) -> st.SearchStrategy[tuple[str, int]]:
    """(source, number of binary + - * operators) for random expression trees."""
    return st.recursive(
        leaves.map(lambda s: (s, 0)),
        lambda inner: st.tuples(inner, _OPS, inner).map(
            lambda t: (f"({t[0][0]} {t[1]} {t[2][0]})", t[0][1] + t[2][1] + 1)
        ),
        max_leaves=12,
    )


@given(_expr(_FLOAT_NAMES))
def test_every_float_operator_is_reported(expr: tuple[str, int]) -> None:
    src, n_ops = expr
    assert len(find_bare_float_ops(f"r = {src}\n", int_names={"x", "y", "n", "w"})) == n_ops


@given(_expr(_INT_NAMES | st.integers(0, 99).map(str)))
def test_pure_index_arithmetic_is_never_reported(expr: tuple[str, int]) -> None:
    src, _ = expr
    assert find_bare_float_ops(f"r = {src}\n", int_names={"x", "y", "n", "w"}) == ()


@pytest.mark.parametrize(
    "name",
    [
        "s4_dd_F6_rn.py.txt",
        "s4_qf_C_rn.py.txt",
        # the golden stand-ins (tests/golden_standin.py)
        "s4_df64_B_r1e6.py.txt",
        "s4_dd_B_r1e6.py.txt",
        "s4_qf_B_r1e6.py.txt",
    ],
)
def test_generated_multiword_modules_are_clean(name: str) -> None:
    src = (FIXTURES / name).read_text()
    assert find_bare_float_ops(src, int_names=S4_INT_NAMES) == ()


@pytest.mark.parametrize("name", ["s4_dd_C_naive.py.txt", "s4_f32_C_plain.py.txt"])
def test_generated_bare_operator_modules_are_flagged(name: str) -> None:
    src = (FIXTURES / name).read_text()
    found = find_bare_float_ops(src, int_names=S4_INT_NAMES)
    assert len(found) >= 10
    # index arithmetic in the kernel body is never among the findings
    assert not any("row0" in f.text or "w // 2" in f.text for f in found)

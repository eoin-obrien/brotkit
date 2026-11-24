"""Static checks on generated kernel source.

`find_bare_float_ops` enforces ADR-0006 on multi-word (df64, dd, qf) source: every float add,
subtract and multiply must be a libdevice `_rn` intrinsic or an explicit `cuda.fma`, because
libnvvm contracts bare `a*b + c` into an FMA even with `fastmath=False`. It is a heuristic over
the AST: it cannot see types, so it trusts an explicit allowlist of integer names that the
printer passes in. The authoritative check is on the PTX (no un-rounded float add/sub/mul,
no FMA other than explicit `fma.rn`), which needs the GPU stack; see
`tests/test_cuda_ptx.py`.
"""

from __future__ import annotations

import ast
from collections.abc import Collection
from dataclasses import dataclass

__all__ = ["BareFloatOp", "find_bare_float_ops"]

_CHECKED_OPS: dict[type[ast.operator], str] = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}


@dataclass(frozen=True, slots=True, kw_only=True)
class BareFloatOp:
    """One `+`, `-` or `*` that isn't provably integer index arithmetic."""

    lineno: int
    col_offset: int
    op: str
    text: str
    function: str | None

    def __str__(self) -> str:
        where = f" in {self.function}()" if self.function else ""
        return (
            f"line {self.lineno}:{self.col_offset}{where}: bare float '{self.op}' in `{self.text}`"
        )


def _is_int_expr(node: ast.expr, int_names: Collection[str]) -> bool:
    """True if `node` is built only from allowlisted int names and int literals."""
    match node:
        case ast.Constant(value=bool()):
            return False
        case ast.Constant(value=int()):
            return True
        case ast.Name(id=name):
            return name in int_names
        case ast.UnaryOp(op=ast.USub() | ast.UAdd() | ast.Invert(), operand=operand):
            return _is_int_expr(operand, int_names)
        case ast.BinOp(op=ast.Div() | ast.Pow() | ast.MatMult()):
            return False
        case ast.BinOp(left=left, right=right):
            return _is_int_expr(left, int_names) and _is_int_expr(right, int_names)
        case _:
            return False


class _Finder(ast.NodeVisitor):
    def __init__(self, source: str, int_names: Collection[str]) -> None:
        self.source = source
        self.int_names = int_names
        self.found: list[BareFloatOp] = []
        self.function: str | None = None

    def _report(self, node: ast.expr | ast.stmt, op: ast.operator) -> None:
        self.found.append(
            BareFloatOp(
                lineno=node.lineno,
                col_offset=node.col_offset,
                op=_CHECKED_OPS[type(op)],
                text=ast.get_source_segment(self.source, node) or ast.unparse(node),
                function=self.function,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        outer, self.function = self.function, node.name
        self.generic_visit(node)
        self.function = outer

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)  # the slice is index arithmetic: allowed

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            return  # range() arguments are loop bounds: allowed
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) in _CHECKED_OPS and not _is_int_expr(node, self.int_names):
            self._report(node, node.op)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if type(node.op) in _CHECKED_OPS and not (
            isinstance(node.target, ast.Name)
            and node.target.id in self.int_names
            and _is_int_expr(node.value, self.int_names)
        ):
            self._report(node, node.op)
        self.generic_visit(node)


def find_bare_float_ops(
    source: str,
    *,
    int_names: Collection[str],
    functions: Collection[str] | None = None,
) -> tuple[BareFloatOp, ...]:
    """Bare `+`, `-`, `*` (and `+=` etc.) in `source` that may operate on floats.

    Allowed: anything inside a subscript slice or a `range()` call, and operations whose
    operands are all names in `int_names` or int literals (recursively). Everything else is
    reported. `functions` limits the check to those top-level function names.
    """
    tree = ast.parse(source)
    finder = _Finder(source, frozenset(int_names))
    if functions is None:
        finder.visit(tree)
    else:
        wanted = frozenset(functions)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                finder.visit(node)
    return tuple(finder.found)

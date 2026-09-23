"""NPT recording must evaluate the calculator on Cartesian, not fractional, positions."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "mmml/cli/run/jaxmd_runner.py"


def test_npt_record_eval_transforms_fractional_positions():
    """``_eval_at_position(..., box=npt_box)`` gets ``space.transform(box, frac)``.

    NPT ``state.position`` is fractional (0..1). Passing it straight to the
    hybrid calculator put every atom within 1 Å: the short-range wall read
    ~8e7 eV, every dimer counted as in range, and the run aborted as an
    "energy blow-up" at the first record.
    """
    tree = ast.parse(RUNNER.read_text())
    boxed_calls: list[ast.Call] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_eval_at_position"
                and any(kw.arg == "box" for kw in node.keywords)
            ):
                boxed_calls.append(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    assert boxed_calls, "expected the NPT record path to call _eval_at_position(box=...)"
    for call in boxed_calls:
        first = call.args[0]
        assert (
            isinstance(first, ast.Call)
            and isinstance(first.func, ast.Attribute)
            and first.func.attr == "transform"
        ), "NPT _eval_at_position must receive space.transform(box, state.position)"

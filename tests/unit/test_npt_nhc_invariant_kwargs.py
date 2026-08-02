"""NPT NHC invariant must not forward box= into jax_md (double kwarg)."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "mmml/cli/run/jaxmd_runner.py"


def test_npt_nose_hoover_invariant_call_omits_box_kwarg():
    """jax_md.npt_nose_hoover_invariant calls energy_fn(..., box=box_fn(V), **kwargs).

    Passing box= in kwargs raises TypeError and forces E_tot fallback.
    """
    tree = ast.parse(RUNNER.read_text())
    hits: list[ast.Call] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name == "npt_nose_hoover_invariant":
                hits.append(node)
            self.generic_visit(node)

    Visitor().visit(tree)
    assert hits, "expected at least one npt_nose_hoover_invariant call"
    for call in hits:
        kw_names = {kw.arg for kw in call.keywords if kw.arg}
        assert "box" not in kw_names, (
            "do not pass box= into npt_nose_hoover_invariant; jax_md supplies it"
        )

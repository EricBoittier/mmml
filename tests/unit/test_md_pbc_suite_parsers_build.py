"""The md_pbc_suite backend parsers must actually build.

`jaxmd.build_parser()` registered --hybrid-hamiltonian and --shared-cutoff twice
in the same function, so it raised before returning:

    argparse.ArgumentError: argument --hybrid-hamiltonian:
    conflicting option string: --hybrid-hamiltonian

Nothing caught it because no test ever called build_parser(), and the failure
only surfaced at the end of a long cluster job -- `mmml md-system --backend
jaxmd` was impossible to run at all. The duplicate was also the stale copy,
offering choices ("handoff", "additive") where md_system.py and ase.py both use
("handoff", "shared_cutoff"), so md-system could forward a value this parser
would have rejected.
"""

import pytest


@pytest.mark.parametrize("module_name", ["jaxmd", "ase"])
def test_backend_parser_builds(module_name):
    import importlib

    mod = importlib.import_module(f"mmml.cli.run.md_pbc_suite.{module_name}")
    parser = mod.build_parser()
    assert parser is not None


@pytest.mark.parametrize("module_name", ["jaxmd", "ase"])
def test_no_duplicate_option_strings(module_name):
    """Every option string is registered exactly once."""
    import ast
    import collections
    import importlib
    import pathlib

    mod = importlib.import_module(f"mmml.cli.run.md_pbc_suite.{module_name}")
    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "build_parser"
    )
    seen = collections.defaultdict(list)
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                    seen[arg.value].append(node.lineno)
    dupes = {opt: lines for opt, lines in seen.items() if len(lines) > 1}
    assert not dupes, f"{module_name}.build_parser registers these twice: {dupes}"


def test_hybrid_hamiltonian_choices_agree_across_parsers():
    """md-system forwards this value verbatim, so the choice sets must match."""
    import re
    from pathlib import Path

    pat = re.compile(
        r'"--hybrid-hamiltonian",\s*\n\s*choices=\(([^)]*)\)', re.MULTILINE
    )
    found = {}
    for rel in (
        "mmml/cli/run/md_system.py",
        "mmml/cli/run/md_pbc_suite/jaxmd.py",
        "mmml/cli/run/md_pbc_suite/ase.py",
    ):
        text = Path(rel).read_text()
        choices = {c.strip().strip('"\'') for c in pat.findall(text)[0].split(",") if c.strip()}
        found[rel] = choices
    distinct = {frozenset(v) for v in found.values()}
    assert len(distinct) == 1, f"choice sets disagree: {found}"


@pytest.mark.parametrize("value", ["handoff", "shared_cutoff"])
def test_jaxmd_accepts_what_md_system_forwards(value):
    from mmml.cli.run.md_pbc_suite import jaxmd

    ns = jaxmd.build_parser().parse_args(["--hybrid-hamiltonian", value])
    assert ns.hybrid_hamiltonian == value

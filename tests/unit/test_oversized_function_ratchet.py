"""Stop the "one enormous function" pattern from spreading.

A full-suite coverage run put mmml at 45.9% of 121,150 statements, and the
audit of what was left found that the binding constraint is often *structure*,
not a missing dependency. The clearest case:

    mlpot/staged_workflow.py -- 896 of 902 uncovered statements sit in
    CHARMM-touching functions, and 699 of those are inside a single
    2,469-line function, ``run_staged_workflow`` (62% of the file).

That file cannot get past ~35% coverage while that function exists, and no
amount of test writing changes it: the function reads 17-23 enclosing locals
and writes 8-28 of them, so it can only be exercised end to end, against live
CHARMM, which CI does not have. Extracting it safely needs the golden-record
loop, not a unit test.

What a test *can* do is keep the problem from getting worse. This file is a
ratchet over every function above ``_MAX_LINES``:

* nothing in the baseline may grow;
* nothing new may join it.

When this fails you have two honest options: split the function, or raise its
recorded number and say why in the commit. Lower the numbers as decomposition
lands -- the baseline is a debt register, not a target.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

import mmml

MMML_ROOT = Path(mmml.__file__).resolve().parent

# Above this, a function is effectively untestable in units: you can only call
# it whole, with whatever environment it needs.
_MAX_LINES = 500


class Function(NamedTuple):
    key: str
    lines: int


def _walk(node: ast.AST, prefix: str, rel: str) -> Iterator[Function]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f"{prefix}{child.name}"
            yield Function(f"{rel}::{qualname}", child.end_lineno - child.lineno + 1)
            yield from _walk(child, f"{qualname}.", rel)
        elif isinstance(child, ast.ClassDef):
            yield from _walk(child, f"{prefix}{child.name}.", rel)
        else:
            yield from _walk(child, prefix, rel)


def _all_functions() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted(MMML_ROOT.rglob("*.py")):
        rel = path.relative_to(MMML_ROOT.parent).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for func in _walk(tree, "", rel):
            found[func.key] = func.lines
    return found


_FUNCTIONS = _all_functions()
_OVERSIZED = {k: v for k, v in _FUNCTIONS.items() if v > _MAX_LINES}

# Measured 2026-08-01. Every entry is technical debt; none may grow.
_BASELINE: dict[str, int] = {
    "mmml/interfaces/pycharmmInterface/mmml_calculator.py::setup_calculator": 3363,
    "mmml/cli/run/jaxmd_runner.py::set_up_nhc_sim_routine": 2522,
    "mmml/interfaces/pycharmmInterface/mlpot/staged_workflow.py::run_staged_workflow": 2469,
    "mmml/cli/run/md_pbc_suite/jaxmd.py::main": 2186,
    "mmml/cli/run/md_system.py::build_parser": 2093,
    "mmml/cli/run/jaxmd_runner.py::set_up_nhc_sim_routine.run_sim": 1986,
    "mmml/interfaces/pycharmmInterface/mlpot/dynamics.py::run_dynamics_with_io": 1587,
    "mmml/interfaces/pycharmmInterface/mm_energy_forces.py::build_mm_energy_forces_fn": 1543,
    "mmml/cli/misc/train_joint.py::plot_validation_results": 1255,
    "mmml/cli/misc/fix_and_split.py::fix_and_split_data": 1032,
    "mmml/utils/hybrid_optimization.py::fit_hybrid_potential_to_training_data_jax": 902,
    "mmml/models/physnetjax/physnetjax/training/training.py::train_model": 751,
    "mmml/models/dcmnet/dcmnet/training.py::train_model": 699,
    "mmml/interfaces/pycharmmInterface/mlpot/cli_common.py::prepare_mlpot_hybrid_state_for_sd": 673,
    "mmml/cli/make/make_training.py::build_parser": 649,
    "mmml/umbrella/sample.py::run_umbrella_nvt": 597,
    "mmml/cli/run/md_pbc_suite/ase.py::main": 592,
    "mmml/models/efield/evaluate.py::main": 582,
    "mmml/cli/run/md_system.py::build_pycharmm_command": 571,
    "mmml/cli/misc/train_joint.py::train_model": 566,
    "mmml/utils/hybrid_optimization.py::create_hybrid_fitting_factory": 514,
    "mmml/models/dcmnet/dcmnet_mcts.py::optimize_dcmnet_combination": 508,
    "mmml/models/efield/ase_md.py::main_batched": 508,
    "mmml/cli/run/md_pbc_suite/ase.py::build_parser": 507,
    "mmml/models/dcmnet/dcmnet/loss.py::esp_mono_loss": 507,
    "mmml/md/drivers/jaxmd.py::JaxmdDriver.run": 506,
}


def test_the_scanner_actually_walks_the_package():
    """A structural guard that finds nothing passes for the wrong reason."""
    assert len(_FUNCTIONS) > 5000
    assert (
        "mmml/interfaces/pycharmmInterface/mlpot/staged_workflow.py::run_staged_workflow"
        in _FUNCTIONS
    )


def test_the_scanner_sees_nested_and_method_definitions():
    """``run_sim`` is nested inside ``set_up_nhc_sim_routine``; ``JaxmdDriver.run``
    is a method. Both are in the baseline, so both must be reachable."""
    assert "mmml/cli/run/jaxmd_runner.py::set_up_nhc_sim_routine.run_sim" in _FUNCTIONS
    assert "mmml/md/drivers/jaxmd.py::JaxmdDriver.run" in _FUNCTIONS


@pytest.mark.parametrize(("key", "cap"), sorted(_BASELINE.items()), ids=lambda v: v)
def test_no_oversized_function_grows(key: str, cap: int):
    actual = _FUNCTIONS.get(key)
    if actual is None:
        pytest.skip(f"{key} no longer exists -- drop its baseline entry")
    assert actual <= cap, (
        f"{key} grew from {cap} to {actual} lines. It is already too large to "
        f"cover in CI; extract the new code into a named function instead of "
        f"appending to it. If the growth is unavoidable, raise the number here "
        f"and say why."
    )


def test_no_new_function_joins_the_oversized_set():
    newcomers = {k: v for k, v in _OVERSIZED.items() if k not in _BASELINE}
    assert not newcomers, (
        f"New function(s) over {_MAX_LINES} lines: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(newcomers.items()))
        + ". Functions this size cannot be unit tested -- they can only be run "
        "whole, against whatever environment they need. Split before merging."
    )


def test_the_baseline_has_no_stale_entries():
    """Entries that shrank below the threshold should be deleted, not left to
    rot -- otherwise the register stops describing the debt."""
    stale = {
        k: _FUNCTIONS[k]
        for k in _BASELINE
        if k in _FUNCTIONS and _FUNCTIONS[k] <= _MAX_LINES
    }
    assert not stale, (
        "These are no longer oversized; remove them from _BASELINE: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(stale.items()))
    )


def test_staged_workflow_is_tracked_as_the_worst_coverage_blocker():
    """The specific finding this file exists for.

    ``run_staged_workflow`` holds 699 of the 902 uncovered statements in its
    module. Any decomposition should move this number down; nothing should
    move it up.
    """
    key = "mmml/interfaces/pycharmmInterface/mlpot/staged_workflow.py::run_staged_workflow"
    module_lines = len(
        (MMML_ROOT / "interfaces/pycharmmInterface/mlpot/staged_workflow.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert _FUNCTIONS[key] <= _BASELINE[key]
    # It was 62% of its file. If that share rises, the file is getting worse
    # even when the absolute count does not.
    assert _FUNCTIONS[key] / module_lines <= 0.66

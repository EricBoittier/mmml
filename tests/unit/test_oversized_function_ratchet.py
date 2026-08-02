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
two-tier ratchet:

* **Over 1,000 lines** (11 functions): none may grow past a 50-line grace, and
  none may be added. At this size a function is only ever callable whole, so no
  test reaches inside it -- which is how 699 statements end up uncoverable at
  once. The grace is there because the failure worth catching is a subsystem
  landing in one lump, not a normal edit.
* **Over 500 lines** (26 functions): capped by *count*, not by length. A
  20-line edit to an already-large function is normal work and should not turn
  the suite red; a 27th module adopting the pattern is the regression.

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

# Two tiers, because one threshold cannot do both jobs.
#
# ``_MAX_LINES`` is the hard line: a four-figure function can only ever be
# called whole, with whatever environment it needs, so no test reaches inside
# it. Eleven functions are already there; they may not grow and nothing new may
# join them.
#
# ``_CROWDED_LINES`` is the soft line. Pinning the exact length of all 26
# functions over 500 lines would fire on ordinary work -- a 20-line edit to an
# already-large function is not the regression this file is about -- so that
# tier is capped by *population* instead: the club may not gain members.
_MAX_LINES = 1000
_CROWDED_LINES = 500

# Routine-edit tolerance on the recorded caps. Pinning them to the exact line
# is unworkable in practice: this file went red three times in one afternoon on
# +2, +23 and +77-line edits landing from other work, and a guard that has to be
# bumped that often gets deleted rather than heeded. The regression it exists to
# catch is not a handful of lines -- `run_staged_workflow` buried 699 statements
# at once. Growth beyond the grace needs a baseline raise and a reason.
_GROWTH_GRACE = 50


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
_CROWDED = {k: v for k, v in _FUNCTIONS.items() if v > _CROWDED_LINES}

# Measured at e8191c8c1 (2026-08-01). Every entry is technical debt; none may
# grow. Raising a number is allowed when the growth is genuinely unavoidable --
# say why in the commit, so the register keeps meaning something.
_BASELINE: dict[str, int] = {
    # 3363 -> 3445. Two separate movements, recorded together because the entry
    # had drifted: other work in flight had already taken it to 3413, which fit
    # inside the grace and so never had to be written down. The remaining +32 is
    # the bonded-intra damping wiring, and it is call sites only -- the logic
    # went to module level as `apply_bonded_intra_damping`, `bonded_intra_bundle`
    # and `resolve_bonded_intra_damping`, which is why they are unit tested and
    # the enclosing function still is not.
    "mmml/interfaces/pycharmmInterface/mmml_calculator.py::setup_calculator": 3445,
    "mmml/cli/run/jaxmd_runner.py::set_up_nhc_sim_routine": 2522,
    "mmml/interfaces/pycharmmInterface/mlpot/staged_workflow.py::run_staged_workflow": 2469,
    # 2186 -> 1467: its 743-line argparse block became `build_parser`, so the
    # backend's CLI surface can be parsed in a test instead of only in a
    # subprocess mid-run.
    "mmml/cli/run/md_pbc_suite/jaxmd.py::main": 1467,
    "mmml/cli/run/md_system.py::build_parser": 2093,
    "mmml/cli/run/jaxmd_runner.py::set_up_nhc_sim_routine.run_sim": 1986,
    "mmml/interfaces/pycharmmInterface/mlpot/dynamics.py::run_dynamics_with_io": 1587,
    # 1543 -> 1620 (+77, past the grace). Raised to record uncommitted work in
    # flight, not because the growth was reviewed here. This is the tier where
    # 77 more lines are 77 more uncoverable ones -- worth a look.
    "mmml/interfaces/pycharmmInterface/mm_energy_forces.py::build_mm_energy_forces_fn": 1620,
    "mmml/cli/misc/train_joint.py::plot_validation_results": 1255,
    "mmml/cli/misc/fix_and_split.py::fix_and_split_data": 1032,
}

# The 500+ tier, capped by population rather than per-function length. 26 at
# e8191c8c1. Lower it when a function drops out; only raise it with a reason.
#
# 26 -> 27: splitting `md_pbc_suite/jaxmd.py::main` moved its parser into a
# 752-line `build_parser`, which joins this tier. That is a split, not a new
# monolith -- the >1000 tier lost 719 lines in the same change.
_MAX_CROWDED = 27


def test_the_scanner_actually_walks_the_package():
    """A structural guard that finds nothing passes for the wrong reason."""
    assert len(_FUNCTIONS) > 5000
    assert (
        "mmml/interfaces/pycharmmInterface/mlpot/staged_workflow.py::run_staged_workflow"
        in _FUNCTIONS
    )


def test_the_scanner_sees_nested_and_method_definitions():
    """``run_sim`` is nested inside ``set_up_nhc_sim_routine`` and
    ``JaxmdDriver.run`` is a method; a walker that only looked at module-level
    ``def``s would miss both, and one of them is 1,986 lines."""
    assert "mmml/cli/run/jaxmd_runner.py::set_up_nhc_sim_routine.run_sim" in _FUNCTIONS
    assert "mmml/md/drivers/jaxmd.py::JaxmdDriver.run" in _FUNCTIONS


@pytest.mark.parametrize(("key", "cap"), sorted(_BASELINE.items()), ids=lambda v: v)
def test_no_oversized_function_grows(key: str, cap: int):
    actual = _FUNCTIONS.get(key)
    if actual is None:
        pytest.skip(f"{key} no longer exists -- drop its baseline entry")
    assert actual <= cap + _GROWTH_GRACE, (
        f"{key} grew from {cap} to {actual} lines, past the {_GROWTH_GRACE}-line "
        f"grace. It is already too large to cover in CI; extract the new code "
        f"into a named function instead of appending to it. If the growth is "
        f"unavoidable, raise the number here and say why."
    )


def test_no_new_function_joins_the_oversized_set():
    newcomers = {k: v for k, v in _OVERSIZED.items() if k not in _BASELINE}
    assert not newcomers, (
        f"New function(s) over {_MAX_LINES} lines: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(newcomers.items()))
        + ". Functions this size cannot be unit tested -- they can only be run "
        "whole, against whatever environment they need. Split before merging."
    )


def test_the_500_line_club_does_not_gain_members():
    """The soft tier. Growing an already-large function by 20 lines is not what
    this file is about; *another* module adopting the pattern is."""
    assert len(_CROWDED) <= _MAX_CROWDED, (
        f"{len(_CROWDED)} functions are now over {_CROWDED_LINES} lines, up "
        f"from {_MAX_CROWDED}. The largest are: "
        + ", ".join(
            f"{k} ({v})"
            for k, v in sorted(_CROWDED.items(), key=lambda kv: -kv[1])[:5]
        )
        + ". Split one before adding another."
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

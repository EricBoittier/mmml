"""`--evaluate-npz` must honour the ML/MM term flags.

`_evaluate_jaxmd_mmml` hardcoded doML/doMM/doML_dimer to True at all three of
its construction and evaluation sites, so single-point evaluation silently
ignored --no-do-ml, --no-include-mm, --do-ml-dimer/--no-do-ml-dimer and
--skip-ml-dimers. Scoring one 600-frame water-dimer grid four ways -- full, no
dimer term, no MM, MM only -- returned BIT-IDENTICAL energies
(max |E - E_full| = 0.000e+00 eV for every arm, including the arm with the ML
model switched off entirely).

That makes a per-term decomposition impossible and, worse, silently misreports
which physics produced a number. These tests pin the resolution, including
--skip-ml-dimers acting as an override on top of --do-ml-dimer, matching what
the jax-md runner does.
"""

from argparse import Namespace

import pytest

from mmml.cli.run.md_evaluate_npz import evaluate_term_flags


def test_defaults_are_everything_on():
    assert evaluate_term_flags(Namespace()) == (True, True, True)


def test_no_do_ml_turns_ml_off():
    assert evaluate_term_flags(Namespace(do_ml=False)) == (False, True, True)


def test_no_include_mm_turns_mm_off():
    assert evaluate_term_flags(Namespace(include_mm=False)) == (True, False, True)


def test_no_do_ml_dimer_turns_the_dimer_term_off():
    assert evaluate_term_flags(Namespace(do_ml_dimer=False)) == (True, True, False)


def test_skip_ml_dimers_overrides_do_ml_dimer():
    """--skip-ml-dimers wins even when --do-ml-dimer is explicitly true."""
    got = evaluate_term_flags(Namespace(do_ml_dimer=True, skip_ml_dimers=True))
    assert got == (True, True, False)


def test_skip_ml_dimers_false_leaves_do_ml_dimer_alone():
    got = evaluate_term_flags(Namespace(do_ml_dimer=True, skip_ml_dimers=False))
    assert got == (True, True, True)


@pytest.mark.parametrize(
    "ns,expected",
    [
        (Namespace(include_mm=True, do_ml=True, skip_ml_dimers=True), (True, True, False)),
        (Namespace(include_mm=False, do_ml=True, do_ml_dimer=True), (True, False, True)),
        (Namespace(include_mm=True, do_ml=False, skip_ml_dimers=True), (False, True, False)),
    ],
)
def test_decomposition_arms_are_distinguishable(ns, expected):
    """The four arms used for the decomposition must not collapse together."""
    assert evaluate_term_flags(ns) == expected


def test_the_four_arms_are_mutually_distinct():
    arms = {
        "full": Namespace(include_mm=True, do_ml=True, do_ml_dimer=True),
        "no_dimer": Namespace(include_mm=True, do_ml=True, skip_ml_dimers=True),
        "no_mm": Namespace(include_mm=False, do_ml=True, do_ml_dimer=True),
        "mm_only": Namespace(include_mm=True, do_ml=False, skip_ml_dimers=True),
    }
    resolved = {k: evaluate_term_flags(v) for k, v in arms.items()}
    assert len(set(resolved.values())) == 4, f"arms collapsed: {resolved}"

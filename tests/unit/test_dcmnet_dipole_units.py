"""Unit conventions for the DCMNet dipole chain.

Three places convert a distributed-multipole sum into a reportable dipole, and
before the units audit they disagreed with each other and with their own
docstrings:

* ``loss.pred_dipole`` multiplied by ``1.88873`` and claimed Debye. The value is
  a transposed-digit typo for the Angstrom -> bohr factor 1.8897261 (5.3e-4
  relative), and the unit is atomic units, not Debye -- both callers in
  ``analysis`` convert the residual with ``au_to_debye`` afterwards.
* ``dcmnet_ase._compute_molecular_dipole`` used the same literal under an
  "atomic units to Debye" comment while its input was e*Angstrom, so the number
  it labelled Debye was ~2.54x too small.
* ``analysis`` carried a third independent literal for e*bohr -> Debye.

None of it was covered by a test, which is why it survived. The assertions here
are anchored on CODATA values computed in the test itself, and on the identity
that ties the three factors together -- not on anything the code under test
returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.data.units import (
    ANGSTROM_TO_BOHR,
    DEBYE_TO_EANGSTROM,
    DEBYE_TO_EBOHR,
    EANGSTROM_TO_DEBYE,
    EBOHR_TO_DEBYE,
)

# CODATA 2018, spelled out so the reference does not come from mmml.data.units.
_BOHR_RADIUS_ANGSTROM = 0.529177210903
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_DEBYE_C_M = 3.335640952e-30

_REF_ANGSTROM_TO_BOHR = 1.0 / _BOHR_RADIUS_ANGSTROM
_REF_EANGSTROM_TO_DEBYE = _ELEMENTARY_CHARGE_C * 1e-10 / _DEBYE_C_M
_REF_EBOHR_TO_DEBYE = _REF_EANGSTROM_TO_DEBYE / _REF_ANGSTROM_TO_BOHR

# The repo's constants are rounded (BOHR_TO_ANGSTROM = 0.529177), so allow a few
# parts per million against the full CODATA values -- but nothing like the
# 5.3e-4 of the old typo, which this tolerance is deliberately tight enough to
# reject.
_TOL = 1e-5


# --- the shared constants ---------------------------------------------------


def test_angstrom_to_bohr_matches_codata():
    assert ANGSTROM_TO_BOHR == pytest.approx(_REF_ANGSTROM_TO_BOHR, rel=_TOL)


def test_eangstrom_to_debye_matches_codata():
    assert EANGSTROM_TO_DEBYE == pytest.approx(_REF_EANGSTROM_TO_DEBYE, rel=_TOL)


def test_ebohr_to_debye_matches_codata():
    assert EBOHR_TO_DEBYE == pytest.approx(_REF_EBOHR_TO_DEBYE, rel=_TOL)


def test_the_three_factors_are_mutually_consistent():
    """e*Angstrom -> Debye must equal (Angstrom -> bohr) x (e*bohr -> Debye).

    This is the invariant that broke when each site carried its own literal.
    """
    assert EANGSTROM_TO_DEBYE == pytest.approx(
        ANGSTROM_TO_BOHR * EBOHR_TO_DEBYE, rel=1e-12
    )


@pytest.mark.parametrize(
    ("forward", "backward"),
    [(EANGSTROM_TO_DEBYE, DEBYE_TO_EANGSTROM), (EBOHR_TO_DEBYE, DEBYE_TO_EBOHR)],
)
def test_dipole_conversions_round_trip(forward, backward):
    assert forward * backward == pytest.approx(1.0, rel=1e-12)


def test_the_old_typo_is_not_reintroduced():
    """1.88873 is neither conversion; it read as plausible for years."""
    for value in (ANGSTROM_TO_BOHR, EANGSTROM_TO_DEBYE, EBOHR_TO_DEBYE):
        assert value != pytest.approx(1.88873, abs=1e-5)


# --- pred_dipole ------------------------------------------------------------


def _pred_dipole(positions, com, charges):
    import jax.numpy as jnp

    from mmml.models.dcmnet.dcmnet.loss import pred_dipole

    return np.asarray(
        pred_dipole(jnp.asarray(positions), jnp.asarray(com), jnp.asarray(charges))
    )


def test_pred_dipole_of_a_unit_point_dipole():
    """+-1 e separated by 1 Angstrom is exactly 1 e*Angstrom by construction."""
    got = _pred_dipole([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.0, 0.0, 0.0], [-1.0, 1.0])

    assert got[0] == pytest.approx(_REF_ANGSTROM_TO_BOHR, rel=_TOL)
    assert got[1:] == pytest.approx([0.0, 0.0], abs=1e-12)
    # ...and in Debye that is the familiar 4.803 D per e*Angstrom.
    assert float(np.linalg.norm(got)) * EBOHR_TO_DEBYE == pytest.approx(
        _REF_EANGSTROM_TO_DEBYE, rel=_TOL
    )


def test_pred_dipole_returns_atomic_units_not_debye():
    """The unit the docstring used to claim would be 2.54x larger."""
    got = _pred_dipole([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.0, 0.0, 0.0], [-1.0, 1.0])
    magnitude = float(np.linalg.norm(got))

    assert magnitude == pytest.approx(1.8897, rel=1e-3)
    assert magnitude != pytest.approx(4.8032, rel=1e-2), "returned Debye, not a.u."


def test_pred_dipole_points_from_negative_to_positive_charge():
    """Physics convention; a sign flip here would be invisible in an L2 loss
    against a same-signed target but wrong against a real reference."""
    got = _pred_dipole([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.5, 0.0, 0.0], [-1.0, 1.0])
    assert got[0] > 0.0


def test_pred_dipole_is_linear_in_charge():
    pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    com = [0.3, 0.3, 0.0]
    q = np.array([-0.8, 0.5, 0.3])

    single = _pred_dipole(pos, com, q)
    doubled = _pred_dipole(pos, com, 2.0 * q)

    assert doubled == pytest.approx(2.0 * single, rel=1e-6)


def test_pred_dipole_of_a_neutral_system_is_translation_invariant():
    """For sum(q) == 0 the dipole does not depend on the origin, so shifting
    the molecule and its centre of mass together must change nothing."""
    pos = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    com = np.array([0.1, 0.2, 0.0])
    q = np.array([-0.834, 0.417, 0.417])
    shift = np.array([5.0, -3.0, 2.0])

    here = _pred_dipole(pos, com, q)
    there = _pred_dipole(pos + shift, com + shift, q)

    assert there == pytest.approx(here, rel=1e-5, abs=1e-6)


def test_pred_dipole_vanishes_without_charge():
    got = _pred_dipole([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.0, 0.0, 0.0], [0.0, 0.0])
    assert got == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_pred_dipole_matches_an_independent_numpy_sum():
    """Hand-rolled reference: sum q_i (r_i - com), then Angstrom -> bohr."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-2.0, 2.0, size=(6, 3))
    q = rng.uniform(-1.0, 1.0, size=6)
    q -= q.mean()  # neutral
    com = pos.mean(axis=0)

    expected = (q[:, None] * (pos - com)).sum(axis=0) * _REF_ANGSTROM_TO_BOHR

    assert _pred_dipole(pos, com, q) == pytest.approx(expected, rel=1e-4, abs=1e-6)


# --- the ASE calculator's reported dipole -----------------------------------


def _ase_dipole(dipole_positions, monopoles, com):
    from mmml.models.dcmnet.dcmnet_ase import DCMNetCalculator

    # The method touches no instance state, so bind it to None rather than
    # standing up a calculator (which would need a trained model).
    return DCMNetCalculator._compute_molecular_dipole(
        None,
        np.asarray(dipole_positions, dtype=float),
        np.asarray(monopoles, dtype=float),
        np.asarray(com, dtype=float),
    )


def test_ase_calculator_reports_debye_as_documented():
    """``get_dcm_data`` documents 'molecular_dipole' in Debye, so it must be."""
    got = _ase_dipole([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], [[-1.0], [1.0]], [0.0] * 3)

    assert got[0] == pytest.approx(_REF_EANGSTROM_TO_DEBYE, rel=_TOL)
    assert float(np.linalg.norm(got)) == pytest.approx(4.8032, rel=1e-3)


def test_ase_calculator_dipole_is_not_the_old_atomic_unit_value():
    """It previously returned ~1.889 D-labelled units -- 2.54x too small."""
    got = _ase_dipole([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], [[-1.0], [1.0]], [0.0] * 3)
    assert float(np.linalg.norm(got)) != pytest.approx(1.88873, rel=1e-3)


def test_ase_and_loss_dipoles_differ_only_by_the_documented_conversion():
    """Same physical dipole, two functions, two units: the ratio must be the
    e*bohr -> Debye factor and nothing else."""
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    charges = np.array([-0.834, 0.417, 0.417])
    com = positions.mean(axis=0)

    in_au = _pred_dipole(positions, com, charges)
    in_debye = _ase_dipole(positions[:, None, :], charges[:, None], com)

    assert in_debye == pytest.approx(np.asarray(in_au) * EBOHR_TO_DEBYE, rel=1e-4)


# --- the reporting path -----------------------------------------------------


def test_analysis_reuses_the_shared_conversion():
    """``analysis.au_to_debye`` was a fourth independent literal; it must now
    be the same object the rest of the chain uses."""
    from mmml.models.dcmnet.dcmnet import analysis

    assert analysis.au_to_debye == pytest.approx(EBOHR_TO_DEBYE, rel=1e-12)

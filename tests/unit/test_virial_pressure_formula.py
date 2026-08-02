"""The virial -> pressure conversion, including units.

`scripts/validate_virial_vs_charmm.py` compares this against CHARMM's PRSI on a
real box. These tests pin the parts that need no CHARMM: the identity itself and
the eV/A^3 -> atm chain, which is where unit bugs of exactly this kind hide.
"""

import numpy as np
import pytest

from scripts.validate_virial_vs_charmm import virial_pressure_atm


def test_ideal_gas_limit_zero_forces():
    """No forces -> pure kinetic pressure, P = 2KE/(3V)."""
    n, V = 100, 1000.0
    ke = 5.0  # eV
    p = virial_pressure_atm(np.zeros((n, 3)), np.zeros((n, 3)), V, kinetic_ev=ke)
    expected = (2 * ke / (3 * V)) * (1.602176634e-19 / 1e-30) / 101325.0
    assert p == pytest.approx(expected, rel=1e-12)


def test_reproduces_the_observed_blowup_pressure():
    """The state that exposed the bug: zero virial, 4059.58 atm measured."""
    kB = 8.617333262e-5
    T, V, n_atoms = 297.87, 21955.3, 2196
    ke = 0.5 * (3 * n_atoms) * kB * T
    p = virial_pressure_atm(np.zeros((n_atoms, 3)), np.zeros((n_atoms, 3)), V, ke)
    assert p == pytest.approx(4059.58, rel=2e-5)


def test_positive_virial_raises_pressure():
    """Outward (repulsive) forces increase the pressure."""
    r = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    f_out = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])  # F.r > 0
    assert virial_pressure_atm(f_out, r, 100.0) > 0.0
    assert virial_pressure_atm(-f_out, r, 100.0) < 0.0


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        virial_pressure_atm(np.zeros((4, 3)), np.zeros((5, 3)), 10.0)


def test_scales_inversely_with_volume():
    r = np.random.default_rng(0).normal(size=(8, 3))
    f = np.random.default_rng(1).normal(size=(8, 3))
    assert virial_pressure_atm(f, r, 100.0) == pytest.approx(
        2.0 * virial_pressure_atm(f, r, 200.0), rel=1e-12
    )

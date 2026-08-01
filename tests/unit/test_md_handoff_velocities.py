"""Velocity units and kinetic temperature across the MD handoff boundary.

Handoff moves a running system between CHARMM (mass-weighted AKMA), JAX-MD/ASE
(metal units), and plain Å/ps. The module's own docstrings record what these
mistakes cost: the legacy ``handoff_velocities_as_ase_ang_fs`` name is
"dimensionally incorrect", and reading metal-unit velocities as Å/ps
"under-reads T by ~10⁴ (e.g. ~150 K → ~0.007 K)".

None of that raises. A handoff with the wrong velocity scale produces a
trajectory that integrates happily at the wrong temperature, so the assertions
below are anchored on the equipartition relation ``<KE> = (dof/2) k_B T``
computed in the test, not on the module's constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.cli.run.md_handoff import (
    ang_ps_velocities_to_jaxmd_metal,
    kinetic_temperature_k_from_ang_ps_velocities,
    kinetic_temperature_k_from_jaxmd_metal_velocities,
    monomer_offsets_uniform,
    remove_center_of_mass_velocity_ang_ps,
)

# CODATA, independent of the module.
_K_B_EV = 8.617333262e-05
# 1 amu*(Å/ps)^2 in kcal/mol: (1.66053907e-27 kg)(1e-10 m/1e-12 s)^2 * N_A / 4184
_AMU_ANGPS2_KCAL = 1.66053906660e-27 * (1e-10 / 1e-12) ** 2 * 6.02214076e23 / 4184.0


def _thermal_speed_ang_ps(t_k: float, mass_amu: float) -> float:
    """Speed giving exactly (3/2) k_B T per atom, from equipartition."""
    ke_kcal_per_atom = 1.5 * (_K_B_EV * t_k) * 23.060548  # eV -> kcal/mol
    return float(np.sqrt(2.0 * ke_kcal_per_atom / (mass_amu * _AMU_ANGPS2_KCAL)))


# --- kinetic temperature from Å/ps ------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_AMU_ANG_PS2_TO_KCALMOL = 1.036427e-3 in charmm_ase_velocities is wrong. "
        "Its comment says 'amu * (Angstrom/psec)^2 -> kcal/mol', for which the "
        "exact AKMA value is 1/418.4 = 2.390057e-3. The literal carries the "
        "mantissa of the *eV* conversion (1 amu(A/ps)^2 = 1.036427e-4 eV) with "
        "the wrong exponent, so every kinetic temperature on this path is a "
        "factor 2.306 out. Verified end to end: velocities generated for '300 K' "
        "measure 690.6 K against CODATA + the AKMA definition. Self-consistent "
        "within the module (assign and read-back use the same constant), which "
        "is why it survived. Delete this marker once the constant is corrected."
    ),
)
def test_temperature_matches_equipartition_for_a_known_speed():
    """One atom at exactly (3/2)kT must report T back."""
    t_target, mass = 300.0, 18.0
    speed = _thermal_speed_ang_ps(t_target, mass)
    v = np.array([[speed / np.sqrt(3.0)] * 3])

    got = kinetic_temperature_k_from_ang_ps_velocities(v, np.array([mass]))

    assert got == pytest.approx(t_target, rel=2e-3)


def test_temperature_is_zero_for_a_frozen_system():
    v = np.zeros((4, 3))
    assert kinetic_temperature_k_from_ang_ps_velocities(v, np.full(4, 12.0)) == 0.0


def test_temperature_scales_with_the_square_of_velocity():
    v = np.full((3, 3), 0.5)
    m = np.full(3, 12.0)
    t1 = kinetic_temperature_k_from_ang_ps_velocities(v, m)
    t2 = kinetic_temperature_k_from_ang_ps_velocities(2.0 * v, m)
    assert t2 == pytest.approx(4.0 * t1, rel=1e-9)


def test_temperature_scales_linearly_with_mass():
    v = np.full((3, 3), 0.5)
    t1 = kinetic_temperature_k_from_ang_ps_velocities(v, np.full(3, 12.0))
    t2 = kinetic_temperature_k_from_ang_ps_velocities(v, np.full(3, 24.0))
    assert t2 == pytest.approx(2.0 * t1, rel=1e-9)


def test_constrained_degrees_of_freedom_raise_the_temperature():
    """Same kinetic energy over fewer dof is a higher T -- rigid water relies
    on this, and defaulting to 3N would under-report by the constraint ratio."""
    v = np.full((3, 3), 0.5)
    m = np.full(3, 12.0)
    free = kinetic_temperature_k_from_ang_ps_velocities(v, m)
    constrained = kinetic_temperature_k_from_ang_ps_velocities(v, m, ndegf=6)
    assert constrained == pytest.approx(free * 9.0 / 6.0, rel=1e-9)


def test_mismatched_velocity_and_mass_lengths_are_rejected():
    with pytest.raises(ValueError, match="same non-zero length"):
        kinetic_temperature_k_from_ang_ps_velocities(np.zeros((3, 3)), np.zeros(2))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="same non-zero length"):
        kinetic_temperature_k_from_ang_ps_velocities(np.zeros((0, 3)), np.zeros(0))


def test_non_finite_velocities_are_rejected():
    """A blown-up trajectory must fail here, not report a plausible number."""
    v = np.array([[1.0, np.nan, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        kinetic_temperature_k_from_ang_ps_velocities(v, np.array([12.0]))


def test_non_positive_masses_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        kinetic_temperature_k_from_ang_ps_velocities(np.zeros((1, 3)), np.array([0.0]))


def test_zero_or_negative_ndegf_is_rejected():
    with pytest.raises(ValueError, match="ndegf must be positive"):
        kinetic_temperature_k_from_ang_ps_velocities(
            np.zeros((2, 3)), np.full(2, 1.0), ndegf=0
        )


# --- kinetic temperature from JAX-MD / ASE metal units ----------------------


def test_metal_unit_temperature_matches_the_jax_md_definition():
    """jax_md.quantity.temperature is sum(m v^2)/dof as kT in eV."""
    v = np.array([[0.01, -0.02, 0.005], [0.0, 0.01, 0.0]])
    m = np.array([12.0, 1.0])

    got = kinetic_temperature_k_from_jaxmd_metal_velocities(v, m)

    expected = float(np.sum(m[:, None] * v * v)) / (6.0 * _K_B_EV)
    assert got == pytest.approx(expected, rel=1e-6)


def test_metal_and_ang_ps_readings_differ_by_the_documented_factor():
    """The bug the docstring warns about: same numbers, two conventions,
    ~10^4 apart. A test that treated them as interchangeable would hide it."""
    v = np.full((4, 3), 0.01)
    m = np.full(4, 16.0)

    as_metal = kinetic_temperature_k_from_jaxmd_metal_velocities(v, m)
    as_ang_ps = kinetic_temperature_k_from_ang_ps_velocities(v, m)

    assert as_metal / as_ang_ps > 1e3


def test_metal_unit_temperature_rejects_bad_input():
    with pytest.raises(ValueError, match="same non-zero length"):
        kinetic_temperature_k_from_jaxmd_metal_velocities(np.zeros((3, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        kinetic_temperature_k_from_jaxmd_metal_velocities(
            np.array([[np.inf, 0.0, 0.0]]), np.array([1.0])
        )


# --- Å/ps <-> metal conversion ----------------------------------------------


def test_ang_ps_to_metal_matches_the_ase_definition():
    from ase import units

    v = np.array([[1.0, -2.0, 0.5]])
    got = ang_ps_velocities_to_jaxmd_metal(v)
    assert got == pytest.approx(v / (1000.0 * float(units.fs)), rel=1e-12)


def test_round_tripping_velocities_preserves_the_temperature():
    """Converting Å/ps to metal units and reading T with the metal formula must
    agree with reading T from the original with the Å/ps formula.

    These are two independent routes to the same physical quantity, so any
    disagreement is a unit bug in one of them.
    """
    v_ang_ps = np.array([[3.0, -1.0, 0.5], [0.0, 2.0, -1.5]])
    m = np.array([16.0, 1.0])

    t_direct = kinetic_temperature_k_from_ang_ps_velocities(v_ang_ps, m)
    t_via_metal = kinetic_temperature_k_from_jaxmd_metal_velocities(
        ang_ps_velocities_to_jaxmd_metal(v_ang_ps), m
    )

    assert t_via_metal == pytest.approx(t_direct, rel=1e-3)


def test_conversion_is_linear():
    v = np.array([[1.0, 2.0, 3.0]])
    assert ang_ps_velocities_to_jaxmd_metal(2.0 * v) == pytest.approx(
        2.0 * ang_ps_velocities_to_jaxmd_metal(v), rel=1e-12
    )


# --- centre-of-mass drift ---------------------------------------------------


def test_com_velocity_is_removed():
    """Left in, COM drift shows up as temperature the system does not have."""
    v = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m = np.full(3, 12.0)

    out = remove_center_of_mass_velocity_ang_ps(v, m)

    assert out == pytest.approx(np.zeros((3, 3)), abs=1e-12)


def test_com_removal_is_mass_weighted():
    v = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    m = np.array([16.0, 1.0])

    out = remove_center_of_mass_velocity_ang_ps(v, m)

    v_com = (16.0 * 1.0 + 1.0 * -1.0) / 17.0
    assert out[0, 0] == pytest.approx(1.0 - v_com)
    assert float(np.sum(m[:, None] * out, axis=0)[0]) == pytest.approx(0.0, abs=1e-12)


def test_com_removal_preserves_relative_motion():
    v = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = remove_center_of_mass_velocity_ang_ps(v, np.full(2, 1.0))
    assert (out[0] - out[1]) == pytest.approx(v[0] - v[1])


def test_com_removal_lowers_the_temperature():
    v = np.full((4, 3), 1.0)  # pure drift
    m = np.full(4, 12.0)
    before = kinetic_temperature_k_from_ang_ps_velocities(v, m)
    after = kinetic_temperature_k_from_ang_ps_velocities(
        remove_center_of_mass_velocity_ang_ps(v, m), m
    )
    assert before > 0.0
    assert after == pytest.approx(0.0, abs=1e-9)


def test_com_removal_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same non-zero length"):
        remove_center_of_mass_velocity_ang_ps(np.zeros((3, 3)), np.zeros(2))


def test_com_removal_rejects_zero_total_mass():
    with pytest.raises(ValueError, match="total mass must be positive"):
        remove_center_of_mass_velocity_ang_ps(np.zeros((2, 3)), np.zeros(2))


# --- monomer partitioning ---------------------------------------------------


def test_uniform_monomer_offsets_are_evenly_spaced():
    assert monomer_offsets_uniform(12, 4).tolist() == [0, 3, 6, 9, 12]


def test_offsets_start_at_zero_and_end_at_n_atoms():
    off = monomer_offsets_uniform(30, 10)
    assert off[0] == 0 and off[-1] == 30


def test_a_single_monomer_spans_everything():
    assert monomer_offsets_uniform(9, 1).tolist() == [0, 9]

"""Unit tests for ASE Maxwell-Boltzmann velocity assignment for CHARMM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
    ase_to_charmm_akma_velocities,
    assign_maxwell_boltzmann_velocities_via_ase,
    charmm_akma_to_ang_fs_velocities,
    estimate_kinetic_temperature_k,
    maybe_assign_velocities_via_ase_if_cold,
    resolve_assignment_temperature_k,
    velocities_are_cold,
)


def test_ase_charmm_velocity_roundtrip():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    v_ase = np.array(
        [
            [0.01, -0.02, 0.03],
            [0.04, 0.05, -0.06],
            [0.07, -0.08, 0.09],
        ],
        dtype=float,
    )
    v_akma = ase_to_charmm_akma_velocities(v_ase, masses)
    back = charmm_akma_to_ang_fs_velocities(v_akma, masses)
    np.testing.assert_allclose(back, v_ase, rtol=1e-12, atol=1e-12)


def test_estimate_kinetic_temperature_k_zero_for_zero_vel():
    masses = np.array([12.0, 1.0], dtype=float)
    vel = np.zeros((2, 3), dtype=float)
    assert estimate_kinetic_temperature_k(vel, masses) == pytest.approx(0.0)
    assert velocities_are_cold(vel, masses_amu=masses, min_temperature_K=1.0)


def test_resolve_assignment_temperature_k_uses_heat_floor_for_corrupt_kw():
    assert resolve_assignment_temperature_k({"firstt": 0.207, "finalt": 1.0}) == pytest.approx(
        60.0
    )
    assert resolve_assignment_temperature_k({"finalt": 240.0, "firstt": 48.0}) == pytest.approx(
        240.0
    )


def test_maybe_assign_velocities_via_ase_if_cold_skips_when_warm():
    kw = {"iasvel": 0, "start": False}
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        return_value=False,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.assign_maxwell_boltzmann_velocities_via_ase",
    ) as assign:
        assert maybe_assign_velocities_via_ase_if_cold(kw) is False
    assign.assert_not_called()


def test_maybe_assign_velocities_via_ase_if_cold_assigns_and_clears_start():
    kw = {"iasvel": 0, "start": True, "finalt": 240.0}
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.assign_maxwell_boltzmann_velocities_via_ase",
        return_value=58.0,
    ) as assign:
        assert maybe_assign_velocities_via_ase_if_cold(kw) is True
    assign.assert_called_once()
    assert kw["iasvel"] == 0
    assert kw["start"] is False


def test_assign_maxwell_boltzmann_velocities_via_ase_syncs_charmm():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    atoms = MagicMock()
    atoms.__len__ = MagicMock(return_value=3)
    atoms.get_velocities.return_value = np.ones((3, 3), dtype=float) * 0.01

    with patch(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.ase_from_pycharmm_state",
        return_value=atoms,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, patch(
        "ase.md.velocitydistribution.MaxwellBoltzmannDistribution",
    ), patch(
        "ase.md.velocitydistribution.Stationary",
    ), patch(
        "ase.md.velocitydistribution.ZeroRotation",
    ):
        assign_maxwell_boltzmann_velocities_via_ase(300.0, quiet=True)

    atoms.set_masses.assert_called_once()
    sync.assert_called_once()

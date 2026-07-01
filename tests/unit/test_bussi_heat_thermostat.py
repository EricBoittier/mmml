"""Unit tests for ASE Bussi heat thermostat integration."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
    apply_bussi_velocity_rescale,
    calculate_bussi_rescale_alpha,
    target_kinetic_energy_kcalmol,
)
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    _requested_heat_thermostat,
    resolve_heat_thermostat,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    _apply_bussi_in_memory_continuation_kw,
    _bussi_heat_chunk_nstep,
    _ensure_bussi_heat_continuation_iasvel,
    bussi_heat_ramp_spec_from_kw,
    heat_ramp_spec_from_kw,
    prepare_bussi_heat_dynamics_kw,
)


def test_requested_heat_thermostat_defaults_to_bussi():
    args = mock.Mock(heat_thermostat="bussi")
    assert _requested_heat_thermostat(args) == "bussi"


def test_calculate_bussi_rescale_alpha_is_stochastic():
    target_ke = target_kinetic_energy_kcalmol(300.0, ndof=30)
    alpha_a = calculate_bussi_rescale_alpha(
        target_ke * 0.8,
        target_kinetic_energy=target_ke,
        ndof=30,
        coupling_time_ps=0.1,
        elapsed_time_ps=0.1,
        rng=np.random.default_rng(1),
    )
    alpha_b = calculate_bussi_rescale_alpha(
        target_ke * 0.8,
        target_kinetic_energy=target_ke,
        ndof=30,
        coupling_time_ps=0.1,
        elapsed_time_ps=0.1,
        rng=np.random.default_rng(2),
    )
    assert alpha_a > 0.0
    assert alpha_b > 0.0
    assert alpha_a != pytest.approx(alpha_b)


def test_prepare_bussi_heat_dynamics_kw_disables_charmm_ihtfrq():
    kw = {
        "firstt": 0.0,
        "finalt": 240.0,
        "timestep": 0.00025,
        "nstep": 1000,
    }
    prepare_bussi_heat_dynamics_kw(
        kw,
        nstep=1000,
        ihtfrq=100,
        timestep_ps=0.00025,
    )
    assert kw["ihtfrq"] == 0
    assert "TEMINC" not in kw
    assert kw["_heat_thermostat"] == "bussi"
    spec = bussi_heat_ramp_spec_from_kw(kw)
    assert spec is not None
    assert spec["thermostat"] == "bussi"
    assert int(spec["ihtfrq"]) == 100
    assert _bussi_heat_chunk_nstep(kw, 1000) == 100
    ramp = heat_ramp_spec_from_kw(kw)
    assert ramp is not None
    assert ramp["thermostat"] == "bussi"


def test_apply_bussi_velocity_rescale_syncs_charmm():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    v_akma = np.ones((3, 3), dtype=float) * 100.0
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_velocities_akma",
        side_effect=[v_akma, v_akma * 1.1],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.calculate_bussi_rescale_alpha",
        return_value=1.1,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_temperature_k",
        return_value=295.0,
    ):
        measured, alpha = apply_bussi_velocity_rescale(
            300.0,
            timestep_ps=0.00025,
            rescale_interval_steps=100,
            quiet=True,
        )
    sync.assert_called_once()
    assert alpha == pytest.approx(1.1)
    assert measured == pytest.approx(295.0)


def test_resolve_heat_thermostat_keeps_bussi_after_pretreat(monkeypatch):
    import argparse

    args = argparse.Namespace(
        heat_thermostat="bussi",
        charmm_mm_pretreat=True,
        setup="pbc_liquid",
        quiet=True,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_charmm_mm_pretreat_for_staged",
        lambda *_a, **_k: True,
    )
    assert resolve_heat_thermostat(args) == "bussi"


def test_apply_bussi_in_memory_continuation_keeps_iasvel_one():
    kw = {
        "firstt": 10.0,
        "finalt": 50.0,
        "timestep": 0.0001,
        "nstep": 50,
    }
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    _apply_bussi_in_memory_continuation_kw(kw)
    assert kw["iasvel"] == 1
    assert kw["start"] is False
    assert kw["_skip_ase_cold_velocity_assign"] is True


def test_ensure_bussi_heat_continuation_iasvel_for_overlap_chunk():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _apply_overlap_chunk_dynamics_kw,
    )

    kw = {
        "firstt": 10.0,
        "finalt": 50.0,
        "timestep": 0.0001,
        "nstep": 50,
        "start": False,
        "iasvel": 0,
    }
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    _apply_overlap_chunk_dynamics_kw(kw, chunk_index=1, has_restart_read=False)
    assert kw["iasvel"] == 1
    assert kw["start"] is False
    _ensure_bussi_heat_continuation_iasvel(kw)
    assert kw["iasvel"] == 1

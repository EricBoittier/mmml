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


def test_capture_charmm_velocities_for_bussi_from_restart(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        capture_charmm_velocities_for_bussi,
    )

    restart = tmp_path / "dyn.res"
    restart.write_text(
        "REST     0     1\n"
        "       1 !NTITLE followed by title\n"
        "* t\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         1           0           0           0           0           0           0\n"
        " !X, Y, Z\n"
        " 0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00\n"
        " !VELOCITIES\n"
        " 1.000000000000000D+02 0.000000000000000D+00 0.000000000000000D+00\n",
        encoding="ascii",
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_cold",
        return_value=False,
    ):
        out = capture_charmm_velocities_for_bussi(restart_path=restart)
    assert out is not None
    assert out.shape == (1, 3)
    sync.assert_called_once()


def test_run_dynamics_captures_bussi_velocities_before_velos_del():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    call_order: list[str] = []

    def _capture(**_kwargs):
        call_order.append("capture")

    def _release():
        call_order.append("release")

    kw = {
        "nstep": 50,
        "start": True,
        "iasvel": 1,
        "firstt": 10.0,
        "finalt": 50.0,
        "timestep": 0.0001,
        "_heat_thermostat": "bussi",
        "_bussi_ramp": {"firstt": 10.0, "finalt": 50.0, "teminc": 0.8, "ihtfrq": 50},
        "_bussi_rescale_interval": 50,
        "_post_dyna_restart_write": "/tmp/fake.res",
    }
    fake_dyn = mock.MagicMock()
    fake_pycharmm = mock.MagicMock()
    fake_pycharmm.DynamicsScript = mock.MagicMock(return_value=fake_dyn)
    with mock.patch.dict("sys.modules", {"pycharmm": fake_pycharmm}), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_via_c_api",
        return_value=fake_dyn,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._dynamics_c_api_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
        side_effect=_release,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._apply_dynamics_io_setters",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.capture_charmm_velocities_for_bussi",
        side_effect=_capture,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.mirror_comparison_velocities_for_dynamics",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.maybe_assign_velocities_via_ase_if_cold",
    ):
        run_dynamics(kw)
    assert call_order == ["capture", "release"]


def test_apply_bussi_velocity_rescale_syncs_charmm():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    v_akma = np.ones((3, 3), dtype=float) * 100.0
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.capture_charmm_velocities_for_bussi",
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


def test_apply_bussi_velocity_rescale_assigns_when_velocities_missing():
    masses = np.array([12.0, 1.0, 1.0], dtype=float)
    v_akma = np.ones((3, 3), dtype=float) * 50.0
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
        return_value=masses,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.capture_charmm_velocities_for_bussi",
        side_effect=[None, v_akma, v_akma],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.assign_maxwell_boltzmann_velocities_via_ase",
        return_value=10.0,
    ) as assign, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma",
    ) as sync, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.calculate_bussi_rescale_alpha",
        return_value=1.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_temperature_k",
        return_value=10.0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.estimate_kinetic_energy_kcalmol",
        return_value=1.0,
    ):
        measured, alpha = apply_bussi_velocity_rescale(
            10.0,
            timestep_ps=0.00025,
            rescale_interval_steps=50,
            quiet=True,
        )
    assign.assert_called_once()
    sync.assert_called_once()
    assert alpha == pytest.approx(1.0)
    assert measured == pytest.approx(10.0)


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


def test_apply_bussi_in_memory_continuation_keeps_iasvel_zero():
    kw = {
        "firstt": 10.0,
        "finalt": 50.0,
        "timestep": 0.0001,
        "nstep": 50,
    }
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    _apply_bussi_in_memory_continuation_kw(kw)
    assert kw["iasvel"] == 0
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
        "iasvel": 1,
    }
    prepare_bussi_heat_dynamics_kw(kw, nstep=50, ihtfrq=50, timestep_ps=0.0001)
    _apply_overlap_chunk_dynamics_kw(kw, chunk_index=1, has_restart_read=False)
    assert kw["iasvel"] == 0
    assert kw["start"] is False
    _ensure_bussi_heat_continuation_iasvel(kw)
    assert kw["iasvel"] == 0


def test_run_dynamics_captures_bussi_velocities_before_velos_del():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    call_order: list[str] = []

    def _capture(**_kwargs):
        call_order.append("capture")

    def _release():
        call_order.append("release")

    kw = {
        "nstep": 50,
        "start": True,
        "iasvel": 1,
        "firstt": 10.0,
        "finalt": 50.0,
        "timestep": 0.0001,
        "_heat_thermostat": "bussi",
        "_bussi_ramp": {"firstt": 10.0, "finalt": 50.0, "teminc": 0.8, "ihtfrq": 50},
        "_bussi_rescale_interval": 50,
        "_post_dyna_restart_write": "/tmp/fake.res",
    }
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_via_c_api",
        return_value=mock.MagicMock(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._dynamics_c_api_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
        side_effect=_release,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.capture_charmm_velocities_for_bussi",
        side_effect=_capture,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.mirror_comparison_velocities_for_dynamics",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.maybe_assign_velocities_via_ase_if_cold",
    ):
        run_dynamics(kw)
    assert call_order == ["capture", "release"]


def test_harmonize_overlap_chunk_preserves_nsavv_when_suppressing_dcd():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _harmonize_overlap_chunk_frequencies,
    )

    kw = {"nsavc": 499, "nsavv": 499, "nprint": 499}
    _harmonize_overlap_chunk_frequencies(kw, 50, global_step_start=0)
    assert kw["_suppress_trajectory"] is True
    assert kw["nsavc"] == 49
    assert kw["nsavv"] == 50
    assert "nprint" not in kw

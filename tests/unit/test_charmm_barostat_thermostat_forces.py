"""CHARMM barostat, thermostat, and force I/O (no MLpot, no libcharmm in CI)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    _apply_hoover_nvt_kwargs,
    _apply_npt_cpt_kwargs,
    build_cpt_equilibration_dynamics,
    build_hoover_heat_dynamics,
    compute_cpt_piston_masses,
    run_dynamics,
)
from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import NptPressureTensor
from tests.unit.pycharmm_stubs import fake_pycharmm_modules


def test_compute_cpt_piston_masses_from_psf_mass():
    fake_select = mock.MagicMock()
    fake_select.get_property.return_value = np.array([12.0, 1.0, 1.0, 16.0] * 3)
    with fake_pycharmm_modules(extra={"pycharmm.select": fake_select}):
        pmass, tmass = compute_cpt_piston_masses()
    total = float(np.sum([12.0, 1.0, 1.0, 16.0] * 3))
    assert pmass == int(total / 50.0)
    assert tmass == int(pmass * 10)


def test_apply_npt_cpt_hoover_barostat_keywords():
    kw: dict = {"nstep": 100}
    _apply_npt_cpt_kwargs(
        kw,
        temp=300.0,
        thermostat="hoover",
        pref=1.0,
        pmass=20,
        tmass=200,
        pgamma=5,
        firstt=280.0,
    )
    assert kw["cpt"] is True
    assert kw["pmass"] == 20
    assert kw["tmass"] == 200
    assert kw["pgamma"] == 5
    assert kw["hoover reft"] == 300.0
    assert kw["firstt"] == 280.0
    assert kw["pint pconst pref"] == 1.0
    assert kw["ihtfrq"] == 0
    assert kw["ieqfrq"] == 0


def test_apply_npt_cpt_berendsen_thermostat_keywords():
    kw: dict = {}
    _apply_npt_cpt_kwargs(
        kw,
        temp=298.15,
        thermostat="berendsen",
        pref=1.0,
        pmass=10,
        tmass=100,
        tcoupling=4.0,
    )
    assert kw["tcons"] is True
    assert kw["tcoupling"] == 4.0
    assert kw["treference"] == 298.15
    assert "hoover reft" not in kw


def test_apply_hoover_nvt_without_cpt_barostat():
    kw: dict = {"nstep": 50}
    _apply_hoover_nvt_kwargs(kw, temp=240.0, tmass=160, firstt=200.0)
    assert "cpt" not in kw
    assert kw["hoover reft"] == 240.0
    assert kw["tmass"] == 160
    assert kw["firstt"] == 200.0
    assert kw["ihtfrq"] == 0


def test_hoover_heat_pbc_uses_cpt_nvt_fixed_volume():
    kw = build_hoover_heat_dynamics(
        temp=30.0,
        firstt=6.0,
        finalt=30.0,
        use_pbc=True,
        tmass=930,
        duration_ps=1.0,
        timestep_ps=0.0002,
    )
    assert kw["cpt"] is True
    assert kw["pmass"] == 0
    assert kw["pgamma"] == 0.0
    assert kw["hoover reft"] == 6.0
    assert kw["tmass"] == 930


def test_cpt_equilibration_anisotropic_pressure_tensor():
    tensor = NptPressureTensor(2.0, 1.5, 1.0, 0.05, -0.1, 0.2)
    kw = build_cpt_equilibration_dynamics(
        pref=1.0,
        pressure_tensor=tensor,
        pmass=16,
        tmass=160,
    )
    assert kw["PRXX"] == 2.0
    assert kw["PRYY"] == 1.5
    assert kw["PRZZ"] == 1.0
    assert kw["PRXY"] == 0.05
    assert "pint pconst pref" not in kw


def test_charmm_grms_after_ener_force_runs_script():
    import mmml.interfaces.pycharmmInterface.mlpot.cli_common as cli_common

    fake_lingo = mock.MagicMock()
    fake_energy = mock.MagicMock()
    fake_energy.get_grms.return_value = 1.23
    with fake_pycharmm_modules(
        energy=fake_energy,
        extra={"pycharmm.lingo": fake_lingo},
    ) as (fake_pycharmm, _coor, _energy), mock.patch.object(
        cli_common, "charmm_grms", return_value=1.23
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_silent_command",
        return_value=__import__("contextlib").nullcontext(),
    ):
        fake_pycharmm.lingo = fake_lingo
        grms = cli_common.charmm_grms_after_ener_force(silent=True)
    fake_lingo.charmm_script.assert_called_once_with("ENER FORCE")
    assert grms == pytest.approx(1.23)


def test_charmm_total_forces_negates_gradient():
    grad = pd.DataFrame(
        {
            "dx": [1.0, -2.0, 0.5],
            "dy": [0.0, 1.0, -0.25],
            "dz": [-3.0, 0.5, 2.0],
        }
    )
    fake_coor = mock.MagicMock()
    fake_coor.get_forces.return_value = grad
    with fake_pycharmm_modules(coor=fake_coor):
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            charmm_total_forces_kcalmol_A,
        )

        forces = charmm_total_forces_kcalmol_A()
    expected = -np.column_stack([grad["dx"], grad["dy"], grad["dz"]])
    np.testing.assert_allclose(forces, expected)


def test_charmm_total_forces_ev_angstrom_unit_conversion():
    physical = np.array([[23.060548867, 0.0, 0.0]], dtype=np.float64)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_total_forces_kcalmol_A",
        return_value=physical,
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            charmm_total_forces_ev_angstrom,
        )

        forces_ev = charmm_total_forces_ev_angstrom()
    np.testing.assert_allclose(forces_ev, [[1.0, 0.0, 0.0]])


def test_run_dynamics_passes_cpt_keywords_to_dynamics_script():
    fake_dyn = mock.MagicMock()
    fake_pycharmm = mock.MagicMock()
    fake_pycharmm.DynamicsScript.return_value = fake_dyn
    kw = build_cpt_equilibration_dynamics(
        temp=300.0,
        pmass=16,
        tmass=160,
        pref=1.0,
        duration_ps=0.0025,
        timestep_ps=0.00025,
    )
    kw["nstep"] = 10
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.clear_comparison_coordinates",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
    ), mock.patch.dict(
        __import__("sys").modules,
        {"pycharmm": fake_pycharmm},
        clear=False,
    ):
        run_dynamics(kw)
    fake_pycharmm.DynamicsScript.assert_called_once()
    passed = fake_pycharmm.DynamicsScript.call_args.kwargs
    assert passed["cpt"] is True
    assert passed["pmass"] == 16
    assert passed["hoover reft"] == 300.0
    fake_dyn.run.assert_called_once()


def test_run_dynamics_clears_comp_when_iasvel_zero_without_start():
    fake_dyn = mock.MagicMock()
    fake_pycharmm = mock.MagicMock()
    fake_pycharmm.DynamicsScript.return_value = fake_dyn
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.clear_comparison_coordinates",
    ) as clear_comp, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers",
    ), mock.patch.dict(
        __import__("sys").modules,
        {"pycharmm": fake_pycharmm},
        clear=False,
    ):
        run_dynamics({"nstep": 5, "iasvel": 0, "start": False})
    clear_comp.assert_called_once()
    fake_dyn.run.assert_called_once()

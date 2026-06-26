"""Unit tests for CHARMM MM pretreat physics resolution and argv forwarding."""

from __future__ import annotations

import argparse

import pytest

from mmml.cli.run.md_system import build_pycharmm_command
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    DEFAULT_CHARMM_MM_PRETREAT_DT_FS,
    resolve_charmm_mm_pretreat_heat_nstep,
    resolve_charmm_mm_pretreat_settings,
    resolve_pretreat_dynamics_print_kwargs,
)
from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
    build_charmm_mm_pretreat_handoff_sections,
)
from tests.unit.test_md_system_pycharmm_cmd import _pycharmm_args


def test_resolve_charmm_mm_pretreat_settings_defaults():
    args = argparse.Namespace(
        temperature=310.0,
        npt_pressure=1.5,
        charmm_mm_pretreat_ps_equi=0.0,
        charmm_mm_pretreat_ps_prod=0.0,
    )
    pretreat = resolve_charmm_mm_pretreat_settings(args)
    assert pretreat.dt_fs == pytest.approx(DEFAULT_CHARMM_MM_PRETREAT_DT_FS)
    assert pretreat.timestep_ps == pytest.approx(0.002)
    assert pretreat.temperature_K == pytest.approx(310.0)
    assert pretreat.pressure_atm == pytest.approx(1.5)


def test_resolve_charmm_mm_pretreat_settings_explicit_overrides():
    args = argparse.Namespace(
        charmm_mm_pretreat_dt_fs=1.5,
        charmm_mm_pretreat_temperature=280.0,
        charmm_mm_pretreat_pressure=2.0,
        temperature=310.0,
        npt_pressure=1.0,
        charmm_mm_pretreat_ps_heat=5.0,
        charmm_mm_pretreat_ps_equi=10.0,
        charmm_mm_pretreat_ps_prod=20.0,
    )
    pretreat = resolve_charmm_mm_pretreat_settings(args)
    assert pretreat.dt_fs == pytest.approx(1.5)
    assert pretreat.temperature_K == pytest.approx(280.0)
    assert pretreat.pressure_atm == pytest.approx(2.0)
    assert pretreat.ps_heat == pytest.approx(5.0)
    assert pretreat.ps_equi == pytest.approx(10.0)
    assert pretreat.ps_prod == pytest.approx(20.0)


def test_resolve_charmm_mm_pretreat_heat_nstep_uses_pretreat_dt():
    args = argparse.Namespace(
        charmm_mm_pretreat_dt_fs=2.0,
        charmm_mm_pretreat_ps_heat=4.0,
        charmm_mm_pretreat_heat_nstep=999,
        temperature=300.0,
        npt_pressure=1.0,
        charmm_mm_pretreat_ps_equi=0.0,
        charmm_mm_pretreat_ps_prod=0.0,
    )
    pretreat = resolve_charmm_mm_pretreat_settings(args)
    assert resolve_charmm_mm_pretreat_heat_nstep(args, settings=pretreat) == 2000


def test_resolve_pretreat_dynamics_print_kwargs_suppresses_status():
    kw = resolve_pretreat_dynamics_print_kwargs(nstep=5000)
    assert kw == {"nprint": 5000, "iprfrq": 5000, "isvfrq": 5000}


def test_build_pretreat_handoff_includes_thermodynamics_section():
    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import CharmmMmPretreatSettings

    pos = np.linspace(0.0, 10.0, 30).reshape(10, 3)
    pretreat = CharmmMmPretreatSettings(
        dt_fs=2.0,
        timestep_ps=0.002,
        temperature_K=300.0,
        pressure_atm=1.0,
        ps_heat=10.0,
        ps_equi=0.0,
        ps_prod=0.0,
    )
    sections = build_charmm_mm_pretreat_handoff_sections(
        pos,
        n_monomers=2,
        tag="dcm_2",
        use_pbc=True,
        workflow_box_side_A=28.0,
        pretreat_settings=pretreat,
        composition={"DCM": 2},
        pretreat_heat_nstep=5000,
    )
    thermo = dict(next(s for title, s in sections if title == "Pretreat thermodynamics"))
    assert thermo["dt_fs"] == "2.000"
    assert thermo["temperature_K"] == "300.00"
    assert thermo["pressure_atm"] == "1.0000"
    assert thermo["heat_ps"] == "10.000"
    assert "density_g/cm³" in thermo


def test_build_pycharmm_command_forwards_pretreat_physics_flags():
    cmd = build_pycharmm_command(
        _pycharmm_args(
            charmm_mm_pretreat=True,
            charmm_mm_pretreat_dt_fs=2.0,
            charmm_mm_pretreat_temperature=295.0,
            charmm_mm_pretreat_pressure=1.2,
        )
    )
    assert "--charmm-mm-pretreat-dt-fs" in cmd
    idx = cmd.index("--charmm-mm-pretreat-dt-fs")
    assert cmd[idx + 1] == "2.0"
    idx = cmd.index("--charmm-mm-pretreat-temperature")
    assert cmd[idx + 1] == "295.0"
    idx = cmd.index("--charmm-mm-pretreat-pressure")
    assert cmd[idx + 1] == "1.2"

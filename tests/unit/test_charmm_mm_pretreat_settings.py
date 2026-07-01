"""Unit tests for CHARMM MM pretreat physics resolution and argv forwarding."""

from __future__ import annotations

import argparse

import pytest

from mmml.cli.run.md_system import build_pycharmm_command
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    DEFAULT_CHARMM_MM_PRETREAT_DT_FS,
    apply_pretreat_dyn_freq_kwargs,
    resolve_charmm_mm_pretreat_heat_nstep,
    resolve_charmm_mm_pretreat_settings,
    resolve_pretreat_dynamics_print_kwargs,
    resolve_pretreat_dyn_inbfrq,
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
    assert pretreat.timestep_ps == pytest.approx(0.001)
    assert pretreat.temperature_K == pytest.approx(310.0)
    assert pretreat.pressure_atm == pytest.approx(1.5)
    assert pretreat.inbfrq == 200
    assert pretreat.imgfrq == 200
    assert pretreat.ixtfrq == 4000


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
    assert kw == {"nprint": 5000, "iprfrq": 5000, "isvfrq": 5000, "nsavv": 5000}


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
        inbfrq=400,
        imgfrq=400,
        ixtfrq=8000,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.charmm_grms",
            lambda: 0.42,
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
            lambda: (28.0, 28.0, 28.0),
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.charmm_crystal_is_active",
            lambda **_: True,
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.probe_charmm_cubic_box_side_A",
            lambda **_: (28.0, "pbound"),
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_charmm_cubic_box_side_A",
            lambda **kw: (float(kw.get("fallback_side_A") or 0.0), "pbound"),
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
            lambda **kw: (float(kw.get("fallback_side_A") or 0.0), "pbound"),
        )
        mp.setattr(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
            lambda: pos.copy(),
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
    assert thermo["inbfrq"] == 400
    assert "density_g/cm³" in thermo


def test_resolve_pretreat_dyn_inbfrq_scales_with_dt():
    args = argparse.Namespace()
    assert resolve_pretreat_dyn_inbfrq(args, dt_fs=0.25) == 50
    assert resolve_pretreat_dyn_inbfrq(args, dt_fs=1.0) == 200
    assert resolve_pretreat_dyn_inbfrq(args, dt_fs=2.0) == 400


def test_resolve_charmm_mm_pretreat_cpt_echeck_defaults_off():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_cpt_echeck,
    )

    args = argparse.Namespace(no_echeck=False, no_scale_echeck=False)
    assert resolve_charmm_mm_pretreat_cpt_echeck(args, echeck=5150.0) == -1.0


def test_resolve_charmm_mm_pretreat_cpt_echeck_no_scale_legacy():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_cpt_echeck,
    )

    args = argparse.Namespace(no_echeck=False, no_scale_echeck=True)
    assert resolve_charmm_mm_pretreat_cpt_echeck(args, echeck=100.0) == pytest.approx(500.0)
    assert resolve_charmm_mm_pretreat_cpt_echeck(args, echeck=8000.0) == pytest.approx(8000.0)


def test_apply_pretreat_dyn_freq_kwargs_pbc():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        apply_pretreat_dyn_freq_kwargs,
    )

    args = argparse.Namespace(charmm_mm_pretreat_dt_fs=1.0)
    kw = {"inbfrq": 50, "imgfrq": 50, "ihbfrq": 50, "ilbfrq": 50, "ixtfrq": 1000}
    apply_pretreat_dyn_freq_kwargs(kw, args, use_pbc=True, dt_fs=1.0)
    assert kw["inbfrq"] == 200
    assert kw["imgfrq"] == 200
    assert kw["ixtfrq"] == 4000


def test_build_pycharmm_command_forwards_pretreat_freq_flags():
    cmd = build_pycharmm_command(
        _pycharmm_args(
            charmm_mm_pretreat=True,
            charmm_mm_pretreat_inbfrq=300,
            charmm_mm_pretreat_imgfrq=250,
        )
    )
    idx = cmd.index("--charmm-mm-pretreat-inbfrq")
    assert cmd[idx + 1] == "300"
    idx = cmd.index("--charmm-mm-pretreat-imgfrq")
    assert cmd[idx + 1] == "250"


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

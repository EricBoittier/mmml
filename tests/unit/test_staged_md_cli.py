"""Tests for staged MLpot CLI stage/PBC resolution."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    recommend_echeck_kcal,
    resolve_charmm_use_pbc,
    resolve_echeck_for_cluster,
    resolve_flat_bottom_selection,
    resolve_loose_pbc,
    resolve_md_stages,
    resolve_mlpot_use_pbc,
    resolve_use_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    CharmmTrajectoryFiles,
    build_hoover_heat_dynamics,
)
from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
    _artifact_paths,
    _build_stage_dynamics_kw,
    _configure_heat_dynamics_start,
    _configure_nve_dynamics_start,
    _overlap_for_stage,
    _prior_restart_for_stage,
)
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import DynamicsOverlapConfig


def test_resolve_md_stages_pycharmm_full():
    args = argparse.Namespace(setup="pycharmm_full", md_stages=None, md_stage=None, phase="staged")
    assert resolve_md_stages(args) == ["mini", "heat", "nve", "equi", "prod"]


def test_resolve_md_stages_free_nve():
    args = argparse.Namespace(setup="free_nve", md_stages=None, md_stage=None, phase="staged")
    assert resolve_md_stages(args) == ["mini", "nve"]


def test_resolve_md_stages_override():
    args = argparse.Namespace(
        setup="pycharmm_full",
        md_stages="mini,heat",
        md_stage=None,
        phase="staged",
    )
    assert resolve_md_stages(args) == ["mini", "heat"]


def test_resolve_use_pbc_from_setup():
    args = argparse.Namespace(setup="pbc_nve", free_space=False, box_size=None)
    assert resolve_use_pbc(args) is True


def test_resolve_use_pbc_free_nvt_setup():
    args = argparse.Namespace(setup="free_nvt", free_space=False, box_size=None)
    assert resolve_use_pbc(args) is False


def test_resolve_use_pbc_free_space():
    args = argparse.Namespace(setup="pbc_nve", free_space=True, box_size=None)
    assert resolve_use_pbc(args) is False


def test_resolve_use_pbc_box_size():
    args = argparse.Namespace(setup="free_nve", free_space=False, box_size=40.0)
    assert resolve_use_pbc(args) is True


def test_loose_pbc_charmm_on_mlpot_off_for_free_setup_with_box():
    args = argparse.Namespace(
        setup="free_nvt",
        free_space=False,
        box_size=55.0,
        mlpot_pbc=False,
    )
    assert resolve_charmm_use_pbc(args) is True
    assert resolve_mlpot_use_pbc(args) is False
    assert resolve_loose_pbc(True, False) is True
    assert resolve_loose_pbc(True, True) is False
    assert resolve_loose_pbc(False, False) is False


def test_resolve_loose_pbc():
    assert resolve_loose_pbc(True, False) is True
    assert resolve_loose_pbc(True, True) is False
    assert resolve_loose_pbc(False, False) is False


def test_mlpot_pbc_flag_enables_mic_on_free_setup_with_box():
    args = argparse.Namespace(
        setup="free_nvt",
        free_space=False,
        box_size=55.0,
        mlpot_pbc=True,
    )
    assert resolve_charmm_use_pbc(args) is True
    assert resolve_mlpot_use_pbc(args) is True


def test_pbc_setup_enables_both():
    args = argparse.Namespace(setup="pbc_nve", free_space=False, box_size=None)
    assert resolve_charmm_use_pbc(args) is True
    assert resolve_mlpot_use_pbc(args) is True


def test_resolve_flat_bottom_selection_dcm_keeps_all_default():
    args = argparse.Namespace(composition="DCM:90", residue="ACO", fb_selection="all")
    assert resolve_flat_bottom_selection(args) == "all"


def test_resolve_flat_bottom_selection_respects_explicit_value():
    args = argparse.Namespace(
        composition="DCM:90",
        residue="ACO",
        fb_selection="TYPE CLGA1",
    )
    assert resolve_flat_bottom_selection(args) == "TYPE CLGA1"


def test_cubic_box_length_from_geometry():
    import numpy as np

    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        cubic_box_length_from_geometry,
    )

    pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    side = cubic_box_length_from_geometry(pos, ml_cutoff=12.0, pad=10.0)
    assert side >= 2.0 * 12.0 + 10.0
    assert side >= 10.0 + 20.0


def test_prior_restart_for_equi_falls_back_to_heat_without_nve(tmp_path: Path):
    paths = _artifact_paths(tmp_path, "dcm_60")
    paths["heat_res"].write_text("heat\n", encoding="utf-8")

    got = _prior_restart_for_stage("equi", paths, restart_from=None)
    assert got == paths["heat_res"]


def test_prior_restart_for_equi_prefers_nve_when_present(tmp_path: Path):
    paths = _artifact_paths(tmp_path, "dcm_60")
    paths["heat_res"].write_text("heat\n", encoding="utf-8")
    paths["nve_res"].write_text("nve\n", encoding="utf-8")

    got = _prior_restart_for_stage("equi", paths, restart_from=None)
    assert got == paths["nve_res"]


def test_recommend_echeck_kcal_single_monomer():
    assert recommend_echeck_kcal(1, 20) == 100.0


def test_recommend_echeck_kcal_dcm9():
    assert recommend_echeck_kcal(9, 45) == 500.0


def test_resolve_heat_firstt_finalt_defaults():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_heat_firstt_finalt,
    )

    args = argparse.Namespace(heat_firstt=None, heat_finalt=None)
    assert resolve_heat_firstt_finalt(args, default_temp=300.0) == (60.0, 300.0)


def test_resolve_heat_firstt_finalt_dcm9_soft():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_heat_firstt_finalt,
    )

    args = argparse.Namespace(heat_firstt=0.0, heat_finalt=240.0)
    assert resolve_heat_firstt_finalt(args, default_temp=300.0) == (0.0, 240.0)


def test_resolve_stage_ps_handles_explicit_none():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_stage_ps

    args = argparse.Namespace(
        ps=1.0,
        ps_prod=None,
        ps_equi=None,
        ps_heat=None,
        ps_nve=None,
    )
    assert resolve_stage_ps(args, "prod") == 1.0
    assert resolve_stage_ps(args, "equi") == 50.0
    assert resolve_stage_ps(args, "heat") == 10.0
    assert resolve_stage_ps(args, "nve") == 1.0


def test_recommend_echeck_kcal_medium_cluster():
    assert recommend_echeck_kcal(20, 100) == 1000.0


def test_recommend_echeck_kcal_dcm90():
    assert recommend_echeck_kcal(90, 450) == 4500.0


def test_resolve_echeck_for_cluster_scales_dcm90():
    args = argparse.Namespace(echeck=500.0, no_echeck=False, no_scale_echeck=False)
    assert resolve_echeck_for_cluster(args, n_atoms=450, n_monomers=90) == 4500.0


def test_resolve_echeck_for_cluster_respects_no_scale():
    args = argparse.Namespace(echeck=500.0, no_echeck=False, no_scale_echeck=True)
    assert resolve_echeck_for_cluster(args, n_atoms=450, n_monomers=90) == 500.0


def test_resolve_echeck_for_cluster_user_floor_above_recommended():
    args = argparse.Namespace(echeck=8000.0, no_echeck=False, no_scale_echeck=False)
    assert resolve_echeck_for_cluster(args, n_atoms=450, n_monomers=90) == 8000.0


def test_build_stage_dynamics_kw_restart_omits_invalid_res_flag():
    args = argparse.Namespace()
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "nve",
        args=args,
        timestep_ps=0.0005,
        nstep=2000,
        save_interval_ps=0.05,
        temp=240.0,
        echeck=10000.0,
        dyn_print=dyn_print,
        restart=True,
    )
    assert kw["restart"] is True
    assert "res" not in kw


def test_build_stage_dynamics_kw_heat_hoover_vacuum_uses_ihtfrq_fallback():
    args = argparse.Namespace(heat_thermostat="hoover", heat_firstt=0.0, heat_finalt=240.0)
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "heat",
        args=args,
        timestep_ps=0.00025,
        nstep=80000,
        save_interval_ps=0.125,
        temp=300.0,
        echeck=100.0,
        dyn_print=dyn_print,
        restart=False,
        use_pbc=False,
    )
    assert "cpt" not in kw
    assert "hoover reft" not in kw
    assert kw["iasors"] == 0
    assert kw["ihtfrq"] > 0
    assert kw["TEMINC"] > 0


def test_build_stage_dynamics_kw_heat_hoover_pbc_disables_ihtfrq_ramp():
    args = argparse.Namespace(heat_thermostat="hoover", heat_firstt=0.0, heat_finalt=240.0)
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.compute_cpt_piston_masses",
        return_value=(80, 800),
    ):
        kw = _build_stage_dynamics_kw(
            "heat",
            args=args,
            timestep_ps=0.00025,
            nstep=80000,
            save_interval_ps=0.125,
            temp=300.0,
            echeck=100.0,
            dyn_print=dyn_print,
            restart=False,
            use_pbc=True,
        )
    assert kw["hoover reft"] == 0.0
    assert kw["tmass"] == 800
    assert kw["echeck"] == 5000.0
    assert kw["pgamma"] == 0.0
    assert kw["cpt"] is True
    assert kw["ihtfrq"] == 0
    assert "TEMINC" not in kw


def test_build_hoover_heat_tmass_floor_for_small_psf_mass():
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.compute_cpt_piston_masses",
        return_value=(8, 80),
    ):
        kw = build_hoover_heat_dynamics(
            firstt=10.0,
            finalt=240.0,
            use_pbc=True,
        )
    assert kw["tmass"] == 400
    assert kw["pgamma"] == 0.0
    assert kw["hoover reft"] == 10.0


def test_resolve_heat_hoover_tmass_explicit_and_clamped_default():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_heat_hoover_tmass

    args = argparse.Namespace(heat_hoover_tmass=600)
    assert resolve_heat_hoover_tmass(args, psf_tmass=80) == 600
    args = argparse.Namespace(heat_hoover_tmass=None)
    assert resolve_heat_hoover_tmass(args, psf_tmass=80) == 400
    assert resolve_heat_hoover_tmass(args, psf_tmass=5000) == 1200


def test_build_stage_dynamics_kw_free_space_equi_uses_charmm_heat_controls():
    args = argparse.Namespace()
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "equi",
        args=args,
        timestep_ps=0.0001,
        nstep=500,
        save_interval_ps=0.04,
        temp=300.0,
        echeck=50000.0,
        dyn_print=dyn_print,
        restart=True,
        use_pbc=False,
    )
    assert kw["restart"] is True
    assert kw["ihtfrq"] == 0
    assert kw["TEMINC"] == 0.0
    assert kw["iasvel"] == 0
    assert "firstt" not in kw
    assert "hoover reft" not in kw
    assert "tmass" not in kw
    assert "cpt" not in kw
    assert "pint pconst pref" not in kw
    assert kw["imgfrq"] == 0


def test_overlap_for_stage_enables_heat_chunking():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.4,
        check_interval=100,
        n_monomers=10,
        use_pbc=False,
    )

    assert _overlap_for_stage("heat", cfg) is cfg
    assert _overlap_for_stage("equi", cfg) is cfg
    assert _overlap_for_stage("prod", cfg) is cfg


def test_configure_heat_dynamics_start_hoover_memory_handoff_no_comp_velocities():
    """Hoover CPT after mini must not use iasvel=0 + start (COMP holds coordinates)."""
    io = CharmmTrajectoryFiles()
    kw = {
        "firstt": 0.0,
        "finalt": 240.0,
        "tbath": 240.0,
        "cpt": True,
        "hoover reft": 240.0,
        "tmass": 2000,
        "start": True,
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign:
        _configure_heat_dynamics_start(
            kw,
            io,
            coords_in_memory=True,
            restart_from_file=False,
            timestep_ps=0.00025,
            use_pbc=True,
            quiet=True,
            heat_thermostat="hoover",
        )

    assign.assert_called_once()
    assert kw["restart"] is False
    assert kw["start"] is False
    assert kw["iasvel"] == 1


def test_configure_heat_dynamics_start_scale_memory_handoff_single_dyna():
    """Scale heat after mini uses one dyna (start=True), not nstep=0 + heat."""
    io = CharmmTrajectoryFiles()
    kw = {
        "firstt": 26.0,
        "finalt": 130.0,
        "tbath": 130.0,
        "ihtfrq": 500,
        "start": False,
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign:
        _configure_heat_dynamics_start(
            kw,
            io,
            coords_in_memory=True,
            restart_from_file=False,
            timestep_ps=0.0002,
            use_pbc=True,
            quiet=True,
            heat_thermostat="scale",
        )

    assign.assert_not_called()
    assert kw["restart"] is False
    assert kw["new"] is False
    assert kw["start"] is True
    assert kw["iasvel"] == 1
    assert kw["iasors"] == 0


def test_configure_nve_dynamics_start_memory_handoff_no_readyn(tmp_path):
    """After mini, NVE must not READYN a 1-step scratch restart (CHARMM EOF)."""
    from unittest.mock import patch

    res = tmp_path / "nve_dcm_5.res"
    io = CharmmTrajectoryFiles(restart_write=res)
    kw = {"restart": True, "start": True, "iasvel": 1, "firstt": 300.0}

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_from_current_state"
    ) as rewrite:
        _configure_nve_dynamics_start(
            kw,
            io,
            coords_in_memory=True,
            restart_from_file=False,
            timestep_ps=0.00025,
            use_pbc=False,
            quiet=True,
            temp=60.0,
        )

    assign.assert_called_once()
    rewrite.assert_not_called()
    assert kw["restart"] is False
    assert kw["start"] is False
    assert kw["iasvel"] == 1
    assert io.restart_read is None


def _write_restartable_res(path: Path, *, jhstrt: int = 250) -> None:
    path.write_text(
        "REST     1       500\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        f"   10     0     500       1      10   {jhstrt}     297       0       0\n",
        encoding="ascii",
    )


def test_configure_heat_dynamics_start_in_place_resume_uses_dyna_restart(tmp_path):
    res = tmp_path / "heat_dcm_20.res"
    _write_restartable_res(res, jhstrt=250)
    io = CharmmTrajectoryFiles(restart_read=res, restart_write=res)
    kw = {
        "firstt": 42.4,
        "finalt": 60.0,
        "cpt": True,
        "hoover reft": 60.0,
        "tmass": 2000,
        "start": True,
    }

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature"
    ) as assign:
        _configure_heat_dynamics_start(
            kw,
            io,
            coords_in_memory=False,
            restart_from_file=True,
            timestep_ps=0.0001,
            use_pbc=True,
            quiet=True,
            heat_thermostat="hoover",
        )

    assign.assert_not_called()
    assert kw["restart"] is True
    assert kw["start"] is False
    assert kw["new"] is False
    assert kw["iasvel"] == 1
    assert io.restart_read == res


def test_reset_stage_restart_preserves_in_place_read(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_restart,
    )

    res = tmp_path / "heat.res"
    _write_restartable_res(res)
    _reset_stage_restart(res, restart_read=res)
    assert res.is_file()

    _reset_stage_restart(res)
    assert not res.is_file()

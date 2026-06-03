"""Tests for staged MLpot CLI stage/PBC resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    recommend_echeck_kcal,
    resolve_echeck_for_cluster,
    resolve_flat_bottom_selection,
    resolve_md_stages,
    resolve_use_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
    _artifact_paths,
    _build_stage_dynamics_kw,
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


def test_resolve_flat_bottom_selection_dcm_uses_one_carbon_type():
    args = argparse.Namespace(composition="DCM:90", residue="ACO", fb_selection="all")
    assert resolve_flat_bottom_selection(args) == "TYPE C"


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


def test_overlap_for_stage_disables_heat_chunking_only():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.4,
        check_interval=100,
        n_monomers=10,
        use_pbc=False,
    )

    assert _overlap_for_stage("heat", cfg) is None
    assert _overlap_for_stage("equi", cfg) is cfg
    assert _overlap_for_stage("prod", cfg) is cfg

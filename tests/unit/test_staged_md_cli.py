"""Tests for staged MLpot CLI stage/PBC resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    resolve_md_stages,
    resolve_use_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
    _artifact_paths,
    _build_stage_dynamics_kw,
    _prior_restart_for_stage,
)


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


def test_resolve_use_pbc_free_space():
    args = argparse.Namespace(setup="pbc_nve", free_space=True, box_size=None)
    assert resolve_use_pbc(args) is False


def test_resolve_use_pbc_box_size():
    args = argparse.Namespace(setup="free_nve", free_space=False, box_size=40.0)
    assert resolve_use_pbc(args) is True


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


def test_build_stage_dynamics_kw_restart_omits_invalid_res_flag():
    args = argparse.Namespace(
        npt_thermostat="hoover",
        npt_pressure=1.0,
        npt_pgamma=5.0,
    )
    dyn_print = {"nprint": 100, "iprfrq": 500, "isvfrq": 500}
    kw = _build_stage_dynamics_kw(
        "equi",
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

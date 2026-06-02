"""Tests for mmml md-system pycharmm argv building and log summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.cli.run.md_system import (
    _pycharmm_run_summary,
    build_pycharmm_command,
)


def _pycharmm_args(**overrides) -> argparse.Namespace:
    base = dict(
        setup="pbc_nvt",
        spacing=5.0,
        ps=1.0,
        dt_fs=0.5,
        temperature=50.0,
        mini_nstep=20,
        residue="MEOH",
        fix_resids="",
        constrain_resids="",
        no_fix=False,
        dyn_nprint=100,
        dyn_iprfrq=500,
        dcd_nsavc=100,
        echeck=100.0,
        ps_heat=2.0,
        ps_equi=2.0,
        n_prod_segments=1,
        composition="DCM:20",
        n_molecules=10,
        md_stage=None,
        md_stages="mini,heat,equi",
        tag=None,
        checkpoint=None,
        output_dir=Path("artifacts/pycharmm_mlpot/dcm20_pbc"),
        box_size=40.0,
        ps_nve=None,
        ps_prod=20.0,
        restart_from=None,
        from_psf=None,
        from_crd=None,
        no_pre_minimize=False,
        skip_cluster_build=False,
        skip_if_crd_exists=False,
        charmm_pre_minimize=True,
        charmm_sd_steps=25,
        charmm_abnr_steps=100,
        charmm_tolenr=0.001,
        charmm_tolgrd=0.001,
        ml_batch_size=None,
        ml_gpu_count=None,
        ml_max_active_dimers=None,
        no_echeck=False,
        skip_energy_show=False,
        show_energy=None,
        quiet=False,
        no_scale_mini_nstep=False,
        no_scale_echeck=False,
        allow_high_grms=False,
        max_grms_before_dyn=50.0,
        test_first=False,
        test_first_tol=0.005,
        test_first_step=1e-4,
        test_first_resids="",
        test_first_charmm=False,
        test_first_update_nbonds=False,
        packmol_sphere=True,
        packmol_radius=10.0,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        extra_args=[],
        seed=123,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_pycharmm_command_omits_residue_when_composition_set():
    cmd = build_pycharmm_command(_pycharmm_args())
    assert "--composition" in cmd
    assert "DCM:20" in cmd
    assert "--residue" not in cmd
    assert "--fix-resids" not in cmd
    assert "--constrain-resids" not in cmd


def test_build_pycharmm_command_includes_residue_without_composition():
    cmd = build_pycharmm_command(_pycharmm_args(composition=None, residue="ACO"))
    assert "--residue" in cmd
    idx = cmd.index("--residue")
    assert cmd[idx + 1] == "ACO"
    assert "--composition" not in cmd
    assert "--n-molecules" in cmd


def test_build_pycharmm_command_includes_ml_max_active_dimers_when_set():
    cmd = build_pycharmm_command(_pycharmm_args(ml_max_active_dimers=1200))
    assert "--ml-max-active-dimers" in cmd
    idx = cmd.index("--ml-max-active-dimers")
    assert cmd[idx + 1] == "1200"


def test_build_pycharmm_command_includes_ml_gpu_count_when_set():
    cmd = build_pycharmm_command(_pycharmm_args(ml_gpu_count=2))
    assert "--ml-gpu-count" in cmd
    idx = cmd.index("--ml-gpu-count")
    assert cmd[idx + 1] == "2"


def test_build_pycharmm_command_includes_fix_resids_when_set():
    cmd = build_pycharmm_command(_pycharmm_args(fix_resids="1,3"))
    assert "--fix-resids" in cmd
    idx = cmd.index("--fix-resids")
    assert cmd[idx + 1] == "1,3"


def test_pycharmm_run_summary_composition():
    summary = _pycharmm_run_summary(_pycharmm_args())
    assert "composition=DCM:20" in summary
    assert "MEOH" not in summary
    assert "fix-resids" not in summary


def test_pycharmm_run_summary_shows_fix_when_explicit():
    summary = _pycharmm_run_summary(_pycharmm_args(fix_resids="1"))
    assert "fix-resids=1" in summary

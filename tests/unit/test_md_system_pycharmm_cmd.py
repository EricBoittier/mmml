"""Tests for mmml md-system pycharmm argv building and log summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmml.cli.run.md_system import (
    _pycharmm_run_summary,
    build_pycharmm_command,
    build_run_manifest,
    resolve_job_name,
    save_job_run_manifest,
)


def _pycharmm_args(**overrides) -> argparse.Namespace:
    base = dict(
        setup="pbc_nvt",
        spacing=5.0,
        ps=1.0,
        dt_fs=0.5,
        temperature=50.0,
        mini_nstep=20,
        nprint=50,
        heat_ihtfrq=0,
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
        npt_thermostat="hoover",
        npt_pressure=1.0,
        npt_pgamma=5.0,
        n_equi_segments=1,
        n_prod_segments=1,
        composition="DCM:20",
        n_molecules=10,
        md_stage=None,
        md_stages="mini,heat,equi",
        tag=None,
        checkpoint=None,
        output_dir=Path("artifacts/pycharmm_mlpot/dcm20_pbc"),
        job_name=None,
        jobs_dir=Path("artifacts/md_system/jobs"),
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
        flat_bottom_selection="all",
        extra_args=[],
        seed=123,
        heat_thermostat="scale",
        dynamics_overlap_action="rescue",
        dynamics_overlap_min_distance=1.5,
        dynamics_overlap_check_interval=50,
        mm_switch_on=8.0,
        mm_switch_width=5.0,
        ml_switch_width=1.5,
        reuse_packmol_cache=True,
        rebuild_packmol=False,
        packmol_cache_dir=None,
        save_run_state=False,
        run_state_dir=None,
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


def test_build_pycharmm_command_derives_n_molecules_from_composition():
    cmd = build_pycharmm_command(
        _pycharmm_args(composition="DCM:90", n_molecules=10)
    )
    idx = cmd.index("--n-molecules")
    assert cmd[idx + 1] == "90"


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


def test_build_pycharmm_command_forwards_intra_monomer_guard():
    cmd = build_pycharmm_command(_pycharmm_args())
    idx = cmd.index("--dynamics-intra-min-distance")
    assert cmd[idx + 1] == "1.0"


def test_build_pycharmm_command_forwards_heat_thermostat():
    cmd = build_pycharmm_command(_pycharmm_args(heat_thermostat="hoover"))
    idx = cmd.index("--heat-thermostat")
    assert cmd[idx + 1] == "hoover"


def test_build_pycharmm_command_forwards_npt_cpt_flags():
    cmd = build_pycharmm_command(
        _pycharmm_args(npt_thermostat="berendsen", npt_pressure=2.5, npt_pgamma=0.0)
    )
    idx = cmd.index("--npt-thermostat")
    assert cmd[idx + 1] == "berendsen"
    idx = cmd.index("--npt-pressure")
    assert cmd[idx + 1] == "2.5"
    idx = cmd.index("--npt-pgamma")
    assert cmd[idx + 1] == "0.0"


def test_build_pycharmm_command_includes_ml_switch_width_default():
    cmd = build_pycharmm_command(_pycharmm_args())
    idx = cmd.index("--ml-switch-width")
    assert cmd[idx + 1] == "1.5"


def test_build_pycharmm_command_includes_ml_gpu_count_when_set():
    cmd = build_pycharmm_command(_pycharmm_args(ml_gpu_count=2))
    assert "--ml-gpu-count" in cmd
    idx = cmd.index("--ml-gpu-count")
    assert cmd[idx + 1] == "2"


def test_build_pycharmm_command_forwards_packmol_cache_and_run_state_flags():
    cmd = build_pycharmm_command(
        _pycharmm_args(
            rebuild_packmol=True,
            save_run_state=True,
            packmol_cache_dir=Path("/tmp/mmml_packmol"),
            run_state_dir=Path("/tmp/run_state"),
            reuse_packmol_cache=False,
        )
    )
    assert "--no-reuse-packmol-cache" in cmd
    assert "--rebuild-packmol" in cmd
    assert "--save-run-state" in cmd
    idx = cmd.index("--packmol-cache-dir")
    assert cmd[idx + 1] == "/tmp/mmml_packmol"
    idx = cmd.index("--run-state-dir")
    assert cmd[idx + 1] == "/tmp/run_state"


def test_build_pycharmm_command_forwards_flat_bottom_selection():
    cmd = build_pycharmm_command(
        _pycharmm_args(flat_bottom_radius=15.0, flat_bottom_selection="TYPE C*")
    )
    assert "--fb-selection" in cmd
    idx = cmd.index("--fb-selection")
    assert cmd[idx + 1] == "TYPE C*"


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


def test_resolve_job_name_from_output_dir():
    assert resolve_job_name(_pycharmm_args()) == "dcm20_pbc"


def test_resolve_job_name_explicit_overrides_output_dir():
    assert resolve_job_name(_pycharmm_args(job_name="dcm90_nvt")) == "dcm90_nvt"


def test_resolve_job_name_skips_generic_default_output_dir():
    assert (
        resolve_job_name(
            _pycharmm_args(output_dir=Path("artifacts/pycharmm_mlpot"), job_name=None)
        )
        is None
    )


def test_save_job_run_manifest_writes_registry_and_output_copy(tmp_path):
    args = _pycharmm_args(
        output_dir=tmp_path / "dcm20_pbc",
        job_name="dcm20_pbc",
        jobs_dir=tmp_path / "jobs",
    )
    manifest = build_run_manifest(
        args,
        backend="pycharmm",
        argv=["--phase", "staged"],
        started_at="2026-06-02T00:00:00+00:00",
        finished_at="2026-06-02T00:01:00+00:00",
        exit_code=0,
    )
    registry_path = save_job_run_manifest(
        args.jobs_dir,
        "dcm20_pbc",
        manifest,
        output_dir=args.output_dir,
    )
    assert registry_path == tmp_path / "jobs" / "dcm20_pbc.json"
    assert (tmp_path / "dcm20_pbc" / "run_manifest.json").is_file()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["job_name"] == "dcm20_pbc"
    assert payload["backend"] == "pycharmm"
    assert payload["setup"] == "pbc_nvt"
    assert payload["args"]["composition"] == "DCM:20"

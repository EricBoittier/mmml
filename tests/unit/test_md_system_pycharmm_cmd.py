"""Tests for mmml md-system pycharmm argv building and log summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from mmml.cli.run.md_system import (
    _apply_charmm_omp_threads_env,
    _pycharmm_run_summary,
    build_pycharmm_command,
    build_run_manifest,
    parse_md_system_args,
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
        dyn_freq_cadence=50,
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
        ml_spatial_mpi=False,
        charmm_omp_threads=None,
        ml_max_active_dimers=None,
        no_echeck=False,
        skip_energy_show=False,
        show_energy=None,
        quiet=False,
        no_scale_mini_nstep=False,
        no_scale_echeck=False,
        allow_high_grms=False,
        no_scale_max_grms=False,
        max_grms_before_dyn=50.0,
        test_first=False,
        test_first_tol=0.005,
        test_first_step=1e-4,
        test_first_resids="",
        test_first_charmm=False,
        test_first_update_nbonds=False,
        packmol_sphere=True,
        packmol_placement=None,
        packmol_radius=10.0,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_selection="all",
        extra_args=[],
        seed=123,
        heat_thermostat="scale",
        heat_mode="ramp",
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
        overlap_run_state_dir=None,
        overlap_run_state_every_chunks=0,
        bonded_mm_mini=True,
        bonded_mm_mini_after="mini,heat",
        bonded_mm_mini_steps=50,
        bonded_recovery_backend="jax",
        bonded_mm_internal_margin=0.0,
        bonded_mm_mini_always=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_pycharmm_command_forwards_dyn_freq_cadence():
    from mmml.cli.run.md_pbc_suite import pycharmm_mlpot

    cmd = build_pycharmm_command(_pycharmm_args(dyn_freq_cadence=50))
    assert "--dyn-freq-cadence" in cmd
    idx = cmd.index("--dyn-freq-cadence")
    assert cmd[idx + 1] == "50"
    parsed = pycharmm_mlpot.parse_args(cmd)
    assert parsed.dyn_freq_cadence == 50


def test_build_pycharmm_command_forwards_include_mm_false():
    cmd = build_pycharmm_command(_pycharmm_args(include_mm=False))
    assert "--no-include-mm" in cmd


def test_build_pycharmm_command_omits_include_mm_when_default_true():
    cmd = build_pycharmm_command(_pycharmm_args())
    assert "--include-mm" not in cmd
    assert "--no-include-mm" not in cmd


def test_build_pycharmm_command_no_include_mm_parses_in_pycharmm_backend():
    from mmml.cli.run.md_pbc_suite import pycharmm_mlpot

    cmd = build_pycharmm_command(_pycharmm_args(include_mm=False))
    assert "--no-include-mm" in cmd
    parsed = pycharmm_mlpot.parse_args(cmd)
    assert parsed.include_mm is False


def test_build_pycharmm_command_fire_min_flags_parse_in_pycharmm_backend():
    from mmml.cli.run.md_pbc_suite import pycharmm_mlpot

    cmd = build_pycharmm_command(
        _pycharmm_args(
            calculator_pre_minimize=True,
            pre_min_steps=50,
            pre_min_fmax=0.1,
            bfgs_maxstep=0.05,
            fire_min_steps=200,
            fire_min_maxstep=0.2,
            rescue_fire_fmax=0.05,
        )
    )
    assert "--fire-min-steps" in cmd
    parsed = pycharmm_mlpot.parse_args(cmd)
    assert parsed.fire_min_steps == 200
    assert parsed.fire_min_maxstep == pytest.approx(0.2)
    assert parsed.rescue_fire_fmax == pytest.approx(0.05)


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


def test_build_pycharmm_command_forwards_bonded_recovery_backend():
    cmd = build_pycharmm_command(_pycharmm_args(bonded_recovery_backend="jax"))
    idx = cmd.index("--bonded-recovery-backend")
    assert cmd[idx + 1] == "jax"


def test_build_pycharmm_command_forwards_intra_monomer_guard():
    cmd = build_pycharmm_command(_pycharmm_args())
    idx = cmd.index("--dynamics-intra-min-distance")
    assert cmd[idx + 1] == "0.5"


def test_build_pycharmm_command_forwards_no_scale_max_grms():
    cmd = build_pycharmm_command(
        _pycharmm_args(no_scale_max_grms=True, max_grms_before_dyn=80.0)
    )
    assert "--no-scale-max-grms" in cmd
    idx = cmd.index("--max-grms-before-dyn")
    assert cmd[idx + 1] == "80.0"


def test_build_pycharmm_command_forwards_heat_thermostat():
    cmd = build_pycharmm_command(_pycharmm_args(heat_thermostat="hoover"))
    idx = cmd.index("--heat-thermostat")
    assert cmd[idx + 1] == "hoover"


def test_build_pycharmm_command_thermalize_setup_and_heat_mode():
    cmd = build_pycharmm_command(
        _pycharmm_args(setup="pbc_thermalize", heat_mode="hold", md_stages=None)
    )
    idx = cmd.index("--setup")
    assert cmd[idx + 1] == "pbc_thermalize"
    idx = cmd.index("--md-stages")
    assert cmd[idx + 1] == "mini,heat,equi"
    idx = cmd.index("--heat-mode")
    assert cmd[idx + 1] == "hold"


def test_parse_md_system_config_accepts_thermalize_setup(tmp_path):
    cfg = tmp_path / "md_system.yaml"
    cfg.write_text(
        "\n".join(
            [
                "backend: pycharmm",
                "setup: pbc_thermalize",
                "composition: DCM:10",
                "heat_mode: hold",
                "temperature: 320.0",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_md_system_args(["--config", str(cfg)])

    assert args.setup == "pbc_thermalize"
    assert args.heat_mode == "hold"
    assert args.temperature == pytest.approx(320.0)


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


def test_build_pycharmm_command_includes_ml_spatial_mpi_when_set():
    cmd = build_pycharmm_command(_pycharmm_args(ml_spatial_mpi=True))
    assert "--ml-spatial-mpi" in cmd


def test_parse_md_system_config_accepts_charmm_omp_threads(tmp_path):
    cfg = tmp_path / "md_system.yaml"
    cfg.write_text(
        "\n".join(
            [
                "backend: pycharmm",
                "setup: pbc_npt",
                "composition: DCM:20",
                "charmm_omp_threads: 4",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_md_system_args(["--config", str(cfg)])

    assert args.charmm_omp_threads == 4


def test_apply_charmm_omp_threads_env_sets_bootstrap_env(monkeypatch):
    monkeypatch.delenv("MMML_CHARMM_OMP_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    monkeypatch.delenv("MMML_JAX_COMPILE_THREADS", raising=False)
    monkeypatch.setenv("MMML_NO_JAX_COMPILE_THREADS", "1")

    applied = _apply_charmm_omp_threads_env(_pycharmm_args(charmm_omp_threads=8))

    assert applied == "8"
    assert os.environ["MMML_CHARMM_OMP_THREADS"] == "8"
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "8"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "8"
    assert os.environ["MMML_JAX_COMPILE_THREADS"] == "8"
    assert os.environ["MMML_NO_JAX_COMPILE_THREADS"] == "0"


def test_apply_charmm_omp_threads_env_preserves_explicit_library_threads(monkeypatch):
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "4")
    monkeypatch.setenv("MMML_JAX_COMPILE_THREADS", "5")

    _apply_charmm_omp_threads_env(_pycharmm_args(charmm_omp_threads=8))

    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "4"
    assert os.environ["MMML_JAX_COMPILE_THREADS"] == "5"


def test_apply_charmm_omp_threads_env_rejects_nonpositive():
    with pytest.raises(ValueError, match="charmm_omp_threads"):
        _apply_charmm_omp_threads_env(_pycharmm_args(charmm_omp_threads=0))


def test_build_pycharmm_command_forwards_ml_compute_dtype_when_set():
    cmd = build_pycharmm_command(_pycharmm_args(ml_compute_dtype="float64"))
    assert "--ml-compute-dtype" in cmd
    idx = cmd.index("--ml-compute-dtype")
    assert cmd[idx + 1] == "float64"


def test_build_pycharmm_command_uses_grid_builder_by_default():
    cmd = build_pycharmm_command(
        _pycharmm_args(packmol_sphere=None, packmol_placement="cube")
    )
    assert "--packmol-placement" not in cmd
    assert "--packmol" not in cmd


def test_build_pycharmm_command_explicit_packmol_argv_parses_in_pycharmm_backend():
    from mmml.cli.run.md_pbc_suite import pycharmm_mlpot

    cmd = build_pycharmm_command(
        _pycharmm_args(packmol=True, packmol_sphere=None, packmol_placement="cube")
    )
    assert "--packmol-placement" in cmd
    idx = cmd.index("--packmol-placement")
    assert cmd[idx + 1] == "cube"
    parsed = pycharmm_mlpot.parse_args(cmd)
    assert parsed.packmol_placement == "cube"


def test_build_pycharmm_command_forwards_pyxtal_flags():
    from mmml.cli.run.md_pbc_suite import pycharmm_mlpot

    cmd = build_pycharmm_command(
        _pycharmm_args(
            pyxtal=True,
            packmol=False,
            pyxtal_spg=4,
            pyxtal_dim=3,
            pyxtal_supercell="2,2,1",
            pyxtal_stoichiometry=[2],
        )
    )
    assert "--pyxtal" in cmd
    assert "--packmol-placement" not in cmd
    idx = cmd.index("--pyxtal-spg")
    assert cmd[idx + 1] == "4"
    idx = cmd.index("--pyxtal-supercell")
    assert cmd[idx + 1] == "2,2,1"
    parsed = pycharmm_mlpot.parse_args(cmd)
    assert parsed.pyxtal is True
    assert parsed.pyxtal_spg == 4


def test_build_pycharmm_command_forwards_packmol_cache_and_run_state_flags():
    cmd = build_pycharmm_command(
        _pycharmm_args(
            rebuild_packmol=True,
            save_run_state=True,
            packmol_cache_dir=Path("/tmp/mmml_packmol"),
            run_state_dir=Path("/tmp/run_state"),
            overlap_run_state_every_chunks=4,
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
    idx = cmd.index("--overlap-run-state-every-chunks")
    assert cmd[idx + 1] == "4"


def test_build_pycharmm_command_forwards_mc_density_flags():
    from mmml.cli.run.md_pbc_suite import pycharmm_mlpot

    cmd = build_pycharmm_command(
        _pycharmm_args(
            box_size=None,
            mc_density_equalize=False,
            mc_density_target_g_cm3=1.1,
            mc_density_steps=12,
            mc_density_step_scale=0.03,
            mc_density_temperature=0.04,
            mc_density_seed=99,
            mc_density_min_scale=0.8,
            mc_density_max_scale=1.2,
        )
    )
    assert "--no-mc-density-equalize" in cmd
    assert "--mc-density-target-g-cm3" in cmd
    assert cmd[cmd.index("--mc-density-target-g-cm3") + 1] == "1.1"
    assert cmd[cmd.index("--mc-density-steps") + 1] == "12"
    assert cmd[cmd.index("--mc-density-step-scale") + 1] == "0.03"
    assert cmd[cmd.index("--mc-density-temperature") + 1] == "0.04"
    assert cmd[cmd.index("--mc-density-seed") + 1] == "99"
    assert cmd[cmd.index("--mc-density-min-scale") + 1] == "0.8"
    assert cmd[cmd.index("--mc-density-max-scale") + 1] == "1.2"
    parsed = pycharmm_mlpot.parse_args(cmd)
    assert parsed.mc_density_equalize is False
    assert parsed.mc_density_target_g_cm3 == 1.1
    assert parsed.mc_density_steps == 12


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


def test_build_pycharmm_command_forwards_periodic_mm_flags():
    cmd = build_pycharmm_command(
        _pycharmm_args(
            mm_nonbond_mode="periodic_external",
            lr_solver="scafacos",
            scafacos_method="p3m",
        )
    )
    assert "--mm-nonbond-mode" in cmd
    assert cmd[cmd.index("--mm-nonbond-mode") + 1] == "periodic_external"
    assert "--lr-solver" in cmd
    assert cmd[cmd.index("--lr-solver") + 1] == "scafacos"
    assert "--scafacos-method" in cmd
    assert cmd[cmd.index("--scafacos-method") + 1] == "p3m"


def test_build_pycharmm_command_forwards_jax_pme_flags():
    cmd = build_pycharmm_command(
        _pycharmm_args(
            mm_nonbond_mode="jax_mic",
            lr_solver="jax_pme",
            jax_pme_method="pme",
            jax_pme_sr_cutoff=7.5,
            jax_pme_dispersion=False,
        )
    )
    assert cmd[cmd.index("--lr-solver") + 1] == "jax_pme"
    assert cmd[cmd.index("--jax-pme-method") + 1] == "pme"
    assert cmd[cmd.index("--jax-pme-sr-cutoff") + 1] == "7.5"
    assert "--no-jax-pme-dispersion" in cmd


def test_build_pycharmm_command_forwards_no_periodic_charmm_vdw():
    cmd = build_pycharmm_command(
        _pycharmm_args(
            mm_nonbond_mode="periodic_external",
            lr_solver="scafacos",
            periodic_charmm_vdw=False,
        )
    )
    assert "--no-periodic-charmm-vdw" in cmd


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


def test_main_skips_parent_manifest_after_campaign_mpi_rerun(monkeypatch, tmp_path):
    from mmml.cli.run import md_system

    cfg = tmp_path / "campaign.yaml"
    cfg.write_text(
        "\n".join(
            [
                "defaults:",
                "  backend: pycharmm",
                "  setup: pbc_npt",
                "  composition: DCM:20",
                "  box_size: 32.0",
                "runs:",
                "  md_run:",
                "    output_dir: artifacts/md_run",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mmml", "--config", str(cfg), "--run-all"],
    )

    def _fake_run_campaign(args):
        args._mpi_rerun_proxy_return = True
        return 2

    with mock.patch("mmml.cli.run.md_campaign.run_campaign", side_effect=_fake_run_campaign), mock.patch(
        "mmml.cli.run.md_system._maybe_save_job_run_manifest"
    ) as save_manifest:
        assert md_system.main() == 2

    save_manifest.assert_not_called()

"""Unit tests for in-process MD campaign runner helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.cli.run.md_config import (
    expand_repeated_jobs,
    merge_campaign_job_config,
    topological_job_order,
)


def _sample_campaign() -> dict:
    return {
        "defaults": {"composition": "DCM:5", "seed": 1, "dt_fs": 0.25},
        "runs": {
            "equil": {"backend": "pycharmm", "setup": "pbc_npt", "output_dir": "a"},
            "prod": {
                "backend": "jaxmd",
                "setup": "pbc_npt",
                "depends_on": "equil",
                "output_dir": "b",
            },
            "nve": {
                "backend": "jaxmd",
                "setup": "pbc_nve",
                "depends_on": "prod",
                "repeat": 2,
                "output_dir": "c",
            },
        },
    }


def test_resolve_campaign_namespace_paths_vs_config(tmp_path: Path) -> None:
    from argparse import Namespace

    from mmml.cli.run.md_campaign import resolve_campaign_namespace_paths

    cfg_dir = tmp_path / "cfgs" / "yaml"
    cfg_dir.mkdir(parents=True)
    policy = tmp_path / "cfgs" / "policy.yaml"
    policy.write_text("schema_version: 1\n", encoding="utf-8")
    cfg = cfg_dir / "campaign.yaml"
    cfg.write_text("runs: {}\n", encoding="utf-8")
    # Relative to yaml/ → ../policy.yaml
    ns = Namespace(interaction_policy="../policy.yaml", from_pdb=None, checkpoint=None)
    resolve_campaign_namespace_paths(ns, config_path=cfg)
    assert Path(ns.interaction_policy) == policy.resolve()


def test_run_single_backend_honors_jaxmd_unified(monkeypatch):
    """Campaign legs must call run_unified_jaxmd when jaxmd_unified is set."""
    from argparse import Namespace
    from unittest import mock

    from mmml.cli.run import md_campaign

    called: dict[str, object] = {}

    def _fake_unified(args):
        called["args"] = args
        return 0

    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.run_unified_jaxmd",
        _fake_unified,
    )
    monkeypatch.setattr(md_campaign, "clear_handoff_context", lambda: None)
    monkeypatch.setattr(md_campaign, "set_handoff_in", lambda *_a, **_k: None)
    monkeypatch.setattr(md_campaign, "set_handoff_out", lambda *_a, **_k: None)
    monkeypatch.setattr(md_campaign, "get_handoff_out", lambda: None)

    ns = Namespace(jaxmd_unified=True, output_dir="/tmp/x")
    rc, handoff, stages = md_campaign.run_single_backend(ns)
    assert rc == 0
    assert called["args"] is ns
    assert handoff is None
    assert stages == []


def test_topological_job_order() -> None:
    order = topological_job_order(_sample_campaign())
    assert order.index("equil") < order.index("prod")
    assert order.index("prod") < order.index("nve")


def test_topological_order_unknown_dep_raises() -> None:
    bad = {"runs": {"a": {"depends_on": "missing"}}}
    with pytest.raises(ValueError, match="unknown job"):
        topological_job_order(bad)


def test_expand_repeated_jobs() -> None:
    expanded = expand_repeated_jobs(_sample_campaign(), ["equil", "prod", "nve"])
    run_ids = [rid for _base, rid, _rep in expanded]
    assert run_ids == ["equil", "prod", "nve.0", "nve.1"]


def test_campaign_continue_from_h5_honors_frame() -> None:
    """HDF5 NVE restarts must pass continue_from_frame into load_handoff."""
    from pathlib import Path

    from mmml.cli.run import md_campaign as mc

    src = Path(mc.__file__).read_text(encoding="utf-8")
    assert 'frame=int(merged.get("continue_from_frame"' in src


def test_jaxmd_pre_min_defaults_to_fire_first() -> None:
    """Rough hybrid surfaces should not open with ASE BFGS by default."""
    from pathlib import Path

    from mmml.cli.run.md_pbc_suite import jaxmd as jaxmd_suite

    src = Path(jaxmd_suite.__file__).read_text(encoding="utf-8")
    assert 'default="fire-first"' in src
    assert "ASE FIRE-first pre-minimization" in src
    assert "_maybe_bfgs_polish" in src


def test_compatible_h5_per_atom_array_drops_stale_natoms() -> None:
    """Traj export must not apply DCM:120 velocities onto a DCM:180 frame."""
    import numpy as np

    from mmml.cli.run.md_pbc_suite.jaxmd import _compatible_h5_per_atom_array

    stale = np.zeros((3, 600, 3), dtype=float)
    ok = np.zeros((3, 900, 3), dtype=float)
    assert (
        _compatible_h5_per_atom_array(
            stale, n_atoms=900, label="velocities", path_name="old.h5"
        )
        is None
    )
    kept = _compatible_h5_per_atom_array(
        ok, n_atoms=900, label="velocities", path_name="new.h5"
    )
    assert kept is not None
    assert kept.shape == (3, 900, 3)


def test_unique_output_dir_if_exists_keeps_missing_path(tmp_path) -> None:
    from mmml.cli.run.md_campaign import _unique_output_dir_if_exists

    missing = tmp_path / "fresh_campaign"
    assert _unique_output_dir_if_exists(missing, resume=False) == missing.resolve()


def test_unique_output_dir_if_exists_adds_uuid_suffix(tmp_path) -> None:
    from mmml.cli.run.md_campaign import _unique_output_dir_if_exists

    existing = tmp_path / "dcm_large_25"
    existing.mkdir()
    got = _unique_output_dir_if_exists(existing, resume=False)
    assert got.parent == existing.parent
    assert got.name.startswith("dcm_large_25_")
    assert len(got.name) == len("dcm_large_25_") + 8
    assert not got.exists()


def test_unique_output_dir_if_exists_honors_resume(tmp_path) -> None:
    from mmml.cli.run.md_campaign import _unique_output_dir_if_exists

    existing = tmp_path / "campaign"
    existing.mkdir()
    assert _unique_output_dir_if_exists(existing, resume=True) == existing.resolve()


def test_resume_requested_cli_aliases() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_config import (
        campaign_resume_enabled,
        normalize_resume_flags,
        resume_requested,
    )

    assert not resume_requested(Namespace(resume=False, resume_campaign=False))

    ns = Namespace(resume=True, resume_campaign=False)
    normalize_resume_flags(ns)
    assert ns.resume is True
    assert ns.resume_campaign is True
    assert resume_requested(ns)

    ns2 = Namespace(resume=False, resume_campaign=True)
    normalize_resume_flags(ns2)
    assert ns2.resume is True
    assert resume_requested(ns2)

    campaign = {"defaults": {"resume": True}, "runs": {"a": {}}}
    assert campaign_resume_enabled(Namespace(resume=False, resume_campaign=False), campaign)
    assert campaign_resume_enabled(
        Namespace(resume=False, resume_campaign=False),
        {"defaults": {"resume_campaign": True}, "runs": {}},
    )


def test_parse_md_system_resume_syncs_campaign_flag() -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    args = parse_md_system_args(["--resume"])
    assert args.resume is True
    assert args.resume_campaign is True

    args2 = parse_md_system_args(["--resume-campaign"])
    assert args2.resume is True
    assert args2.resume_campaign is True


def test_parse_md_system_tracks_explicit_lr_cli_flags() -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    args = parse_md_system_args(
        [
            "--lr-solver",
            "jax_pme",
            "--mm-nonbond-mode",
            "periodic_external",
            "--no-periodic-charmm-vdw",
        ]
    )
    assert args.lr_solver == "jax_pme"
    assert args.mm_nonbond_mode == "periodic_external"
    assert args.periodic_charmm_vdw is False
    assert "lr_solver" in args._cli_explicit
    assert "mm_nonbond_mode" in args._cli_explicit
    assert "periodic_charmm_vdw" in args._cli_explicit


def test_lookup_resolved_output_dir_prefers_in_run_path(tmp_path) -> None:
    from mmml.cli.run.md_campaign import _lookup_resolved_output_dir

    campaign = _sample_campaign()
    resolved = {"equil": (tmp_path / "equil_run_abc12345").resolve()}
    got = _lookup_resolved_output_dir(resolved, campaign, "equil")
    assert got == resolved["equil"]


def test_resolve_output_dir_repeat_subdirs(tmp_path) -> None:
    from mmml.cli.run.md_campaign import _resolve_output_dir

    merged = {"output_dir": str(tmp_path / "jaxmd_nve"), "repeat": 4}
    assert _resolve_output_dir(merged, "jaxmd_nve.0", rep=0) == (
        tmp_path / "jaxmd_nve" / "rep00"
    ).resolve()
    assert _resolve_output_dir(merged, "jaxmd_nve.3", rep=3) == (
        tmp_path / "jaxmd_nve" / "rep03"
    ).resolve()
    single = {"output_dir": str(tmp_path / "jaxmd_prod"), "repeat": 1}
    assert _resolve_output_dir(single, "jaxmd_prod", rep=0) == (
        tmp_path / "jaxmd_prod"
    ).resolve()


def test_merge_campaign_job_config_defaults() -> None:
    merged = merge_campaign_job_config(_sample_campaign(), "prod")
    assert merged["composition"] == "DCM:5"
    assert merged["depends_on"] == "equil"
    assert merged["backend"] == "jaxmd"


PLACEHOLDER_CKPT = "/path/to/ckpts/tag/<run>/params.json"


def _campaign_yaml_with_placeholder_checkpoint(tmp_path: Path) -> Path:
    cfg = tmp_path / "campaign.yaml"
    cfg.write_text(
        "defaults:\n"
        f"  checkpoint: {PLACEHOLDER_CKPT}\n"
        "  composition: DCM:2\n"
        "runs:\n"
        "  liquid_nvt:\n"
        "    backend: pycharmm\n"
        "    setup: pbc_nvt\n",
        encoding="utf-8",
    )
    return cfg


def test_cli_checkpoint_overrides_unresolvable_config_placeholder(tmp_path: Path) -> None:
    """A CLI --checkpoint must win before the config value is checked for existence."""
    from mmml.cli.run.md_system import parse_md_system_args

    cfg = _campaign_yaml_with_placeholder_checkpoint(tmp_path)
    real = tmp_path / "params.json"
    real.write_text("{}", encoding="utf-8")

    args = parse_md_system_args(
        ["--config", str(cfg), "--checkpoint", str(real), "--job-id", "liquid_nvt"]
    )
    assert Path(args.checkpoint) == real
    assert "checkpoint" in args._cli_explicit


def test_config_placeholder_checkpoint_still_rejected_without_cli(tmp_path: Path) -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    cfg = _campaign_yaml_with_placeholder_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        parse_md_system_args(["--config", str(cfg), "--job-id", "liquid_nvt"])


def test_merge_campaign_job_config_does_not_require_checkpoint(tmp_path: Path) -> None:
    """Merge expands only; the CLI still gets to replace the value afterwards."""
    campaign = {
        "defaults": {"checkpoint": PLACEHOLDER_CKPT},
        "runs": {"liquid_nvt": {"backend": "pycharmm"}},
    }
    merged = merge_campaign_job_config(campaign, "liquid_nvt")
    assert merged["checkpoint"].endswith("params.json")


def test_validate_campaign_checkpoint(tmp_path: Path) -> None:
    from mmml.cli.run.md_config import validate_campaign_checkpoint

    real = tmp_path / "params.json"
    real.write_text("{}", encoding="utf-8")
    validate_campaign_checkpoint(str(real), job_id="liquid_nvt")
    validate_campaign_checkpoint(None)

    with pytest.raises(FileNotFoundError, match="liquid_nvt"):
        validate_campaign_checkpoint(str(tmp_path / "missing.json"), job_id="liquid_nvt")
    with pytest.raises(FileNotFoundError, match="placeholder"):
        validate_campaign_checkpoint(PLACEHOLDER_CKPT)


def test_apply_campaign_cli_overrides_ml_flags() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_campaign import apply_campaign_cli_overrides

    merged = {"backend": "pycharmm", "ml_gpu_count": 1, "ml_batch_size": 64}
    parent = Namespace(
        ml_batch_size=128,
        ml_gpu_count=2,
        ml_max_active_dimers=None,
        charmm_omp_threads=16,
        ml_spatial_mpi=False,
        skip_jit_warmup=True,
        handoff_pre_minimize=False,
        dt_fs=0.25,
        nvt_integrator="langevin",
        checkpoint="/tmp/ck.json",
        jaxmd_minimize_steps=250,
        _cli_explicit=set(),
    )
    apply_campaign_cli_overrides(merged, parent)
    assert merged["ml_batch_size"] == 128
    assert merged["ml_gpu_count"] == 2
    assert merged["charmm_omp_threads"] == 16
    assert merged["skip_jit_warmup"] is True
    assert merged["handoff_pre_minimize"] is False
    assert merged["dt_fs"] == 0.25
    assert merged["nvt_integrator"] == "langevin"
    assert merged["checkpoint"] == "/tmp/ck.json"
    assert merged["jaxmd_minimize_steps"] == 250

    merged2 = {"backend": "pycharmm"}
    parent2 = Namespace(
        ml_batch_size=None,
        ml_gpu_count=None,
        ml_max_active_dimers=900,
        charmm_omp_threads=None,
        ml_spatial_mpi=True,
        skip_jit_warmup=False,
        handoff_pre_minimize=True,
        dt_fs=None,
        nvt_integrator=None,
        checkpoint=None,
        jaxmd_minimize_steps=None,
        _cli_explicit=set(),
    )
    apply_campaign_cli_overrides(merged2, parent2)
    assert merged2["ml_max_active_dimers"] == 900
    assert merged2["ml_spatial_mpi"] is True
    assert "ml_gpu_count" not in merged2
    assert "dt_fs" not in merged2


def test_apply_campaign_cli_overrides_lr_flags_when_explicit() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_campaign import apply_campaign_cli_overrides

    merged = {
        "backend": "pycharmm",
        "lr_solver": "auto",
        "mm_nonbond_mode": "jax_mic",
    }
    parent = Namespace(
        lr_solver="jax_pme",
        jax_pme_method="ewald",
        jax_pme_sr_cutoff=6.0,
        scafacos_method=None,
        mm_nonbond_mode="periodic_external",
        periodic_charmm_vdw=False,
        include_mm=True,
        ml_batch_size=None,
        ml_gpu_count=None,
        ml_max_active_dimers=None,
        ml_spatial_mpi=False,
        skip_jit_warmup=False,
        handoff_pre_minimize=False,
        _cli_explicit={
            "lr_solver",
            "jax_pme_method",
            "mm_nonbond_mode",
            "periodic_charmm_vdw",
        },
    )
    apply_campaign_cli_overrides(merged, parent)
    assert merged["lr_solver"] == "jax_pme"
    assert merged["jax_pme_method"] == "ewald"
    assert merged["mm_nonbond_mode"] == "periodic_external"
    assert merged["periodic_charmm_vdw"] is False


def test_apply_campaign_cli_overrides_lr_flags_skip_implicit_defaults() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_campaign import apply_campaign_cli_overrides

    merged = {
        "backend": "pycharmm",
        "mm_nonbond_mode": "periodic_external",
        "lr_solver": "jax_pme",
    }
    parent = Namespace(
        lr_solver=None,
        jax_pme_method=None,
        jax_pme_sr_cutoff=None,
        scafacos_method=None,
        mm_nonbond_mode="jax_mic",
        periodic_charmm_vdw=True,
        include_mm=True,
        ml_batch_size=None,
        ml_gpu_count=None,
        ml_max_active_dimers=None,
        ml_spatial_mpi=False,
        skip_jit_warmup=False,
        handoff_pre_minimize=False,
        _cli_explicit=set(),
    )
    apply_campaign_cli_overrides(merged, parent)
    assert merged["mm_nonbond_mode"] == "periodic_external"
    assert merged["lr_solver"] == "jax_pme"


def test_namespace_from_merged_preserves_no_periodic_charmm_vdw() -> None:
    from mmml.cli.run import md_system
    from mmml.cli.run.md_campaign import namespace_from_merged

    args = namespace_from_merged(
        {
            "backend": "pycharmm",
            "setup": "pbc_npt",
            "composition": "DCM:30",
            "box_size": 28.0,
            "mm_nonbond_mode": "periodic_external",
            "lr_solver": "jax_pme",
            "jax_pme_method": "ewald",
            "periodic_charmm_vdw": False,
            "checkpoint": "/tmp/ckpt.json",
            "output_dir": "/tmp/out",
        }
    )
    assert args.periodic_charmm_vdw is False
    cmd = md_system.build_pycharmm_command(args)
    assert "--no-periodic-charmm-vdw" in cmd
    assert "--periodic-charmm-vdw" not in cmd


def test_namespace_from_merged_defaults_bonded_mm_mini_off_pycharmm() -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    args = namespace_from_merged(
        {
            "backend": "pycharmm",
            "setup": "pbc_npt",
            "composition": "BENZ:10",
        }
    )
    assert args.bonded_mm_mini is False
    assert args.bonded_mm_mini_after == "mini,heat"


def test_namespace_from_merged_bonded_mm_mini_always() -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    merged = {
        "backend": "pycharmm",
        "setup": "pbc_npt",
        "composition": "BENZ:10",
        "bonded_mm_mini": True,
        "bonded_mm_mini_after": "mini,heat,equi",
        "bonded_mm_mini_always": True,
    }
    args = namespace_from_merged(merged)
    assert args.bonded_mm_mini is True
    assert args.bonded_mm_mini_after == "mini,heat,equi"
    assert args.bonded_mm_mini_always is True


def test_namespace_from_merged_keeps_extra_args_last() -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    merged = merge_campaign_job_config(_sample_campaign(), "prod")
    merged["continue_from"] = "/tmp/handoff/state.npz"
    merged["extra_args"] = ["--steps-per-recording", "800"]
    args = namespace_from_merged(merged)
    assert getattr(args, "continue_from", None) is None
    assert args.extra_args == ["--steps-per-recording", "800"]
    assert "--job-id" not in args.extra_args
    assert "--continue-from" not in args.extra_args


def test_namespace_from_merged_charmm_image_min_distance_flag() -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    args = namespace_from_merged(
        {
            "composition": "DCM:2",
            "box_size": 32.0,
            "charmm_image_mlpot_min_distance": 3.5,
        }
    )
    assert args.charmm_image_mlpot_min_distance == pytest.approx(3.5)


def test_build_command_jaxmd_liquid_packmol_default_includes_placement_argv() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_system import build_command

    args = Namespace(
        backend="jaxmd",
        setup="pbc_nvt",
        composition="BENZ:30",
        spacing=5.0,
        ps=50.0,
        dt_fs=0.25,
        temperature=260.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=30,
        box_size=32.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=123,
        min_intermonomer_atom_distance=0.1,
        packmol=None,
        packmol_placement=None,
        packmol_sphere=None,
        packmol_radius=None,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        nvt_integrator="nhc",
        traj_export_molecular_wrap=False,
        extra_args=["--steps-per-recording", "800", "--jax-md-update-interval", "1"],
    )
    backend, argv = build_command(args)
    assert backend == "jaxmd"
    assert "--packmol-placement" in argv
    assert argv[argv.index("--packmol-placement") + 1] == "cube"
    assert "--steps-per-recording" in argv
    assert "--jax-md-update-interval" in argv


def test_build_command_jaxmd_grid_builder_omits_packmol_argv() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_system import build_command

    args = Namespace(
        backend="jaxmd",
        setup="pbc_nvt",
        composition="BENZ:30",
        builder="liquid",
        spacing=5.0,
        ps=50.0,
        dt_fs=0.25,
        temperature=260.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=30,
        box_size=32.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=123,
        min_intermonomer_atom_distance=0.1,
        packmol=None,
        packmol_placement=None,
        packmol_sphere=None,
        packmol_radius=None,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        nvt_integrator="nhc",
        traj_export_molecular_wrap=False,
        extra_args=[],
    )
    backend, argv = build_command(args)
    assert backend == "jaxmd"
    assert "--packmol-placement" not in argv


def test_build_command_forwards_jaxmd_neighbor_tuning_args() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_system import build_command

    args = Namespace(
        backend="jaxmd",
        setup="pbc_nvt",
        composition="DCM:20",
        spacing=5.0,
        ps=1.0,
        dt_fs=0.25,
        temperature=160.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=1,
        box_size=40.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=123,
        min_intermonomer_atom_distance=0.1,
        packmol_sphere=None,
        packmol_radius=None,
        packmol=None,
        packmol_placement=None,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        nvt_integrator="nhc",
        traj_export_molecular_wrap=False,
        jax_md_update_interval=10,
        jax_md_skin_distance=0.5,
        extra_args=[],
    )
    backend, argv = build_command(args)

    assert backend == "jaxmd"
    assert argv[argv.index("--jax-md-update-interval") + 1] == "10"
    assert argv[argv.index("--jax-md-skin-distance") + 1] == "0.5"


def test_build_command_forwards_skip_jit_warmup() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_system import build_command

    args = Namespace(
        backend="jaxmd",
        setup="pbc_nvt",
        composition="DCM:20",
        spacing=5.0,
        ps=1.0,
        dt_fs=0.25,
        temperature=160.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=1,
        box_size=40.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=123,
        min_intermonomer_atom_distance=0.1,
        packmol_sphere=True,
        packmol_radius=10.0,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        nvt_integrator="nhc",
        traj_export_molecular_wrap=False,
        skip_jit_warmup=True,
        extra_args=[],
    )
    backend, argv = build_command(args)
    assert backend == "jaxmd"
    assert "--skip-jit-warmup" in argv


def test_build_command_filters_campaign_flags_from_extra_args() -> None:
    from argparse import Namespace

    from mmml.cli.run.md_system import build_command

    args = Namespace(
        backend="jaxmd",
        setup="pbc_nvt",
        composition="DCM:20",
        spacing=5.0,
        ps=1.0,
        dt_fs=0.25,
        temperature=160.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=1,
        box_size=40.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=123,
        min_intermonomer_atom_distance=0.1,
        packmol_sphere=True,
        packmol_radius=10.0,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        nvt_integrator="nhc",
        traj_export_molecular_wrap=False,
        extra_args=[
            "--steps-per-recording",
            "800",
            "--job-id",
            "jaxmd_prod",
            "--continue-from",
            "/tmp/state.npz",
        ],
    )
    backend, argv = build_command(args)
    assert backend == "jaxmd"
    assert "--job-id" not in argv
    assert "--continue-from" not in argv
    assert "--steps-per-recording" in argv


def test_resolve_dcd_nsavc_caps_frames_for_short_legs() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_dcd_nsavc

    assert resolve_dcd_nsavc(dcd_nsavc=1, nstep=400, dcd_max_frames=25) == 16
    assert resolve_dcd_nsavc(dcd_nsavc=1, nstep=400, dcd_max_frames=0) == 1
    assert resolve_dcd_nsavc(dcd_nsavc=500, nstep=4000, dcd_max_frames=25) == 500


def test_strip_campaign_metadata_keys_drops_setup_compare_fields() -> None:
    from mmml.cli.run.md_campaign import strip_campaign_metadata_keys

    raw = {
        "setup_variant": "liquid_prep_dense",
        "setup_description": "liquid_prep + density_prep_ladder",
        "prep_sweep_id": "baseline",
        "prep_sweep_overrides": {"spacing": 6.0},
        "campaign_output": "/tmp/out",
        "composition": "DCM:127",
        "box_size": 30.0,
    }
    stripped = strip_campaign_metadata_keys(raw)
    assert "setup_variant" not in stripped
    assert "setup_description" not in stripped
    assert "prep_sweep_id" not in stripped
    assert "prep_sweep_overrides" not in stripped
    assert "campaign_output" not in stripped
    assert stripped["composition"] == "DCM:127"

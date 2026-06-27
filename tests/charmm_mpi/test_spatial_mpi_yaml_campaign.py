"""YAML campaign + env sync for Tier 2 spatial MPI."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest
import yaml

from mmml.cli.run.md_campaign import apply_campaign_cli_overrides, namespace_from_merged
from mmml.cli.run.md_config import load_yaml_config, merge_campaign_job_config
from mmml.cli.run.md_system import build_pycharmm_command, parse_md_system_args
from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
    spatial_mpi_enabled,
    sync_spatial_mpi_env_from_campaign,
    sync_spatial_mpi_env_from_args,
)


def test_spatial_mpi_example_yaml_parses(tmp_path: Path) -> None:
    example = (
        Path(__file__).resolve().parents[2]
        / "mmml/cli/run/md_system.spatial_mpi.example.yaml"
    )
    raw = load_yaml_config(example)
    assert raw["ml_spatial_mpi"] is True
    assert raw["ml_gpu_count"] == 1
    assert "mini" in str(raw["md_stages"])


def test_parse_md_system_config_sets_spatial_mpi_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    cfg = tmp_path / "spatial.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "composition": "DCM:20",
                "ml_spatial_mpi": True,
                "md_stages": "mini",
            }
        )
    )
    args = parse_md_system_args(["--config", str(cfg)])
    assert args.ml_spatial_mpi is True
    assert os.environ.get("MMML_MLPOT_SPATIAL_MPI") == "1"
    assert spatial_mpi_enabled(args.ml_spatial_mpi)


def test_build_pycharmm_command_from_yaml_config(tmp_path: Path) -> None:
    cfg = tmp_path / "spatial.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "backend": "pycharmm",
                "setup": "pbc_npt",
                "composition": "DCM:20",
                "output_dir": "artifacts/x",
                "box_size": 32.0,
                "ml_spatial_mpi": True,
                "ml_gpu_count": 1,
                "ml_batch_size": 64,
                "md_stages": "mini",
            }
        )
    )
    args = parse_md_system_args(["--config", str(cfg)])
    cmd = build_pycharmm_command(args)
    assert "--ml-spatial-mpi" in cmd
    assert "--ml-gpu-count" in cmd
    assert "1" in cmd


def test_campaign_defaults_spatial_mpi_namespace() -> None:
    campaign = {
        "defaults": {
            "composition": "DCM:20",
            "ml_spatial_mpi": True,
            "ml_gpu_count": 1,
            "backend": "pycharmm",
            "setup": "pbc_npt",
            "md_stages": "mini",
        },
        "runs": {
            "spatial_mini": {
                "output_dir": "results/spatial_mini",
            }
        },
    }
    merged = merge_campaign_job_config(campaign, "spatial_mini")
    args = namespace_from_merged(merged)
    assert args.ml_spatial_mpi is True
    assert args.ml_gpu_count == 1


def test_campaign_cli_override_spatial_mpi() -> None:
    merged = {"backend": "pycharmm", "setup": "pbc_npt"}
    parent = Namespace(
        ml_batch_size=None,
        ml_gpu_count=None,
        ml_max_active_dimers=None,
        ml_spatial_mpi=True,
        skip_jit_warmup=False,
        handoff_pre_minimize=False,
    )
    apply_campaign_cli_overrides(merged, parent)
    args = namespace_from_merged(merged)
    assert args.ml_spatial_mpi is True


def test_sync_spatial_mpi_env_from_campaign_defaults(monkeypatch) -> None:
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    campaign = {
        "defaults": {"ml_spatial_mpi": True},
        "runs": {"a": {"backend": "pycharmm"}},
    }
    assert sync_spatial_mpi_env_from_campaign(campaign, None) is True
    assert os.environ["MMML_MLPOT_SPATIAL_MPI"] == "1"


def test_sync_spatial_mpi_env_from_args_false_does_not_set(monkeypatch) -> None:
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    assert sync_spatial_mpi_env_from_args(Namespace(ml_spatial_mpi=False)) is False
    assert "MMML_MLPOT_SPATIAL_MPI" not in os.environ


def test_md_system_spatial_mpi_mini_dry_run_script() -> None:
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/07_md_system_spatial_mpi_mini.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "PASS dry-run" in proc.stdout

"""Unit tests for ``mmml md-system`` YAML config parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.cli.run.md_config import apply_mapping_to_namespace, load_yaml_config
from mmml.cli.run.md_system import parse_md_system_args


def test_namespace_from_yaml_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "md.yaml"
    cfg.write_text(
        """
setup: pbc_nve
backend: jaxmd
composition: DCM:5
checkpoint: /tmp/ckpt
output_dir: results/run
dt-fs: 0.25
ps: 2.0
temperature: 260
seed: 7
""".strip()
    )
    args = parse_md_system_args(["--config", str(cfg)])
    assert args.setup == "pbc_nve"
    assert args.backend == "jaxmd"
    assert args.composition == "DCM:5"
    assert args.dt_fs == pytest.approx(0.25)
    assert args.ps == pytest.approx(2.0)
    assert args.seed == 7
    assert str(args.output_dir).endswith("results/run")


def test_cli_overrides_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "md.yaml"
    cfg.write_text("ps: 5.0\nseed: 1\n")
    args = parse_md_system_args(["--config", str(cfg), "--ps", "10", "--seed", "99"])
    assert args.ps == pytest.approx(10.0)
    assert args.seed == 99


def test_campaign_yaml_ignored_for_flat_parse(tmp_path: Path) -> None:
    cfg = tmp_path / "campaign.yaml"
    cfg.write_text(
        """
defaults:
  composition: DCM:20
runs:
  job_a:
    backend: jaxmd
    setup: pbc_nve
""".strip()
    )
    raw = load_yaml_config(cfg)
    assert "defaults" in raw
    flat = {k: v for k, v in raw.items() if k not in {"defaults", "runs", "jobs"}}
    assert flat == {}


def test_campaign_output_alias(tmp_path: Path) -> None:
    cfg = tmp_path / "campaign.yaml"
    cfg.write_text(
        """
campaign_output: artifacts/my_campaign
defaults:
  composition: DCM:5
runs:
  job_a:
    backend: jaxmd
    setup: pbc_nve
""".strip()
    )
    args = parse_md_system_args(["--config", str(cfg), "--run-all"])
    assert str(args.campaign_output_dir).endswith("artifacts/my_campaign")


def test_apply_mapping_hyphen_keys() -> None:
    from mmml.cli.run.md_system import parse_args

    args = parse_args([])
    apply_mapping_to_namespace(
        args,
        {"continue-from": "/tmp/state.npz", "handoff-write-res": False},
        source="test",
    )
    assert str(args.continue_from) == "/tmp/state.npz"
    assert args.handoff_write_res is False

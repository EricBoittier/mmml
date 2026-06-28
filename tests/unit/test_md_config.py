"""Tests for md-system YAML config loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def test_load_yaml_config_merges_include_defaults(tmp_path: Path) -> None:
    from mmml.cli.run.md_config import load_yaml_config

    preset = tmp_path / "preset.yaml"
    preset.write_text(
        textwrap.dedent(
            """
            defaults:
              dt_fs: 0.25
              heat_thermostat: hoover
              no_echeck_heat: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    main = tmp_path / "campaign.yaml"
    main.write_text(
        textwrap.dedent(
            f"""
            include:
              - {preset.name}
            defaults:
              composition: "DCM:52"
              heat_thermostat: scale
            runs:
              equil:
                backend: pycharmm
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_yaml_config(main)
    assert cfg["defaults"]["dt_fs"] == pytest.approx(0.25)
    assert cfg["defaults"]["no_echeck_heat"] is True
    assert cfg["defaults"]["composition"] == "DCM:52"
    assert cfg["defaults"]["heat_thermostat"] == "scale"
    assert "equil" in cfg["runs"]


def test_load_yaml_config_chained_includes(tmp_path: Path) -> None:
    from mmml.cli.run.md_config import load_yaml_config

    base = tmp_path / "base.yaml"
    base.write_text("defaults:\n  dt_fs: 0.5\n", encoding="utf-8")
    heat = tmp_path / "heat.yaml"
    heat.write_text(
        "include:\n  - base.yaml\ndefaults:\n  no_echeck_heat: true\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.yaml"
    main.write_text(
        f"include:\n  - {heat.name}\ndefaults:\n  composition: DCM:103\n",
        encoding="utf-8",
    )

    cfg = load_yaml_config(main)
    assert cfg["defaults"]["dt_fs"] == pytest.approx(0.5)
    assert cfg["defaults"]["no_echeck_heat"] is True
    assert cfg["defaults"]["composition"] == "DCM:103"


def test_repo_preset_heat_conservative_loads() -> None:
    from mmml.cli.run.md_config import load_yaml_config

    repo = Path(__file__).resolve().parents[2]
    preset = repo / "mmml/cli/run/presets/heat-dt0.25-conservative.yaml"
    cfg = load_yaml_config(preset)
    assert cfg["defaults"]["heat_thermostat"] == "hoover"
    assert cfg["defaults"]["no_echeck_heat"] is True


def test_dcm_liquid_workflow_resilient_merges_presets() -> None:
    from mmml.cli.run.md_config import load_yaml_config

    repo = Path(__file__).resolve().parents[2]
    cfg = load_yaml_config(repo / "mmml/cli/run/dcm_liquid_workflow.resilient.yaml")
    d = cfg["defaults"]
    assert d["md_stages"] == "mini,heat,equi"
    assert d["calculator_pre_minimize"] is True
    assert d["bonded_mm_mini"] is True
    assert d["mini_box_equil_ps"] == pytest.approx(5.0)
    assert d["mini_lattice_abnr_steps"] == 300
    assert d["heat_thermostat"] == "hoover"
    assert d["dynamics_max_monomer_extent"] == pytest.approx(12.0)
    assert d["liquid_prep"] is True
    assert d["density_prep_ladder"] is True


def test_parse_md_system_args_resilient_workflow_config() -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    repo = Path(__file__).resolve().parents[2]
    cfg = repo / "mmml/cli/run/dcm_liquid_workflow.resilient.yaml"
    args = parse_md_system_args(
        [
            "--config",
            str(cfg),
            "--output-dir",
            "/tmp/dcm60_liquid",
            "--checkpoint",
            "/tmp/ckpt.json",
            "--composition",
            "DCM:60",
        ]
    )
    assert args.calculator_pre_minimize is True
    assert args.fire_min_steps == 200
    assert args.fire_min_maxstep == pytest.approx(0.2)
    assert args.rescue_fire_fmax == pytest.approx(0.05)
    assert args.quiet_bfgs is False
    assert args.md_stages == "mini,heat,equi"


def test_dcm103_example_campaign_merges_presets() -> None:
    from mmml.cli.run.md_config import load_yaml_config

    repo = Path(__file__).resolve().parents[2]
    cfg = load_yaml_config(repo / "mmml/cli/run/md_system.dcm103_equil.example.yaml")
    d = cfg["defaults"]
    assert d["composition"] == "DCM:103"
    assert d["calculator_pre_minimize"] is True
    assert d["mini_box_equil_ps"] == pytest.approx(2.0)
    assert d["heat_thermostat"] == "hoover"
    assert d["dynamics_max_monomer_extent"] == pytest.approx(12.0)
    assert d["heat_overlap_segment_boundary_only"] is True
    assert "dcm103_equil" in cfg["runs"]


def test_config_is_campaign() -> None:
    from mmml.cli.run.md_config import config_is_campaign

    assert config_is_campaign({"defaults": {"dt_fs": 0.25}}) is False
    assert config_is_campaign({"runs": {"a": {"backend": "pycharmm"}}}) is True
    assert config_is_campaign({"jobs": {"a": {}}}) is True
    assert config_is_campaign({"runs": {}}) is False


def test_parse_md_system_args_applies_defaults_block(tmp_path: Path) -> None:
    from mmml.cli.run.md_system import parse_md_system_args

    cfg = tmp_path / "heat.conf"
    cfg.write_text(
        textwrap.dedent(
            """
            defaults:
              dt_fs: 0.25
              no_echeck_heat: true
              composition: "DCM:52"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = parse_md_system_args(
        [
            "--config",
            str(cfg),
            "--output-dir",
            str(tmp_path / "out"),
            "--backend",
            "pycharmm",
        ]
    )
    assert args.dt_fs == pytest.approx(0.25)
    assert args.no_echeck_heat is True
    assert args.composition == "DCM:52"


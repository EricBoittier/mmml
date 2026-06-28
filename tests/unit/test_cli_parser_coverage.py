"""Tests for MMML CLI registry, parser coverage, and configure presets."""

from __future__ import annotations

from pathlib import Path

from mmml.cli import __main__ as cli_main
from mmml.cli.configure_presets import PRESET_BY_KEY, apply_preset
from mmml.cli.parser_utils import get_subcommand_parser, parser_available, parsers_with_flags
from mmml.cli.registry import COMMAND_REGISTRY, _DISPATCH_COMMANDS, format_audit_report


def test_dispatch_matches_registry():
    assert set(cli_main._DISPATCH_COMMANDS) == set(_DISPATCH_COMMANDS)


def test_all_dispatch_commands_in_registry():
    registry_names = {spec.name for spec in COMMAND_REGISTRY}
    assert set(_DISPATCH_COMMANDS).issubset(registry_names)


def test_parser_coverage_major_commands():
    must_have_flags = {
        "md-system",
        "physnet-train",
        "fix-and-split",
        "pyscf-evaluate",
        "configure",
        "liquid-box",
        "warmup-mlpot-jax",
        "ef-train",
        "ef-evaluate",
        "validate",
    }
    missing = sorted(c for c in must_have_flags if not parser_available(c))
    assert not missing, f"missing build_parser: {missing}"


def test_md_system_parser_has_setup():
    parser = get_subcommand_parser("md-system")
    assert parser is not None
    assert any(a.dest == "setup" for a in parser._actions)
    box_actions = [
        a
        for a in parser._actions
        if any(opt == "--box-size" for opt in getattr(a, "option_strings", ()))
    ]
    assert len(box_actions) == 1


def test_audit_report_mentions_deprecated():
    text = format_audit_report()
    assert "deprecated" in text.lower() or "legacy" in text.lower()
    assert "physnet-train" in text


def test_configure_preset_spatial_mpi(tmp_path: Path):
    preset = PRESET_BY_KEY["cpu-spatial-mpi-mini"]
    paths = apply_preset(preset, tmp_path)
    assert len(paths) == 1
    assert "spatial_mpi" in paths[0].name
    assert "ml_spatial_mpi" in paths[0].read_text()


def test_configure_preset_md_benchmark_tree(tmp_path: Path):
    preset = PRESET_BY_KEY["cpu-md-benchmark"]
    paths = apply_preset(preset, tmp_path)
    names = {p.name for p in paths}
    assert "Snakefile" in names
    assert "config.yaml" in names
    slurm = tmp_path / "md_benchmark_workflow" / "profiles" / "slurm" / "config.yaml"
    assert slurm.is_file()


def test_parsers_with_flags_count():
    assert len(parsers_with_flags()) >= 30

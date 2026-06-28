"""Tests for slim CLI help, configure wizard, and completion."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import yaml

from mmml.cli import __main__ as cli_main
from mmml.cli.commands_help import commands_main, examples_main
from mmml.cli.completion import MMML_COMMANDS, get_subcommand_parser
from mmml.cli.configure import configure_main, run_wizard
from mmml.cli.help_text import format_top_level_help


def test_mmml_commands_match_dispatch():
    dispatch = set(cli_main._DISPATCH_COMMANDS)
    listed = {c for c in MMML_COMMANDS if c != "completion"}
    assert dispatch == listed


def test_top_level_help_is_compact():
    text = format_top_level_help()
    assert "MMML: Machine Learning" in text
    assert "mmml commands" in text
    assert "mmml configure" in text
    assert "make-res    Generate residue" not in text
    assert len(text.splitlines()) < 25


def test_commands_main_lists_md_system(capsys):
    assert commands_main([]) == 0
    out = capsys.readouterr().out
    assert "md-system" in out
    assert "configure" in out


def test_examples_main(capsys):
    assert examples_main([]) == 0
    out = capsys.readouterr().out
    assert "mmml configure" in out


def test_configure_non_interactive(capsys):
    assert configure_main(["--non-interactive"]) == 0
    out = capsys.readouterr().out
    assert "preset-menu" in out or "md-single" in out
    assert "list-presets" in out or "Presets" in out


def test_configure_list_presets(capsys):
    assert configure_main(["--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "cpu-spatial-mpi-mini" in out
    assert "cpu-md-benchmark" in out


def test_configure_md_single_writes_yaml(tmp_path: Path):
    answers = iter(
        [
            "1",  # goal: liquid
            "5",  # setup: pbc_npt
            "1",  # backend: pycharmm
            "DCM:10",
            "${MMML_CKPT}",
            "260",
            "artifacts/test_run",
            "30",
        ]
    )
    with patch("builtins.input", lambda _p="": next(answers)):
        paths = run_wizard("md-single", tmp_path)
    assert len(paths) == 1
    cfg = yaml.safe_load(paths[0].read_text())
    assert cfg["setup"] == "pbc_npt"
    assert cfg["composition"] == "DCM:10"
    assert cfg["box_size"] == 30.0


def test_configure_physnet_train(tmp_path: Path):
    answers = iter(
        [
            "splits/train.npz",
            "splits/valid.npz",
            "./ckpts/x",
            "tag1",
            "2",  # medium scale
        ]
    )
    with patch("builtins.input", lambda _p="": next(answers)):
        paths = run_wizard("physnet-train", tmp_path)
    cfg = yaml.safe_load(paths[0].read_text())
    assert cfg["num_epochs"] == 500
    assert cfg["tag"] == "tag1"


def test_configure_snakemake_scaffold(tmp_path: Path):
    answers = iter(
        [
            "wf_test",
            "DCM:5",
            "y",  # pycharmm
            "n",  # jaxmd
        ]
    )
    with patch("builtins.input", lambda _p="": next(answers)):
        paths = run_wizard("snakemake-md", tmp_path)
    wf = tmp_path / "wf_test"
    assert (wf / "Snakefile").is_file()
    assert (wf / "config.yaml").is_file()
    assert (wf / "scripts" / "job_shell.sh").is_file()


def test_md_system_subcommand_parser_builds():
    parser = get_subcommand_parser("md-system")
    assert parser is not None
    assert any(a.dest == "setup" for a in parser._actions)


def test_configure_parser_for_completion():
    parser = get_subcommand_parser("configure")
    assert parser is not None
    assert any(a.dest == "workflow" for a in parser._actions)


def test_main_help_no_args(capsys):
    with patch.object(cli_main.sys, "argv", ["mmml"]):
        rc = cli_main.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "mmml commands" in out


def test_main_commands_dispatch(capsys):
    with patch.object(cli_main.sys, "argv", ["mmml", "commands"]):
        rc = cli_main.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "md-system" in out
    assert "Structure" in out or "MD & campaigns" in out


def test_main_configure_dispatch(capsys):
    with patch.object(cli_main.sys, "argv", ["mmml", "configure", "--non-interactive"]):
        rc = cli_main.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "preset-menu" in out or "md-single" in out


def test_commands_audit_is_fast(capsys):
    import time

    t0 = time.perf_counter()
    assert commands_main(["--audit"]) == 0
    elapsed = time.perf_counter() - t0
    out = capsys.readouterr().out
    assert "Deprecated / legacy" in out
    assert elapsed < 2.0, f"audit took {elapsed:.1f}s (should not import JAX/PySCF)"

"""CLI / config merge tests for umbrella commands (no MD)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmml.cli.misc.umbrella_mbar import build_parser as build_mbar_parser
from mmml.cli.misc.umbrella_sample import (
    _config_from_args,
    build_parser as build_sample_parser,
)


def test_sample_parser_targets():
    parser = build_sample_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--structure",
            "mol.npz",
            "-o",
            "out",
            "--atoms",
            "0,2",
            "--targets",
            "1.0,1.5,2.0",
            "--k",
            "12.5",
            "--nsteps",
            "50",
            "--seed-mode",
            "frames",
            "--structure-index",
            "2",
            "--overwrite",
        ]
    )
    cfg = _config_from_args(args)
    assert cfg.atom_i == 0
    assert cfg.atom_j == 2
    assert cfg.resolve_targets() == (1.0, 1.5, 2.0)
    assert cfg.resolve_force_constants() == (12.5, 12.5, 12.5)
    assert cfg.nsteps == 50
    assert cfg.overwrite is True
    assert cfg.seed_mode == "frames"
    assert cfg.structure_index == 2


def test_sample_parser_grid_from_config_file(tmp_path: Path):
    cfg_path = tmp_path / "umb.json"
    cfg_path.write_text(
        json.dumps(
            {
                "checkpoint": "ckpt",
                "structure": "mol.xyz",
                "output_dir": "out",
                "atom_i": 1,
                "atom_j": 3,
                "xi_min": 1.0,
                "xi_max": 2.0,
                "n_windows": 3,
                "k_ev_A2": 8,
            }
        ),
        encoding="utf-8",
    )
    parser = build_sample_parser()
    args = parser.parse_args(["--config", str(cfg_path), "--temperature", "310"])
    cfg = _config_from_args(args)
    assert cfg.atom_i == 1
    assert cfg.atom_j == 3
    assert cfg.temperature_K == 310.0
    assert len(cfg.resolve_targets()) == 3
    assert cfg.resolve_targets()[0] == pytest.approx(1.0)
    assert cfg.resolve_targets()[-1] == pytest.approx(2.0)


def test_sample_parser_missing_required():
    parser = build_sample_parser()
    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        _config_from_args(args)


def test_mbar_parser():
    parser = build_mbar_parser()
    args = parser.parse_args(["--run-dir", "out/umb", "--mbar-verbose"])
    assert args.run_dir == Path("out/umb")
    assert args.mbar_verbose is True


def test_sample_parser_2d():
    parser = build_sample_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--structure",
            "mol.xyz",
            "-o",
            "out",
            "--atoms",
            "0,2",
            "--atoms2",
            "1,2",
            "--xi-min",
            "1.5",
            "--xi-max",
            "2.5",
            "--n-windows",
            "3",
            "--yi-min",
            "1.8",
            "--yi-max",
            "2.8",
            "--n-windows-y",
            "2",
            "--ky",
            "15",
            "--overwrite",
        ]
    )
    cfg = _config_from_args(args)
    assert cfg.is_2d
    sched = cfg.resolve_schedule()
    assert sched.n_windows == 6
    assert sched.grid_shape == (3, 2)
    assert cfg.atom_k == 1 and cfg.atom_l == 2


def test_registry_lists_umbrella_commands():
    from mmml.cli.registry import command_by_name

    assert command_by_name("umbrella-sample") is not None
    assert command_by_name("umbrella-mbar") is not None
    assert command_by_name("umbrella-sample").module == "mmml.cli.misc.umbrella_sample"
    assert command_by_name("umbrella-mbar").module == "mmml.cli.misc.umbrella_mbar"


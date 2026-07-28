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


def test_sample_parser_move_with_and_timestep_default():
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
            "2,1",
            "--move-with",
            "1,3,4,5",
            "--targets",
            "2.0",
            "--overwrite",
        ]
    )
    cfg = _config_from_args(args)
    assert cfg.atom_i == 2 and cfg.atom_j == 1
    assert cfg.move_with == (1, 3, 4, 5)
    assert cfg.timestep_fs == pytest.approx(0.1)


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


def test_format_pmf_report_2d():
    from mmml.cli.misc.umbrella_mbar import _format_pmf_report

    result = {
        "ndim": 2,
        "xi0": [1.8, 1.8, 2.0, 2.0],
        "yi0": [1.8, 2.0, 1.8, 2.0],
        "pmf_rel_kcal_mol": [1.0, 0.0, 2.0, 3.0],
        "d_pmf_rel_kcal_mol": [0.1, 0.1, 0.1, 0.1],
        "grid_shape": [2, 2],
        "pmf_rel_kcal_mol_2d": [[1.0, 0.0], [2.0, 3.0]],
    }
    lines = _format_pmf_report(result)
    assert any("η₀=" in line for line in lines)
    assert any("PMF grid" in line for line in lines)


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


def test_sample_parser_hybrid_engine():
    parser = build_sample_parser()
    args = parser.parse_args(
        [
            "--engine",
            "hybrid_jaxmd",
            "--checkpoint",
            "ckpt",
            "--from-psf",
            "box/model.psf",
            "--from-pdb",
            "box/model.pdb",
            "--box-size",
            "30",
            "-o",
            "out",
            "--atom-name-i",
            "C1",
            "--atom-name-j",
            "N1",
            "--ml-resnames",
            "AMM1,CH3CL",
            "--xi-min",
            "2.0",
            "--xi-max",
            "3.0",
            "--n-windows",
            "3",
            "--overwrite",
        ]
    )
    cfg = _config_from_args(args)
    assert cfg.engine == "hybrid_jaxmd"
    assert cfg.from_psf == Path("box/model.psf")
    assert cfg.from_pdb == Path("box/model.pdb")
    assert cfg.box_size == pytest.approx(30.0)
    assert cfg.atom_name_i == "C1"
    assert cfg.atom_name_j == "N1"
    assert cfg.ml_resnames == ("AMM1", "CH3CL")
    assert cfg.structure is None


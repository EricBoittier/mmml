"""Validate DCM NVE scaling Snakemake workflow config and argv."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

WORKFLOW_ROOT = (
    Path(__file__).resolve().parents[2] / "workflows" / "dcm_nve_scaling"
)
SCRIPTS = WORKFLOW_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scaling_lib import (  # noqa: E402
    build_md_system_argv,
    composition_string,
    load_config,
    packmol_radius_A,
    paths_for_size,
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(WORKFLOW_ROOT / "config.yaml")


def test_cluster_sizes(cfg: dict) -> None:
    assert cfg["cluster_sizes"] == [5, 6, 7, 8, 9, 10]


def test_per_step_output_required(cfg: dict) -> None:
    assert cfg["dcd_nsavc"] == 1
    assert cfg["dyn_nprint"] == 1
    assert cfg["nprint"] == 1


def test_packmol_radius_scaling() -> None:
    assert packmol_radius_A(5) == pytest.approx(7.7, abs=0.2)
    assert packmol_radius_A(90) == pytest.approx(21.0, abs=0.2)


def test_build_md_system_argv_per_step_flags(cfg: dict, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    os.environ["MMML_CKPT"] = str(ckpt)
    argv = build_md_system_argv(cfg, 7)
    assert "--composition" in argv
    assert composition_string(7) in argv
    idx = argv.index("--dcd-nsavc")
    assert argv[idx + 1] == "1"
    idx = argv.index("--dyn-nprint")
    assert argv[idx + 1] == "1"
    idx = argv.index("--nprint")
    assert argv[idx + 1] == "1"
    assert "--save-forces-npz" in argv
    idx = argv.index("--forces-npz-interval")
    assert argv[idx + 1] == "1"
    assert "--setup" in argv and "free_nve" in argv
    assert "--md-stages" in argv
    idx = argv.index("--md-stages")
    assert argv[idx + 1] == "mini,nve"


def test_paths_for_size(cfg: dict) -> None:
    p = paths_for_size(cfg, 9)
    assert p["nve_dcd"].name == "nve_dcm_9.dcd"
    assert p["mini_crd"].name == "mini_full_mlpot_dcm_9.crd"

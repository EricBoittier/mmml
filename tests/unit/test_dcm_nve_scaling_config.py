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
    expected_nve_nstep,
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


def test_nve_boltzmann_temp_below_full_temperature(cfg: dict) -> None:
    assert float(cfg["nve_boltzmann_temp"]) < float(cfg["temperature"])


def test_expected_nve_nstep_matches_config(cfg: dict) -> None:
    dt_ps = float(cfg["dt_fs"]) * 1e-3
    assert expected_nve_nstep(cfg) == int(round(float(cfg["ps_nve"]) / dt_ps))


def test_nve_overlap_single_chunk(cfg: dict) -> None:
    """One overlap chunk for the full NVE leg (no scratch restart between chunks)."""
    dt_ps = float(cfg["dt_fs"]) * 1e-3
    nstep = int(round(float(cfg["ps_nve"]) / dt_ps))
    assert int(cfg["dynamics_overlap_check_interval"]) >= nstep


def test_conservative_minimize_and_echeck(cfg: dict) -> None:
    assert int(cfg["mini_nstep"]) >= 1000
    assert bool(cfg.get("no_echeck")) is True


def test_packmol_radius_scaling() -> None:
    assert packmol_radius_A(5) == pytest.approx(7.9, abs=0.2)
    assert packmol_radius_A(90) == pytest.approx(20.6, abs=0.5)


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
    assert "--no-echeck" in argv
    assert "--echeck" not in argv


def test_paths_for_size(cfg: dict) -> None:
    p = paths_for_size(cfg, 9)
    assert p["nve_dcd"].name == "nve_dcm_9.dcd"
    assert p["mini_crd"].name == "mini_full_mlpot_dcm_9.crd"

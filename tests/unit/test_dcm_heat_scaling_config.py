"""Unit tests for workflows/dcm_heat_scaling/scripts/heat_lib.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / "workflows" / "dcm_heat_scaling"
sys.path.insert(0, str(_WORKFLOW / "scripts"))

from heat_lib import (  # noqa: E402
    build_md_system_argv,
    dt_fs_from_slug,
    dt_fs_slug,
    pick_slurm_gpu_node,
    run_output_dir,
    run_seed,
)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MMML_CKPT", str(tmp_path))
    (tmp_path / "params.json").write_text("{}")
    return {
        "checkpoint": "${MMML_CKPT}",
        "output_root": "artifacts/pycharmm_mlpot",
        "seed_base": 123456,
        "setup": "pycharmm_full",
        "backend": "pycharmm",
        "md_stages": "mini,heat",
        "box_size": 180.0,
        "ps_heat": 1000.0,
        "n_heat_segments": 4000,
        "heat_thermostat": "hoover",
        "flat_bottom_radius": 55.0,
        "packmol_radius": 15.0,
        "temperature": 220.0,
        "dcd_nsavc": 500,
        "dynamics_overlap_action": "rescue",
        "dynamics_intra_min_distance": 0.5,
        "ml_gpu_count": 1,
        "ml_batch_size": 2056,
        "no_echeck": True,
        "slurm_gpu_nodes": ["gpu08", "gpu09"],
    }


def test_dt_slugs_roundtrip():
    assert dt_fs_slug(0.25) == "dt025"
    assert dt_fs_slug(0.125) == "dt0125"
    assert dt_fs_from_slug("dt025") == 0.25
    assert dt_fs_from_slug("dt0125") == 0.125


def test_unique_seeds_per_dt_and_repeat(cfg):
    s1 = run_seed(25, 1, 0.25, seed_base=cfg["seed_base"])
    s2 = run_seed(25, 1, 0.125, seed_base=cfg["seed_base"])
    s3 = run_seed(25, 2, 0.25, seed_base=cfg["seed_base"])
    assert len({s1, s2, s3}) == 3


def test_build_md_system_argv_heat_flags(cfg):
    argv = build_md_system_argv(cfg, 30, 1, 0.25)
    assert "--setup" in argv and "pycharmm_full" in argv
    assert "--md-stages" in argv and "mini,heat" in argv
    assert "--dt-fs" in argv
    assert argv[argv.index("--dt-fs") + 1] == "0.25"
    assert "--dcd-nsavc" in argv
    assert argv[argv.index("--dcd-nsavc") + 1] == "500"
    assert "--no-echeck" in argv
    assert str(run_seed(30, 1, 0.25, seed_base=cfg["seed_base"])) in argv


def test_output_dir_layout(cfg):
    out = run_output_dir(cfg, 45, 2, 0.125)
    assert out.name == "dt0125"
    assert out.parent.name == "dcm45_npt_x64_2"


def test_pick_slurm_gpu_node_stable(cfg):
    n1 = pick_slurm_gpu_node(cfg, 10, 1, "dt025")
    n2 = pick_slurm_gpu_node(cfg, 10, 1, "dt025")
    assert n1 in ("gpu08", "gpu09")
    assert n1 == n2

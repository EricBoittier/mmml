"""Unit tests for workflows/pbc_liquid_density_dyn."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "pbc_liquid_density_dyn"
SCRIPTS = WORKFLOW / "scripts"
_MODULE_NAME = "pbc_liquid_density_dyn_campaign_lib"


def _load_campaign_lib() -> ModuleType:
    """Load this workflow's campaign_lib without clashing with pbc_solvent_burst."""
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    path = SCRIPTS / "campaign_lib.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load_campaign_lib()
RunCell = cl.RunCell
build_campaign = cl.build_campaign
build_md_system_campaign_argv = cl.build_md_system_campaign_argv
campaign_job_order = cl.campaign_job_order
cell_bulk_density_fraction = cl.cell_bulk_density_fraction
cell_from_cli = cl.cell_from_cli
cell_run_tag = cl.cell_run_tag
cluster_name = cl.cluster_name
iter_matrix_cells = cl.iter_matrix_cells
load_config = cl.load_config
matrix_job_count = cl.matrix_job_count
warmup_mlpot_argv = cl.warmup_mlpot_argv
mlpot_device_name = cl.mlpot_device_name
scheduler_mode = cl.scheduler_mode
slurm_launch_jobs = cl.slurm_launch_jobs
slurm_resources_cli = cl.slurm_resources_cli
workflow_root = cl.workflow_root


@pytest.fixture
def cfg() -> dict:
    raw = yaml.safe_load((WORKFLOW / "config.yaml").read_text(encoding="utf-8"))
    return {
        **raw,
        "checkpoint": "/tmp/fake_ckpt.json",
        "bulk_density_fractions": [0.9, 1.0],
        "temperatures": [300.0],
        "box_sizes": [32.0],
        "solvents": ["DCM"],
        "pycharmm_equi_legs": 3,
        "pycharmm_prod_legs": 2,
    }


@pytest.fixture
def cell(cfg: dict) -> RunCell:
    return cell_from_cli(cfg, "DCM", 277, temperature=300.0, box_size=32.0)


def test_campaign_job_order_init_equi_prod(cfg: dict) -> None:
    order = campaign_job_order(cfg)
    assert order[0] == "pycharmm_init"
    assert order[-1] == "pycharmm_prod_02"
    assert len(order) == 1 + 3 + 2


def test_build_campaign_liquid_prep_and_ladder(cfg: dict, cell: RunCell) -> None:
    camp = build_campaign(cfg, cell)
    defaults = camp["defaults"]
    assert defaults["liquid_prep"] is True
    assert defaults["density_prep_ladder"] is True
    assert defaults["target_density_g_cm3"] == pytest.approx(1.326)
    assert "bulk_density_fraction" in defaults
    init = camp["runs"]["pycharmm_init"]
    assert init["md_stages"] == "mini,heat"
    assert init["cleanup"] is True
    equi = camp["runs"]["pycharmm_equi_01"]
    assert equi["depends_on"] == "pycharmm_init"
    prod = camp["runs"]["pycharmm_prod_02"]
    assert prod["md_stage"] == "prod"
    assert prod["depends_on"] == "pycharmm_prod_01"


def test_build_md_system_argv_includes_resume(cfg: dict, cell: RunCell, tmp_path: Path) -> None:
    cfg = {**cfg, "output_root": str(tmp_path / "out")}
    argv = build_md_system_campaign_argv(cfg, cell)
    assert "--run-all" in argv
    assert "--resume" in argv


def test_matrix_counts_bulk_fractions(cfg: dict) -> None:
    assert matrix_job_count(cfg) == 2


def test_warmup_argv_matches_cell(cfg: dict, cell: RunCell) -> None:
    argv = warmup_mlpot_argv(cfg, cell)
    assert argv[0] == "warmup-mlpot-jax"
    assert "--n-monomers" in argv
    if cfg.get("warmup_do_mm", True):
        assert "--do-mm" in argv
    else:
        assert "--do-mm" not in argv


def test_cell_bulk_density_fraction(cfg: dict, cell: RunCell) -> None:
    frac = cell_bulk_density_fraction(cell, cfg)
    assert frac is not None
    assert 0.85 <= frac <= 0.95


def test_run_tag_includes_tl_when_sweep(cfg: dict) -> None:
    cfg2 = {**cfg, "temperatures": [280.0, 300.0]}
    tags = {cell_run_tag(c, cfg2) for c in iter_matrix_cells(cfg2)}
    assert any("_t280_" in t for t in tags)


def test_cpu_scheduler_resources() -> None:
    cfg = {
        "scheduler": "cpu",
        "slurm_max_concurrent": 12,
        "bulk_density_fractions": [0.9, 1.0],
        "solvents": ["DCM", "ACO"],
        "temperatures": [300.0],
        "box_sizes": [28.0, 32.0],
    }
    cap = matrix_job_count(cfg)
    assert cap >= 4
    expected = min(12, cap)
    assert slurm_launch_jobs(cfg) == expected
    assert slurm_resources_cli(cfg) == f"charmm_slot={expected}"


def test_mlpot_device_cpu_when_scheduler_cpu(cfg: dict) -> None:
    cpu_cfg = {**cfg, "scheduler": "cpu"}
    assert scheduler_mode(cpu_cfg) == "cpu"
    assert mlpot_device_name(cpu_cfg) == "cpu"


def test_cluster_name_pc_bach() -> None:
    cfg = load_config(workflow_root() / "config.pc-bach.cpu.yaml")
    assert cluster_name(cfg) == "pc-bach"
    assert scheduler_mode(cfg) == "cpu"

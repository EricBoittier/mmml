"""Unit tests for workflows/dcm_density_setup_compare campaign generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "dcm_density_setup_compare"
SCRIPTS = WORKFLOW / "scripts"
_CAMPAIGN_MOD = "dcm_density_setup_compare_campaign_lib"
_SETUP_MOD = "dcm_density_setup_compare_setup_variants"


def _load_script_module(name: str, path: Path) -> ModuleType:
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old_path = sys.path[:]
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old_path
    return mod


cl = _load_script_module(_CAMPAIGN_MOD, SCRIPTS / "campaign_lib.py")
sv = _load_script_module(_SETUP_MOD, SCRIPTS / "setup_variants.py")

RunCell = cl.RunCell
build_campaign = cl.build_campaign
build_md_system_campaign_argv = cl.build_md_system_campaign_argv
campaign_job_order = cl.campaign_job_order
cell_bulk_density_fraction = cl.cell_bulk_density_fraction
cell_from_cli = cl.cell_from_cli
cell_from_tag = cl.cell_from_tag
cell_run_tag = cl.cell_run_tag
composition_string = cl.composition_string
iter_matrix_cells = cl.iter_matrix_cells
load_config = cl.load_config
matrix_job_count = cl.matrix_job_count
matrix_setup_ids = cl.matrix_setup_ids
slurm_launch_jobs = cl.slurm_launch_jobs
slurm_resources_cli = cl.slurm_resources_cli
resolve_setup_variant = sv.resolve_setup_variant


@pytest.fixture
def cfg() -> dict:
    raw = yaml.safe_load((WORKFLOW / "config.yaml").read_text(encoding="utf-8"))
    return {
        **raw,
        "checkpoint": "/tmp/fake_ckpt.json",
        "setups": ["minimal", "burst_hybrid"],
        "bulk_density_fractions": [0.25, 0.5],
        "temperatures": [300.0],
        "box_sizes": [28.0, 32.0],
    }


@pytest.fixture
def cell(cfg: dict) -> RunCell:
    return cell_from_cli(cfg, "minimal", "DCM", 52, temperature=300.0, box_size=28.0)


def test_campaign_job_order_is_mini_only(cfg: dict) -> None:
    assert campaign_job_order(cfg) == ["pycharmm_mini"]


def test_run_tag_includes_setup(cfg: dict, cell: RunCell) -> None:
    assert cell_run_tag(cell, cfg) == "minimal_dcm_52_t300_l28"


def test_matrix_expands_setups_and_density(cfg: dict) -> None:
    tags = {cell_run_tag(c, cfg) for c in iter_matrix_cells(cfg)}
    assert "minimal_dcm_52_t300_l28" in tags
    assert "burst_hybrid_dcm_154_t300_l32" in tags
    # 2 setups × 2 fractions × 2 boxes = 8 cells
    assert matrix_job_count(cfg) == 8


def test_cell_from_tag_round_trip(cfg: dict, cell: RunCell) -> None:
    tag = cell_run_tag(cell, cfg)
    resolved = cell_from_tag(cfg, tag)
    assert resolved == cell


def test_build_campaign_mini_only(cfg: dict, cell: RunCell) -> None:
    campaign = build_campaign(cfg, cell)
    assert list(campaign["runs"]) == ["pycharmm_mini"]
    mini = campaign["runs"]["pycharmm_mini"]
    assert mini["md_stages"] == "mini"
    assert mini["backend"] == "pycharmm"
    assert mini["setup"] == "pbc_npt"
    assert campaign["defaults"]["setup_variant"] == "minimal"
    assert campaign["defaults"]["composition"] == composition_string(cell)


def test_minimal_setup_disables_prep(cfg: dict, cell: RunCell) -> None:
    mini = build_campaign(cfg, cell)["runs"]["pycharmm_mini"]
    assert mini.get("liquid_prep") is False
    assert mini.get("charmm_mm_pretreat") is False
    assert mini.get("calculator_pre_minimize") is False
    assert mini.get("bonded_mm_mini") is False


def test_burst_hybrid_enables_pretreat(cfg: dict) -> None:
    cell = cell_from_cli(cfg, "burst_hybrid", "DCM", 77, temperature=300.0, box_size=32.0)
    mini = build_campaign(cfg, cell)["runs"]["pycharmm_mini"]
    assert mini.get("charmm_mm_pretreat") is True
    assert mini.get("dynamics_overlap_action") == "rescue"
    assert mini.get("bonded_mm_mini") is True


def test_liquid_prep_dense_flags(cfg: dict) -> None:
    cell = RunCell(
        setup_id="liquid_prep_dense",
        solvent="DCM",
        n_monomers=77,
        temperature=300.0,
        box_size=32.0,
    )
    mini = build_campaign(cfg, cell)["runs"]["pycharmm_mini"]
    assert mini.get("liquid_prep") is True
    assert mini.get("density_prep_ladder") is True


def test_bulk_density_fraction(cfg: dict, cell: RunCell) -> None:
    frac = cell_bulk_density_fraction(cell, cfg)
    assert frac == pytest.approx(0.25, rel=0.05)


def test_setup_variants_known_ids() -> None:
    for sid in ("minimal", "calculator_pre_sd", "liquid_prep_dense", "burst_hybrid", "resilient"):
        v = resolve_setup_variant(sid)
        assert v.id == sid
        assert v.description


def test_build_campaign_resolves_mmml_ckpt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: dict, cell: RunCell) -> None:
    ckpt = tmp_path / "params.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MMML_CKPT", str(ckpt))
    cfg = dict(cfg)
    cfg["checkpoint"] = "${MMML_CKPT}"
    campaign = build_campaign(cfg, cell)
    assert campaign["defaults"]["checkpoint"] == str(ckpt.resolve())


def test_build_md_system_campaign_argv(tmp_path: Path, cfg: dict, cell: RunCell) -> None:
    cfg = dict(cfg)
    cfg["output_root"] = str(tmp_path / "out")
    argv = build_md_system_campaign_argv(cfg, cell)
    campaign_path = Path(argv[argv.index("--config") + 1])
    loaded = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    assert loaded["defaults"]["composition"] == composition_string(cell)
    assert loaded["defaults"]["setup_variant"] == "minimal"


def test_matrix_setup_ids_from_config() -> None:
    cfg = load_config(WORKFLOW / "config.yaml")
    ids = matrix_setup_ids(cfg)
    assert "minimal" in ids
    assert "resilient" in ids
    assert len(ids) == 5


def test_default_config_matrix_job_count() -> None:
    cfg = load_config(WORKFLOW / "config.yaml")
    # 5 setups × 2 fractions × 3 boxes = 30
    assert matrix_job_count(cfg) == 30


def test_slurm_resources_cli(cfg: dict) -> None:
    cli = slurm_resources_cli(cfg)
    assert "gpu_fast=" in cli
    assert "charmm_slot=" in cli
    assert slurm_launch_jobs(cfg) == 18

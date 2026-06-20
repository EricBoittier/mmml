"""Unit tests for workflows/pbc_solvent_burst campaign generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "pbc_solvent_burst"
SCRIPTS = WORKFLOW / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from campaign_lib import (  # noqa: E402
    RunCell,
    build_campaign,
    build_md_system_campaign_argv,
    campaign_job_order,
    cell_from_cli,
    cell_from_tag,
    cell_run_tag,
    cell_slurm_tier,
    composition_string,
    iter_matrix_cells,
    load_config,
    matrix_job_count,
    slurm_launch_jobs,
    slurm_max_concurrent,
    slurm_nodelist_for_tier,
    slurm_tier_enabled,
    slurm_tier_resource_pools,
    total_jaxmd_ps,
    total_pycharmm_equi_ps,
)
from cleanup_strategy import resolve_cleanup_strategy  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    raw = yaml.safe_load((WORKFLOW / "config.yaml").read_text(encoding="utf-8"))
    # Unit tests assume one temperature and box unless a test overrides these lists.
    return {**raw, "temperatures": [300.0], "box_sizes": [32.0], "pycharmm_equi_ps": 10.0}


@pytest.fixture
def sweep_cfg(cfg: dict) -> dict:
    return {
        **cfg,
        "temperatures": [280.0, 300.0],
        "box_sizes": [28.0, 32.0],
    }


@pytest.fixture
def cell(cfg: dict) -> RunCell:
    return cell_from_cli(cfg, "DCM", 10)


def test_campaign_job_order_has_eleven_legs(cfg: dict) -> None:
    order = campaign_job_order(cfg)
    assert len(order) == 11
    assert order[0] == "pycharmm_init"
    assert order[-1] == "jaxmd_burst_05"


def test_run_tag_short_when_single_T_and_box(cfg: dict, cell: RunCell) -> None:
    assert cell_run_tag(cell, cfg) == "dcm_10"


def test_matrix_expands_temperature_and_box(sweep_cfg: dict) -> None:
    tags = {cell_run_tag(c, sweep_cfg) for c in iter_matrix_cells(sweep_cfg)}
    assert "dcm_10_t280_l28" in tags
    assert "dcm_10_t300_l32" in tags
    assert "dcm_10" not in tags
    assert matrix_job_count(sweep_cfg) == 2 * 5 * 2 * 2


def test_exclude_run_tags_skips_cells(sweep_cfg: dict) -> None:
    cfg = {**sweep_cfg, "exclude_run_tags": ["dcm_10_t300_l32", "aco_10_t280_l28"]}
    tags = {cell_run_tag(c, cfg) for c in iter_matrix_cells(cfg)}
    assert "dcm_10_t300_l32" not in tags
    assert "aco_10_t280_l28" not in tags
    assert "dcm_10_t280_l28" in tags
    assert matrix_job_count(cfg) == matrix_job_count(sweep_cfg) - 2


def test_cell_from_tag_accepts_long_form_when_short_is_canonical(cfg: dict, cell: RunCell) -> None:
    assert cell_from_tag(cfg, "dcm_10") == cell
    assert cell_from_tag(cfg, "dcm_10_t300_l32") == cell


def test_cell_from_tag_full_tag_on_sweep(sweep_cfg: dict) -> None:
    cell = cell_from_tag(sweep_cfg, "dcm_10_t300_l32")
    assert cell.solvent == "DCM"
    assert cell.n_monomers == 10
    assert cell.temperature == pytest.approx(300.0)
    assert cell.box_size == pytest.approx(32.0)


def test_depends_on_chain(cfg: dict, cell: RunCell) -> None:
    runs = build_campaign(cfg, cell)["runs"]
    assert runs["pycharmm_equi_00"]["depends_on"] == "pycharmm_init"
    assert runs["jaxmd_burst_01"]["depends_on"] == "pycharmm_equi_00"
    assert runs["jaxmd_burst_05"]["depends_on"] == "pycharmm_equi_04"


def test_cleanup_strategy_maps_to_pycharmm_and_jaxmd(cfg: dict) -> None:
    strategy = resolve_cleanup_strategy(cfg)
    assert strategy.name == "pbc_hybrid_default"
    campaign = build_campaign(cfg, cell_from_cli(cfg, "DCM", 10))
    init = campaign["runs"]["pycharmm_init"]
    burst = campaign["runs"]["jaxmd_burst_01"]
    assert init["dynamics_overlap_action"] == "rescue"
    assert init["bonded_mm_mini"] is True
    assert burst["handoff_quality_gate"] is True
    assert burst["jaxmd_pbc_minimize_steps"] == 200


def test_pretreat_from_cleanup_strategy(cfg: dict, cell: RunCell) -> None:
    enabled = dict(cfg)
    enabled["cleanup_strategy"] = dict(cfg["cleanup_strategy"])
    enabled["cleanup_strategy"]["charmm_mm"] = dict(cfg["cleanup_strategy"]["charmm_mm"])
    enabled["cleanup_strategy"]["charmm_mm"]["pretreat_on_pycharmm"] = True
    campaign = build_campaign(enabled, cell)
    init = campaign["runs"]["pycharmm_init"]
    assert init["charmm_mm_pretreat"] is True
    assert init["charmm_mm_pretreat_ps_equi"] == pytest.approx(100.0)
    assert campaign["runs"]["pycharmm_equi_00"]["charmm_mm_pretreat"] is True


def test_campaign_output_dirs_under_cell_tag(cfg: dict, cell: RunCell) -> None:
    campaign = build_campaign(cfg, cell)
    assert campaign["defaults"]["output_root"].endswith("dcm_10")
    init = campaign["runs"]["pycharmm_init"]
    assert init["output_dir"].endswith("dcm_10/pycharmm_init")
    from mmml.cli.run.md_campaign import _resolve_output_dir, merge_campaign_job_config

    merged = merge_campaign_job_config(campaign, "pycharmm_init")
    out = _resolve_output_dir(merged, "pycharmm_init")
    assert str(out).endswith("dcm_10/pycharmm_init")


def test_heat_finalt_follows_cell_temperature(cfg: dict) -> None:
    hot = RunCell(solvent="DCM", n_monomers=10, temperature=320.0, box_size=32.0)
    init = build_campaign(cfg, hot)["runs"]["pycharmm_init"]
    assert init["heat_finalt"] == pytest.approx(320.0)


def test_jaxmd_burst_quality_gate(cfg: dict, cell: RunCell) -> None:
    burst = build_campaign(cfg, cell)["runs"]["jaxmd_burst_02"]
    assert burst["backend"] == "jaxmd"
    assert burst["handoff_quality_gate"] is True
    assert "--steps-per-recording" in burst["extra_args"]


def test_build_md_system_campaign_argv(tmp_path: Path, cfg: dict, monkeypatch) -> None:
    monkeypatch.setenv("MMML_CKPT", str(tmp_path))
    (tmp_path / "ckpt.json").write_text("{}", encoding="utf-8")
    cfg = dict(cfg)
    cfg["checkpoint"] = "${MMML_CKPT}"
    cfg["output_root"] = str(tmp_path / "out")
    cell = RunCell(solvent="DCM", n_monomers=30, temperature=300.0, box_size=32.0)
    argv = build_md_system_campaign_argv(cfg, cell)
    campaign_path = Path(argv[argv.index("--config") + 1])
    loaded = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    assert loaded["defaults"]["composition"] == composition_string(cell)


def test_namespace_from_merged_jaxmd_burst(cfg: dict, cell: RunCell) -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    merged = build_campaign(cfg, cell)["runs"]["jaxmd_burst_01"]
    merged.update(build_campaign(cfg, cell)["defaults"])
    merged["job_id"] = "jaxmd_burst_01"
    ns = namespace_from_merged(merged)
    assert ns.backend == "jaxmd"
    assert ns.jaxmd_pbc_minimize_steps == 200


def test_slurm_max_concurrent(cfg: dict) -> None:
    assert matrix_job_count(cfg) == 10
    assert slurm_max_concurrent(cfg) == 10


def test_total_ps_budget(cfg: dict) -> None:
    assert total_jaxmd_ps(cfg) == pytest.approx(1000.0)
    assert total_pycharmm_equi_ps(cfg) == pytest.approx(50.0)


def test_load_config_accepts_string_path() -> None:
    cfg = load_config(str(WORKFLOW / "config.yaml"))
    assert "solvents" in cfg


def test_slurm_tier_routing(cfg: dict, cell: RunCell) -> None:
    tiered = {
        **cfg,
        "slurm_gpu_nodes_fast": ["gpu08"],
        "slurm_gpu_nodes_slow": ["gpu01"],
        "slurm_small_cluster_max_n": 30,
    }
    small = RunCell(solvent="DCM", n_monomers=10, temperature=300.0, box_size=32.0)
    large = RunCell(solvent="DCM", n_monomers=50, temperature=300.0, box_size=32.0)
    assert slurm_tier_enabled(tiered)
    assert cell_slurm_tier(small, tiered) == "slow"
    assert cell_slurm_tier(large, tiered) == "fast"
    assert "gpu01" in slurm_nodelist_for_tier(tiered, "slow")
    assert "gpu08" in slurm_nodelist_for_tier(tiered, "fast")


def test_slurm_tier_resource_pools(cfg: dict) -> None:
    tiered = {
        **cfg,
        "slurm_gpu_nodes_fast": ["gpu08", "gpu09"],
        "slurm_gpu_nodes_slow": ["gpu01"],
        "slurm_max_concurrent_fast": 4,
        "slurm_max_concurrent_slow": 2,
    }
    pools = slurm_tier_resource_pools(tiered)
    assert pools == {"gpu_fast": 4, "gpu_slow": 2, "charmm_slot": 6}
    assert slurm_launch_jobs(tiered) == 6

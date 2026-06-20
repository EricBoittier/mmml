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
    build_campaign,
    build_md_system_campaign_argv,
    campaign_job_order,
    composition_string,
    load_config,
    total_jaxmd_ps,
    total_pycharmm_equi_ps,
)


@pytest.fixture
def cfg() -> dict:
    return yaml.safe_load((WORKFLOW / "config.yaml").read_text(encoding="utf-8"))


def test_campaign_job_order_has_eleven_legs(cfg: dict) -> None:
    order = campaign_job_order(cfg)
    assert len(order) == 11
    assert order[0] == "pycharmm_init"
    assert order[1] == "pycharmm_equi_00"
    assert order[-1] == "jaxmd_burst_05"
    assert order.index("jaxmd_burst_03") < order.index("pycharmm_equi_03")


def test_depends_on_chain(cfg: dict) -> None:
    campaign = build_campaign(cfg, "DCM", 30)
    runs = campaign["runs"]
    assert runs["pycharmm_equi_00"]["depends_on"] == "pycharmm_init"
    assert runs["jaxmd_burst_01"]["depends_on"] == "pycharmm_equi_00"
    assert runs["pycharmm_equi_01"]["depends_on"] == "jaxmd_burst_01"
    assert runs["jaxmd_burst_05"]["depends_on"] == "pycharmm_equi_04"


def test_total_ps_budget(cfg: dict) -> None:
    assert total_jaxmd_ps(cfg) == pytest.approx(1000.0)
    assert total_pycharmm_equi_ps(cfg) == pytest.approx(50.0)


def test_defaults_composition_and_box(cfg: dict) -> None:
    campaign = build_campaign(cfg, "ACO", 50)
    assert campaign["defaults"]["composition"] == "ACO:50"
    assert campaign["defaults"]["box_size"] == pytest.approx(32.0)


def test_pycharmm_init_gentle_heat_and_repair(cfg: dict) -> None:
    init = build_campaign(cfg, "DCM", 10)["runs"]["pycharmm_init"]
    assert init["backend"] == "pycharmm"
    assert init["md_stages"] == "mini,heat"
    assert init["ps_heat"] == pytest.approx(30.0)
    assert init["n_heat_segments"] == 8
    assert init["heat_thermostat"] == "hoover"
    assert init["dynamics_overlap_action"] == "rescue"
    assert init["bonded_mm_mini"] is True
    assert "charmm_mm_pretreat" not in init


def test_pycharmm_init_charmm_mm_pretreat_when_enabled(cfg: dict) -> None:
    enabled = dict(cfg)
    enabled["charmm_mm_pretreat"] = True
    enabled["charmm_mm_pretreat_ps_equi"] = 10.0
    enabled["charmm_mm_pretreat_ps_prod"] = 5.0
    init = build_campaign(enabled, "DCM", 10)["runs"]["pycharmm_init"]
    assert init["charmm_mm_pretreat"] is True
    assert init["charmm_mm_pretreat_ps_heat"] == pytest.approx(30.0)
    assert init["charmm_mm_pretreat_ps_equi"] == pytest.approx(10.0)
    assert init["charmm_mm_pretreat_ps_prod"] == pytest.approx(5.0)
    assert "CHARMM MM pretreat" in init["description"]


def test_namespace_from_merged_pycharmm_init_pretreat(cfg: dict) -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    enabled = dict(cfg)
    enabled["charmm_mm_pretreat"] = True
    enabled["charmm_mm_pretreat_ps_equi"] = 10.0
    enabled["charmm_mm_pretreat_ps_prod"] = 5.0
    merged = build_campaign(enabled, "DCM", 10)["runs"]["pycharmm_init"]
    merged.update(build_campaign(enabled, "DCM", 10)["defaults"])
    merged["job_id"] = "pycharmm_init"
    ns = namespace_from_merged(merged)
    assert ns.charmm_mm_pretreat is True
    assert float(ns.charmm_mm_pretreat_ps_heat) == pytest.approx(30.0)
    assert float(ns.charmm_mm_pretreat_ps_equi) == pytest.approx(10.0)
    assert float(ns.charmm_mm_pretreat_ps_prod) == pytest.approx(5.0)


def test_jaxmd_burst_quality_gate_and_pbc_fire(cfg: dict) -> None:
    burst = build_campaign(cfg, "DCM", 30)["runs"]["jaxmd_burst_02"]
    assert burst["backend"] == "jaxmd"
    assert burst["setup"] == "pbc_nvt"
    assert burst["ps"] == pytest.approx(200.0)
    assert burst["handoff_quality_gate"] is True
    assert burst["jaxmd_pbc_minimize_steps"] == 200
    assert burst["dynamics_overlap_action"] == "warn"
    assert "--steps-per-recording" in burst["extra_args"]
    assert "800" in burst["extra_args"]


def test_optional_size_marks_final_burst(cfg: dict) -> None:
    burst = build_campaign(cfg, "ACO", 100)["runs"]["jaxmd_burst_05"]
    assert burst.get("optional") is True


def test_build_md_system_campaign_argv(tmp_path: Path, cfg: dict, monkeypatch) -> None:
    monkeypatch.setenv("MMML_CKPT", str(tmp_path))
    (tmp_path / "ckpt.json").write_text("{}", encoding="utf-8")
    cfg = dict(cfg)
    cfg["checkpoint"] = "${MMML_CKPT}"
    cfg["output_root"] = str(tmp_path / "out")
    argv = build_md_system_campaign_argv(cfg, "DCM", 30)
    assert "--run-all" in argv
    assert "--resume-campaign" in argv
    campaign_path = Path(argv[argv.index("--config") + 1])
    assert campaign_path.is_file()
    loaded = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    assert loaded["defaults"]["composition"] == composition_string("DCM", 30)


def test_namespace_from_merged_jaxmd_burst(cfg: dict) -> None:
    from mmml.cli.run.md_campaign import namespace_from_merged

    merged = build_campaign(cfg, "DCM", 30)["runs"]["jaxmd_burst_01"]
    merged.update(build_campaign(cfg, "DCM", 30)["defaults"])
    merged["job_id"] = "jaxmd_burst_01"
    ns = namespace_from_merged(merged)
    assert ns.backend == "jaxmd"
    assert ns.setup == "pbc_nvt"
    assert float(ns.ps) == pytest.approx(200.0)
    assert ns.jaxmd_pbc_minimize_steps == 200


def test_load_config_matches_repo(cfg: dict) -> None:
    loaded = load_config(WORKFLOW / "config.yaml")
    assert loaded["solvents"] == ["DCM", "ACO"]
    assert loaded["cluster_sizes"] == [10, 30, 50, 80, 100]


def test_slurm_max_concurrent(cfg: dict) -> None:
    from campaign_lib import matrix_job_count, slurm_max_concurrent

    assert matrix_job_count(cfg) == 10
    assert slurm_max_concurrent(cfg) == 10
    capped = dict(cfg)
    capped["slurm_max_concurrent"] = 99
    assert slurm_max_concurrent(capped) == 10
    low = dict(cfg)
    low["slurm_max_concurrent"] = 4
    assert slurm_max_concurrent(low) == 4

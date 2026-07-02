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
dynamics_campaign_enabled = cl.dynamics_campaign_enabled
init_job_id = cl.init_job_id
iter_matrix_cells = cl.iter_matrix_cells
load_config = cl.load_config
matrix_job_count = cl.matrix_job_count
matrix_setup_ids = cl.matrix_setup_ids
matrix_heat_thermostats = cl.matrix_heat_thermostats
heat_compare_enabled = cl.heat_compare_enabled
parse_dynamics_legs = cl.parse_dynamics_legs
prep_sweep_enabled = cl.prep_sweep_enabled
prep_sweep_variant_ids = cl.prep_sweep_variant_ids
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
        "heat_thermostats": [],
        "dynamics_legs": {
            "pycharmm_equi": False,
            "pycharmm_prod": False,
            "jaxmd": False,
            "ase": False,
        },
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
    cfg_bh = {k: v for k, v in cfg.items() if k not in ("bonded_mm_mini", "heat_thermostats")}
    cell = cell_from_cli(cfg_bh, "burst_hybrid", "DCM", 77, temperature=300.0, box_size=32.0)
    mini = build_campaign(cfg_bh, cell)["runs"]["pycharmm_mini"]
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


def test_resilient_disables_bonded_mm_mini_for_mini_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ckpt = tmp_path / "params.json"
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MMML_CKPT", str(ckpt))
    cfg = load_config(WORKFLOW / "config.yaml")
    cell = cell_from_cli(
        cfg,
        "resilient",
        "DCM",
        52,
        temperature=100.0,
        box_size=28.0,
        heat_thermostat="bussi",
    )
    init_id = init_job_id(cfg)
    init = build_campaign(cfg, cell)["runs"][init_id]
    assert init.get("liquid_prep") is True
    assert init.get("calculator_pre_minimize") is True
    assert init.get("bonded_mm_mini") is False
    assert init.get("charmm_mm_pretreat") is True
    assert init.get("md_stages") == "mini,heat"
    assert init.get("heat_thermostat") == "bussi"
    assert init.get("ps_heat") == 5.0


def test_heat_compare_coerces_scale_when_pretreat(cfg: dict) -> None:
    cfg_ht = {
        **cfg,
        "heat_thermostats": ["scale"],
        "setups": ["burst_hybrid"],
        "temperatures": [100.0],
        "box_sizes": [28.0],
        "bulk_density_fractions": [0.25],
    }
    cell = cell_from_cli(
        cfg_ht,
        "burst_hybrid",
        "DCM",
        52,
        temperature=100.0,
        box_size=28.0,
        heat_thermostat="scale",
    )
    mini = build_campaign(cfg_ht, cell)["runs"]["pycharmm_mini"]
    assert mini["heat_thermostat"] == "hoover"


def test_heat_compare_run_tag_suffix(cfg: dict) -> None:
    cfg_ht = {
        **cfg,
        "heat_thermostats": ["bussi", "hoover"],
        "temperatures": [100.0],
        "box_sizes": [28.0],
        "bulk_density_fractions": [0.25],
        "setups": ["minimal"],
    }
    cell = cell_from_cli(
        cfg_ht,
        "minimal",
        "DCM",
        52,
        temperature=100.0,
        box_size=28.0,
        heat_thermostat="hoover",
    )
    assert cell_run_tag(cell, cfg_ht) == "minimal_dcm_52_t100_l28_ht_hoover"
    campaign = build_campaign(cfg_ht, cell)
    mini = campaign["runs"]["pycharmm_mini"]
    assert mini["md_stages"] == "mini,heat"
    assert mini["heat_thermostat"] == "hoover"
    assert mini["heat_finalt"] == 100.0


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
    assert ids == ["resilient"]


def test_default_config_matrix_job_count() -> None:
    cfg = load_config(WORKFLOW / "config.yaml")
    assert heat_compare_enabled(cfg)
    assert matrix_heat_thermostats(cfg) == ["bussi", "hoover", "scale"]
    assert dynamics_campaign_enabled(cfg)
    assert parse_dynamics_legs(cfg) == {
        "pycharmm_equi": True,
        "pycharmm_prod": True,
        "jaxmd": True,
        "ase": True,
    }
    assert campaign_job_order(cfg) == [
        "pycharmm_init",
        "pycharmm_equi_01",
        "pycharmm_prod_01",
        "jaxmd_prod",
        "ase_prod",
    ]
    # resilient × 4 fractions × 3 T × 3 L × 3 thermostats = 108
    assert matrix_job_count(cfg) == 108


def test_full_dynamics_campaign_chain(cfg: dict, cell: RunCell) -> None:
    cfg_dyn = {
        **cfg,
        "dynamics_legs": {
            "pycharmm_equi": True,
            "pycharmm_prod": True,
            "jaxmd": True,
            "ase": True,
        },
        "pycharmm_equi_ps": 5.0,
        "pycharmm_prod_ps": 8.0,
        "jaxmd_ps": 3.0,
        "ase_ps": 4.0,
    }
    order = campaign_job_order(cfg_dyn)
    campaign = build_campaign(cfg_dyn, cell)
    assert list(campaign["runs"]) == order
    init = campaign["runs"]["pycharmm_init"]
    assert init["md_stages"] == "mini,heat"
    assert init["heat_thermostat"] == "bussi"
    equi = campaign["runs"]["pycharmm_equi_01"]
    prod = campaign["runs"]["pycharmm_prod_01"]
    jaxmd = campaign["runs"]["jaxmd_prod"]
    ase = campaign["runs"]["ase_prod"]
    assert equi["depends_on"] == "pycharmm_init"
    assert equi["ps_equi"] == 5.0
    assert prod["depends_on"] == "pycharmm_equi_01"
    assert prod["ps_prod"] == 8.0
    assert jaxmd["depends_on"] == "pycharmm_prod_01"
    assert jaxmd["backend"] == "jaxmd"
    assert jaxmd["ps"] == 3.0
    assert ase["depends_on"] == "jaxmd_prod"
    assert ase["backend"] == "ase"
    assert ase["nvt_integrator"] == "nhc"
    assert ase["ps"] == 4.0
    assert campaign["defaults"]["handoff_write_res"] is True
    assert campaign["defaults"]["continue_velocities"] is True
    paths = cl.paths_for_run(cfg_dyn, cell)
    assert paths["final_handoff"].as_posix().endswith("ase_prod/handoff/state.npz")


def test_slurm_resources_cli(cfg: dict) -> None:
    cli = slurm_resources_cli(cfg)
    assert "gpu_fast=" in cli
    assert "charmm_slot=" in cli
    assert slurm_launch_jobs(cfg) == 18


def test_prep_sweep_expands_variants(cfg: dict) -> None:
    sweep_cfg = {
        **cfg,
        "setups": ["resilient"],
        "checkpoint": cfg["checkpoint"],
        "prep_sweep": {
            "enabled": True,
            "stages": "mini",
            "anchor": {
                "setup_id": "resilient",
                "bulk_density_fraction": 0.25,
                "temperature": 50.0,
                "box_size": 28.0,
            },
            "variants": {
                "baseline": {},
                "dt050": {"dt_fs": 0.5},
            },
        },
    }
    tags = [cell_run_tag(c, sweep_cfg) for c in iter_matrix_cells(sweep_cfg)]
    assert len(tags) == 2
    assert tags[0].endswith("_sw_baseline")
    assert tags[1].endswith("_sw_dt050")
    assert matrix_job_count(sweep_cfg) == 2


def test_prep_sweep_applies_overrides_to_campaign(cfg: dict) -> None:
    sweep_cfg = {
        **cfg,
        "setups": ["resilient"],
        "checkpoint": cfg["checkpoint"],
        "prep_sweep": {
            "enabled": True,
            "stages": "mini",
            "anchor": {
                "setup_id": "resilient",
                "n_monomers": 52,
                "temperature": 50.0,
                "box_size": 28.0,
            },
            "variants": {
                "baseline": {},
                "pmtol30": {"packmol_tolerance": 3.0},
            },
        },
    }
    cell = cell_from_tag(sweep_cfg, "resilient_dcm_52_t50_l28_sw_pmtol30")
    campaign = build_campaign(sweep_cfg, cell)
    assert campaign["defaults"]["prep_sweep_id"] == "pmtol30"
    assert campaign["defaults"]["packmol_tolerance"] == 3.0
    assert list(campaign["runs"]) == ["pycharmm_mini"]
    mini = campaign["runs"]["pycharmm_mini"]
    assert mini["md_stages"] == "mini"


def test_prep_sweep_tag_unknown_in_default_config(cfg: dict) -> None:
    sweep_cfg = {
        **cfg,
        "prep_sweep": {
            "enabled": True,
            "stages": "mini",
            "anchor": {"setup_id": "resilient", "n_monomers": 52, "temperature": 50.0, "box_size": 28.0},
            "variants": {"baseline": {}},
        },
    }
    tag = "resilient_dcm_52_t50_l28_sw_baseline"
    with pytest.raises(KeyError):
        cell_from_tag(cfg, tag)
    cell = cell_from_tag(sweep_cfg, tag)
    assert cell.sweep_id == "baseline"


def test_prep_sweep_mini_heat_stage(cfg: dict) -> None:
    sweep_cfg = {
        **cfg,
        "setups": ["resilient"],
        "checkpoint": cfg["checkpoint"],
        "prep_sweep": {
            "enabled": True,
            "stages": "mini,heat",
            "anchor": {
                "setup_id": "resilient",
                "n_monomers": 52,
                "temperature": 50.0,
                "box_size": 28.0,
                "heat_thermostat": "hoover",
            },
            "variants": {"baseline": {}},
        },
    }
    cell = cell_from_tag(sweep_cfg, "resilient_dcm_52_t50_l28_ht_hoover_sw_baseline")
    campaign = build_campaign(sweep_cfg, cell)
    mini = campaign["runs"]["pycharmm_mini"]
    assert mini["md_stages"] == "mini,heat"
    assert mini["heat_thermostat"] == "hoover"

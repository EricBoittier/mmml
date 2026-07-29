"""Unit tests for workflows/pbc_methane_ewald campaign generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "pbc_methane_ewald"
SCRIPTS = WORKFLOW / "scripts"
_MOD = "pbc_methane_ewald_campaign_lib"


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


cl = _load_script_module(_MOD, SCRIPTS / "campaign_lib.py")
sys.modules["campaign_lib"] = cl

RunCell = cl.RunCell
build_campaign = cl.build_campaign
cell_from_tag = cl.cell_from_tag
cell_run_tag = cl.cell_run_tag
checkpoint_map = cl.checkpoint_map
composition_string = cl.composition_string
iter_matrix_cells = cl.iter_matrix_cells
load_config = cl.load_config
matrix_backends = cl.matrix_backends
matrix_job_count = cl.matrix_job_count
matrix_temperatures = cl.matrix_temperatures
campaign_job_order = cl.campaign_job_order
methane_n_bulk = cl.methane_n_bulk


@pytest.fixture
def cfg() -> dict:
    raw = yaml.safe_load((WORKFLOW / "config.yaml").read_text(encoding="utf-8"))
    return {
        **raw,
        "checkpoints": {
            "des": "examples/ckpts_json/DESdimers_params.json",
            "sppoky10": "examples/sppoky-epoch-0010_params.json",
        },
        "temperatures": [100.0, 200.0, 300.0],
        "box_sizes": [20.0],
        "bulk_density_fractions": [1.0],
        "backends": ["pycharmm", "jaxmd"],
        "cluster_sizes": None,
    }


def test_temperatures_match_request(cfg: dict) -> None:
    assert matrix_temperatures(cfg) == [100.0, 200.0, 300.0]


def test_matrix_expands_t_ckpt_backend(cfg: dict) -> None:
    tags = {cell_run_tag(c, cfg) for c in iter_matrix_cells(cfg)}
    assert any(t.endswith("_des_pycharmm") and "_t100_" in t for t in tags)
    assert any(t.endswith("_sppoky10_jaxmd") and "_t300_" in t for t in tags)
    # 1 solvent × 1 N × 3 T × 2 ckpt × 2 backend
    assert matrix_job_count(cfg) == 3 * 2 * 2


def test_campaign_defaults_use_ewald(cfg: dict) -> None:
    cell = next(c for c in iter_matrix_cells(cfg) if c.backend == "pycharmm")
    campaign = build_campaign(cfg, cell)
    defaults = campaign["defaults"]
    assert defaults["lr_solver"] == "ewald"
    assert defaults["mm_nonbond_mode"] == "periodic_external"
    assert defaults["composition"].startswith("METH:")
    assert Path(defaults["checkpoint"]).name.endswith(".json")


def test_pycharmm_and_jaxmd_job_orders(cfg: dict) -> None:
    py_cell = next(c for c in iter_matrix_cells(cfg) if c.backend == "pycharmm")
    jx_cell = next(c for c in iter_matrix_cells(cfg) if c.backend == "jaxmd")
    py_order = campaign_job_order(cfg, py_cell)
    jx_order = campaign_job_order(cfg, jx_cell)
    assert py_order[0] == "pycharmm_init"
    assert py_order[-1].startswith("pycharmm_prod_")
    assert jx_order == ["pycharmm_init", "jaxmd_equi", "jaxmd_prod"]
    py_camp = build_campaign(cfg, py_cell)
    jx_camp = build_campaign(cfg, jx_cell)
    assert set(py_order) == set(py_camp["runs"])
    assert set(jx_order) == set(jx_camp["runs"])
    assert jx_camp["runs"]["jaxmd_prod"]["backend"] == "jaxmd"
    assert jx_camp["runs"]["jaxmd_prod"]["setup"] == "pbc_nvt"


def test_cell_from_tag_roundtrip(cfg: dict) -> None:
    cell = next(iter_matrix_cells(cfg))
    tag = cell_run_tag(cell, cfg)
    again = cell_from_tag(cfg, tag)
    assert again == cell
    assert composition_string(again) == f"METH:{again.n_monomers}"


def test_smoke_config(cfg: dict) -> None:
    del cfg
    smoke = load_config(WORKFLOW / "config.smoke.yaml")
    assert smoke["lr_solver"] == "ewald"
    assert smoke["temperatures"] == [100.0, 200.0, 300.0]
    assert set(checkpoint_map(smoke)) == {"des", "sppoky10"}
    assert matrix_backends(smoke) == ["pycharmm", "jaxmd"]
    assert matrix_job_count(smoke) == 3 * 2 * 2


def test_methane_bulk_count_positive() -> None:
    assert methane_n_bulk(20.0, 1.0) >= 100


def test_meth_residue_and_monomer_bundled() -> None:
    from mmml.analysis.residue_geometry import bundled_monomer_pdb, known_solvent_density_kg_m3
    from mmml.interfaces.pycharmmInterface.cgenff_residues import (
        normalize_cgenff_residue_name,
        require_cgenff_residue_name,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms

    assert normalize_cgenff_residue_name("CH4") == "METH"
    assert normalize_cgenff_residue_name("methane") == "METH"
    assert require_cgenff_residue_name("METH") == "METH"
    pdb = bundled_monomer_pdb("METH")
    assert pdb is not None and pdb.is_file()
    assert known_solvent_density_kg_m3("METH") == pytest.approx(422.6)
    assert estimate_ml_atoms(10, solvent="METH") == 50

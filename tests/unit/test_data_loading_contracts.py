"""Ingest vs train NPZ contracts and data-loading CI guards (synthetic only)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mmml.data import convert_spice_alpha_hdf5, load_npz
from mmml.data.npz_schema import REQUIRED_KEYS
from mmml.data.spice_alpha import TRAIN_NPZ_UNITS, assert_train_npz_contract
from mmml.data.units import HARTREE_TO_EV, units_from_npz
from spice_alpha_fixtures import write_spice_h5

pytestmark = pytest.mark.data_loading

_REPO = Path(__file__).resolve().parents[2]


def _http_literals(path: Path) -> list[str]:
    prefixes = ("ht" + "tp://", "ht" + "tps://", "ft" + "p://")
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip().lower()
            if value.startswith(prefixes) and "://" in value[4:]:
                found.append(node.value)
    return found


def test_ingest_schema_labels_are_not_train_units():
    """``mmml validate`` / npz_schema document PySCF ingest, not physnet-train."""
    assert "Hartree" in REQUIRED_KEYS["E"]
    assert TRAIN_NPZ_UNITS["E"] == "ev"
    assert TRAIN_NPZ_UNITS["F"] == "ev_angstrom"
    assert HARTREE_TO_EV == pytest.approx(27.211386, rel=0, abs=1e-6)


def test_converted_npz_loads_and_embeds_train_units(tmp_path):
    src = write_spice_h5(tmp_path / "spice.hdf5", extra_group=False)
    out = tmp_path / "train.npz"
    convert_spice_alpha_hdf5([src], out)
    loaded = load_npz(out, validate=True)
    assert_train_npz_contract(loaded)
    assert loaded["E"][0] == pytest.approx(-100.0)
    manifest = units_from_npz(out)
    assert manifest is not None
    arrays = {key.lower(): str(val).lower() for key, val in manifest.arrays.items()}
    assert arrays["e"] == "ev"
    assert "ev" in arrays["f"]


def test_converter_source_does_not_download_datasets():
    src = _REPO / "mmml" / "data" / "spice_alpha.py"
    text = src.read_text()
    assert "urllib" not in text
    assert "requests" not in text
    assert "urllib.request" not in text
    assert "zenodo.org/api" not in text
    assert "Never download Zenodo" in text


def test_data_loading_tests_have_no_remote_dataset_urls():
    roots = [
        _REPO / "tests" / "unit" / "test_spice_alpha.py",
        _REPO / "tests" / "unit" / "test_training_npz_contract.py",
        _REPO / "tests" / "unit" / "test_data_loading_contracts.py",
        _REPO / "tests" / "unit" / "spice_alpha_fixtures.py",
    ]
    leaked: list[str] = []
    for path in roots:
        leaked.extend(f"{path.name}:{lit}" for lit in _http_literals(path))
    assert leaked == []


def test_workflow_selects_marker_and_forbids_dataset_pulls():
    workflow = (_REPO / ".github" / "workflows" / "data-loading.yml").read_text()
    assert "-m data_loading" in workflow
    assert "synthetic" in workflow.lower()
    assert "curl " not in workflow
    assert "wget " not in workflow
    assert "zenodo.org" not in workflow
    assert "huggingface.co" not in workflow

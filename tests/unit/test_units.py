"""Unit tests for mmml.data.units."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.data.units import (
    CANONICAL_ENERGY_UNIT,
    HARTREE_TO_EV,
    UnitsManifestV2,
    convert_coords,
    convert_energy,
    convert_forces,
    energy_to_ev,
    find_units_manifest,
    forces_to_ev_angstrom,
    load_units_manifest,
    normalize_to_canonical,
    subtract_atom_refs,
    units_from_npz,
)


def test_convert_energy_roundtrip() -> None:
    ha = np.array([-1.0, -2.5])
    ev = convert_energy(ha, "hartree", "ev")
    np.testing.assert_allclose(ev, ha * HARTREE_TO_EV)
    back = convert_energy(ev, "ev", "hartree")
    np.testing.assert_allclose(back, ha, rtol=1e-12)


def test_convert_forces_roundtrip() -> None:
    ha_b = np.array([[0.01, -0.02, 0.03]])
    ev_a = convert_forces(ha_b, "hartree_bohr", "ev_angstrom")
    back = convert_forces(ev_a, "ev_angstrom", "hartree_bohr")
    np.testing.assert_allclose(back, ha_b, rtol=1e-10)


def test_convert_coords_bohr_angstrom() -> None:
    ang = np.array([1.0, 2.0])
    bohr = convert_coords(ang, "angstrom", "bohr")
    np.testing.assert_allclose(convert_coords(bohr, "bohr", "angstrom"), ang, rtol=1e-5)


def test_energy_to_ev_scalar() -> None:
    assert energy_to_ev(-1.0, "hartree") == pytest.approx(-HARTREE_TO_EV)


def test_subtract_atom_refs_respects_energy_unit() -> None:
    z = np.array([[1, 1]], dtype=np.int32)
    e_ha = np.array([-1.5])
    e_ev = convert_energy(e_ha, "hartree", "ev")
    ref_ha = subtract_atom_refs(e_ha, z, energy_unit="hartree")
    ref_ev = subtract_atom_refs(e_ev, z, energy_unit="ev")
    np.testing.assert_allclose(
        convert_energy(ref_ev, "ev", "hartree"),
        ref_ha,
        rtol=1e-6,
    )


def test_units_manifest_v1_upgrade() -> None:
    v1 = {
        "coords_in": "angstrom",
        "coords_out": "angstrom",
        "coords_detected": "angstrom",
        "energy_in": "hartree",
        "energy_out": "ev",
        "force_in": "hartree-bohr",
        "force_out": "ev-angstrom",
        "dipole_in": "debye",
        "dipole_out": "e-angstrom",
        "grid_coords_in": "bohr",
        "grid_coords_out": "angstrom",
        "esp_values": "hartree/e",
        "flip_forces": True,
        "preserve_units": False,
        "notes": [],
    }
    manifest = UnitsManifestV2.from_dict(v1)
    assert manifest.schema_version == 2
    assert manifest.arrays["E"] == "ev"
    assert manifest.arrays["F"] == "ev-angstrom"
    assert manifest.energy_unit() == "ev"


def test_load_units_manifest_from_directory(tmp_path: Path) -> None:
    manifest = UnitsManifestV2(
        arrays={"E": "ev", "F": "ev_angstrom", "R": "angstrom"},
        preserve_units=False,
    )
    path = tmp_path / "units_manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    loaded = load_units_manifest(tmp_path)
    assert loaded is not None
    assert loaded.energy_unit() == "ev"
    assert find_units_manifest(tmp_path / "dummy.npz") is not None


def test_units_from_npz_embedded_metadata(tmp_path: Path) -> None:
    meta = json.dumps({"E": "hartree", "F": "hartree_bohr", "R": "angstrom"})
    path = tmp_path / "data.npz"
    np.savez_compressed(path, E=np.array([-1.0]), _mmml_units=np.array(meta))
    manifest = units_from_npz(path)
    assert manifest is not None
    assert manifest.energy_unit() == "hartree"


def test_infer_reference_units_from_force_magnitudes(tmp_path: Path) -> None:
    from mmml.data.units import infer_reference_energy_unit, infer_reference_force_unit

    ev_path = tmp_path / "ev_ref.npz"
    np.savez_compressed(
        ev_path,
        E=np.array([-43.0, -44.0]),
        F=np.random.default_rng(0).normal(size=(2, 10, 3)) * 2.0,
        R=np.zeros((2, 10, 3)),
        Z=np.ones((2, 10), dtype=int),
        N=np.array([10, 10], dtype=int),
    )
    assert infer_reference_energy_unit(ev_path) == "ev"
    assert infer_reference_force_unit(ev_path) == "ev_angstrom"

    ha_path = tmp_path / "ha_ref.npz"
    np.savez_compressed(
        ha_path,
        E=np.array([-43.0]),
        F=np.random.default_rng(1).normal(size=(1, 10, 3)) * 0.05,
        R=np.zeros((1, 10, 3)),
        Z=np.ones((1, 10), dtype=int),
        N=np.array([10], dtype=int),
    )
    assert infer_reference_energy_unit(ha_path) == "hartree"
    assert infer_reference_force_unit(ha_path) == "hartree_bohr"


def test_normalize_to_canonical_converts_hartree(tmp_path: Path) -> None:
    manifest = UnitsManifestV2(
        arrays={"E": "hartree", "F": "hartree_bohr", "R": "angstrom"},
        energy_out="hartree",
        force_out="hartree_bohr",
        coords_out="angstrom",
    )
    data = {
        "E": np.array([-1.0]),
        "F": np.array([[[0.01, 0.0, 0.0]]]),
        "R": np.array([[[0.0, 0.0, 0.0]]]),
    }
    out = normalize_to_canonical(data, manifest, allow_hartree=True)
    assert out["E"][0] == pytest.approx(-HARTREE_TO_EV)
    assert CANONICAL_ENERGY_UNIT == "ev"

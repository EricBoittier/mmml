"""Unit tests for fix_and_split unit manifest and efield conversions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.cli.misc.fix_and_split import (
    convert_energy_array,
    convert_force_array,
    UnitsManifest,
)
from mmml.data.units import UnitsManifestV2, load_units_manifest


def test_convert_energy_array_hartree_to_ev() -> None:
    e_ha = np.array([-1.0, -2.0])
    e_ev = convert_energy_array(e_ha, "hartree", "ev")
    assert e_ev[0] == pytest.approx(-27.211386, rel=1e-5)


def test_units_manifest_v2_from_v1_fields() -> None:
    v1 = UnitsManifest(
        coords_in="angstrom",
        coords_out="angstrom",
        coords_detected="angstrom",
        energy_in="hartree",
        energy_out="ev",
        force_in="hartree_bohr",
        force_out="ev_angstrom",
        dipole_in="debye",
        dipole_out="e_angstrom",
        grid_coords_in="bohr",
        grid_coords_out="angstrom",
        esp_values="hartree/e",
        flip_forces=False,
        preserve_units=False,
        notes=["test"],
    )
    from dataclasses import asdict

    manifest = UnitsManifestV2.from_dict(asdict(v1))
    assert manifest.schema_version == 2
    assert manifest.arrays["E"] == "ev"
    assert manifest.arrays["efield_energy"] == "ev"


def test_fix_and_split_writes_manifest_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test manifest v2 emission via minimal synthetic NPZ."""
    from mmml.cli.misc import fix_and_split as fas

    n_samples = 4
    n_atoms = 2
    efd = tmp_path / "efd.npz"
    np.savez_compressed(
        efd,
        R=np.random.default_rng(0).random((n_samples, n_atoms, 3)) + 1.0,
        E=np.full(n_samples, -1.5),
        F=np.zeros((n_samples, n_atoms, 3)),
        N=np.full(n_samples, n_atoms, dtype=np.int32),
        Z=np.tile(np.array([1, 1], dtype=np.int32), (n_samples, 1)),
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(fas, "validate_fixed_data", lambda *a, **k: True)
    ok = fas.fix_and_split_data(
        efd_file=efd,
        grid_file=None,
        output_dir=out_dir,
        verbose=False,
        skip_validation=True,
        energy_in="hartree",
        energy_out="ev",
        force_in="hartree_bohr",
        force_out="ev_angstrom",
        coords_in="angstrom",
        coords_out="same",
    )
    assert ok is not False
    manifest = load_units_manifest(out_dir)
    assert manifest is not None
    assert manifest.schema_version == 2
    assert manifest.energy_unit() == "ev"
    train_npz = out_dir / "energies_forces_dipoles_train.npz"
    if train_npz.is_file():
        with np.load(train_npz, allow_pickle=True) as data:
            assert "_mmml_units" in data.files
            arrays = json.loads(str(data["_mmml_units"].item()))
            assert arrays["E"] == "ev"

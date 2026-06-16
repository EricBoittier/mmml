"""Unit tests for fix-and-split EFD merge (no CHARMM / full pipeline)."""

from pathlib import Path

import numpy as np
import pytest

from mmml.cli.misc.fix_and_split import (
    _load_and_merge_efd,
    atomic_ref_sum_hartree,
    check_atomic_ref_subtraction,
    diagnose_energy_unit_for_atomic_refs,
    expected_atomic_ref_units,
    subtract_atomic_references,
)


def _write_efd(path: Path, n_samples: int, n_atoms: int = 4, z_1d: bool = False) -> None:
    rng = np.random.default_rng(0)
    z_row = np.array([6, 1, 1, 1], dtype=np.int32)
    if z_1d:
        z = z_row
    else:
        z = np.broadcast_to(z_row.reshape(1, -1), (n_samples, n_atoms)).copy()
    np.savez(
        path,
        R=rng.normal(size=(n_samples, n_atoms, 3)),
        E=rng.normal(size=(n_samples,)),
        F=rng.normal(size=(n_samples, n_atoms, 3)),
        N=np.full(n_samples, n_atoms, dtype=np.int32),
        Z=z,
    )


def test_merge_efd_concatenates_z_per_sample(tmp_path):
    f1 = tmp_path / "a.npz"
    f2 = tmp_path / "b.npz"
    _write_efd(f1, n_samples=20, z_1d=False)
    _write_efd(f2, n_samples=10, z_1d=False)

    merged = _load_and_merge_efd([f1, f2])

    assert merged["R"].shape[0] == 30
    assert merged["E"].shape[0] == 30
    assert merged["Z"].shape == (30, 4)


def test_merge_efd_broadcasts_shared_z_then_concatenates(tmp_path):
    f1 = tmp_path / "a.npz"
    f2 = tmp_path / "b.npz"
    _write_efd(f1, n_samples=20, z_1d=True)
    _write_efd(f2, n_samples=10, z_1d=True)

    merged = _load_and_merge_efd([f1, f2])

    assert merged["Z"].shape == (30, 4)


def test_subtract_atomic_references_matches_z_rows(tmp_path):
    n_samples = 5
    n_atoms = 4
    z = np.broadcast_to(np.array([[6, 1, 1, 1]], dtype=np.int32), (n_samples, n_atoms))
    e = np.linspace(-100.0, -90.0, n_samples)

    corrected = subtract_atomic_references(e, z, "pbe0/def2-tzvp", ref_units="ev")

    assert corrected.shape == (n_samples,)
    assert np.all(np.isfinite(corrected))


def test_merge_efd_z_rows_match_r_after_concat(tmp_path):
    """After merge, Z rows must match R rows (regression guard)."""
    f1 = tmp_path / "a.npz"
    f2 = tmp_path / "b.npz"
    _write_efd(f1, n_samples=20, z_1d=False)
    _write_efd(f2, n_samples=10, z_1d=False)

    merged = _load_and_merge_efd([f1, f2])
    n_samples = merged["R"].shape[0]
    z = merged["Z"]
    if z.ndim == 1:
        z_expanded = np.broadcast_to(z[np.newaxis, :], (n_samples, z.shape[0]))
    else:
        z_expanded = z

    assert z_expanded.shape[0] == n_samples


def test_expected_atomic_ref_units_for_def2tzvp():
    assert expected_atomic_ref_units("pbe0/def2-tzvp") == "hartree"


def test_check_atomic_ref_subtraction_catches_wrong_units():
    n_atoms = 4
    n_samples = 3
    z = np.broadcast_to(np.array([[6, 1, 1, 1]], dtype=np.int32), (n_samples, n_atoms))

    e_before_large = np.full(n_samples, -35130.0)
    e_wrong = subtract_atomic_references(e_before_large, z, "pbe0/def2-tzvp", ref_units="ev")
    with pytest.raises(ValueError, match="atomic-ref-units hartree"):
        check_atomic_ref_subtraction(
            e_before_large,
            e_wrong,
            scheme="pbe0/def2-tzvp",
            ref_units="ev",
        )

    ref_offset = float(
        subtract_atomic_references(np.zeros(n_samples), z, "pbe0/def2-tzvp", ref_units="hartree")[0]
    )
    e_before = np.full(n_samples, -ref_offset - 1.0)
    e_right = subtract_atomic_references(e_before, z, "pbe0/def2-tzvp", ref_units="hartree")
    check_atomic_ref_subtraction(
        e_before,
        e_right,
        scheme="pbe0/def2-tzvp",
        ref_units="hartree",
    )
    assert np.allclose(e_right, -1.0)


def test_diagnose_energy_unit_detects_ev_totals():
    n_atoms = 40
    n_samples = 3
    z_row = np.array([6] * 34 + [1] * 6, dtype=np.int32)
    z = np.broadcast_to(z_row.reshape(1, -1), (n_samples, n_atoms))
    ref_ha = atomic_ref_sum_hartree(z, "pbe0/def2-tzvp", "hartree")
    e_ev = np.full(n_samples, -(ref_ha * 27.211386 + 5.0))

    hint = diagnose_energy_unit_for_atomic_refs(
        e_ev, z, "pbe0/def2-tzvp", "hartree", "hartree"
    )
    assert hint is not None
    assert "--energy-in ev" in hint

    e_ha = np.full(n_samples, -(ref_ha + 5.0))
    corrected = subtract_atomic_references(e_ha, z, "pbe0/def2-tzvp", ref_units="hartree")
    assert np.allclose(corrected, -5.0)
    assert diagnose_energy_unit_for_atomic_refs(e_ha, z, "pbe0/def2-tzvp", "hartree", "hartree") is None

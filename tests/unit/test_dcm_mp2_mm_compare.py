"""Unit tests for DCM MP2 vs MM comparison helpers (no PyCHARMM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.dcm_mp2_mm_compare import (
    DCM_PSF_MONOMER_PERM,
    HybridEvalResult,
    Mp2Frame,
    MmFrameResult,
    aggregate_comparison,
    apply_atom_permutation,
    compare_mm_to_mp2_frame,
    forces_kcal_to_ev,
    load_monomer_mean_energy_eV,
    load_mp2_frames,
    parse_monomer_permutation,
    repeat_monomer_permutation,
    select_frame_indices,
)
from mmml.data.units import convert_forces


def _write_synthetic_mp2_npz(path: Path, *, n_dimer: int = 4, n_mono: int = 2) -> None:
    """Minimal NPZ: dimers N=10, monomers N=5, Hartree E, eV/Å F."""
    n_total = n_dimer + n_mono
    counts = np.array([10] * n_dimer + [5] * n_mono, dtype=int)
    nat_pad = 10
    z = np.zeros((n_total, nat_pad), dtype=np.int32)
    r = np.zeros((n_total, nat_pad, 3), dtype=np.float64)
    f = np.zeros((n_total, nat_pad, 3), dtype=np.float64)
    for i in range(n_total):
        n = int(counts[i])
        mono_z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
        if n == 10:
            z[i, :n] = np.concatenate([mono_z, mono_z])
        else:
            z[i, :n] = mono_z
        r[i, :n] = np.arange(n * 3, dtype=np.float64).reshape(n, 3) * 0.1
        f[i, :n] = 0.01 * (i + 1)
    e = -43.0 + 0.001 * np.arange(n_total, dtype=np.float64)
    np.savez(path, N=counts, Z=z, R=r, E=e, F=f)


def test_parse_and_repeat_monomer_permutation() -> None:
    perm = parse_monomer_permutation("0,3,4,1,2")
    assert np.array_equal(perm, DCM_PSF_MONOMER_PERM)
    full = repeat_monomer_permutation(10, perm)
    assert full.tolist() == [0, 3, 4, 1, 2, 5, 8, 9, 6, 7]
    with pytest.raises(ValueError, match="zero-based"):
        parse_monomer_permutation("0,1,3")


def test_apply_atom_permutation() -> None:
    z = np.array([6, 1, 1, 17, 17], dtype=np.int32)
    r = np.arange(15, dtype=np.float64).reshape(5, 3)
    f = np.ones((5, 3), dtype=np.float64)
    perm = DCM_PSF_MONOMER_PERM
    z2, r2, f2 = apply_atom_permutation(z, r, f, perm)
    assert z2.tolist() == [6, 17, 17, 1, 1]
    assert np.array_equal(r2, r[perm])
    assert np.array_equal(f2, f[perm])


def test_select_frame_indices_stride_and_cap() -> None:
    idx = select_frame_indices(50, max_frames=None, stride=10, seed=0)
    assert idx.tolist() == [0, 10, 20, 30, 40]
    idx2 = select_frame_indices(1000, max_frames=3, stride=1, seed=42)
    assert len(idx2) == 3
    assert np.all(idx2 < 1000)


def test_load_mp2_frames_applies_permutation(tmp_path: Path) -> None:
    npz = tmp_path / "ref.npz"
    _write_synthetic_mp2_npz(npz, n_dimer=3, n_mono=1)
    frames, meta = load_mp2_frames(
        npz,
        n_atoms=10,
        reference_energy_unit="hartree",
        reference_force_unit="ev_angstrom",
        max_frames=None,
        stride=1,
    )
    assert len(frames) == 3
    assert meta["n_available"] == 3
    assert frames[0].z.tolist() == [6, 17, 17, 1, 1] * 2
    assert frames[0].f_ref_ev_A is not None
    assert frames[0].f_ref_ev_A.shape == (10, 3)


def test_load_monomer_mean_energy_eV(tmp_path: Path) -> None:
    npz = tmp_path / "ref.npz"
    _write_synthetic_mp2_npz(npz, n_dimer=2, n_mono=3)
    mean_ev = load_monomer_mean_energy_eV(npz, reference_energy_unit="hartree")
    from mmml.data.units import energy_to_ev

    mono_e = -43.0 + 0.001 * np.arange(2, 5, dtype=np.float64)
    expected = float(np.mean(energy_to_ev(mono_e, "hartree")))
    assert mean_ev == pytest.approx(expected)


def test_forces_kcal_to_ev_roundtrip() -> None:
    f_kcal = np.array([[1.0, 0.0, -0.5]], dtype=np.float64)
    f_ev = forces_kcal_to_ev(f_kcal)
    back = convert_forces(f_ev, "ev_angstrom", "kcal_mol_angstrom")
    assert np.allclose(back, f_kcal, rtol=1e-10)


def test_compare_and_aggregate_frame_metrics() -> None:
    frame = Mp2Frame(
        index=0,
        source_index=7,
        n_atoms=10,
        z=np.ones(10, dtype=np.int32),
        r=np.zeros((10, 3)),
        e_ref_raw=-43.0,
        e_ref_eV=-1170.0,
        f_ref_ev_A=np.ones((10, 3), dtype=np.float64),
    )
    mm = MmFrameResult(
        index=0,
        jax_energy_kcal=-50000.0,
        charmm_energy_kcal=-50000.01,
        jax_forces_kcal_A=np.ones((10, 3)) * 0.1,
        charmm_forces_kcal_A=np.ones((10, 3)) * 0.1,
        interaction_energy_kcal=0.5,
        mp2_interaction_eV=0.02,
    )
    row = compare_mm_to_mp2_frame(mm, frame)
    assert row["source_index"] == 7
    assert "mp2_jax_force_rmse_ev_A" in row
    assert "interaction_delta_eV" in row
    summary = aggregate_comparison([row, row])
    assert summary["n_frames"] == 2
    assert summary["mp2_jax_force_rmse_ev_A"]["n"] == 2

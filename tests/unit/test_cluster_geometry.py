"""Unit tests for cluster geometry helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cluster_geometry import (
    atoms_from_reference_npz,
    reference_frame_geometry,
)


def test_reference_frame_geometry_trajectory_npz(tmp_path: Path) -> None:
    path = tmp_path / "ref.npz"
    z_row = np.array([[6, 17, 17, 1, 1, 6, 17, 17, 1, 1]], dtype=int)
    r_row = np.random.default_rng(0).normal(size=(1, 10, 3))
    np.savez_compressed(
        path,
        N=np.array([10], dtype=int),
        Z=z_row,
        R=r_row,
        E_eV=np.array([-43.0]),
    )
    z, r = reference_frame_geometry(path, frame=0)
    assert z.shape == (10,)
    assert r.shape == (10, 3)
    np.testing.assert_array_equal(z, z_row[0])
    np.testing.assert_allclose(r, r_row[0])


def test_reference_frame_geometry_respects_n_per_frame(tmp_path: Path) -> None:
    path = tmp_path / "padded.npz"
    z_full = np.array([[6, 17, 17, 1, 1, 0, 0, 0, 0, 0]], dtype=int)
    r_full = np.arange(30, dtype=float).reshape(1, 10, 3)
    np.savez_compressed(
        path,
        N=np.array([5], dtype=int),
        Z=z_full,
        R=r_full,
    )
    z, r = reference_frame_geometry(path, frame=0)
    assert z.shape == (5,)
    assert r.shape == (5, 3)
    np.testing.assert_array_equal(z, z_full[0, :5])


def test_atoms_from_reference_npz(tmp_path: Path) -> None:
    pytest.importorskip("ase")
    path = tmp_path / "ref.npz"
    z_row = np.array([[6, 17, 17, 1, 1]], dtype=int)
    r_row = np.zeros((1, 5, 3))
    np.savez_compressed(path, N=np.array([5], dtype=int), Z=z_row, R=r_row)
    atoms = atoms_from_reference_npz(path)
    assert len(atoms) == 5
    np.testing.assert_array_equal(atoms.get_atomic_numbers(), z_row[0])


def test_ensure_monomer_3d_coords_breaks_collinear() -> None:
    from mmml.interfaces.pycharmmInterface.cluster_geometry import ensure_monomer_3d_coords

    flat = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=float)
    spread = ensure_monomer_3d_coords(flat)
    span = np.ptp(spread, axis=0)
    assert float(span[1]) >= 0.3
    assert float(span[2]) >= 0.3


def test_ensure_charmm_session_ready_sets_bomlev(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface import cluster_geometry as cg

    cg._charmm_session_ready = False
    calls: list[int] = []

    def _fake_apply(*, prnlev: int, warnlev: int, bomlev: int) -> dict[str, int]:
        calls.append(int(bomlev))
        return {"prnlev": prnlev, "warnlev": warnlev, "bomlev": bomlev}

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.apply_charmm_verbosity",
        _fake_apply,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.utils.set_up_directories",
        lambda: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.prepare_charmm_vacuum",
        lambda: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.reset_block",
        lambda: None,
    )

    cg.ensure_charmm_session_ready()
    assert calls == [-2]
    cg.ensure_charmm_session_ready()
    assert calls == [-2]

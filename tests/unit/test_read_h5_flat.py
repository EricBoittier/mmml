"""
Unit tests for flat (concatenated-atom) HDF5 loading and spooky batch construction.

Run:
  JAX_PLATFORMS=cpu pytest tests/unit/test_read_h5_flat.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

h5py = pytest.importorskip("h5py")
jax = pytest.importorskip("jax")
e3x = pytest.importorskip("e3x")

from mmml.models.physnetjax.physnetjax.data.read_h5 import (
    _concatenate_flat_data_dicts,
    _subset_flat_dataset,
    load_h5_flat,
    prepare_h5_datasets_flat,
)
from mmml.models.physnetjax.physnetjax.training.spooky_training import (
    build_spooky_batch_from_flat_data,
)


def _write_qcell_like_h5(
    path: Path,
    mols: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]],
) -> None:
    """Write mol_000001, ... with atomic_numbers, positions, formation_energy, total_forces."""
    with h5py.File(path, "w") as f:
        for i, (z, r, e, frc) in enumerate(mols, start=1):
            name = f"mol_{i:06d}"
            g = f.create_group(name)
            g.create_dataset("atomic_numbers", data=z.astype(np.int32))
            g.create_dataset("positions", data=r.astype(np.float64))
            g.create_dataset("formation_energy", data=np.float64(e))
            g.create_dataset("total_forces", data=frc.astype(np.float64))


def _synthetic_flat_dict() -> dict[str, np.ndarray]:
    """Two molecules: 2 atoms and 3 atoms."""
    R = np.zeros((5, 3), dtype=np.float64)
    Z = np.array([1, 6, 1, 6, 8], dtype=np.int32)
    F = np.zeros((5, 3), dtype=np.float64)
    mo = np.array([0, 2, 5], dtype=np.int32)
    return {
        "mol_offsets": mo,
        "R": R,
        "Z": Z,
        "F": F,
        "E": np.array([[1.0], [2.0]], dtype=np.float64),
        "N": np.array([[2], [3]], dtype=np.int32),
        "Q": np.array([[0.0], [0.0]], dtype=np.float64),
        "S": np.array([[1.0], [1.0]], dtype=np.float64),
    }


def test_concatenate_flat_data_dicts_merges_offsets(tmp_path: Path) -> None:
    d1 = {
        "mol_offsets": np.array([0, 2, 5], dtype=np.int32),
        "R": np.zeros((5, 3)),
        "Z": np.arange(5, dtype=np.int32),
        "F": np.zeros((5, 3)),
        "E": np.array([[0.0], [0.0]]),
        "N": np.array([[2], [3]], dtype=np.int32),
        "Q": np.zeros((2, 1)),
        "S": np.ones((2, 1)),
    }
    d2 = {
        "mol_offsets": np.array([0, 4], dtype=np.int32),
        "R": np.ones((4, 3)),
        "Z": np.ones(4, dtype=np.int32) * 7,
        "F": np.ones((4, 3)),
        "E": np.array([[3.0]]),
        "N": np.array([[4]], dtype=np.int32),
        "Q": np.zeros((1, 1)),
        "S": np.ones((1, 1)),
    }
    out = _concatenate_flat_data_dicts([d1, d2])
    assert out["mol_offsets"].tolist() == [0, 2, 5, 9]
    assert out["R"].shape == (9, 3)
    assert out["E"].shape == (3, 1)
    assert np.allclose(out["R"][:5, 0], 0.0)
    assert np.allclose(out["R"][5:, 0], 1.0)


def test_subset_flat_dataset_order() -> None:
    full = _synthetic_flat_dict()
    sub = _subset_flat_dataset(full, np.array([1, 0], dtype=np.int64))
    assert sub["mol_offsets"].tolist() == [0, 3, 5]
    assert sub["Z"].tolist() == [1, 6, 8, 1, 6]
    assert sub["E"].reshape(-1).tolist() == [2.0, 1.0]


def test_load_h5_flat_no_cache(tmp_path: Path) -> None:
    z1 = np.array([1, 1], dtype=np.int32)
    r1 = np.random.randn(2, 3)
    f1 = np.zeros((2, 3))
    z2 = np.array([6, 6, 8], dtype=np.int32)
    r2 = np.random.randn(3, 3)
    f2 = np.zeros((3, 3))
    _write_qcell_like_h5(
        tmp_path / "t.h5",
        [(z1, r1, -1.5, f1), (z2, r2, -2.5, f2)],
    )
    data = load_h5_flat(
        tmp_path / "t.h5",
        natoms=10,
        cache=False,
        verbose=False,
    )
    assert data["mol_offsets"].tolist() == [0, 2, 5]
    assert data["R"].shape == (5, 3)
    assert data["Z"].shape == (5,)
    assert data["E"].shape == (2, 1)
    assert np.all(np.diff(data["mol_offsets"]) == data["N"].flatten())


def test_prepare_h5_datasets_flat_split(tmp_path: Path) -> None:
    mols = []
    for _ in range(4):
        z = np.array([1, 6], dtype=np.int32)
        r = np.random.randn(2, 3)
        mols.append((z, r, float(np.random.randn()), np.zeros((2, 3))))
    _write_qcell_like_h5(tmp_path / "s.h5", mols)
    key = jax.random.PRNGKey(0)
    train, valid, natoms = prepare_h5_datasets_flat(
        key,
        filepath=tmp_path / "s.h5",
        train_size=2,
        valid_size=2,
        natoms=5,
        cache=False,
        verbose=False,
    )
    assert natoms == 5
    assert len(train["E"]) == 2
    assert len(valid["E"]) == 2
    assert train["mol_offsets"].shape == (3,)
    assert valid["mol_offsets"].shape == (3,)
    assert train["R"].shape[0] == 4
    assert valid["R"].shape[0] == 4


def test_build_spooky_batch_from_flat_data_pair_indices() -> None:
    d = _synthetic_flat_dict()
    batch = build_spooky_batch_from_flat_data(d, np.array([0, 1], dtype=np.int64))
    assert int(batch["batch_size"]) == 2
    assert batch["Z"].shape[0] == 5
    assert batch["Z"].shape[0] == batch["R"].shape[0] == batch["F"].shape[0]
    n0, n1 = 2, 3
    ld0, _ = e3x.ops.sparse_pairwise_indices(n0)
    ld1, _ = e3x.ops.sparse_pairwise_indices(n1)
    expected_pairs = np.asarray(ld0).size + np.asarray(ld1).size
    assert batch["dst_idx"].shape[0] == expected_pairs
    assert batch["batch_mask"].shape[0] == expected_pairs
    assert int(np.max(np.asarray(batch["dst_idx"]))) < 5
    assert np.allclose(np.asarray(batch["batch_mask"]), 1.0)
    seg = np.asarray(batch["batch_segments"])
    assert np.all(seg[:n0] == 0)
    assert np.all(seg[n0:] == 1)


def test_build_spooky_batch_empty_raises() -> None:
    d = _synthetic_flat_dict()
    with pytest.raises(ValueError, match="non-empty"):
        build_spooky_batch_from_flat_data(d, np.array([], dtype=np.int64))


def test_flat_no_zero_z_in_real_atoms(tmp_path: Path) -> None:
    z = np.array([1, 6], dtype=np.int32)
    r = np.zeros((2, 3))
    f = np.zeros((2, 3))
    _write_qcell_like_h5(tmp_path / "one.h5", [(z, r, 0.0, f)])
    data = load_h5_flat(tmp_path / "one.h5", natoms=2, cache=False)
    a0, a1 = int(data["mol_offsets"][0]), int(data["mol_offsets"][1])
    assert np.all(data["Z"][a0:a1] > 0)

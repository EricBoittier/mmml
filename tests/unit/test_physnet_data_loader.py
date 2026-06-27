"""Unit tests for PhysNetJAX NPZ dataset loading."""

import json

import jax
import numpy as np
import pytest

from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets


def test_prepare_datasets_skips_mmml_units_metadata(tmp_path):
    """fix-and-split embeds _mmml_units; loader must not crash on 0-d metadata."""
    n_samples, natoms = 8, 3
    payload = {
        "R": np.random.randn(n_samples, natoms, 3),
        "Z": np.tile(np.array([6, 1, 1], dtype=np.int32), (n_samples, 1)),
        "F": np.random.randn(n_samples, natoms, 3),
        "E": np.random.randn(n_samples),
        "N": np.full(n_samples, natoms, dtype=np.int32),
        "D": np.random.randn(n_samples, 3) * 0.05,
        "Q": np.zeros(n_samples, dtype=np.float64),
        "_mmml_units": np.array(json.dumps({"D": "e_angstrom"})),
    }
    npz_path = tmp_path / "train.npz"
    np.savez_compressed(npz_path, **payload)

    train_data, valid_data = prepare_datasets(
        jax.random.PRNGKey(0),
        train_size=5,
        valid_size=2,
        files=str(npz_path),
        natoms=natoms,
    )

    assert train_data["R"].shape == (5, natoms, 3)
    assert valid_data["R"].shape == (2, natoms, 3)
    assert train_data["Q"].shape == (5, 1)
    assert "_mmml_units" not in train_data


def test_prepare_datasets_q_reshaped_to_n_samples_by_one(tmp_path):
    n_samples, natoms = 4, 2
    payload = {
        "R": np.zeros((n_samples, natoms, 3)),
        "Z": np.tile(np.array([6, 1], dtype=np.int32), (n_samples, 1)),
        "F": np.zeros((n_samples, natoms, 3)),
        "E": np.zeros(n_samples),
        "N": np.full(n_samples, natoms, dtype=np.int32),
        "Q": np.arange(n_samples, dtype=np.float64),
    }
    npz_path = tmp_path / "train.npz"
    np.savez_compressed(npz_path, **payload)

    train_data, _ = prepare_datasets(
        jax.random.PRNGKey(1),
        train_size=3,
        valid_size=1,
        files=str(npz_path),
        natoms=natoms,
    )

    assert train_data["Q"].shape == (3, 1)

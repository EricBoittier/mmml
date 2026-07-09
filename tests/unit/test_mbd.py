from __future__ import annotations

import e3x
import jax
import jax.numpy as jnp
import numpy as np

from mmml.models.mbd import E3xMBDModel, mbd_energy_and_forces, qdo_pairwise_dispersion
from scripts.cache_qcml_mbd_orbax import preprocess_examples
from scripts.train_qcml_mbd import (
    bucket_indices,
    eligible_indices,
    limit_cache,
    make_batch,
)


def test_qdo_pairwise_dispersion_counts_directed_pair_once() -> None:
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(2)
    coefficients = jnp.tile(jnp.array([[6.0, 8.0, 10.0]]), (2, 1))
    damping_radii = jnp.ones(2)

    energy = qdo_pairwise_dispersion(
        positions,
        dst_idx,
        src_idx,
        coefficients,
        damping_radii,
    )

    expected = -(6.0 / 65.0 + 8.0 / 257.0 + 10.0 / 1025.0)
    np.testing.assert_allclose(energy, expected, rtol=1e-6)


def test_mbd_cache_join_and_padding() -> None:
    key_hash = np.array(b"same")
    geometry = {
        "key_hash": key_hash,
        "positions": np.zeros((2, 3), dtype=np.float32),
        "atomic_numbers": np.array([6, 1], dtype=np.uint8),
        "charge": np.array(0),
        "multiplicity": np.array(1),
    }
    c6 = {"key_hash": key_hash, "mbd_c6_coefficients": np.array([4.0, 1.0])}
    correction = {
        "key_hash": key_hash,
        "mbd_energy": np.array(-0.01),
        "mbd_forces": np.zeros((2, 3), dtype=np.float32),
    }
    alpha = {"key_hash": key_hash, "mbd_polarizabilities": np.array([3.0, 0.5])}

    cache = preprocess_examples([(geometry, c6, correction, alpha)])

    assert cache["R"].shape == (1, 2, 3)
    assert cache["C6_mbd"].shape == (1, 2)
    assert cache["alpha_mbd"].shape == (1, 2)
    assert cache["E_mbd"][0] == -0.01


def test_mbd_model_outputs_positive_properties_and_conservative_forces() -> None:
    cache = {
        "R": np.array([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]], dtype=np.float32),
        "Z": np.array([[6, 1]], dtype=np.int32),
        "Q": np.array([0.0], dtype=np.float32),
        "S": np.array([1.0], dtype=np.float32),
        "E_mbd": np.array([-0.01], dtype=np.float32),
        "F_mbd": np.zeros((1, 2, 3), dtype=np.float32),
        "C6_mbd": np.ones((1, 2), dtype=np.float32),
        "alpha_mbd": np.ones((1, 2), dtype=np.float32),
        "atom_mask": np.ones((1, 2), dtype=np.float32),
    }
    batch = make_batch(cache, np.array([0]), np.ones(1, dtype=np.float32))
    model = E3xMBDModel(features=4, num_iterations=1, num_basis_functions=3)
    keys = (
        "positions", "atomic_numbers", "charge", "spin", "dst_idx", "src_idx",
        "batch_segments", "batch_size", "atom_mask", "edge_mask",
    )
    inputs = {key: batch[key] for key in keys}
    variables = model.init(jax.random.key(0), **inputs)

    output, forces = mbd_energy_and_forces(model, variables["params"], **inputs)

    assert output["energy"].shape == (1,)
    assert output["c6_coefficients"].shape == (2,)
    assert output["polarizabilities"].shape == (2,)
    assert forces.shape == (2, 3)
    assert np.all(np.asarray(output["c6_coefficients"]) > 0)
    assert np.all(np.asarray(output["polarizabilities"]) > 0)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-6)


def test_mbd_training_cache_limit() -> None:
    cache = {
        "R": np.zeros((4, 2, 3)),
        "E_mbd": np.zeros(4),
        "scalar_metadata": np.array(4),
    }

    limited = limit_cache(cache, 2)

    assert limited["R"].shape[0] == 2
    assert limited["E_mbd"].shape[0] == 2
    assert limited["scalar_metadata"] == 4


def test_mbd_atom_buckets_crop_atomic_targets() -> None:
    atom_mask = np.array(
        [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]],
        dtype=np.float32,
    )
    cache = {
        "R": np.zeros((2, 6, 3), dtype=np.float32),
        "Z": np.zeros((2, 6), dtype=np.int32),
        "Q": np.zeros(2),
        "S": np.ones(2),
        "E_mbd": np.zeros(2),
        "F_mbd": np.zeros((2, 6, 3)),
        "C6_mbd": np.zeros((2, 6)),
        "alpha_mbd": np.zeros((2, 6)),
        "atom_mask": atom_mask,
    }
    indices = eligible_indices(cache, max_atoms=5)
    buckets = bucket_indices(cache, indices, bucket_width=2)
    batch = make_batch(
        cache,
        np.array([0]),
        np.ones(1, dtype=np.float32),
        max_atoms=2,
    )

    np.testing.assert_array_equal(indices, np.array([0, 1]))
    assert set(buckets) == {2, 6}
    assert batch["target_forces"].shape == (2, 3)
    assert batch["target_c6"].shape == (2,)

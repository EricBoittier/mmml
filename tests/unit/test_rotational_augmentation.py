from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mmml.models.EF.training import prepare_batches as prepare_ef_batches
from mmml.models.dcmnet.dcmnet.data import prepare_batches as prepare_dcmnet_batches
from mmml.utils.rotations import sample_random_rotations


def test_random_rotation_perturbation_zero_returns_identity() -> None:
    rots = sample_random_rotations(jax.random.PRNGKey(0), num=4, perturbation=0.0)
    expected = jnp.tile(jnp.eye(3, dtype=rots.dtype)[None, :, :], (4, 1, 1))
    np.testing.assert_allclose(np.asarray(rots), np.asarray(expected), atol=1e-6)


def test_ef_prepare_batches_rot_aug_perturbation_zero_is_identity() -> None:
    key = jax.random.PRNGKey(1)
    data = {
        "atomic_numbers": jnp.array([[1, 8], [6, 1]], dtype=jnp.int32),
        "positions": jnp.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "electric_field": jnp.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=jnp.float32),
        "energies": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "forces": jnp.array(
            [[[0.2, 0.0, 0.1], [0.0, -0.1, 0.3]], [[-0.2, 0.4, 0.0], [0.1, 0.0, -0.3]]],
            dtype=jnp.float32,
        ),
        "D": jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=jnp.float32),
    }
    batches_plain = prepare_ef_batches(key, data, batch_size=2, shuffle=False, rot_augment=False)
    batches_aug = prepare_ef_batches(
        key,
        data,
        batch_size=2,
        shuffle=False,
        rot_augment=True,
        rot_perturbation=0.0,
    )
    np.testing.assert_allclose(batches_plain[0]["positions"], batches_aug[0]["positions"], atol=1e-6)
    np.testing.assert_allclose(batches_plain[0]["forces"], batches_aug[0]["forces"], atol=1e-6)
    np.testing.assert_allclose(batches_plain[0]["electric_field"], batches_aug[0]["electric_field"], atol=1e-6)
    np.testing.assert_allclose(batches_plain[0]["dipoles"], batches_aug[0]["dipoles"], atol=1e-6)
    np.testing.assert_allclose(batches_plain[0]["energies"], batches_aug[0]["energies"], atol=1e-6)


def test_dcmnet_prepare_batches_rot_aug_keeps_scalar_targets() -> None:
    key = jax.random.PRNGKey(2)
    num_atoms = 2
    data = {
        "R": jnp.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=jnp.float32,
        ),
        "Z": jnp.array([[1, 8], [6, 1]], dtype=jnp.int32),
        "N": jnp.array([2, 2], dtype=jnp.int32),
        "mono": jnp.array([[0.1, -0.1], [0.2, -0.2]], dtype=jnp.float32),
        "esp": jnp.array([[0.01, 0.02], [0.03, 0.04]], dtype=jnp.float32),
        "vdw_surface": jnp.array(
            [
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0]],
            ],
            dtype=jnp.float32,
        ),
        "n_grid": jnp.array([2, 2], dtype=jnp.int32),
    }
    base = prepare_dcmnet_batches(key, data, batch_size=2, num_atoms=num_atoms, rot_augment=False)[0]
    aug = prepare_dcmnet_batches(
        key,
        data,
        batch_size=2,
        num_atoms=num_atoms,
        rot_augment=True,
        rot_perturbation=0.0,
    )[0]
    np.testing.assert_allclose(base["mono"], aug["mono"], atol=1e-6)
    np.testing.assert_allclose(base["esp"], aug["esp"], atol=1e-6)
    np.testing.assert_allclose(base["n_grid"], aug["n_grid"], atol=1e-6)

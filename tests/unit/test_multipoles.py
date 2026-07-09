from __future__ import annotations

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.multipoles import E3xMultipoleModel, irrep_blocks_to_traceless
from scripts.cache_qcml_multipoles_orbax import preprocess_examples


def test_irrep_blocks_to_traceless_shapes_and_traces() -> None:
    converted = irrep_blocks_to_traceless(jnp.arange(16, dtype=jnp.float32))

    assert converted["l0_irrep"].shape == (1,)
    assert converted["l1_dipole_tensor"].shape == (3,)
    assert converted["l2_quadrupole_tensor"].shape == (3, 3)
    assert converted["l3_octupole_tensor"].shape == (3, 3, 3)
    np.testing.assert_allclose(
        jnp.trace(converted["l2_quadrupole_tensor"]),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.trace(converted["l3_octupole_tensor"], axis1=-2, axis2=-1),
        jnp.zeros(3),
        atol=1e-6,
    )


def test_irrep_blocks_reject_wrong_width() -> None:
    with pytest.raises(ValueError, match="Expected 16"):
        irrep_blocks_to_traceless(jnp.zeros(15))


def test_e3x_multipole_model_output_shapes() -> None:
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.2, 0.8, 0.0]]
    )
    atomic_numbers = jnp.array([8, 1, 1])
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(3)
    model = E3xMultipoleModel(
        features=8,
        num_iterations=1,
        num_basis_functions=4,
    )
    arguments = (
        positions,
        atomic_numbers,
        jnp.array([0.0]),
        jnp.array([1.0]),
        dst_idx,
        src_idx,
    )

    variables = model.init(jax.random.key(0), *arguments)
    prediction = model.apply(variables, *arguments)

    assert prediction["multipoles"].shape == (1, 16)
    assert prediction["l1_dipole_tensor"].shape == (1, 3)
    assert prediction["l2_quadrupole_tensor"].shape == (1, 3, 3)
    assert prediction["l3_octupole_tensor"].shape == (1, 3, 3, 3)


def test_qcml_pair_preprocessing_joins_and_pads() -> None:
    irreps = {
        degree: jnp.arange(2 * degree + 1, dtype=jnp.float32)
        for degree in range(1, 4)
    }
    geometry = {
        "atomic_numbers": np.array([8, 1], dtype=np.uint8),
        "positions": np.zeros((2, 3), dtype=np.float32),
        "charge": np.array(-1, dtype=np.int64),
        "multiplicity": np.array(2, dtype=np.int64),
        "key_hash": np.array(b"same-key"),
    }
    moments = {
        "key_hash": np.array(b"same-key"),
        "pbe0_dipole": np.asarray(e3x.so3.irreps_to_tensor(irreps[1], degree=1)),
        "pbe0_quadrupole": np.asarray(e3x.so3.irreps_to_tensor(irreps[2], degree=2)),
        "pbe0_octupole": np.asarray(e3x.so3.irreps_to_tensor(irreps[3], degree=3)),
    }

    cache = preprocess_examples([(geometry, moments)])

    assert cache["R"].shape == (1, 2, 3)
    assert cache["atom_mask"].shape == (1, 2)
    assert cache["multipoles"].shape == (1, 16)
    assert cache["multipoles"][0, 0] == -1
    np.testing.assert_allclose(cache["multipoles"][0, 1:4], irreps[1], atol=1e-6)

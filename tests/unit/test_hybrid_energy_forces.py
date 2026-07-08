import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.jaxmdInterface.hybrid_energy import make_monomer_energy_fn


class DummyModel:
    def apply(
        self,
        params,
        *,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        compute_forces=True,
    ):
        del params, atomic_numbers, dst_idx, src_idx, compute_forces
        return {
            "energy": jnp.sum(positions * positions),
            "forces": jnp.zeros_like(positions),
        }


def test_monomer_energy_forces_follow_scalar_energy_not_model_forces():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    atomic_numbers = jnp.ones(4, dtype=jnp.int32)
    monomer_indices = [
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([2, 3], dtype=jnp.int32),
    ]

    def displacement_fn(a, b):
        return a - b

    energy_fn = make_monomer_energy_fn(
        DummyModel(),
        params={},
        jax_z=atomic_numbers,
        monomer_indices=monomer_indices,
        displacement_fn=displacement_fn,
    )

    forces = -jax.grad(energy_fn)(positions)

    np.testing.assert_allclose(
        np.asarray(forces),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0],
            ]
        ),
    )

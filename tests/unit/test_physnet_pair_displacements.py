"""PhysNet evaluated on pair vectors given by the caller (e.g. an MD engine)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from karml.models.physnetjax.physnetjax.models.model import PhysNet

N_ATOMS = 5


@pytest.fixture(scope="module")
def setup():
    model = PhysNet(
        features=8,
        max_degree=1,
        num_iterations=1,
        num_basis_functions=8,
        cutoff=5.0,
        max_atomic_number=9,
        charges=True,
        include_electrostatics=True,
        max_padded_atoms=N_ATOMS,
    )
    Z = jnp.array([8, 1, 1, 6, 1], dtype=jnp.int32)
    R = jnp.asarray(np.random.default_rng(0).uniform(0.0, 2.5, (N_ATOMS, 3)), jnp.float32)
    dst, src = (jnp.asarray(i, jnp.int32) for i in np.nonzero(~np.eye(N_ATOMS, dtype=bool)))
    params = model.init(jax.random.PRNGKey(0), Z, R, dst, src)

    def call(R, displacements=None, compute_forces=True):
        return model.apply(
            params,
            Z,
            R,
            dst,
            src,
            jnp.zeros(N_ATOMS, jnp.int32),
            1,
            jnp.ones(dst.shape[0]),
            jnp.ones(N_ATOMS),
            compute_forces=compute_forces,
            displacements=displacements,
        )

    return call, R, dst, src


def test_same_energy_and_forces(setup):
    call, R, dst, src = setup
    reference = call(R)
    displacements = R[src] - R[dst]
    energy = lambda d: call(R, d, compute_forces=False)["energy"].sum()  # noqa: E731
    np.testing.assert_allclose(energy(displacements), reference["energy"].sum(), rtol=1e-6)

    # forces through the chain rule d(R[src] - R[dst]) / dR
    gradient = jax.grad(energy)(displacements)
    from_displacements = jax.ops.segment_sum(gradient, dst, N_ATOMS) - jax.ops.segment_sum(gradient, src, N_ATOMS)
    np.testing.assert_allclose(from_displacements, reference["forces"], atol=1e-5)


def test_periodic_images(setup):
    # moving an atom by a lattice vector does not change the given pair vectors,
    # so the energy (and charges) must not change either
    call, R, dst, src = setup
    displacements = R[src] - R[dst]
    moved = R.at[0].add(jnp.array([7.0, 0.0, 0.0]))
    a = call(R, displacements, compute_forces=False)
    b = call(moved, displacements, compute_forces=False)
    np.testing.assert_allclose(a["energy"], b["energy"], rtol=1e-6)
    np.testing.assert_allclose(a["charges"], b["charges"], rtol=1e-6)


def test_forces_need_positions(setup):
    call, R, dst, src = setup
    with pytest.raises(ValueError, match="compute_forces=False"):
        call(R, R[src] - R[dst])

"""Mirror / inversion equivariance of EFieldPhysNet with pseudotensors.

With ``parity_correct_field`` and ``parity_correct_dipole`` the model must give
mirror images identical energies, mirrored forces and dipoles, and a mirrored
polarizability, and its pseudoscalar features must flip sign. The defaults keep
the historical field embedding (constant in the pseudoscalar slot, E in the
axial slot), which breaks this; ``test_default_field_embedding_breaks_parity``
documents that so a silent change of the default is caught.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from karml.models.efield.packed import PackSpec, iter_packed_batches
from karml.models.efield.training import EFieldPhysNet

S = np.diag([1.0, 1.0, -1.0])
CONV = 0.001 * 51.42206747632595  # Ef_input -> V/Å


@pytest.fixture(scope="module", autouse=True)
def _float32():
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def _model(**kw) -> EFieldPhysNet:
    return EFieldPhysNet(features=8, max_degree=2, num_iterations=2, num_basis_functions=8, cutoff=5.0,
                         include_pseudotensors=True, field_scale=0.001, zbl=False,
                         electrostatics_damping_sigma=4.0, packed=True, **kw)


def _chiral_molecule():
    """Irregular six-atom cluster (fixed seed): no improper symmetry, so strongly chiral."""
    Z = np.array([6, 7, 8, 9, 17, 1], np.int32)
    R = np.random.default_rng(7).normal(scale=1.1, size=(6, 3))
    return Z, R - R.mean(0)


def _batch(Z, R):
    d = {"Z": Z, "R": R.astype(np.float32), "F": np.zeros_like(R, np.float32), "E": np.zeros(1),
         "D": np.zeros((1, 3), np.float32), "polar": np.zeros((1, 3, 3), np.float32),
         "subset": np.zeros(1, np.int8), "offsets": np.array([0, len(Z)])}
    return {k: jnp.asarray(v) for k, v in next(iter_packed_batches(d, np.array([0]), PackSpec(2, 8, 64, 5.0))).items()}


def _params(model, b):
    p = model.init(jax.random.PRNGKey(0), atomic_numbers=b["atomic_numbers"], positions=b["positions"],
                   Ef=jnp.zeros((2, 3)), dst_idx_flat=b["dst_idx_flat"], src_idx_flat=b["src_idx_flat"],
                   batch_segments=b["batch_segments"], batch_size=2)["params"]
    # Output heads start at zero; randomise them so charges, dipoles and energy are non-trivial.
    leaves, tree = jax.tree_util.tree_flatten(p)
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    leaves = [jnp.where(jnp.all(x == 0), 0.3 * jax.random.normal(k, x.shape), x) for x, k in zip(leaves, keys)]
    return {"params": jax.tree_util.tree_unflatten(tree, leaves)}


def _outputs(model, params, Z, R, ef):
    b = _batch(Z, R)

    def apply(pos, e):
        (u, mu), st = model.apply(params, atomic_numbers=b["atomic_numbers"], positions=pos,
                                  Ef=jnp.zeros((2, 3)).at[0].set(e), dst_idx_flat=b["dst_idx_flat"],
                                  src_idx_flat=b["src_idx_flat"], batch_segments=b["batch_segments"],
                                  batch_size=2, mutable=["intermediates"], capture_intermediates=True)
        return u[0], mu[0], st["intermediates"]

    ef = jnp.asarray(ef, jnp.float32)
    u, mu, inter = apply(b["positions"], ef)
    forces = -jax.grad(lambda p: apply(p, ef)[0])(b["positions"])[: len(Z)]
    alpha = jax.jacfwd(lambda e: apply(b["positions"], e)[1])(ef) / CONV
    ps = np.asarray(inter["TensorDense_1"]["__call__"][0])[: len(Z), 1, 0, :]  # pseudoscalars after iteration 2
    return float(u), np.asarray(forces), np.asarray(mu), np.asarray(alpha), ps


@pytest.mark.parametrize("field", [0.0, 5.0])
def test_parity_correct_model_is_mirror_equivariant(field):
    Z, R = _chiral_molecule()
    model = _model(parity_correct_field=True, parity_correct_dipole=True)
    params = _params(model, _batch(Z, R))
    ef = field * np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    u0, f0, mu0, a0, p0 = _outputs(model, params, Z, R, ef)
    u1, f1, mu1, a1, p1 = _outputs(model, params, Z, R @ S, S @ ef)
    assert abs(u1 - u0) <= 1e-5 * max(abs(u0), 1.0)
    np.testing.assert_allclose(f1, f0 @ S, atol=1e-4 * np.abs(f0).max())
    np.testing.assert_allclose(mu1, S @ mu0, atol=1e-4 * np.abs(mu0).max())
    np.testing.assert_allclose(a1, S @ a0 @ S, atol=1e-4 * np.abs(a0).max())
    assert np.linalg.norm(p0) > 1e-4, "pseudoscalars should be non-zero for a chiral molecule"
    assert np.linalg.norm(p1 + p0) <= 1e-3 * np.linalg.norm(p0)


def test_parity_correct_model_is_rotation_equivariant():
    Z, R = _chiral_molecule()
    model = _model(parity_correct_field=True, parity_correct_dipole=True)
    params = _params(model, _batch(Z, R))
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    ef = np.array([1.0, 2.0, -1.5])
    u0, f0, mu0, a0, p0 = _outputs(model, params, Z, R, ef)
    u1, f1, mu1, a1, p1 = _outputs(model, params, Z, R @ q.T, q @ ef)
    assert abs(u1 - u0) <= 1e-5 * max(abs(u0), 1.0)
    np.testing.assert_allclose(f1, f0 @ q.T, atol=1e-4 * np.abs(f0).max())
    np.testing.assert_allclose(mu1, q @ mu0, atol=1e-4 * np.abs(mu0).max())
    np.testing.assert_allclose(a1, q @ a0 @ q.T, atol=1e-4 * np.abs(a0).max())
    assert np.linalg.norm(p1 - p0) <= 1e-3 * np.linalg.norm(p0)


def test_default_field_embedding_breaks_parity():
    """Historical default: pseudoscalars do not flip and the field response is not mirror-equivariant."""
    Z, R = _chiral_molecule()
    model = _model()
    params = _params(model, _batch(Z, R))
    _, _, _, a0, p0 = _outputs(model, params, Z, R, np.zeros(3))
    _, _, _, a1, p1 = _outputs(model, params, Z, R @ S, np.zeros(3))
    assert np.linalg.norm(p1 + p0) > 0.5 * np.linalg.norm(p0)          # mostly parity-even
    assert np.linalg.norm(a1 - S @ a0 @ S) > 1e-2 * np.linalg.norm(a0)

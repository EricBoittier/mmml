"""Regression test: gradients must flow through the DCMNet distributed-charge positions.

The l=1 block was read as ``x[:, 1, 1:4, :]`` on a parity axis of size 1.  The forward
pass clamps the index to 0, but jax 0.8 returns an exactly zero gradient through it, so
training never optimised the charge displacements and autodiff forces missed that term.
"""
import numpy as np
import jax
import jax.numpy as jnp

from karml.models.dcmnet.dcmnet.modules import DCMNetCharges


def _setup():
    n = 5
    Z = jnp.asarray([8, 6, 1, 1, 1], jnp.int32)
    R = jnp.asarray(np.random.default_rng(0).normal(size=(n, 3)) * 0.8, jnp.float32)
    ii, jj = np.nonzero(~np.eye(n, dtype=bool))
    model = DCMNetCharges(features=8, max_degree=1, num_iterations=2, num_basis_functions=8, cutoff=5.0, n_dcm=2)
    params = model.init(jax.random.PRNGKey(0), Z, R, ii, jj)
    return model, params, Z, R, ii, jj


def test_dipo_position_gradient_matches_finite_difference():
    model, params, Z, R, ii, jj = _setup()
    W = jnp.asarray(np.random.default_rng(1).normal(size=model.apply(params, Z, R, ii, jj)[1].shape), jnp.float32)

    def f(R_):  # displacement only (dipo minus the atom position), so the identity term cannot mask a lost gradient
        dipo = model.apply(params, Z, R_, ii, jj)[1]
        Rb = R_[:, None, :] if dipo.shape[-1] == 3 else R_[:, :, None]
        return jnp.sum(W * (dipo - jnp.broadcast_to(Rb, dipo.shape)))

    u = jnp.asarray(np.random.default_rng(2).normal(size=R.shape), jnp.float32)
    u = u / jnp.linalg.norm(u)
    ad = float(jnp.sum(jax.grad(f)(R) * u))
    h = 1e-2
    fd = float((f(R + h * u) - f(R - h * u)) / (2 * h))
    assert abs(fd) > 1e-6, fd
    assert abs(ad - fd) < 2e-2 * abs(fd) + 1e-6, (ad, fd)


def test_dipo_displacement_receives_parameter_gradient():
    model, params, Z, R, ii, jj = _setup()

    def disp_loss(p):
        dipo = model.apply(p, Z, R, ii, jj)[1]
        return jnp.sum((dipo - jnp.broadcast_to(R[:, None, :] if dipo.shape[-1] == 3 else R[:, :, None], dipo.shape)) ** 2)

    g = jax.grad(disp_loss)(params)
    total = sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree_util.tree_leaves(g))
    assert total > 0.0

"""Unit tests for JAX PBC molecular wrapping helpers."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
    cart_coords,
    frac_coords,
    group_ids_from_groups,
    wrap_groups,
    wrap_groups_by_id,
)


def _reference_wrap_groups_loop(R, groups, cell, mass=None):
    """Old per-group implementation used as a semantic oracle."""
    R_out = R
    for g in groups:
        if mass is not None:
            m_g = mass[g]
            com = jnp.sum(R[g] * m_g[:, None], axis=0, dtype=R.dtype) / jnp.sum(m_g)
        else:
            com = jnp.sum(R[g], axis=0, dtype=R.dtype) / g.shape[0]
        S_com = frac_coords(com[None, :], cell)[0]
        lattice_shift = -jnp.floor(S_com)
        cart_shift = cart_coords(lattice_shift[None, :], cell)[0]
        R_out = R_out.at[g].add(cart_shift)
    return R_out


@pytest.mark.unit
def test_group_ids_from_noncontiguous_groups() -> None:
    groups = [
        jnp.array([0, 2], dtype=jnp.int32),
        jnp.array([1, 3], dtype=jnp.int32),
    ]

    group_id = group_ids_from_groups(groups, n_atoms=4)

    np.testing.assert_array_equal(np.asarray(group_id), np.array([0, 1, 0, 1]))


@pytest.mark.unit
def test_wrap_groups_matches_reference_for_irregular_groups() -> None:
    cell = jnp.diag(jnp.array([10.0, 12.0, 14.0], dtype=jnp.float32))
    groups = [
        jnp.array([0, 2], dtype=jnp.int32),
        jnp.array([1, 3, 4], dtype=jnp.int32),
    ]
    R = jnp.array(
        [
            [9.5, 1.0, 1.0],
            [-1.0, 11.8, 2.0],
            [10.5, 1.5, 1.0],
            [-0.5, 12.2, 2.5],
            [-1.5, 11.4, 2.0],
            [15.0, 15.0, 15.0],
        ],
        dtype=jnp.float32,
    )

    actual = wrap_groups(R, groups, cell)
    expected = _reference_wrap_groups_loop(R, groups, cell)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual[5]), np.asarray(R[5]), atol=1e-6)


@pytest.mark.unit
def test_wrap_groups_by_id_matches_mass_weighted_reference() -> None:
    cell = jnp.diag(jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32))
    groups = [
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([2, 3], dtype=jnp.int32),
    ]
    group_id = group_ids_from_groups(groups, n_atoms=4)
    R = jnp.array(
        [
            [8.0, 1.0, 1.0],
            [12.0, 1.0, 1.0],
            [-2.0, 5.0, 5.0],
            [1.0, 5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    mass = jnp.array([1.0, 9.0, 4.0, 1.0], dtype=jnp.float32)

    actual = wrap_groups_by_id(R, group_id, len(groups), cell, mass=mass)
    expected = _reference_wrap_groups_loop(R, groups, cell, mass=mass)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)


@pytest.mark.unit
def test_wrap_groups_by_id_jitted_matches_eager() -> None:
    cell = jnp.diag(jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32))
    groups = [
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([2, 3], dtype=jnp.int32),
    ]
    group_id = group_ids_from_groups(groups, n_atoms=4)
    mass = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    R = jnp.array(
        [
            [10.5, 0.0, 0.0],
            [11.5, 0.0, 0.0],
            [2.0, -2.0, 5.0],
            [2.0, -1.0, 5.0],
        ],
        dtype=jnp.float32,
    )

    eager = wrap_groups_by_id(R, group_id, len(groups), cell, mass=mass)
    jitted = jax.jit(lambda pos: wrap_groups_by_id(pos, group_id, len(groups), cell, mass=mass))(R)

    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-6)

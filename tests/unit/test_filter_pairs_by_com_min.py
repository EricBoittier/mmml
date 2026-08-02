"""COM-distance pair filter: Cartesian contract + monomer indexing."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    _filter_pairs_by_com_min,
    _filter_pairs_by_com_min_jax,
)


def _two_monomer_system():
    """Two dimers along x; COM–COM distance = 5.0 Å in a 20 Å cubic cell."""
    r_cart = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    monomer_id = np.array([0, 0, 1, 1], dtype=np.int32)
    offsets = np.array([0, 2, 4], dtype=np.int32)
    box = np.diag([20.0, 20.0, 20.0])
    return r_cart, monomer_id, offsets, box


def test_numpy_filter_keeps_or_drops_by_com_distance():
    r_cart, monomer_id, offsets, box = _two_monomer_system()
    # Atom 1 (mol 0) ↔ atom 3 (mol 1): same monomers as 0↔2, COM dist 5 Å.
    pair_i = np.array([1], dtype=np.int32)
    pair_j = np.array([3], dtype=np.int32)
    mask = np.array([True])

    keep = _filter_pairs_by_com_min(
        r_cart, pair_i, pair_j, mask, offsets, monomer_id, 4.0, pbc_cell=box
    )
    drop = _filter_pairs_by_com_min(
        r_cart, pair_i, pair_j, mask, offsets, monomer_id, 6.0, pbc_cell=box
    )
    assert keep.tolist() == [True]
    assert drop.tolist() == [False]


def test_jax_filter_matches_numpy_on_cartesian_non_identity_atom_indices():
    """Must index COMs by monomer_id[atom], not by the atom index itself."""
    r_cart, monomer_id, offsets, box = _two_monomer_system()
    pair_i = np.array([1], dtype=np.int32)
    pair_j = np.array([3], dtype=np.int32)
    mask = np.array([True])

    expected = _filter_pairs_by_com_min(
        r_cart, pair_i, pair_j, mask, offsets, monomer_id, 4.0, pbc_cell=box
    )
    got = _filter_pairs_by_com_min_jax(
        jnp.asarray(r_cart),
        jnp.asarray(pair_i),
        jnp.asarray(pair_j),
        jnp.asarray(mask),
        jnp.asarray(monomer_id),
        4.0,
        2,
        pbc_cell=jnp.asarray(box),
    )
    np.testing.assert_array_equal(np.asarray(got), expected)

    got_drop = _filter_pairs_by_com_min_jax(
        jnp.asarray(r_cart),
        jnp.asarray(pair_i),
        jnp.asarray(pair_j),
        jnp.asarray(mask),
        jnp.asarray(monomer_id),
        6.0,
        2,
        pbc_cell=jnp.asarray(box),
    )
    assert bool(np.asarray(got_drop)[0]) is False


def test_jax_filter_fractional_positions_need_cartesian_conversion():
    """Callers must convert fractional R before the Cartesian MIC filter."""
    r_cart, monomer_id, _offsets, box = _two_monomer_system()
    r_frac = r_cart @ np.linalg.inv(box)
    pair_i = np.array([1], dtype=np.int32)
    pair_j = np.array([3], dtype=np.int32)
    mask = np.array([True])

    converted = _filter_pairs_by_com_min_jax(
        jnp.asarray(r_frac @ box),
        jnp.asarray(pair_i),
        jnp.asarray(pair_j),
        jnp.asarray(mask),
        jnp.asarray(monomer_id),
        4.0,
        2,
        pbc_cell=jnp.asarray(box),
    )
    assert bool(np.asarray(converted)[0]) is True

    # Fractional R + Cartesian MIC → tiny bogus distance → false drop.
    naive = _filter_pairs_by_com_min_jax(
        jnp.asarray(r_frac),
        jnp.asarray(pair_i),
        jnp.asarray(pair_j),
        jnp.asarray(mask),
        jnp.asarray(monomer_id),
        4.0,
        2,
        pbc_cell=jnp.asarray(box),
    )
    assert bool(np.asarray(naive)[0]) is False

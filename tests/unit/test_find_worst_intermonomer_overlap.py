"""`find_worst_intermonomer_overlap` must stay exact while getting fast.

It was an O(n_monomers^2) Python loop that called ``_mic`` -- and therefore
``numpy.linalg.inv`` on a *constant* cell -- once per monomer pair. A cProfile
of the ASE pre-minimisation on a 2,196-atom water box attributed 444 s of a
1,054 s job (42%) to this one function: 31 calls, 8,293,926 ``_mic`` calls,
8,293,954 matrix inversions, 26,512,505 ``numpy.round`` calls.

The rewrite vectorises over atoms and masks intra-monomer pairs. Speed is
worthless if the answer moves, so the original implementation is kept here
verbatim as ``_reference_pair_loop`` and the two are compared on random
geometries, with and without a periodic cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.utils.geometry_checks import (
    _cell_matrix,
    _mic,
    find_worst_intermonomer_overlap,
)


def _reference_pair_loop(positions, monomer_offsets, *, cell=None):
    """The original O(n_monomers^2) implementation, unchanged."""
    pos = np.asarray(positions, dtype=float)
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    if n_monomers <= 1:
        return float("inf"), None
    cell_mat = _cell_matrix(cell)
    best_dist = float("inf")
    best = None
    for mi in range(n_monomers):
        si, ei = int(offsets[mi]), int(offsets[mi + 1])
        ri = pos[si:ei]
        for mj in range(mi + 1, n_monomers):
            sj, ej = int(offsets[mj]), int(offsets[mj + 1])
            rj = pos[sj:ej]
            disp = _mic(ri[:, None, :] - rj[None, :, :], cell_mat)
            d2 = np.sum(disp * disp, axis=-1)
            flat_idx = int(np.argmin(d2))
            local_i, local_j = np.unravel_index(flat_idx, d2.shape)
            dist = float(np.sqrt(d2[local_i, local_j]))
            if dist < best_dist:
                best_dist = dist
                best = (mi, mj, si + int(local_i), sj + int(local_j), dist)
    return best_dist, best


def _random_box(rng, n_monomers, atoms_per, L):
    offsets = np.arange(n_monomers + 1) * atoms_per
    pos = rng.uniform(0.0, L, size=(n_monomers * atoms_per, 3))
    return pos, offsets


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("use_cell", [False, True])
def test_matches_reference_distance(seed, use_cell):
    rng = np.random.default_rng(seed)
    L = 12.0
    pos, offsets = _random_box(rng, n_monomers=9, atoms_per=3, L=L)
    cell = np.diag([L, L, L]) if use_cell else None

    ref_dist, ref = _reference_pair_loop(pos, offsets, cell=cell)
    got_dist, got = find_worst_intermonomer_overlap(pos, offsets, cell=cell)

    assert got_dist == pytest.approx(ref_dist, rel=0, abs=1e-10)
    assert got is not None and ref is not None
    # The identified contact must be the same pair of monomers.
    assert (got.monomer_i, got.monomer_j) == (ref[0], ref[1])
    # And the reported distance must belong to the reported atoms.
    d = pos[got.atom_i] - pos[got.atom_j]
    if cell is not None:
        d = _mic(d[None, :], cell)[0]
    assert float(np.linalg.norm(d)) == pytest.approx(got_dist, abs=1e-10)


def test_atoms_belong_to_the_reported_monomers():
    rng = np.random.default_rng(7)
    pos, offsets = _random_box(rng, n_monomers=6, atoms_per=4, L=10.0)
    _, got = find_worst_intermonomer_overlap(pos, offsets, cell=np.diag([10.0] * 3))
    assert got is not None
    assert offsets[got.monomer_i] <= got.atom_i < offsets[got.monomer_i + 1]
    assert offsets[got.monomer_j] <= got.atom_j < offsets[got.monomer_j + 1]
    assert got.monomer_i < got.monomer_j


def test_minimum_image_is_actually_applied():
    """Two atoms across a periodic face are close, not a box-length apart."""
    L = 10.0
    pos = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    offsets = np.array([0, 1, 2])
    d_pbc, _ = find_worst_intermonomer_overlap(pos, offsets, cell=np.diag([L] * 3))
    d_free, _ = find_worst_intermonomer_overlap(pos, offsets, cell=None)
    assert d_pbc == pytest.approx(1.0, abs=1e-10)
    assert d_free == pytest.approx(9.0, abs=1e-10)


def test_intra_monomer_contacts_are_ignored():
    """Two atoms of the SAME monomer may be very close without counting."""
    pos = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 0.0, 0.0]])
    offsets = np.array([0, 2, 3])  # monomer 0 = atoms 0,1 (0.01 A apart)
    dist, got = find_worst_intermonomer_overlap(pos, offsets, cell=None)
    assert dist == pytest.approx(4.99, abs=1e-10)
    assert got is not None and (got.monomer_i, got.monomer_j) == (0, 1)


def test_single_monomer_has_no_intermonomer_contact():
    pos = np.zeros((4, 3))
    assert find_worst_intermonomer_overlap(pos, np.array([0, 4])) == (float("inf"), None)


def test_chunking_does_not_change_the_answer():
    """Enough atoms to cross the internal row-chunk boundary."""
    rng = np.random.default_rng(11)
    pos, offsets = _random_box(rng, n_monomers=40, atoms_per=3, L=18.0)
    cell = np.diag([18.0] * 3)
    ref_dist, ref = _reference_pair_loop(pos, offsets, cell=cell)
    got_dist, got = find_worst_intermonomer_overlap(pos, offsets, cell=cell)
    assert got_dist == pytest.approx(ref_dist, abs=1e-10)
    assert (got.monomer_i, got.monomer_j) == (ref[0], ref[1])


def test_mic_accepts_precomputed_inverse():
    cell = np.diag([7.0, 8.0, 9.0])
    d = np.array([[6.5, 0.0, 0.0]])
    a = _mic(d, cell)
    b = _mic(d, cell, np.linalg.inv(cell))
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    assert float(np.linalg.norm(b[0])) == pytest.approx(0.5, abs=1e-12)

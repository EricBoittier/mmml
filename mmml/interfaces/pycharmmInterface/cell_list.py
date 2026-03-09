"""JAX-compatible cell-list neighbour search for molecular systems.

This module provides spatial-hashing-based pair generation that replaces the
naive O(N^2) all-pairs Cartesian product with an O(N)-scaling cell list.  Only
atoms in the same or adjacent cells (26 neighbours in 3-D) are considered,
dramatically reducing the number of pairs for large systems.

Key design decisions for JAX compatibility
------------------------------------------
* **Fixed-size output arrays** – ``max_pairs`` is pre-allocated so the function
  signature is static and amenable to ``jax.jit``.  Invalid slots are masked
  with ``pair_mask``.
* **PBC-aware** – fractional coordinates are used for cell assignment; the
  26-neighbour search wraps around the periodic boundaries.
* **NumPy pre-computation** – the cell list *build* step runs in NumPy so it
  can use dynamic shapes.  Only the final pair/mask arrays are converted to JAX.
  This means the cell list is rebuilt outside JIT and passed in as static data,
  which is the "build once, rebuild on request" approach.
"""

from __future__ import annotations

from itertools import product as _product
from typing import Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp
except ModuleNotFoundError:
    jnp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_groups_np(
    positions: np.ndarray,
    cell_matrix: np.ndarray,
    monomer_offsets: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrap each monomer as a unit into the primary cell (keeps molecules intact).

    For each monomer: wrap center of mass to [0,1)^3, apply same shift to all atoms in group.
    Uses mass-weighted COM when masses provided; otherwise uses mean of positions.
    """
    R = np.asarray(positions, dtype=np.float64).copy()
    inv_cell = np.linalg.inv(cell_matrix)
    n_monomers = len(monomer_offsets) - 1
    m = np.asarray(masses, dtype=np.float64) if masses is not None else None
    for mi in range(n_monomers):
        start, end = int(monomer_offsets[mi]), int(monomer_offsets[mi + 1])
        g = np.arange(start, end)
        if m is not None:
            m_g = m[g].reshape(-1, 1)
            com = (R[g] * m_g).sum(axis=0) / m_g.sum()
        else:
            com = R[g].mean(axis=0)
        frac_com = (inv_cell.T @ com)  # fractional coords of COM
        frac_wrapped = frac_com - np.floor(frac_com)
        com_wrapped = frac_wrapped @ cell_matrix
        shift = com_wrapped - com
        R[g] += shift
    return R


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cell_list(
    positions: np.ndarray,
    cell_matrix: np.ndarray,
    cutoff: float,
    *,
    monomer_offsets: Optional[np.ndarray] = None,
) -> dict:
    """Assign atoms to 3-D grid cells.

    Parameters
    ----------
    positions : (N, 3) array
        Cartesian atom positions.
    cell_matrix : (3, 3) array
        Lattice vectors as rows (or columns – we use ``inv`` so both work as
        long as you are consistent with how ``positions`` were generated).
    cutoff : float
        Interaction cutoff distance.  Cell side length ``>= cutoff``.
    monomer_offsets : (n_monomers+1,) int array, optional
        Cumulative atom offsets per monomer.  When provided the returned pair
        list will only contain *inter*-monomer pairs (atoms from different
        monomers).

    Returns
    -------
    dict with keys
        ``"cell_atoms"``   – list of lists: atoms in each cell
        ``"n_cells"``      – (3,) int array: grid dimensions
        ``"cell_matrix"``  – the input lattice
        ``"cutoff"``       – the input cutoff
        ``"monomer_offsets"`` – the input offsets (or None)
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell_matrix = np.asarray(cell_matrix, dtype=np.float64)
    N = positions.shape[0]
    inv_cell = np.linalg.inv(cell_matrix)

    # When monomer_offsets provided: wrap by molecule for binning (keeps molecules intact)
    if monomer_offsets is not None:
        positions = _wrap_groups_np(positions, cell_matrix, np.asarray(monomer_offsets))

    # Fractional coordinates for cell assignment
    frac = positions @ inv_cell.T
    frac = frac - np.floor(frac)  # wrap into [0, 1)

    # Grid dimensions (at least 3 cells per axis to avoid self-image)
    lengths = np.linalg.norm(cell_matrix, axis=1)
    n_cells = np.maximum(np.floor(lengths / cutoff).astype(int), 3)

    # Assign atoms to cells
    cell_idx = np.floor(frac * n_cells).astype(int)
    cell_idx = np.clip(cell_idx, 0, n_cells - 1)

    # Build cell -> atom list
    total_cells = int(np.prod(n_cells))
    cell_atoms: list[list[int]] = [[] for _ in range(total_cells)]
    for i in range(N):
        flat_idx = int(
            cell_idx[i, 0] * n_cells[1] * n_cells[2]
            + cell_idx[i, 1] * n_cells[2]
            + cell_idx[i, 2]
        )
        cell_atoms[flat_idx].append(i)

    return {
        "cell_atoms": cell_atoms,
        "n_cells": n_cells,
        "cell_matrix": cell_matrix,
        "cutoff": cutoff,
        "monomer_offsets": monomer_offsets,
    }


def cell_list_pairs(
    positions: np.ndarray,
    cell_matrix: np.ndarray,
    cutoff: float,
    max_pairs: int,
    *,
    monomer_offsets: Optional[np.ndarray] = None,
    atoms_per_monomer_list: Optional[list[int]] = None,
    exclude_intra_monomer: bool = True,
) -> Tuple:
    """Generate inter-monomer atom-atom pairs within *cutoff* using a cell list.

    Parameters
    ----------
    positions : (N, 3)
        Cartesian atom positions.
    cell_matrix : (3, 3)
        Lattice vectors.
    cutoff : float
        Pair-distance cutoff.
    max_pairs : int
        Maximum number of pairs to return.  Output arrays are padded to this
        size; invalid entries are masked out.
    monomer_offsets : (n_monomers+1,) int, optional
        Cumulative atom offsets.  Required when ``exclude_intra_monomer=True``.
    atoms_per_monomer_list : list[int], optional
        Atom count per monomer (redundant with offsets but convenient).
    exclude_intra_monomer : bool
        If ``True`` (default), exclude pairs where both atoms belong to the
        same monomer.

    Returns
    -------
    pair_i, pair_j : (max_pairs,) int32 arrays (JAX or NumPy)
        Atom indices for each pair.
    pair_mask : (max_pairs,) bool array
        ``True`` for valid pairs, ``False`` for padding.
    n_valid : int
        Number of valid pairs found.
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell_matrix = np.asarray(cell_matrix, dtype=np.float64)
    N = positions.shape[0]
    inv_cell = np.linalg.inv(cell_matrix)

    # Build cell list
    cl = build_cell_list(positions, cell_matrix, cutoff, monomer_offsets=monomer_offsets)
    cell_atoms = cl["cell_atoms"]
    n_cells = cl["n_cells"]

    # Build monomer-id lookup for intra-monomer exclusion
    monomer_id = None
    if exclude_intra_monomer and monomer_offsets is not None:
        n_monomers = len(monomer_offsets) - 1
        monomer_id = np.empty(N, dtype=np.int32)
        for mi in range(n_monomers):
            monomer_id[monomer_offsets[mi]:monomer_offsets[mi + 1]] = mi

    # Enumerate cell neighbours (self + 26 neighbours with PBC wrapping)
    neighbour_offsets = list(_product([-1, 0, 1], repeat=3))

    # Collect pairs
    pair_i_list = []
    pair_j_list = []
    cutoff_sq = cutoff * cutoff

    nx, ny, nz = int(n_cells[0]), int(n_cells[1]), int(n_cells[2])

    for cx in range(nx):
        for cy in range(ny):
            for cz in range(nz):
                flat_c = cx * ny * nz + cy * nz + cz
                atoms_c = cell_atoms[flat_c]
                if not atoms_c:
                    continue

                for dx, dy, dz in neighbour_offsets:
                    ncx = (cx + dx) % nx
                    ncy = (cy + dy) % ny
                    ncz = (cz + dz) % nz
                    flat_n = ncx * ny * nz + ncy * nz + ncz
                    atoms_n = cell_atoms[flat_n]
                    if not atoms_n:
                        continue

                    for ai in atoms_c:
                        for aj in atoms_n:
                            if ai >= aj:
                                continue  # avoid self and double counting

                            # Intra-monomer exclusion
                            if monomer_id is not None and monomer_id[ai] == monomer_id[aj]:
                                continue

                            # MIC distance check
                            dr = positions[ai] - positions[aj]
                            # Apply minimum image convention
                            frac_dr = dr @ inv_cell.T
                            frac_dr = frac_dr - np.round(frac_dr)
                            dr_mic = frac_dr @ cell_matrix
                            dist_sq = float(np.dot(dr_mic, dr_mic))

                            if dist_sq < cutoff_sq:
                                pair_i_list.append(ai)
                                pair_j_list.append(aj)

    n_valid = len(pair_i_list)

    # Pad to max_pairs
    if n_valid > max_pairs:
        # Truncate (shouldn't happen in practice if max_pairs is set correctly)
        import warnings
        warnings.warn(
            f"cell_list_pairs: found {n_valid} pairs but max_pairs={max_pairs}. "
            f"Truncating. Increase max_pairs for correctness.",
            RuntimeWarning,
            stacklevel=2,
        )
        pair_i_list = pair_i_list[:max_pairs]
        pair_j_list = pair_j_list[:max_pairs]
        n_valid = max_pairs

    pair_i = np.zeros(max_pairs, dtype=np.int32)
    pair_j = np.zeros(max_pairs, dtype=np.int32)
    pair_mask = np.zeros(max_pairs, dtype=bool)

    pair_i[:n_valid] = pair_i_list
    pair_j[:n_valid] = pair_j_list
    pair_mask[:n_valid] = True

    if jnp is not None:
        return jnp.array(pair_i), jnp.array(pair_j), jnp.array(pair_mask), n_valid
    return pair_i, pair_j, pair_mask, n_valid


def estimate_max_pairs(
    n_atoms: int,
    density_estimate: float = 0.03,
    cutoff: float = 10.0,
    safety_factor: float = 1.5,
) -> int:
    """Heuristic for ``max_pairs`` based on expected atom density.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms.
    density_estimate : float
        Approximate number density (atoms / A^3).  Default 0.03 is reasonable
        for condensed-phase organic liquids.
    cutoff : float
        Pair cutoff in Angstroms.
    safety_factor : float
        Multiplicative safety margin.

    Returns
    -------
    int : suggested ``max_pairs``.
    """
    import math
    vol_sphere = (4.0 / 3.0) * math.pi * cutoff**3
    avg_neighbours = density_estimate * vol_sphere
    # Each atom has ~avg_neighbours neighbours; total unique pairs ~ N * avg / 2
    est = int(n_atoms * avg_neighbours / 2.0 * safety_factor)
    return max(est, n_atoms)  # at least n_atoms pairs

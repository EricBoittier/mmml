"""jax_md-based neighbor list for MM pair generation.

Provides JAX-native, GPU-capable neighbor lists with incremental updates.
Falls back to the custom cell list when jax_md is unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp
    from jax_md import partition
    from jax_md import space
    _HAVE_JAX_MD = True
except ImportError:
    _HAVE_JAX_MD = False
    jnp = None


def have_jax_md() -> bool:
    """Return True if jax_md is available."""
    return _HAVE_JAX_MD


def _pbc_cell_to_box(pbc_cell: np.ndarray):
    """Convert pbc_cell (3x3, (3,), or scalar) to jax_md box format for space.periodic.

    space.periodic expects a float (cubic) or ndarray of shape [spatial_dim].
    A 3x3 diagonal matrix is converted to [Lx, Ly, Lz].
    """
    cell = np.asarray(pbc_cell, dtype=np.float64)
    if cell.ndim == 0:
        L = float(cell)
        return jnp.array([L, L, L])
    if cell.shape == (3,):
        return jnp.array(cell)
    if cell.shape == (3, 3):
        return jnp.array([float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])])
    raise ValueError(f"pbc_cell must be scalar, (3,) or (3,3); got {cell.shape}")


def create_jax_md_neighbor_list(
    pbc_cell: np.ndarray,
    r_cutoff: float,
    monomer_offsets: np.ndarray,
    dr_threshold: float = 0.5,
    capacity_multiplier: float = 1.25,
):
    """Create jax_md neighbor list with inter-monomer filtering.

    Returns (neighbor_fn, filter_inter_monomer_fn, monomer_id_jnp) or None if jax_md unavailable.
    """
    if not _HAVE_JAX_MD:
        return None

    box = _pbc_cell_to_box(pbc_cell)
    displacement, _ = space.periodic(box)

    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
        r_cutoff,
        dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier,
        format=partition.NeighborListFormat.OrderedSparse,
    )

    n_monomers = len(monomer_offsets) - 1
    monomer_id = np.empty(int(monomer_offsets[-1]), dtype=np.int32)
    for mi in range(n_monomers):
        monomer_id[int(monomer_offsets[mi]) : int(monomer_offsets[mi + 1])] = mi
    monomer_id_jnp = jnp.array(monomer_id)

    def filter_inter_monomer(idx):
        """Filter to inter-monomer pairs. idx shape (2, max_neighbors). Returns (pair_i, pair_j, mask)."""
        pair_i = idx[0]
        pair_j = idx[1]
        N = monomer_id_jnp.shape[0]
        valid_slot = (pair_i < N) & (pair_j < N) & (pair_i < pair_j)
        mid_i = jnp.where(pair_i < N, monomer_id_jnp[pair_i], -1)
        mid_j = jnp.where(pair_j < N, monomer_id_jnp[pair_j], -1)
        inter_monomer = mid_i != mid_j
        mask = valid_slot & inter_monomer
        return pair_i, pair_j, mask

    return neighbor_fn, filter_inter_monomer, monomer_id_jnp

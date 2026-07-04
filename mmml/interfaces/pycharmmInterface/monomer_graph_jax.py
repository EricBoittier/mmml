"""Monomer COM / pair helpers under PBC (legacy thin wrapper).

Prefer :func:`~mmml.interfaces.pycharmmInterface.calculator_utils.monomer_coms_segment`
for JIT-safe COMs.  This module keeps MIC pair-distance utilities for older scripts.
"""
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.calculator_utils import monomer_coms_segment
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
    cart_coords,
    frac_coords,
    group_ids_from_groups,
)

Array = jnp.ndarray


def monomer_COMs(R: Array, groups: list[Array], masses: Array | None, cell: Array) -> Array:
    """COMs for monomers (wrapped consistently), mass-weighted if masses provided."""
    from mmml.interfaces.pycharmmInterface.legacy_paths import DEPRECATED, warn_legacy

    warn_legacy(
        "mmml.interfaces.pycharmmInterface.monomer_graph_jax.monomer_COMs",
        DEPRECATED["mmml.interfaces.pycharmmInterface.monomer_graph_jax.monomer_COMs"],
    )
    group_id = group_ids_from_groups(groups, n_atoms=R.shape[0])
    n_groups = len(groups)
    if masses is None:
        coms = monomer_coms_segment(R, group_id, n_groups)
    else:
        coms = monomer_coms_segment(R, group_id, n_groups, masses=masses)
    # Wrap COMs into primary cell for consistency
    S = frac_coords(coms, cell)
    S = S - jnp.floor(S)
    return cart_coords(S, cell)


def monomer_pairwise_mic(coms: Array, cell: Array):
    """All-pairs MIC displacement and distances between monomer COMs."""
    dR = coms[None, :, :] - coms[:, None, :]
    dS = frac_coords(dR, cell)
    dS_mic = dS - jnp.round(dS)
    dR_mic = cart_coords(dS_mic, cell)
    dij = jnp.linalg.norm(dR_mic + 1e-18, axis=-1)
    return dR_mic, dij


def monomer_pairs_within_cutoff(coms: Array, cell: Array, cutoff: float):
    """Boolean mask (M,M) with True for pairs <= cutoff (excludes diagonal)."""
    _, dij = monomer_pairwise_mic(coms, cell)
    M = coms.shape[0]
    mask = (dij <= cutoff) & (~jnp.eye(M, dtype=bool))
    return mask

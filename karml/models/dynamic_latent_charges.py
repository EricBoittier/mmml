"""Pure scatter/weighted-average core for Mode E (``latent_dynamic``) charges.

See :mod:`mmml.models.mm_charge_mode` (Mode E) for the physical picture: for
each MD step, a monomer with several currently-active ML-dimer partners has
several independent per-atom ``q_ML`` estimates (one per pairwise AB forward),
weighted by that pair's ``ml_switch_scale`` and averaged.

This module holds only the generic "weighted scatter-average" arithmetic,
factored out of :mod:`mmml.interfaces.pycharmmInterface.mmml_calculator`
(``_aggregate_dynamic_latent_charges``) so it can be unit-tested without a
live model/CHARMM session -- that function does the geometry (COM
separations -> weights) and padding/index bookkeeping specific to the MD
calculator's dimer batch layout, then calls :func:`weighted_scatter_average`
here for the actual reduction.
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray

__all__ = ["weighted_scatter_average"]


def weighted_scatter_average(
    values: Array,
    global_idx: Array,
    weights: Array,
    mask: Array,
    n_atoms: int,
) -> Array:
    """Per-atom weighted average of per-slot local values, scattered to global atoms.

    Parameters
    ----------
    values
        ``(n_slots, max_atoms)`` local per-atom values (e.g. ``q_ML`` for each
        active dimer slot, in that dimer's local atom ordering).
    global_idx
        ``(n_slots, max_atoms)`` int, the global atom index each local atom
        slot maps to.
    weights
        ``(n_slots,)`` scalar weight per slot (e.g. ``ml_switch_scale`` of
        that dimer's COM separation), broadcast over its atoms.
    mask
        ``(n_slots, max_atoms)`` bool, local-atom validity (``False`` for
        padding).
    n_atoms
        Total number of global atoms to scatter onto.

    Returns
    -------
    ``(n_atoms,)`` weighted average value per global atom;  ``0`` for atoms
    that received zero total weight (no active slot referenced them, or all
    referencing slots had zero weight).
    """
    w = weights[:, None] * mask.astype(values.dtype)
    contrib = values * w

    flat_idx = global_idx.reshape(-1)
    flat_contrib = contrib.reshape(-1)
    flat_w = w.reshape(-1)

    value_sum = jnp.zeros((n_atoms,), dtype=values.dtype).at[flat_idx].add(flat_contrib)
    weight_sum = jnp.zeros((n_atoms,), dtype=values.dtype).at[flat_idx].add(flat_w)
    return value_sum / jnp.maximum(weight_sum, 1e-10)

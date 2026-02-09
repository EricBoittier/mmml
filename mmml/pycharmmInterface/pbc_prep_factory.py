# pbc_prep_factory.py
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from typing import Callable, Iterable, List, Optional, Sequence

from mmml.pycharmmInterface.pbc_utils_jax import coregister_groups, unwrap_groups, wrap_groups

Array = jnp.ndarray


class PBCMapper:
    """PBC mapper that keeps monomers intact and transforms forces via chain rule.

    map_positions(R): unwrap → coregister → wrap (molecular groups preserved).
    transform_forces(R, F_mapped): applies J^T to map forces from R_mapped space back to R.
    """

    def __init__(self, map_positions_fn: Callable[[Array], Array]):
        self._map_positions_fn = map_positions_fn

    def __call__(self, R: Array) -> Array:
        """Map positions into the primary cell (keeps monomers intact)."""
        return self._map_positions_fn(R)

    def map_positions(self, R: Array) -> Array:
        """Map positions into the primary cell (keeps monomers intact)."""
        return self._map_positions_fn(R)

    def transform_forces(self, R: Array, F_mapped: Array) -> Array:
        """Transform forces from R_mapped space back to R space (chain rule: F_orig = J^T F_mapped)."""
        _, vjp_fn = jax.vjp(self._map_positions_fn, R)
        F_orig = vjp_fn(F_mapped)[0]
        return F_orig


def _validate_cell(cell: Array) -> Array:
    cell = jnp.asarray(cell, dtype=jnp.float64)
    if cell.shape != (3, 3):
        raise ValueError(f"Cell must be (3,3); got {cell.shape}.")
    if not bool(jnp.all(jnp.isfinite(cell))):
        raise ValueError("Cell contains non-finite entries.")
    det = float(jnp.linalg.det(cell))
    if not math.isfinite(det) or det <= 0.0:
        raise ValueError(f"Cell determinant must be positive; got {det}.")
    return cell


def _groups_from_mol_id(mol_id: Iterable[int] | Array) -> List[Array]:
    mol_id_arr = jnp.asarray(mol_id, dtype=jnp.int32).reshape(-1)
    if mol_id_arr.size == 0:
        return []
    unique_ids = jnp.unique(mol_id_arr)
    return [jnp.where(mol_id_arr == m)[0] for m in unique_ids.tolist()]


def _groups_by_chunking(n_atoms: int, n_monomers: int) -> List[Array]:
    if n_monomers is None:
        raise ValueError("n_monomers must be provided when mol_id is None.")
    if n_atoms < n_monomers:
        raise ValueError(f"Cannot split {n_atoms} atoms into {n_monomers} monomers.")

    base = n_atoms // n_monomers
    remainder = n_atoms % n_monomers
    counts = [base + (1 if i < remainder else 0) for i in range(n_monomers)]

    groups: List[Array] = []
    idx = 0
    for count in counts:
        groups.append(jnp.arange(idx, idx + count, dtype=jnp.int32))
        idx += count
    return groups


def make_pbc_mapper(
    cell: Array,
    mol_id: Optional[Sequence[int] | Array],
    n_monomers: Optional[int] = None,
):
    """Factory for a jittable map that unwraps, coregisters, and rewraps positions under PBCs."""

    cell = _validate_cell(cell)

    static_groups: Optional[List[Array]]
    if mol_id is not None:
        static_groups = _groups_from_mol_id(mol_id)
        if not static_groups:
            raise ValueError("mol_id produced no groups.")
    else:
        static_groups = None
        if n_monomers is None:
            raise ValueError("Provide mol_id or n_monomers to build groups.")

    def _runtime_groups(n_atoms: int) -> List[Array]:
        if static_groups is not None:
            return static_groups
        return _groups_by_chunking(n_atoms, n_monomers)  # type: ignore[arg-type]

    @jax.jit
    def map_positions(R: Array) -> Array:
        if R.ndim != 2 or R.shape[1] != 3:
            raise ValueError(f"positions must be (N,3); got {R.shape}.")
        R = jax.lax.cond(
            jnp.all(jnp.isfinite(R)),
            lambda x: x,
            lambda x: jnp.full_like(x, jnp.nan),
            R,
        )
        groups = _runtime_groups(R.shape[0])
        R1 = unwrap_groups(R, groups, cell)
        R2 = coregister_groups(R1, groups, cell)
        R3 = wrap_groups(R2, groups, cell)
        return R3

    return PBCMapper(map_positions)

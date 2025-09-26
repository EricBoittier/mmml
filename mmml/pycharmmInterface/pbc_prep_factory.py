# pbc_prep_factory.py
import jax
import jax.numpy as jnp
from typing import Optional, Sequence
from pbc_utils_jax import wrap_positions, unwrap_groups, coregister_groups

Array = jnp.ndarray

def make_pbc_mapper(cell: Array,
                    mol_id: Optional[Array],
                    n_monomers: Optional[int] = None):
    """
    Build a jittable function map_positions(R) that:
      1) unwraps each monomer (group) locally,
      2) co-registers all monomers into the same image,
      3) wraps the final coordinates into the primary cell.
    Groups are derived from mol_id if provided, otherwise from equal chunking with n_monomers.
    """
    cell = jnp.asarray(cell)

    if mol_id is None:
        assert n_monomers is not None, "Provide mol_id or n_monomers"
        def groups_from_counts(N):
            counts = jnp.full((n_monomers,), N // n_monomers, dtype=jnp.int32)
            counts = counts.at[: (N % n_monomers)].add(1)
            idxs = jnp.arange(N, dtype=jnp.int32)
            splits = jnp.cumsum(counts)[:-1]
            return jnp.split(idxs, splits.tolist())
        static_groups = None
    else:
        mol_id = jnp.asarray(mol_id, dtype=jnp.int32)
        # Build python lists of indices per monomer (static for JIT if mol_id size fixed)
        def _groups_from_molid(molid):
            mmax = int(jnp.max(molid)) + 1
            groups = [jnp.where(molid == m)[0] for m in range(mmax)]
            return groups
        static_groups = _groups_from_molid(mol_id)

    @jax.jit
    def map_positions(R: Array) -> Array:
        if static_groups is None:
            groups = groups_from_counts(R.shape[0])
        else:
            groups = static_groups
        R1 = unwrap_groups(R, groups, cell)
        R2 = coregister_groups(R1, groups, cell)
        R3 = wrap_positions(R2, cell)
        return R3

    return map_positions

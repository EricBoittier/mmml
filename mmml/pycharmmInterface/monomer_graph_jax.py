# monomer_graph_jax.py
import jax
import jax.numpy as jnp
from pbc_utils_jax import frac_coords, cart_coords

Array = jnp.ndarray

@jax.jit
def monomer_COMs(R: Array, groups: list[Array], masses: Array | None, cell: Array) -> Array:
    """COMs for monomers (wrapped consistently), mass-weighted if masses provided."""
    if masses is None:
        coms = jnp.stack([R[g].mean(axis=0) for g in groups], axis=0)
    else:
        coms = jnp.stack([ (masses[g][:,None]*R[g]).sum(0) / masses[g].sum() for g in groups ], axis=0)
    # Wrap COMs into primary cell for consistency
    S = frac_coords(coms, cell)
    S = S - jnp.floor(S)
    return cart_coords(S, cell)

@jax.jit
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
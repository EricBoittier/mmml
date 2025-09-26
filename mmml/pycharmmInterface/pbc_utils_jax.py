# pbc_utils_jax.py
import jax
import jax.numpy as jnp

Array = jnp.ndarray

@jax.jit
def frac_coords(R: Array, cell: Array) -> Array:
    """Cartesian -> fractional (row-vectors)."""
    CinvT = jnp.linalg.inv(cell).T
    return R @ CinvT

@jax.jit
def cart_coords(S: Array, cell: Array) -> Array:
    """Fractional -> Cartesian (row-vectors)."""
    return S @ cell.T

@jax.jit
def wrap_positions(R: Array, cell: Array) -> Array:
    """Wrap positions into the primary cell [0,1)^3 (C∞ in interiors; piecewise at faces)."""
    S = frac_coords(R, cell)
    S_wrapped = S - jnp.floor(S)
    return cart_coords(S_wrapped, cell)

@jax.jit
def mic_displacement(Ri: Array, Rj: Array, cell: Array) -> Array:
    """
    Minimum-image displacement vector r_j - r_i under PBC.
    Works for any parallelepiped cell.
    """
    dR = Rj - Ri
    dS = frac_coords(dR, cell)
    dS_mic = dS - jnp.round(dS)
    return cart_coords(dS_mic, cell)

@jax.jit
def pairwise_mic(R: Array, cell: Array):
    """All-pairs MIC displacement and distance. Returns (dR_ij, d_ij)."""
    dR = R[None, :, :] - R[:, None, :]
    dS = frac_coords(dR, cell)
    dS_mic = dS - jnp.round(dS)
    dR_mic = cart_coords(dS_mic, cell)
    dij = jnp.linalg.norm(dR_mic + 1e-18, axis=-1)  # tiny eps avoids NaNs at i=j
    return dR_mic, dij

@jax.jit
def unwrap_group(R: Array, idx: Array, cell: Array) -> Array:
    """
    Unwrap a *single* monomer so its atoms are contiguous:
    uses the first atom in idx as a reference.
    """
    ref = idx[0]
    dR = R[idx] - R[ref]
    dS = frac_coords(dR, cell)
    dS_mic = dS - jnp.round(dS)
    return R[ref] + cart_coords(dS_mic, cell)

def unwrap_groups(R: Array, groups: list[Array], cell: Array) -> Array:
    """Unwrap all groups and return positions in original atom order."""
    # Build concatenated list in group order, then invert to original order
    unwrapped_list = [unwrap_group(R, g, cell) for g in groups]
    R_concat = jnp.concatenate(unwrapped_list, axis=0)
    order = jnp.concatenate(groups, axis=0)
    inv = jnp.empty_like(order)
    inv = inv.at[order].set(jnp.arange(order.shape[0]))
    return R_concat[inv]

def coregister_groups(R: Array, groups: list[Array], cell: Array) -> Array:
    """
    Place all groups in a common image:
    anchor group 0; for each other group, shift by the MIC of COMs.
    """
    # COMs (equal masses; swap for mass-weighting if needed)
    coms = jnp.stack([R[g].mean(axis=0) for g in groups], axis=0)
    anchor = coms[0]
    def shift_group(carry_R, inputs):
        m, g = inputs
        dv = coms[m] - anchor
        # MIC shift to bring com[m] near anchor
        dS = frac_coords(dv, cell)
        shift = -jnp.round(dS)
        dR = cart_coords(shift, cell)
        return carry_R.at[g].add(dR), None
    R_out, _ = jax.lax.scan(shift_group, R, (jnp.arange(len(groups)), jnp.array(groups, dtype=object)))
    return R_out

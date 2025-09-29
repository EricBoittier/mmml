# pbc_utils_jax.py
import jax
import jax.numpy as jnp
import jax.scipy.linalg

Array = jnp.ndarray


@jax.jit
def frac_coords(R: Array, cell: Array) -> Array:
    """Cartesian -> fractional (row-vectors) using a stable linear solve."""
    S_T = jax.scipy.linalg.solve(cell.T, R.T, assume_a='gen')
    return S_T.T


@jax.jit
def cart_coords(S: Array, cell: Array) -> Array:
    """Fractional -> Cartesian (row-vectors)."""
    return S @ cell


@jax.jit
def wrap_positions(R: Array, cell: Array) -> Array:
    """Wrap positions into the primary cell [0,1)^3 (C∞ in interiors; piecewise at faces)."""
    S = frac_coords(R, cell)
    S_wrapped = S - jnp.floor(S)
    return cart_coords(S_wrapped, cell)


@jax.jit
def mic_displacement(Ri: Array, Rj: Array, cell: Array) -> Array:
    """Minimum-image displacement vector r_j - r_i under PBC."""
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
    """Unwrap a *single* group so its atoms are contiguous; uses idx[0] as reference."""
    ref = idx[0]
    dR = R[idx] - R[ref]
    dS = frac_coords(dR, cell)
    dS_mic = dS - jnp.round(dS)
    return R[ref] + cart_coords(dS_mic, cell)


def unwrap_groups(R: Array, groups: list[Array], cell: Array) -> Array:
    """Unwrap all groups and return positions in original atom order."""
    unwrapped_list = [unwrap_group(R, g, cell) for g in groups]
    R_concat = jnp.concatenate(unwrapped_list, axis=0)
    order = jnp.concatenate(groups, axis=0)
    inv = jnp.empty_like(order)
    inv = inv.at[order].set(jnp.arange(order.shape[0]))
    return R_concat[inv]


def coregister_groups(R: Array, groups: list[Array], cell: Array) -> Array:
    """Place all groups in a common image by aligning COMs via MIC shifts."""
    coms = [R[g].mean(axis=0) for g in groups]
    anchor = coms[0]

    R_out = R
    for i, g in enumerate(groups):
        if i == 0:
            continue
        dv = coms[i] - anchor
        dS = frac_coords(dv[None, :], cell)[0]
        shift_frac = -jnp.round(dS)
        shift = cart_coords(shift_frac[None, :], cell)[0]
        R_out = R_out.at[g].add(shift)

    return R_out

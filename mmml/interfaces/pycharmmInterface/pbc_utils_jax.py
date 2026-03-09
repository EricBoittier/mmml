# pbc_utils_jax.py
"""PBC utilities for JAX. Includes smooth MIC for differentiable optimization under PBC."""
import jax
import jax.numpy as jnp
import jax.scipy.linalg

Array = jnp.ndarray

# Sharpness of smooth MIC transition (higher = closer to exact round, narrower transition)
SMOOTH_MIC_K = 20.0


def _smooth_round_frac(frac: Array, k: float = SMOOTH_MIC_K) -> Array:
    """Smooth approximation to round for fractional part in [0, 1).

    Replaces the discontinuous round at frac=0.5 with a tanh transition.
    Output: 0 when frac < 0.5, 1 when frac > 0.5, smooth in between.
    """
    return 0.5 + 0.5 * jnp.tanh(k * (frac - 0.5))


def _smooth_frac_to_mic(frac: Array, k: float = SMOOTH_MIC_K) -> Array:
    """Smooth fractional part to (-0.5, 0.5] for MIC.

    Replaces: frac - round(frac) which jumps at 0.5.
    Uses: frac - 0.5 - 0.5*tanh(k*(frac-0.5)) for smooth transition.
    """
    return frac - 0.5 - 0.5 * jnp.tanh(k * (frac - 0.5))


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
    """Wrap positions into the primary cell [0,1)^3 (C∞ in interiors; piecewise at faces).
    Wraps each atom individually - use wrap_groups for molecular wrapping."""
    S = frac_coords(R, cell)
    S_wrapped = S - jnp.floor(S)
    return cart_coords(S_wrapped, cell)


def wrap_groups(R: Array, groups: list[Array], cell: Array) -> Array:
    """Wrap each group (monomer) as a unit into the primary cell.
    Keeps monomers intact: shift all atoms by the same integer-lattice translation
    so the group's CoM lands in [0,1)^3.

    Numerically stable: the shift is ``cart_coords(-floor(S_com))``, which is
    exactly zero for in-box molecules (floor==0) and an exact lattice vector
    otherwise. This avoids the frac->cart round-trip noise of the naive
    ``cart_coords(S_wrapped) - com`` formulation."""
    R_out = R
    for g in groups:
        com = jnp.sum(R[g], axis=0, dtype=R.dtype) / g.shape[0]
        S_com = frac_coords(com[None, :], cell)[0]
        lattice_shift = -jnp.floor(S_com)
        cart_shift = cart_coords(lattice_shift[None, :], cell)[0]
        R_out = R_out.at[g].add(cart_shift)
    return R_out


@jax.jit
def mic_displacement(Ri: Array, Rj: Array, cell: Array) -> Array:
    """Minimum-image displacement vector r_j - r_i under PBC."""
    dR = Rj - Ri
    dS = frac_coords(dR, cell)
    dS_mic = dS - jnp.round(dS)
    return cart_coords(dS_mic, cell)


@jax.jit
def mic_displacement_smooth(Ri: Array, Rj: Array, cell: Array, k: float = SMOOTH_MIC_K) -> Array:
    """Smooth (differentiable) MIC displacement. Use for BFGS/minimization under PBC."""
    dR = Rj - Ri
    dS = frac_coords(dR, cell)
    frac = dS - jnp.floor(dS)  # fractional part in [0, 1)
    dS_mic = _smooth_frac_to_mic(frac, k)
    return cart_coords(dS_mic, cell)


@jax.jit
def mic_displacements_batched(positions_dst: Array, positions_src: Array, cell: Array) -> Array:
    """MIC displacement for batched pairs. positions_dst/src shape (n_edges, 3)."""
    dR = positions_src - positions_dst
    dS = frac_coords(dR, cell)
    dS_mic = dS - jnp.round(dS)
    return cart_coords(dS_mic, cell)


@jax.jit
def mic_displacements_batched_smooth(
    positions_dst: Array, positions_src: Array, cell: Array, k: float = SMOOTH_MIC_K
) -> Array:
    """Smooth MIC displacement for batched pairs. Use for minimization under PBC."""
    dR = positions_src - positions_dst
    dS = frac_coords(dR, cell)
    frac = dS - jnp.floor(dS)
    dS_mic = _smooth_frac_to_mic(frac, k)
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

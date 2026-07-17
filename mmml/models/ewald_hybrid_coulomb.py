"""Hybrid training: full-box jit-native Ewald Coulomb (LJ off).

Same contract as :mod:`mmml.models.nvalchemiops_hybrid_coulomb`'s
``hybrid_nvalchemiops_pme_coulomb_energy`` -- a drop-in alternative that needs
no external PME library (pure JAX, via :mod:`mmml.interfaces.pycharmmInterface.
ewald_native`), useful wherever ``nvalchemiops`` isn't installed (e.g. CPU-only
clusters) or a dependency-free reference is preferred::

    E_MM = E_ewald(all atoms, q_CGenFF)

No exclusion lists, no intra-monomer subtraction, no COM MM taper on this
term -- one Ewald sum over the whole charged (padded) structure, all pairs
included via O(N^2) minimum-image distances (cheap at training-time batch
sizes; not intended for large liquid boxes). Intramolecular Coulomb is
intentionally left in (the model trains against that same operator) -- see
``mm_nonbonded.py``'s docstring for the same design choice at MD time.

Forces come from ``jax.value_and_grad`` of this energy (see
:func:`mmml.models.hybrid_energy.hybrid_forward`) -- unlike ``jax_pme``, this
has no host callback, so plain autodiff works.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.ewald_native import (
    build_kspace_integers,
    default_ewald_alpha,
    ewald_reciprocal_energy,
    ewald_self_energy,
)
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

Array = jnp.ndarray
COULOMB_KCAL = 332.063711

__all__ = ["hybrid_ewald_coulomb_energy"]


def hybrid_ewald_coulomb_energy(
    positions: Array,
    mol_id: Array,
    charges: Array,
    *,
    box_length_A: float,
    accuracy: float = 1e-6,
    real_space_cutoff_A: float | None = None,
    mm_switch_on: float = 0.0,
    mm_switch_width: float = 0.0,
    ml_switch_width: float = 0.0,
    complementary_handoff: bool = True,
    n_monomers: int = 2,
) -> Array:
    """Full-box Ewald Coulomb for one padded structure (kcal/mol).

    Padding atoms (``mol_id < 0``) contribute zero charge. Switch / monomer
    kwargs are accepted for call-site compatibility with the MIC hybrid path
    (and the nvalchemiops variant) but are **not** applied -- many-to-many
    Ewald does not taper this term.

    ``box_length_A``/``accuracy``/``real_space_cutoff_A`` must all be static
    Python values (part of the frozen ``HybridConfig``, a jit static arg) --
    the k-space integer grid is built once, host-side, from them.
    """
    del mm_switch_on, mm_switch_width, ml_switch_width, complementary_handoff
    del n_monomers

    mid = jnp.asarray(mol_id).reshape(-1)
    q = jnp.asarray(charges).reshape(-1)
    pos = jnp.asarray(positions)
    valid = mid >= 0
    q_full = jnp.where(valid, q, 0.0)

    L = float(box_length_A)
    cell_np = np.diag([L, L, L])
    cell = jnp.asarray(cell_np)

    # accuracy -> exponent: erfc(x) ~ 1e-accuracy_digits at x ~ sqrt(-ln(accuracy)).
    # Reuses the same "accuracy_exponent" convention as ewald_native's default.
    # `accuracy` is a static Python float (jit static arg), so plain math here
    # keeps this out of the traced graph entirely.
    accuracy_exponent = math.sqrt(max(-math.log(float(accuracy)), 1.0))
    rcut_for_alpha = float(real_space_cutoff_A) if real_space_cutoff_A is not None else L / 2.0
    alpha = default_ewald_alpha(rcut_for_alpha, accuracy_exponent=accuracy_exponent)
    n_int = jnp.asarray(build_kspace_integers(cell_np, alpha, accuracy_exponent=accuracy_exponent))

    # all-pairs MIC distance matrix, built from the single-pair primitive
    # directly (pbc_utils_jax.pairwise_mic mis-shapes frac_coords's batched
    # solve for this (N,N,3) case -- a pre-existing bug, unrelated to this
    # module; tracked separately rather than relied on here).
    disp = jax.vmap(jax.vmap(lambda a, b: mic_displacement(a, b, cell), in_axes=(None, 0)), in_axes=(0, None))(pos, pos)
    disp_sq = jnp.sum(disp * disp, axis=-1)
    # padding atoms are all placed at the same (0, 0, 0) -- clamp the SQUARED
    # distance before sqrt, not after: jnp.linalg.norm's gradient is 0/0 (NaN)
    # exactly at disp=0, so sqrt must never see a true zero on any traced
    # path, even one whose *value* later gets masked to zero by qq=0. Masking
    # only the output (dij/energy) does not stop the NaN subgradient from
    # propagating through the unmasked branch during backprop.
    coincident = disp_sq < 1e-18
    safe_sq = jnp.where(coincident, 1.0, disp_sq)
    dij = jnp.sqrt(safe_sq)
    qq = q_full[:, None] * q_full[None, :]
    real_mat = qq * jax.scipy.special.erfc(alpha * dij) / dij
    real_mat = jnp.where(coincident, 0.0, real_mat)
    if real_space_cutoff_A is not None:
        real_mat = jnp.where(dij < float(real_space_cutoff_A), real_mat, 0.0)
    e_real = 0.5 * jnp.sum(real_mat) * COULOMB_KCAL

    e_recip = ewald_reciprocal_energy(pos, q_full, cell, n_int, alpha) * COULOMB_KCAL
    e_self = ewald_self_energy(q_full, alpha) * COULOMB_KCAL

    return e_real + e_recip + e_self

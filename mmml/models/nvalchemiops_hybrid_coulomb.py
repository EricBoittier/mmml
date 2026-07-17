"""Hybrid training: cross-monomer nvalchemiops PME Coulomb (LJ off).

Matches the jax-pme hybrid correction pattern used on the MD side::

    E_MM = s(r_com) * (E_pme(all) - Σ_m E_pme(monomer_m))

Intra-monomer PME is subtracted so ML intramolecular electrostatics are not
double-counted. Forces should come from ``jax.value_and_grad`` of this energy
(see :func:`mmml.models.hybrid_energy.hybrid_forward`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.calculator_utils import mm_switch_scale
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    nvalchemiops_pme_coulomb_energy_jax,
)
from mmml.models.cgenff_mm import monomer_centroids

Array = jnp.ndarray

__all__ = [
    "hybrid_nvalchemiops_pme_coulomb_energy",
]


def hybrid_nvalchemiops_pme_coulomb_energy(
    positions: Array,
    mol_id: Array,
    charges: Array,
    *,
    box_length_A: float,
    accuracy: float = 1e-6,
    real_space_cutoff_A: float | None = None,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_switch_width: float,
    complementary_handoff: bool = True,
    n_monomers: int = 2,
) -> Array:
    """Switched cross-monomer PME Coulomb for one padded dimer (kcal/mol).

    Padding atoms (``mol_id < 0``) contribute zero charge. Monomers (a single
    occupied ``mol_id``) have no intermolecular Coulomb and return 0.
    """
    mid = jnp.asarray(mol_id).reshape(-1)
    q = jnp.asarray(charges).reshape(-1)
    pos = jnp.asarray(positions)
    valid = mid >= 0
    q_full = jnp.where(valid, q, 0.0)

    def _pme(q_use):
        return nvalchemiops_pme_coulomb_energy_jax(
            pos,
            q_use,
            box_length_A=float(box_length_A),
            accuracy=float(accuracy),
            real_space_cutoff_A=real_space_cutoff_A,
            compute_forces=False,
        )

    e_full = _pme(q_full)

    def _intra_one(m):
        q_m = jnp.where(valid & (mid == m), q, 0.0)
        occupied = jnp.any(valid & (mid == m))
        return jnp.where(occupied, _pme(q_m), 0.0)

    e_intra = jnp.sum(
        jax.vmap(_intra_one)(jnp.arange(int(n_monomers), dtype=mid.dtype))
    )
    e_cross = e_full - e_intra

    coms = monomer_centroids(pos, mid, n_monomers=int(n_monomers))
    d_com = coms[1] - coms[0]
    r_com = jnp.sqrt(jnp.maximum(jnp.sum(d_com * d_com), 1e-20))
    scale = mm_switch_scale(
        r_com,
        mm_switch_on=mm_switch_on,
        mm_switch_width=mm_switch_width,
        ml_switch_width=ml_switch_width,
        complementary_handoff=complementary_handoff,
    )
    # Need at least two occupied monomer ids for an intermolecular term.
    def _occupied(m):
        return jnp.any(valid & (mid == m)).astype(jnp.int32)

    n_occ = jnp.sum(
        jax.vmap(_occupied)(jnp.arange(int(n_monomers), dtype=mid.dtype))
    )
    has_inter = n_occ >= 2
    return jnp.where(has_inter, scale * e_cross, 0.0)

"""Hybrid training: full-box nvalchemiops PME Coulomb (LJ off).

Matches the fast MD ``periodic_external`` / many-to-many Ewald path::

    E_MM = E_pme(all atoms, q_CGenFF)

No exclusion lists, no intra-monomer subtraction, no COM MM taper on this
term — one PME call over the whole charged system.  Intramolecular Coulomb
is intentionally left in (the model trains against that same operator).

Forces should come from ``jax.value_and_grad`` of this energy
(see :func:`mmml.models.hybrid_energy.hybrid_forward`).
"""

from __future__ import annotations

import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.long_range_backend import (
    nvalchemiops_pme_coulomb_energy_jax,
)

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
    mm_switch_on: float = 0.0,
    mm_switch_width: float = 0.0,
    ml_switch_width: float = 0.0,
    complementary_handoff: bool = True,
    n_monomers: int = 2,
) -> Array:
    """Full-box PME Coulomb for one padded structure (kcal/mol).

    Padding atoms (``mol_id < 0``) contribute zero charge.  Switch / monomer
    kwargs are accepted for call-site compatibility with the MIC hybrid path
    but are **not** applied — MD many-to-many Ewald does not taper this term.
    """
    del mm_switch_on, mm_switch_width, ml_switch_width, complementary_handoff
    del n_monomers

    mid = jnp.asarray(mol_id).reshape(-1)
    q = jnp.asarray(charges).reshape(-1)
    pos = jnp.asarray(positions)
    valid = mid >= 0
    q_full = jnp.where(valid, q, 0.0)

    return nvalchemiops_pme_coulomb_energy_jax(
        pos,
        q_full,
        box_length_A=float(box_length_A),
        accuracy=float(accuracy),
        real_space_cutoff_A=real_space_cutoff_A,
        compute_forces=False,
    )

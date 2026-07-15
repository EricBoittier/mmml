"""Fixed-coefficient pairwise dispersion (QDO baseline) for rigid-body sampling.

Per-atom C6/C8/C10 and damping radii are frozen at build time (from an MBD
checkpoint prediction or injected via ``EnergyContext.options``). The neural
MBD model is not re-evaluated during sampling; only
:func:`mmml.models.mbd.qdo_pairwise_dispersion` runs on the current geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mmml.md.energy.capacity import COMPUTE_DTYPE
from mmml.md.energy.registry import EnergyContext, NeighborRequest, TermFns, register_term
from mmml.md.system import MolecularSystem

__all__ = ["MBDDispersionTerm", "DEFAULT_MBD_CUTOFF_A"]

DEFAULT_MBD_CUTOFF_A = 12.0
HARTREE_TO_EV = 27.211386245988


@register_term("mbd")
class MBDDispersionTerm:
    """Frozen C6/C8/C10 pairwise dispersion (registered name ``mbd``)."""

    name = "mbd"

    def __init__(self, cutoff_A: float | None = None):
        self.cutoff_A = cutoff_A

    def neighbor_request(self, system: MolecularSystem):
        cut = float(self.cutoff_A) if self.cutoff_A is not None else DEFAULT_MBD_CUTOFF_A
        return NeighborRequest(cutoff_A=cut, kind="intermolecular")

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        from mmml.models.mbd.qdo import qdo_pairwise_dispersion

        opts = dict(ctx.options)
        fixed = opts.get("fixed_dispersion")
        if fixed is None:
            raise ValueError(
                "mbd term requires ctx.options['fixed_dispersion'] "
                "(coefficients_per_atom shape (N,3), damping_radii shape (N,))"
            )

        coeffs_atom = jnp.asarray(fixed["coefficients_per_atom"], dtype=COMPUTE_DTYPE)
        damp_atom = jnp.asarray(fixed["damping_radii"], dtype=COMPUTE_DTYPE)
        if coeffs_atom.shape[0] != system.n_atoms or damp_atom.shape[0] != system.n_atoms:
            raise ValueError("fixed_dispersion arrays must match system.n_atoms")
        weight = float(opts.get("mbd_weight", fixed.get("weight", 1.0)))
        mol_id = jnp.asarray(system.mol_id, dtype=jnp.int32)
        cutoff = float(
            self.cutoff_A
            if self.cutoff_A is not None
            else opts.get("mbd_cutoff", DEFAULT_MBD_CUTOFF_A)
        )

        def energy_fn(R, *, pair_i=None, pair_j=None, pair_mask=None, box=None, **kwargs) -> Any:
            if pair_i is None or pair_j is None:
                raise ValueError("mbd jax face requires pair_i/pair_j (neighbor list)")
            pi = jnp.asarray(pair_i, dtype=jnp.int32)
            pj = jnp.asarray(pair_j, dtype=jnp.int32)
            pos = jnp.asarray(R, dtype=COMPUTE_DTYPE)

            inter = mol_id[pi] != mol_id[pj]
            undirected = pi < pj
            keep = inter & undirected
            if pair_mask is not None:
                keep = keep & (jnp.asarray(pair_mask) > 0)

            # Geometric-mean combining rules for pair coefficients / damping.
            c_pair = jnp.sqrt(jnp.maximum(coeffs_atom[pi] * coeffs_atom[pj], 0.0))
            d_pair = 0.5 * (damp_atom[pi] + damp_atom[pj])

            # Zero out padded / intramolecular edges for qdo's dst<src filter.
            # qdo already counts dst_idx < src_idx once; feed only kept edges by
            # masking coefficients to zero on discarded pairs.
            c_pair = jnp.where(keep[:, None], c_pair, 0.0)
            d_pair = jnp.where(keep, d_pair, 1.0)

            e_ha = qdo_pairwise_dispersion(pos, pi, pj, c_pair, d_pair)
            return jnp.asarray(weight, dtype=COMPUTE_DTYPE) * e_ha * HARTREE_TO_EV

        return TermFns(
            jax_energy_fn=energy_fn,
            neighbor_request=NeighborRequest(cutoff_A=cutoff, kind="intermolecular"),
        )

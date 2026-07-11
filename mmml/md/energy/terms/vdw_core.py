"""Short-range repulsive LJ "core" wall between a core group and molecule groups.

When intergroup interactions are handled by an ML model, this term adds back only
the *repulsive* branch of the CHARMM LJ potential (smoothly switched to zero at a
short cutoff) between a leading "core" atom group and a set of molecule groups, to
stop atoms collapsing into ML blind spots.

Extracted and generalized from ``examples/cg_jaxmd.py``
(``_smooth_cutoff_weight``, ``compute_peptide_water_core_vdw_energy``): the
peptide/water specifics are lifted into constructor args — ``n_trialanine`` ->
``n_core_atoms``, ``_pw_water_indices_jax`` -> ``group_indices``, ``_eps_jax`` /
``_rmin_jax`` -> ``lj_epsilon`` / ``lj_rmin_half`` (the builder populates these
from ``FFParams``; passed explicitly here so the term is testable without CHARMM).
Assumes an orthorhombic cell (minimum image via the box diagonal), matching the
original.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.system import MolecularSystem

__all__ = ["RepulsiveCoreVdwTerm"]


def _pair_lj_epsilon(ep_i, ep_j):
    """CHARMM geometric ε_ij = sqrt(|ε_i ε_j|)."""
    import jax.numpy as jnp

    return jnp.sqrt(jnp.abs(ep_i * ep_j))


def _smooth_cutoff_weight(dist, cutoff, width):
    """Smoothstep taper: 1 below (cutoff-width), 0 above cutoff."""
    import jax.numpy as jnp

    width = float(max(0.0, min(float(width), float(cutoff))))
    if width <= 0.0:
        return jnp.where(dist < cutoff, 1.0, 0.0)
    switch_on = float(cutoff) - width
    t = jnp.clip((dist - switch_on) / width, 0.0, 1.0)
    smoothstep = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    return 1.0 - smoothstep


@register_term("vdw_core")
class RepulsiveCoreVdwTerm:
    """Repulsive LJ wall between a leading core atom group and molecule groups.

    The core group occupies the first ``n_core_atoms`` indices; ``group_indices``
    is ``(n_groups, atoms_per_group)`` for the other molecules (e.g. solvent).
    Only the repulsive LJ branch is kept, smoothly switched off past ``cutoff_A``.
    """

    name = "vdw_core"

    def __init__(
        self,
        n_core_atoms: int,
        group_indices: Sequence[Sequence[int]],
        lj_epsilon: Sequence[float],
        lj_rmin_half: Sequence[float],
        cutoff_A: float,
        switch_width_A: float,
    ):
        self.n_core_atoms = int(n_core_atoms)
        self.group_indices = np.asarray(group_indices, dtype=int)
        self.lj_epsilon = np.asarray(lj_epsilon, dtype=float)
        self.lj_rmin_half = np.asarray(lj_rmin_half, dtype=float)
        self.cutoff_A = float(cutoff_A)
        self.switch_width_A = float(switch_width_A)

    def neighbor_request(self, system: MolecularSystem):
        # Dense core × group block; padding handled via the optional
        # ``active_group_slots``/``active_group_mask`` kwargs, not a driver list.
        return None

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        if system.box is None:
            raise ValueError("vdw_core requires a periodic box (orthorhombic).")

        n_core = self.n_core_atoms
        group_idx = jnp.asarray(self.group_indices, dtype=jnp.int32)
        eps = jnp.asarray(self.lj_epsilon)
        rmin_half = jnp.asarray(self.lj_rmin_half)
        box_diag = jnp.asarray(np.diag(np.asarray(system.box)))
        cutoff = self.cutoff_A
        width = self.switch_width_A
        core_idx = jnp.arange(n_core, dtype=jnp.int32)

        def energy_fn(R, *, active_group_slots=None, active_group_mask=None, **kwargs) -> Any:
            if group_idx.shape[0] == 0:
                return jnp.asarray(0.0)

            if active_group_slots is None:
                groups = group_idx
                mask = jnp.ones((group_idx.shape[0],))
            else:
                groups = group_idx[jnp.asarray(active_group_slots, dtype=jnp.int32)]
                mask = jnp.asarray(active_group_mask)

            core_pos = R[:n_core]
            group_pos = R[groups]
            disp = group_pos[:, None, :, :] - core_pos[None, :, None, :]
            disp = disp - box_diag * jnp.round(disp / box_diag)
            dist = jnp.sqrt(jnp.maximum(jnp.sum(disp * disp, axis=-1), 1e-12))

            ep = _pair_lj_epsilon(
                eps[core_idx][None, :, None],
                eps[groups][:, None, :],
            )
            sig = rmin_half[core_idx][None, :, None] + rmin_half[groups][:, None, :]
            sig_r6 = (sig / jnp.maximum(dist, 1e-10)) ** 6
            vdw_full = ep * (sig_r6 * sig_r6 - 2.0 * sig_r6)
            repulsive = jnp.maximum(vdw_full, 0.0)
            weights = _smooth_cutoff_weight(dist, cutoff, width)
            return jnp.sum(mask[:, None, None] * weights * repulsive) * KCAL_MOL_TO_EV

        return TermFns(jax_energy_fn=energy_fn, ase_contribution=None, neighbor_request=None)

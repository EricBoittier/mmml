"""Short-range repulsive LJ "core" wall for ML-treated peptide-water pairs.

Extracted from ``examples/cg_jaxmd.py`` (``_smooth_cutoff_weight``,
``compute_peptide_water_core_vdw_energy``). When peptide-water interactions are
handled by the ML model, this term adds back only the *repulsive* branch of the
CHARMM LJ potential (smoothly switched to zero at a short cutoff) to stop atoms
collapsing into ML blind spots.

CHARMM globals lifted to constructor args: ``n_trialanine`` -> ``n_peptide``,
``_pw_water_indices_jax`` -> ``water_indices``, ``_eps_jax`` / ``_rmin_jax`` ->
``lj_epsilon`` / ``lj_rmin_half`` (the builder populates these from
``FFParams``; passed explicitly here so the term is testable without CHARMM).
Assumes an orthorhombic cell (minimum image via the box diagonal), matching the
original.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.system import MolecularSystem

__all__ = ["PeptideWaterCoreVdwTerm"]


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
class PeptideWaterCoreVdwTerm:
    """Repulsive LJ wall between the peptide and ML-treated waters."""

    name = "vdw_core"

    def __init__(
        self,
        n_peptide: int,
        water_indices: Sequence[Sequence[int]],
        lj_epsilon: Sequence[float],
        lj_rmin_half: Sequence[float],
        cutoff_A: float,
        switch_width_A: float,
    ):
        self.n_peptide = int(n_peptide)
        self.water_indices = np.asarray(water_indices, dtype=int)
        self.lj_epsilon = np.asarray(lj_epsilon, dtype=float)
        self.lj_rmin_half = np.asarray(lj_rmin_half, dtype=float)
        self.cutoff_A = float(cutoff_A)
        self.switch_width_A = float(switch_width_A)

    def neighbor_request(self, system: MolecularSystem):
        # Dense peptide × water block; padding handled via the optional
        # ``active_water_slots``/``active_water_mask`` kwargs, not a driver list.
        return None

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        if system.box is None:
            raise ValueError("vdw_core requires a periodic box (orthorhombic).")

        n_pep = self.n_peptide
        water_idx = jnp.asarray(self.water_indices, dtype=jnp.int32)
        eps = jnp.asarray(self.lj_epsilon)
        rmin_half = jnp.asarray(self.lj_rmin_half)
        box_diag = jnp.asarray(np.diag(np.asarray(system.box)))
        cutoff = self.cutoff_A
        width = self.switch_width_A
        pep_idx = jnp.arange(n_pep, dtype=jnp.int32)

        def energy_fn(R, *, active_water_slots=None, active_water_mask=None, **kwargs) -> Any:
            if water_idx.shape[0] == 0:
                return jnp.asarray(0.0)

            if active_water_slots is None:
                waters = water_idx
                mask = jnp.ones((water_idx.shape[0],))
            else:
                waters = water_idx[jnp.asarray(active_water_slots, dtype=jnp.int32)]
                mask = jnp.asarray(active_water_mask)

            pep_pos = R[:n_pep]
            water_pos = R[waters]
            disp = water_pos[:, None, :, :] - pep_pos[None, :, None, :]
            disp = disp - box_diag * jnp.round(disp / box_diag)
            dist = jnp.sqrt(jnp.maximum(jnp.sum(disp * disp, axis=-1), 1e-12))

            ep = _pair_lj_epsilon(
                eps[pep_idx][None, :, None],
                eps[waters][:, None, :],
            )
            sig = rmin_half[pep_idx][None, :, None] + rmin_half[waters][:, None, :]
            sig_r6 = (sig / jnp.maximum(dist, 1e-10)) ** 6
            vdw_full = ep * (sig_r6 * sig_r6 - 2.0 * sig_r6)
            repulsive = jnp.maximum(vdw_full, 0.0)
            weights = _smooth_cutoff_weight(dist, cutoff, width)
            return jnp.sum(mask[:, None, None] * weights * repulsive) * KCAL_MOL_TO_EV

        return TermFns(jax_energy_fn=energy_fn, ase_contribution=None, neighbor_request=None)

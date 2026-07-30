"""Electrostatic embedding: ML-predicted fluctuating charges against MM charges.

This is the term that makes solvent reorganisation happen. With fixed
force-field charges on the solute (mechanical embedding) the solvent cannot
respond to charge being created, so an Sn2 reaction that converts neutral
reactants into an ion pair gets almost none of its catalysis. Here the solute's
partial charges are predicted by the ML model at every step and therefore change
along the reaction coordinate: for NH3 + CH3Cl the chloride charge goes from
about -0.5 e at the transition state to -0.9 e once the ion pair forms, and the
solvent feels that.

Because the charges are a function of the coordinates, ``dq/dR`` contributes to
the force. The energy is written as a plain jittable function of ``R`` and the
forces are taken with ``jax.grad`` over the whole expression, so that term is
included automatically -- a hand-written force that differentiated only the
``1/r`` factor would be missing real physics.

Double counting
---------------
``mm_nonbonded`` also computes Coulomb over the same intermolecular pairs. Set
the solute's charges to zero in :class:`~mmml.md.system.FFParams` when using
this term, which removes the solute's electrostatics from ``mm_nonbonded``
without touching its Lennard-Jones (LJ reads epsilon / Rmin, not charge). The
builder in ``examples/menshutkin/jaxmd_box.py`` does this via
``solute_charges="ml"``.

Charge neutrality
-----------------
``charge_mode="q0"`` subtracts the mean residual so the ML charges on the solute
sum exactly to ``total_charge``. Neural-network charges are only approximately
conserving, and in a periodic box a small net charge on the solute is a physical
error that grows with system size rather than a rounding detail.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.md.energy.registry import EnergyContext, NeighborRequest, TermFns, register_term
from mmml.md.energy.terms._common import ase_contribution_from_jax, resolve_ml_model
from mmml.md.system import MolecularSystem

__all__ = ["MLMMElectrostaticTerm"]

# CHARMM's CCELEC, kcal*A/(mol*e^2).
COULOMB_KCAL = 332.063711


@register_term("ml_mm_elec")
class MLMMElectrostaticTerm:
    """Coulomb between ML-charged solute atoms and fixed-charge MM atoms."""

    name = "ml_mm_elec"

    def __init__(
        self,
        ml_atoms: Sequence[int] | None = None,
        model: Any = None,
        params: Any = None,
        cutoff_A: float = 12.0,
        switch_on_A: float = 10.0,
        charge_mode: str = "q0",
        total_charge: float = 0.0,
        damping_sigma_A: float = 1.0,
        charge_clip: float | None = 1.0,
        charge_gradient: bool = True,
        capacity_hint: int | None = None,
    ):
        self.ml_atoms = None if ml_atoms is None else tuple(int(a) for a in ml_atoms)
        self.model = model
        self.params = params
        self.cutoff_A = float(cutoff_A)
        self.switch_on_A = float(switch_on_A)
        # Short-range bound on the solute-solvent Coulomb. Without it the term is
        # a bare 1/r between an ML charge and an MM point charge, and MM
        # hydrogens have almost no Lennard-Jones core to stop the collapse --
        # TIP3's H has Rmin/2 = 0.2245 A, so against a chloride approaching -0.9 e
        # the attraction is tens of kcal/mol with nothing opposing it. Observed
        # directly: a water H reached 1.586 A from the solute Cl and dragged its
        # oxygen into the methyl group (0.640 A H-H) before the energy diverged.
        # erf(r/sigma)/r is the same damping the models in mmml/models use for
        # their own learned-charge electrostatics; at sigma = 1 A it is within
        # 2.5 % of 1/r beyond 1.6 A, so it bounds the singularity without
        # touching the physical interaction range. Set 0 to disable.
        self.damping_sigma_A = float(damping_sigma_A)
        if self.damping_sigma_A < 0.0:
            raise ValueError(
                f"damping_sigma_A must be >= 0 (got {self.damping_sigma_A})"
            )
        if self.switch_on_A >= self.cutoff_A:
            raise ValueError(
                f"switch_on_A ({self.switch_on_A}) must be below cutoff_A ({self.cutoff_A})"
            )
        if charge_mode not in ("raw", "q0"):
            raise ValueError(f"charge_mode must be raw|q0 (got {charge_mode!r})")
        self.charge_mode = charge_mode
        self.total_charge = float(total_charge)
        # Bound on |q| for the embedding charges.
        #
        # The solute-solvent coupling is a feedback loop: the solvent pulls the
        # chloride out, the model responds with more charge transfer, the larger
        # charge pulls the solvent in harder. Measured, its gain exceeds unity
        # somewhere near 0.6 of full coupling -- ramping the coupling on in 5,
        # 10 and 20 stages failed at 0.80, 0.70 and 0.60 respectively, i.e. the
        # more time spent near the threshold the sooner it is found, which is a
        # threshold rather than a rate. q(Cl) reached -1.035 just before the
        # collapse.
        #
        # Clipping caps the gain exactly where the runaway passes through, and
        # costs nothing in normal operation: a hard clip has unit gradient
        # inside the bound, so charges in their physical range are untouched and
        # dq/dR survives there. The reference dipoles never imply more than
        # about 1 e of charge separation, so +-1 is the physical bound, not a
        # tuning knob. Set None to disable.
        self.charge_clip = None if charge_clip is None else float(charge_clip)
        if self.charge_clip is not None and self.charge_clip <= 0:
            raise ValueError(f"charge_clip must be positive (got {self.charge_clip})")
        # Whether dq/dR contributes to the force.
        #
        # True is the physically complete form and the reason this term exists:
        # the solvent feels the charges, and the charges' response to geometry
        # feeds back into the forces.
        #
        # It is also the gain of a feedback loop -- solvent pulls the chloride
        # out, model transfers more charge, larger charge pulls harder -- whose
        # gain exceeds unity near 0.6 of full coupling. Ramping the coupling on
        # in 5, 10 and 20 stages failed at 0.80, 0.70 and 0.60: the more time
        # spent near the threshold, the sooner it is found, so it is a threshold
        # and not a rate. Clipping |q| at 1 does not help, because q(Cl) only
        # reached -1.035 before collapsing -- the runaway is already underway at
        # |q| ~ 0.85, inside the physical range.
        #
        # Setting this False keeps the charges recomputed every step and still
        # in the energy, but stops the gradient through them. NOTE this is an
        # APPROXIMATION, unlike the stop_gradient on the MIC lattice shift
        # documented in devtools/CLAUDE.md, which is exact: piecewise-constant
        # shifts genuinely have zero derivative. Here a real term is being
        # dropped, so the dynamics is no longer strictly variational and the
        # forces are not the gradient of the energy being reported.
        self.charge_gradient = bool(charge_gradient)
        self.capacity_hint = capacity_hint

    def neighbor_request(self, system: MolecularSystem) -> NeighborRequest:
        # Shares the intermolecular pair family with mm_nonbonded so the driver
        # builds one list and both terms read it.
        return NeighborRequest(
            cutoff_A=self.cutoff_A, kind="intermolecular",
            capacity_hint=self.capacity_hint,
        )

    def _resolve_ml_atoms(self, system: MolecularSystem, ctx: EnergyContext) -> np.ndarray:
        if self.ml_atoms is not None:
            return np.asarray(self.ml_atoms, dtype=np.int64)
        from_ctx = dict(getattr(ctx, "options", {}) or {}).get("ml_atoms")
        if from_ctx is None:
            raise ValueError(
                "ml_mm_elec needs ml_atoms (the solute indices), either on the "
                "term or in EnergyContext.options['ml_atoms']"
            )
        return np.asarray(list(from_ctx), dtype=np.int64)

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import e3x
        import jax
        import jax.numpy as jnp

        model, params = resolve_ml_model(self, ctx)
        ml_idx = self._resolve_ml_atoms(system, ctx)
        n_ml = int(ml_idx.shape[0])
        if system.ff_params is None:
            raise ValueError("ml_mm_elec needs system.ff_params for the MM charges")

        mm_charges = np.asarray(system.ff_params.charges, dtype=np.float64)
        if float(np.abs(mm_charges[ml_idx]).max()) > 1e-8:
            raise ValueError(
                "the solute still carries non-zero MM charges; zero them in "
                "FFParams or mm_nonbonded will double-count the solute "
                "electrostatics this term computes"
            )

        # Boolean membership, so the pair filter works inside jit.
        is_ml = np.zeros(system.n_atoms, dtype=bool)
        is_ml[ml_idx] = True
        is_ml_j = jnp.asarray(is_ml)
        ml_idx_j = jnp.asarray(ml_idx, dtype=jnp.int32)
        # Position of each atom within the ML block, for gathering its charge.
        ml_slot = np.full(system.n_atoms, -1, dtype=np.int64)
        ml_slot[ml_idx] = np.arange(n_ml)
        ml_slot_j = jnp.asarray(ml_slot, dtype=jnp.int32)
        mm_q_j = jnp.asarray(mm_charges)

        z_ml = jnp.asarray(np.asarray(system.Z)[ml_idx], dtype=jnp.int32)
        dst, src = e3x.ops.sparse_pairwise_indices(n_ml)
        dst, src = jnp.asarray(dst, jnp.int32), jnp.asarray(src, jnp.int32)

        cell = None if system.box is None else jnp.asarray(system.box)
        r_on, r_off = self.switch_on_A, self.cutoff_A
        sigma = self.damping_sigma_A
        mode, q_tot = self.charge_mode, self.total_charge
        q_clip = self.charge_clip
        use_charge_grad = self.charge_gradient

        def unfold(pos, cell_used):
            """Make the solute contiguous across the periodic boundary.

            The integrator wraps coordinates into the primary cell, so a solute
            that drifts across a box face comes back with its atoms on opposite
            sides -- tens of Angstroms apart in raw coordinates. Handing that to
            the model destroys it: every atom falls outside every other atom's
            8 A cutoff, the graph disconnects, and charge conservation fails.
            Observed as sum(q) = +5.37 e on a 30 A box, with forces of
            2363 eV/A, roughly 100 fs into a window.

            Rebuilding each atom's position as (first atom) + (minimum-image
            displacement to it) restores the molecule. This is what
            ``resolve_displacement_fn`` already does for ``ml_intra`` and
            ``mm_bonded``; this term was missing it.
            """
            if cell_used is None:
                return pos
            lengths = jnp.diag(cell_used)
            d = pos - pos[0]
            return pos[0] + (d - lengths * jnp.round(d / lengths))

        def ml_charges(R, box=None):
            """Per-atom ML charges for the solute at the current geometry."""
            out = model.apply(
                params,
                atomic_numbers=z_ml,
                positions=unfold(R[ml_idx_j],
                                 cell if box is None else jnp.asarray(box)),
                dst_idx=dst,
                src_idx=src,
                compute_forces=False,
            )
            q = jnp.reshape(jnp.asarray(out["charges"]), (n_ml,))
            # Clip before the neutrality correction, so the solute still sums
            # exactly to total_charge -- a net charge in a periodic box is a
            # physical error that grows with system size, and it matters more
            # than the last few thousandths on an individual atom.
            if q_clip is not None:
                q = jnp.clip(q, -q_clip, q_clip)
            if mode == "q0":
                q = q - (jnp.sum(q) - q_tot) / n_ml
            return q

        def switch(r):
            """CHARMM-style energy switch: 1 below r_on, 0 above r_off."""
            x = jnp.clip((r_off**2 - r**2), 0.0, None)
            num = x * x * (r_off**2 + 2.0 * r**2 - 3.0 * r_on**2)
            den = (r_off**2 - r_on**2) ** 3
            return jnp.where(r <= r_on, 1.0, jnp.where(r >= r_off, 0.0, num / den))

        def energy_fn(R, *, pair_i=None, pair_j=None, pair_mask=None, box=None,
                      elec_scale=None, **kwargs):
            """``elec_scale`` scales the whole solute-solvent Coulomb.

            Exists so the coupling can be switched on gradually. Turning it on
            at full strength against a freshly packed box drives a runaway: the
            solvent pulls the chloride out, the model responds by transferring
            more charge (q(Cl) went -0.80 -> -1.03 in 50 fs), the stronger
            charge pulls the solvent in harder, and nothing in the loop opposes
            it. Ramping over the equilibration leg lets the first solvation
            shell form before the feedback can run.

            Threaded as a traced scalar rather than a Python float so changing
            it between legs does not trigger an XLA recompilation.
            """
            del kwargs
            if pair_i is None:
                raise ValueError("ml_mm_elec requires the intermolecular pair list")
            q_ml = ml_charges(R, box=box)

            i = jnp.asarray(pair_i, dtype=jnp.int32)
            j = jnp.asarray(pair_j, dtype=jnp.int32)
            mask = (
                jnp.ones(i.shape, dtype=R.dtype)
                if pair_mask is None
                else jnp.asarray(pair_mask, dtype=R.dtype)
            )
            # Keep pairs with exactly one ML atom: ML-ML is the model's own
            # business and MM-MM belongs to mm_nonbonded.
            cross = jnp.logical_xor(is_ml_j[i], is_ml_j[j])
            mask = mask * cross.astype(R.dtype)

            # Orient each pair as (ml, mm) so the charge lookup is unambiguous.
            i_is_ml = is_ml_j[i]
            a = jnp.where(i_is_ml, i, j)   # ML side
            b = jnp.where(i_is_ml, j, i)   # MM side

            disp = R[b] - R[a]
            cell_used = cell if box is None else jnp.asarray(box)
            if cell_used is not None:
                lengths = jnp.diag(cell_used)
                disp = disp - lengths * jnp.round(disp / lengths)
            # Clamp so padded/masked slots cannot produce 1/0 or NaN gradients.
            r = jnp.sqrt(jnp.sum(disp * disp, axis=-1) + 1e-12)
            r_safe = jnp.where(mask > 0, r, r_off)

            qa = q_ml[ml_slot_j[a]]
            if not use_charge_grad:
                qa = jax.lax.stop_gradient(qa)
            qb = mm_q_j[b]
            damping = (
                1.0
                if sigma <= 0.0
                else jax.scipy.special.erf(r_safe / sigma)
            )
            e_pair = COULOMB_KCAL * qa * qb * damping / r_safe * switch(r_safe)
            scale = 1.0 if elec_scale is None else elec_scale
            return scale * jnp.sum(e_pair * mask) * KCAL_MOL_TO_EV

        energy_fn.ml_charges = ml_charges  # type: ignore[attr-defined]
        energy_fn.n_ml_atoms = n_ml  # type: ignore[attr-defined]

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=self.neighbor_request(system),
        )

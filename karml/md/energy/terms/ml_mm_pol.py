"""Induced polarisation of the ML solute by the MM environment.

``ml_mm_elec`` computes the *static* part of the solute-solvent electrostatics:
ML-predicted charges against fixed MM charges. That is one-way. The solute's
charges respond to its own geometry, but never to the field the solvent puts on
it, because the model only ever sees the solute's own atoms. In QM/MM language
``ml_mm_elec`` alone is closer to mechanical than to electrostatic embedding --
the MM charges never enter the electronic problem.

This term supplies the missing half:

    E_pol = -1/2 sum_i alpha_i |E_i|^2

with ``E_i`` the field at solute atom ``i`` from the MM charges. It is the
classical induction energy, and it is what EMLE (J. Chem. Theory Comput. 2023,
19, 1417) calls the *induced* component -- there obtained from a Thole model
over atomic polarisabilities, exactly as here. EMLE's central point is that this
requires **no QM/MM training data**: in-vacuo atomic properties suffice.

Why it matters here, measured on the Menshutkin campaign
--------------------------------------------------------
Without it, the NH3 + CH3Cl barrier in water and in acetonitrile came out at
26.17 and 26.24 kcal/mol -- a 0.07 kcal/mol gap where Turan et al. report 2.6,
and a total solvent effect (-8.4) about half of both Turan (-17.8) and an AM1
QM/MM study (-20.1). The solvent *structure* was right (water put 4.7 hydrogens
inside 3 A of the developing chloride at 2.26 A; acetonitrile, which cannot
donate an H-bond, put none), but that structure bought almost no free energy,
because the solute's charges could not deepen in response to the field those
H-bonds create.

Evaluated post-hoc on those trajectories this term gives, relative to reactants,
-4.2 kcal/mol in water against -1.0 in acetonitrile at the transition state: the
right sign and roughly the right size to restore the missing differential.

Two properties make it behave correctly by construction:

* it is always stabilising (``-|E|^2``), and
* it scales as the field *squared*, so it is near zero for the neutral reactant
  and grows as charge separates -- which is exactly where the error grew.

Caveats worth keeping in view
-----------------------------
* **Not self-consistent with the solvent.** The MM charges are fixed, so this is
  induction of the solute by the solvent and not mutual polarisation. Induced
  dipoles here also do not polarise each other (no dipole-dipole coupling); that
  is a further refinement.
* **No double counting.** Worth stating because it is the obvious worry: the
  PhysNet charges are trained on *gas-phase* B3LYP/def2-SVPD, so they carry no
  condensed-phase pre-polarisation, and ``ml_mm_elec`` is pure static Coulomb.
  TIP3P's enhanced charges (2.35 D vs 1.85 D in gas) represent water polarising
  *water*, not the solute. Only the 9 solute atoms are polarised here.
* **The polarisabilities are element-based constants.** Chloride is the weak
  point: covalent Cl and Cl- differ by roughly 2x in polarisability, and along
  this reaction the atom becomes the anion. ``alpha_by_charge`` interpolates on
  the ML charge as a crude stand-in for the volume-scaled polarisabilities EMLE
  gets from MBIS partitioning.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mmml.md.energy.registry import EnergyContext, NeighborRequest, TermFns, register_term
from mmml.md.energy.terms._common import ase_contribution_from_jax
from mmml.md.system import MolecularSystem

__all__ = ["MLMMPolarisationTerm"]

# CHARMM's CCELEC, kcal*A/(mol*e^2). Same constant ml_mm_elec uses.
COULOMB_KCAL = 332.0637
KCAL_MOL_TO_EV = 1.0 / 23.060547830619027

#: Atomic polarisabilities in A^3 (Thole / van Duijnen). Chlorine is the
#: covalently bound value; see ``alpha_by_charge`` for the anion.
ALPHA_A3 = {1: 0.514, 6: 1.405, 7: 1.105, 8: 0.862, 17: 2.315, 35: 3.130, 53: 5.350}

#: Polarisability of the free halide anion, A^3. Chloride is roughly 1.6x the
#: covalent atom; the electron it gains is diffuse and easily distorted.
ALPHA_ANION_A3 = {17: 3.760, 35: 4.770, 53: 6.900}

#: Thole exponential damping parameter. Damping is not cosmetic: an undamped
#: 1/r^2 field diverges as an MM hydrogen approaches, and -alpha|E|^2 then
#: diverges *downward*, which is an attractive singularity the integrator will
#: happily fall into.
THOLE_A = 0.39


@register_term("ml_mm_pol")
class MLMMPolarisationTerm:
    """Induction energy of the ML solute in the field of the MM charges."""

    def __init__(
        self,
        *,
        ml_atoms: Sequence[int] | None = None,
        cutoff_A: float = 12.0,
        alpha_A3: dict[int, float] | None = None,
        alpha_by_charge: bool = True,
        thole_a: float = THOLE_A,
        capacity_hint: int | None = None,
        scale: float = 1.0,
    ) -> None:
        self.ml_atoms = None if ml_atoms is None else list(ml_atoms)
        self.cutoff_A = float(cutoff_A)
        self.alpha_A3 = dict(ALPHA_A3 if alpha_A3 is None else alpha_A3)
        self.alpha_by_charge = bool(alpha_by_charge)
        self.thole_a = float(thole_a)
        self.capacity_hint = capacity_hint
        self.scale = float(scale)

    def neighbor_request(self, system: MolecularSystem) -> NeighborRequest:
        # Shares the intermolecular family with mm_nonbonded and ml_mm_elec, so
        # the driver builds one list for all three.
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
                "ml_mm_pol needs ml_atoms (the solute indices), either on the "
                "term or in EnergyContext.options['ml_atoms']"
            )
        return np.asarray(list(from_ctx), dtype=np.int64)

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        ml_idx = self._resolve_ml_atoms(system, ctx)
        n_ml = int(ml_idx.shape[0])
        if system.ff_params is None:
            raise ValueError("ml_mm_pol needs system.ff_params for the MM charges")

        mm_charges = np.asarray(system.ff_params.charges, dtype=np.float64)
        z_all = np.asarray(system.Z)
        z_ml = z_all[ml_idx]

        missing = sorted({int(z) for z in z_ml} - set(self.alpha_A3))
        if missing:
            raise ValueError(
                f"no polarisability for atomic numbers {missing}; extend "
                "ALPHA_A3 or pass alpha_A3="
            )
        alpha_neutral = np.array([self.alpha_A3[int(z)] for z in z_ml])
        alpha_anion = np.array([
            ALPHA_ANION_A3.get(int(z), self.alpha_A3[int(z)]) for z in z_ml
        ])

        is_ml = np.zeros(system.n_atoms, dtype=bool)
        is_ml[ml_idx] = True
        is_ml_j = jnp.asarray(is_ml)

        ml_slot = np.full(system.n_atoms, -1, dtype=np.int64)
        ml_slot[ml_idx] = np.arange(n_ml)
        ml_slot_j = jnp.asarray(ml_slot, dtype=jnp.int32)
        mm_q_j = jnp.asarray(mm_charges)
        a_neu = jnp.asarray(alpha_neutral)
        a_ani = jnp.asarray(alpha_anion)

        cell = None if system.box is None else jnp.asarray(system.box)
        thole_a = self.thole_a
        by_charge = self.alpha_by_charge
        scale_const = self.scale

        def energy_fn(R, *, pair_i=None, pair_j=None, pair_mask=None, box=None,
                      elec_scale=None, ml_charges=None, **kwargs) -> Any:
            """Induction energy. ``ml_charges`` (optional) scales alpha.

            ``elec_scale`` is honoured so this term ramps on with the rest of the
            solute-solvent coupling. Switching full induction on against an
            unequilibrated box would drive the same runaway the static term does.
            """
            del kwargs
            if pair_i is None:
                raise ValueError("ml_mm_pol requires the intermolecular pair list")

            i = jnp.asarray(pair_i, dtype=jnp.int32)
            j = jnp.asarray(pair_j, dtype=jnp.int32)
            mask = (
                jnp.ones(i.shape, dtype=R.dtype)
                if pair_mask is None
                else jnp.asarray(pair_mask, dtype=R.dtype)
            )
            cross = jnp.logical_xor(is_ml_j[i], is_ml_j[j])
            mask = mask * cross.astype(R.dtype)

            i_is_ml = is_ml_j[i]
            a = jnp.where(i_is_ml, i, j)      # ML side
            b = jnp.where(i_is_ml, j, i)      # MM side

            disp = R[b] - R[a]               # points from the ML atom to the MM one
            cell_used = cell if box is None else jnp.asarray(box)
            if cell_used is not None:
                lengths = jnp.diag(cell_used)
                disp = disp - lengths * jnp.round(disp / lengths)
            r2 = jnp.sum(disp * disp, axis=-1) + 1e-12
            r = jnp.sqrt(r2)
            # Masked slots must not reach the 1/r^3, or their zero-weighted
            # contribution still poisons the gradient (0 * NaN = NaN).
            r_safe = jnp.where(mask > 0, r, 1.0)

            slot = ml_slot_j[a]
            alpha_site = a_neu[slot]
            if by_charge and ml_charges is not None:
                # Interpolate covalent -> anionic on the ML charge. q = 0 gives
                # the neutral value, q = -1 the anion; clamped so a transient
                # overshoot cannot inflate alpha without bound.
                q_site = jnp.clip(-jnp.asarray(ml_charges)[slot], 0.0, 1.0)
                alpha_site = a_neu[slot] + q_site * (a_ani[slot] - a_neu[slot])

            # Thole exponential damping of the charge-induced-dipole field.
            u = r_safe / jnp.cbrt(alpha_site)
            damp = 1.0 - jnp.exp(-thole_a * u * u * u)

            # Field at the ML atom from this MM charge: E = q * (R_ml - R_mm)/r^3,
            # and (R_ml - R_mm) = -disp.
            contrib = (-(mm_q_j[b] * damp / (r_safe * r2))[:, None] * disp)
            contrib = contrib * mask[:, None]

            field = jnp.zeros((n_ml, 3), dtype=R.dtype).at[slot].add(contrib)

            # alpha per site, recomputed off the scattered charges if given.
            if by_charge and ml_charges is not None:
                q_ml_c = jnp.clip(-jnp.asarray(ml_charges), 0.0, 1.0)
                alpha = a_neu + q_ml_c * (a_ani - a_neu)
            else:
                alpha = a_neu

            e_pol = -0.5 * COULOMB_KCAL * jnp.sum(alpha * jnp.sum(field * field, axis=-1))
            s = 1.0 if elec_scale is None else elec_scale
            return scale_const * s * e_pol * KCAL_MOL_TO_EV

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=self.neighbor_request(system),
        )

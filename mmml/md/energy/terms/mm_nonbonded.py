"""MM intermolecular nonbonded term (switched VDW + Coulomb).

Extracted from the (B) block of ``examples/cg_jaxmd.py::hybrid_energy_fn`` and the
reference ``mm_system_energy.nonbonded_energy_and_forces``. Reuses the shared
CHARMM switching kernels (``_pair_vdw_energy`` / ``_pair_elec_energy`` and their
coefficient helpers) so this is a thin, faithful wrapper rather than a
re-derivation.

Two faces:

- **jax** — a jittable padded-pair energy: the driver's neighbor list supplies
  ``pair_i`` / ``pair_j`` (+ optional ``pair_mask`` / ``e14_scale`` /
  ``vdw14_scale``) as kwargs each step (decision B). Masked/padded pairs are
  distance-clamped to avoid NaN gradients. Supports ``lr_solver="mic"``
  (real-space minimum-image) and ``lr_solver="ewald"`` (jit-native Ewald
  summation — see ``ewald_native.py`` — real-space erfc term over the same
  pair list plus a reciprocal-space structure-factor sum and self-energy, all
  differentiable via plain ``jax.grad``). No exclusion correction: real-space
  MM is already intermolecular-only (the ML model owns intramolecular
  interactions; ``mol_id`` filtering already drops those pairs from the pair
  list), so there is no bonded-exclusion list to reconcile the reciprocal-space
  sum against. This leaves a small, uncorrected leak of intramolecular
  electrostatics through k-space (which has no notion of molecules) — an
  accepted tradeoff; see ``ewald_native.ewald_exclusion_correction`` if that
  needs revisiting. These are the only long-range electrostatics backends that
  are jit-compatible in this term (see ``lr_solver`` below for why the others
  aren't).
- **ASE** — wraps ``nonbonded_energy_and_forces`` (rebuilds the pair list each
  call), matching ``JAXIntermolecularCalculator``. Supports every
  ``lr_solver`` the reference function does (``mic``, ``jax_pme``,
  ``nvalchemiops_pme``, ``scafacos``).

Per-atom force-field data comes from ``system.ff_params`` (``FFParams``, decision
A); the CHARMM cutoffs/switch modes come from a ``CharmmNbondSettings`` passed in
the constructor or via ``ctx.options``.

``lr_solver`` (default ``"mic"``): selects the long-range electrostatics
backend. ``"mic"`` (real-space minimum-image) and ``"ewald"`` (jit-native
Ewald, ``ewald_native.py``) are available through the jax/jit face.
``"jax_pme"`` / ``"nvalchemiops_pme"`` / ``"scafacos"`` remain host-orchestrated
(e.g. jax-pme's evaluator builds an ASE ``Atoms`` and its own neighbor list on
the host via ``jax.pure_callback``, which has no VJP -- incompatible with the
``jax.grad``-derived forces ``jax_md``'s integrators require inside an
existing ``jax.jit`` graph, not merely unimplemented). Requesting one of
those three still raises ``NotImplementedError`` if the jax energy function is
actually called; use ``as_ase_calculator()`` instead, which supports all
solvers including those three (see ``docs/hybrid-mlmm-decomposition.md`` and
``docs/md-cg-unification-design.md`` §11 for the underlying limitation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.md.energy.capacity import COMPUTE_DTYPE
from mmml.md.energy.registry import EnergyContext, NeighborRequest, TermFns, register_term
from mmml.md.system import FFParams, MolecularSystem

__all__ = ["MMNonbondedTerm"]

_DEFAULT_CUTOFFS = {"cutnb": 12.0, "ctonnb": 10.0, "ctofnb": 12.0}


def _nbdata_from_ffparams(ff: FFParams):
    """Reconstruct a ``NonbondedSystemData`` (the reference's input) from FFParams."""
    from mmml.interfaces.pycharmmInterface.mm_system_energy import NonbondedSystemData

    return NonbondedSystemData(
        charges=np.asarray(ff.charges, dtype=np.float64),
        at_codes=np.asarray(ff.at_codes, dtype=np.int32),
        epsilon=np.asarray(ff.epsilon, dtype=np.float64),
        rmin=np.asarray(ff.rmin_half, dtype=np.float64),
        excluded_pairs=frozenset(map(tuple, ff.exclusions.tolist())),
        e14_pairs=frozenset(map(tuple, ff.e14_pairs.tolist())),
        psf_path=None,
        psf_bonds=None,
    )


@register_term("mm_nonbonded")
class MMNonbondedTerm:
    """Switched intermolecular VDW + Coulomb over a (padded) pair list."""

    name = "mm_nonbonded"

    def __init__(
        self,
        settings: Any | None = None,
        lr_solver: str = "mic",
        jax_pme_method: str | None = None,
        jax_pme_sr_cutoff_A: float = 6.0,
        ewald_alpha: float | None = None,
        ewald_accuracy_exponent: float = 3.5,
    ):
        self.settings = settings
        self.lr_solver = str(lr_solver)
        self.jax_pme_method = jax_pme_method
        self.jax_pme_sr_cutoff_A = float(jax_pme_sr_cutoff_A)
        self.ewald_alpha = ewald_alpha
        self.ewald_accuracy_exponent = float(ewald_accuracy_exponent)

    def _resolve_settings(self, ctx: EnergyContext):
        if self.settings is not None:
            return self.settings
        from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings

        opts = dict(ctx.options)
        return CharmmNbondSettings(
            cutnb=float(opts.get("cutnb", _DEFAULT_CUTOFFS["cutnb"])),
            ctonnb=float(opts.get("ctonnb", _DEFAULT_CUTOFFS["ctonnb"])),
            ctofnb=float(opts.get("ctofnb", _DEFAULT_CUTOFFS["ctofnb"])),
            eps=float(opts.get("eps", 1.0)),
            e14fac=float(opts.get("e14fac", 1.0)),
            vdw14fac=float(opts.get("vdw14fac", 0.0)),
        )

    def neighbor_request(self, system: MolecularSystem):
        cut = self.settings.cutnb if self.settings is not None else _DEFAULT_CUTOFFS["cutnb"]
        return NeighborRequest(cutoff_A=float(cut), kind="intermolecular")

    def _host_pairs(self, pos, settings, ff: FFParams, mol_id):
        """Build (pair_i, pair_j, e14_scale, vdw14_scale) exactly as the reference."""
        from mmml.interfaces.pycharmmInterface.mm_system_energy import _build_pair_indices

        excluded = frozenset(map(tuple, ff.exclusions.tolist()))
        cell = np.asarray(self._cell_np)
        pi, pj = _build_pair_indices(np.asarray(pos), cell, excluded, settings.cutnb)
        if mol_id is not None:
            mol_id = np.asarray(mol_id, dtype=np.int32)
            inter = mol_id[pi] != mol_id[pj]
            pi, pj = pi[inter], pj[inter]
        e14_set = frozenset(map(tuple, ff.e14_pairs.tolist()))
        e14 = np.ones(len(pi), dtype=np.float64)
        vdw14 = np.ones(len(pi), dtype=np.float64)
        for k, (i, j) in enumerate(zip(pi, pj, strict=True)):
            if (int(i), int(j)) in e14_set:
                e14[k] = settings.e14fac
                vdw14[k] = settings.vdw14fac
        return pi, pj, e14, vdw14

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        if system.box is None:
            raise ValueError("mm_nonbonded requires a periodic box.")
        if system.ff_params is None:
            raise ValueError("mm_nonbonded requires system.ff_params (FFParams).")

        import jax
        import jax.numpy as jnp

        from mmml.interfaces.pycharmmInterface.mm_system_energy import (
            COULOMB_KCAL,
            _pair_elec_energy,
            _pair_lj_epsilon,
            _pair_vdw_energy,
            charmm_fswitch_coeffs,
            charmm_vfswitch_coeffs,
        )
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

        ff = system.ff_params
        settings = self._resolve_settings(ctx)
        self._cell_np = np.asarray(system.box, dtype=np.float64)
        cell = jnp.asarray(self._cell_np)
        q = jnp.asarray(ff.charges, dtype=jnp.float64)
        eps_tbl = jnp.asarray(ff.epsilon, dtype=jnp.float64)
        rm_tbl = jnp.asarray(ff.rmin_half, dtype=jnp.float64)
        vf = charmm_vfswitch_coeffs(settings)
        fs = charmm_fswitch_coeffs(settings)
        c2of = settings.c2ofnb
        eps_scale = float(settings.eps)

        if self.lr_solver == "ewald":
            from mmml.interfaces.pycharmmInterface.ewald_native import (
                build_kspace_integers,
                default_ewald_alpha,
                ewald_reciprocal_energy,
                ewald_self_energy,
            )

            # No exclusion correction: real-space MM is already intermolecular
            # -only (mol_id filter in _host_pairs / the driver's neighbor list),
            # so there is no formal bonded-exclusion list to reconcile against.
            # This does leave a small, uncorrected leak of intramolecular
            # electrostatics through the reciprocal-space sum (which has no
            # notion of molecules) -- accepted tradeoff for now; see
            # ewald_native.ewald_exclusion_correction if that needs revisiting.
            ewald_alpha = float(
                self.ewald_alpha
                if self.ewald_alpha is not None
                else default_ewald_alpha(settings.cutnb, accuracy_exponent=self.ewald_accuracy_exponent)
            )
            n_int_np = build_kspace_integers(
                self._cell_np, ewald_alpha, accuracy_exponent=self.ewald_accuracy_exponent
            )
            n_int = jnp.asarray(n_int_np)
            ewald_self = ewald_self_energy(q, ewald_alpha) * COULOMB_KCAL

        def _pair_energy(R, pi, pj, e14, vdw14, mask, cell_used):
            ri = R[pi]
            rj = R[pj]
            disp = jax.vmap(lambda a, b: mic_displacement(a, b, cell_used))(ri, rj)
            disp_sq = jnp.sum(disp * disp, axis=-1)
            # clamp masked/padded pairs so sqrt / 1-r never sees a true zero
            safe_sq = jnp.where(mask > 0.5, disp_sq, 1.0)
            dist = jnp.sqrt(jnp.maximum(safe_sq, 1e-12))
            within = (dist * dist < c2of) * mask
            safe_dist = jnp.where(mask > 0.5, dist, 1.0)

            ep = _pair_lj_epsilon(eps_tbl[pi], eps_tbl[pj])
            sig = rm_tbl[pi] + rm_tbl[pj]
            qq = q[pi] * q[pj] * e14 / eps_scale
            vdw = _pair_vdw_energy(safe_dist, ep, sig, settings, vf, use_jax_pme_dispersion=False)
            vdw = vdw * vdw14
            elec = _pair_elec_energy(safe_dist, qq, settings, fs)
            vdw = jnp.where(within, vdw, 0.0)
            elec = jnp.where(within, elec, 0.0)
            return (jnp.sum(vdw) + jnp.sum(elec)) * KCAL_MOL_TO_EV

        if self.lr_solver == "ewald":

            def _pair_energy_ewald(R, pi, pj, e14, vdw14, mask, cell_used):
                import jax.scipy.special as jsp

                ri = R[pi]
                rj = R[pj]
                disp = jax.vmap(lambda a, b: mic_displacement(a, b, cell_used))(ri, rj)
                disp_sq = jnp.sum(disp * disp, axis=-1)
                safe_sq = jnp.where(mask > 0.5, disp_sq, 1.0)
                dist = jnp.sqrt(jnp.maximum(safe_sq, 1e-12))
                within = (dist * dist < c2of) * mask
                safe_dist = jnp.where(mask > 0.5, dist, 1.0)

                ep = _pair_lj_epsilon(eps_tbl[pi], eps_tbl[pj])
                sig = rm_tbl[pi] + rm_tbl[pj]
                qq = q[pi] * q[pj] * e14 / eps_scale
                vdw = _pair_vdw_energy(safe_dist, ep, sig, settings, vf, use_jax_pme_dispersion=False)
                vdw = vdw * vdw14
                # real-space Ewald term: erfc-damped Coulomb, natural (fast) decay
                # to ~0 by rcut given `ewald_alpha` -- no CHARMM switching needed.
                elec_real = qq * jsp.erfc(ewald_alpha * safe_dist) / safe_dist * COULOMB_KCAL
                vdw = jnp.where(within, vdw, 0.0)
                elec_real = jnp.where(within, elec_real, 0.0)

                recip = ewald_reciprocal_energy(R, q, cell_used, n_int, ewald_alpha) * COULOMB_KCAL
                return (jnp.sum(vdw) + jnp.sum(elec_real) + recip + ewald_self) * KCAL_MOL_TO_EV

        def energy_fn(
            R,
            *,
            pair_i=None,
            pair_j=None,
            e14_scale=None,
            vdw14_scale=None,
            pair_mask=None,
            molecule_id=None,
            box=None,
            **kwargs,
        ) -> Any:
            # box-aware for NPT: use the current cell when supplied, else the
            # build-time cell (NVE/NVT fixed box).
            cell_used = cell if box is None else jnp.asarray(box)
            if pair_i is None or pair_j is None:
                # host build (non-jit convenience / validation path)
                mid = system.mol_id if molecule_id is None else molecule_id
                pi_np, pj_np, e14_np, vdw14_np = self._host_pairs(
                    np.asarray(R), settings, ff, mid
                )
                pi = jnp.asarray(pi_np, dtype=jnp.int32)
                pj = jnp.asarray(pj_np, dtype=jnp.int32)
                e14 = jnp.asarray(e14_np)
                vdw14 = jnp.asarray(vdw14_np)
                mask = jnp.ones(pi.shape, dtype=COMPUTE_DTYPE)
            else:
                pi = jnp.asarray(pair_i, dtype=jnp.int32)
                pj = jnp.asarray(pair_j, dtype=jnp.int32)
                e14 = jnp.ones(pi.shape, dtype=COMPUTE_DTYPE) if e14_scale is None else jnp.asarray(e14_scale)
                vdw14 = jnp.ones(pi.shape, dtype=COMPUTE_DTYPE) if vdw14_scale is None else jnp.asarray(vdw14_scale)
                mask = jnp.ones(pi.shape, dtype=COMPUTE_DTYPE) if pair_mask is None else jnp.asarray(pair_mask).astype(COMPUTE_DTYPE)
            if self.lr_solver == "mic":
                return _pair_energy(R, pi, pj, e14, vdw14, mask, cell_used)
            if self.lr_solver == "ewald":
                return _pair_energy_ewald(R, pi, pj, e14, vdw14, mask, cell_used)
            # jax_pme/nvalchemiops_pme/scafacos are host-orchestrated (e.g.
            # jax-pme's evaluator builds an ASE Atoms + its own neighbor list
            # on the host) -- not jit-traceable, so fail loudly here rather
            # than silently falling back to mic under the driver's jit.
            raise NotImplementedError(
                f"mm_nonbonded's jax/jit face only supports lr_solver='mic' or "
                f"'ewald'; got {self.lr_solver!r}. Use as_ase_calculator() instead, "
                "which supports mic/jax_pme/nvalchemiops_pme/scafacos."
            )

        def ase_contribution(atoms):
            from mmml.interfaces.pycharmmInterface.mm_system_energy import (
                nonbonded_energy_and_forces,
            )

            terms, forces = nonbonded_energy_and_forces(
                np.asarray(atoms.get_positions(), dtype=np.float64),
                _nbdata_from_ffparams(ff),
                self._cell_np,
                settings,
                molecule_id=np.asarray(system.mol_id, dtype=np.int32),
                lr_solver=self.lr_solver,
                jax_pme_method=self.jax_pme_method,
                jax_pme_sr_cutoff_A=self.jax_pme_sr_cutoff_A,
            )
            energy = float(terms.get("total", sum(float(v) for v in terms.values())))
            return energy * KCAL_MOL_TO_EV, np.asarray(forces, dtype=np.float64) * KCAL_MOL_TO_EV

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution,
            neighbor_request=self.neighbor_request(system),
        )

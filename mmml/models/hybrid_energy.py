"""Assemble hybrid ML/MM energy + forces for training.

Mirrors what the MD hybrid calculator evaluates, so a model trained here is
trained on the energy it will later be deployed with::

    E_total = ml_switch_scale(r_com) * E_ML  +  E_MM(switched LJ + electrostatics)

Both scale factors come from the shared single source of truth
(:mod:`mmml.interfaces.pycharmmInterface.calculator_utils`), and ``E_MM`` from
:mod:`mmml.models.cgenff_mm` (CHARMM-free, formula- and parameter-matched to
``mm_energy_forces``; see ``tests/unit/test_cgenff_lj_parity.py`` and
``tests/unit/test_cgenff_mm_energy.py``).

Forces need care.  ``ml_switch_scale`` depends on the *positions* (through the
monomer COMs), so the ML contribution is **not** simply ``scale * F_ML``:

.. math::

    F_{total} = -\\frac{d}{dR}\\big[s(R)\\,E_{ML}(R) + E_{MM}(R)\\big]
              = s\\,F_{ML} \\;-\\; E_{ML}\\,\\frac{ds}{dR} \\;+\\; F_{MM}

The middle term is the force exerted by the handoff itself; dropping it (a
tempting shortcut, since the model already hands back ``F_ML``) silently breaks
energy conservation in the handoff region.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_scale
from mmml.models.cgenff_mm import cgenff_mm_energy, monomer_centroids

Array = jnp.ndarray

__all__ = [
    "HYBRID_MM_BATCH_KEYS",
    "HybridEnergyForces",
    "apply_hybrid_mm_to_output",
    "hybrid_energy_forces",
    "ml_scale_from_positions",
]

#: Per-atom dataset fields the hybrid mode needs in each training batch.
#: ``prepare_batches_jit`` passes these through unreshaped as ``(batch, natoms)``
#: (only R/F/E/Z get special-cased), which is exactly the layout we want.
#: The ``cgenff_master_*`` tables are ``(n_types,)`` -- not per-sample -- so they
#: are handed in separately rather than batched.
HYBRID_MM_BATCH_KEYS = ("cgenff_type_idx", "mol_id", "cgenff_charge")


class HybridEnergyForces(NamedTuple):
    """Assembled hybrid result (kcal/mol, kcal/mol/A) plus its components."""

    energy: Array
    forces: Array
    ml_scale: Array
    e_mm: Array


def ml_scale_from_positions(
    positions: Array,
    mol_id: Array,
    *,
    mm_switch_on: float,
    ml_switch_width: float,
) -> Array:
    """ML taper for one structure, as a differentiable function of positions.

    Monomers (a single ``mol_id``) have no second centroid, so ML stays fully on.
    """
    coms = monomer_centroids(positions, mol_id, n_monomers=2)
    # sqrt(max(., eps)) rather than norm(): a monomer's centroid coincides with
    # the (all-zero) second centroid when there is no second monomer, and
    # d|x|/dx is undefined at 0 -- that NaN would propagate into every force.
    d_com = coms[1] - coms[0]
    r_com = jnp.sqrt(jnp.maximum(jnp.sum(d_com * d_com), 1e-20))
    scale = ml_switch_scale(
        r_com, mm_switch_on=mm_switch_on, ml_switch_width=ml_switch_width
    )
    is_dimer = jnp.any(mol_id == 1)
    return jnp.where(is_dimer, scale, 1.0)


def hybrid_energy_forces(
    e_ml: Array,
    f_ml: Array,
    positions: Array,
    type_idx: Array,
    mol_id: Array,
    charges: Array,
    master_sigmas: Array,
    master_epsilons: Array,
    *,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_switch_width: float,
    complementary_handoff: bool = True,
) -> HybridEnergyForces:
    """Combine a model's ``E_ML``/``F_ML`` with the switched CGenFF MM term.

    Parameters
    ----------
    e_ml : scalar ML energy for this structure (kcal/mol).
    f_ml : (n_atoms, 3) ML forces, i.e. ``-dE_ML/dR`` (kcal/mol/A).
    positions : (n_atoms, 3)
    type_idx, mol_id : (n_atoms,) padded with ``-1``.
    charges : (n_atoms,) CGenFF charges.
    master_sigmas, master_epsilons : (n_types,) dataset LJ tables.

    Returns
    -------
    HybridEnergyForces with the total energy/forces and the components.

    Padding-safe and vmap-safe; differentiable w.r.t. ``e_ml``/``f_ml``.
    """

    def _scale(pos: Array) -> Array:
        return ml_scale_from_positions(
            pos, mol_id, mm_switch_on=mm_switch_on, ml_switch_width=ml_switch_width
        )

    def _emm(pos: Array) -> Array:
        return cgenff_mm_energy(
            pos,
            type_idx,
            mol_id,
            charges,
            master_sigmas,
            master_epsilons,
            mm_switch_on=mm_switch_on,
            mm_switch_width=mm_switch_width,
            ml_switch_width=ml_switch_width,
            complementary_handoff=complementary_handoff,
        )

    scale, dscale_dR = jax.value_and_grad(_scale)(positions)
    e_mm, demm_dR = jax.value_and_grad(_emm)(positions)

    energy = scale * e_ml + e_mm
    # F = s*F_ML - E_ML*ds/dR + F_MM     (F_MM = -dE_MM/dR)
    forces = scale * f_ml - e_ml * dscale_dR - demm_dR

    # Padding must not carry force.
    valid = (mol_id >= 0)[:, None]
    forces = jnp.where(valid, forces, 0.0)

    return HybridEnergyForces(energy=energy, forces=forces, ml_scale=scale, e_mm=e_mm)


def apply_hybrid_mm_to_output(
    output: dict,
    batch: dict,
    batch_size: int,
    master_sigmas: Array,
    master_epsilons: Array,
    *,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_switch_width: float,
    complementary_handoff: bool = True,
) -> dict:
    """Replace a model's ``energy``/``forces`` with the hybrid ML/MM totals.

    ``prepare_batches_jit`` flattens ``R``/``F`` to ``(batch*natoms, 3)`` but
    leaves the per-atom CGenFF fields as ``(batch, natoms)``, so reshape the
    former and vmap over structures.  Returns a shallow copy of ``output`` with
    ``energy``/``forces`` replaced (same shapes) plus ``ml_scale``/``e_mm``.
    """
    r_flat = batch["R"]
    n_atoms = r_flat.shape[0] // int(batch_size)

    pos = r_flat.reshape(batch_size, n_atoms, 3)
    f_ml = output["forces"].reshape(batch_size, n_atoms, 3)
    e_ml = output["energy"].reshape(batch_size)

    def _one(e, f, p, t, m, q):
        return hybrid_energy_forces(
            e,
            f,
            p,
            t,
            m,
            q,
            master_sigmas,
            master_epsilons,
            mm_switch_on=mm_switch_on,
            mm_switch_width=mm_switch_width,
            ml_switch_width=ml_switch_width,
            complementary_handoff=complementary_handoff,
        )

    hyb = jax.vmap(_one)(
        e_ml,
        f_ml,
        pos,
        batch["cgenff_type_idx"],
        batch["mol_id"],
        batch["cgenff_charge"],
    )

    out = dict(output)
    out["energy"] = hyb.energy.reshape(output["energy"].shape)
    out["forces"] = hyb.forces.reshape(output["forces"].shape)
    out["ml_scale"] = hyb.ml_scale
    out["e_mm"] = hyb.e_mm
    return out

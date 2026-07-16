"""MM electrostatic charge modes for hybrid ML/MM (train + MD).

These modes control **only** the per-atom ``q`` that enters intermolecular
``E_MM`` Coulomb.  They are orthogonal to:

* ``--charges`` / the PhysNet charge head (may exist for dipoles even in Mode A)
* ``include_electrostatics`` (Coulomb from ``q_ML`` **inside** ``E_ML``)

Modes
-----
**fixed** (Mode A)
    ``q_MM = q_CGenFF``.  Default hybrid train and today's MD calculator.

**latent** (Mode B) — **deferred**
    ``q_MM = neutralize_per_monomer(q_ML)``.  Not implemented in hybrid train
    or MD.  Named here so CLI/docs do not grow a third flag by accident.
    Closest cousin: ``examples/cg_jaxmd.py`` peptide–water embedding (different
    product).

**fixed_plus_latent** (Mode C)
    ``q_MM = q_CGenFF + neutralize_per_monomer(q_ML)``.
    Train: ``--mm-charge-correction`` / ``HybridMMConfig.charge_correction``.
    ``q_ML`` comes from the **AB dimer** forward in training.  MD v1 supports
    dimers only (``n_monomers == 2``); liquid ``q_ML`` context is undefined.

Liquid follow-up (not implemented): choose L1 monomer-context charges, L2
aggregate active-dimer charges, or L3 liquid-aware training before
``md-system`` deployment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import jax.numpy as jnp

from mmml.models.cgenff_mm import neutralize_per_monomer

Array = jnp.ndarray

__all__ = [
    "MMChargeMode",
    "apply_mm_charge_mode",
    "mm_charge_mode_from_charge_correction",
    "parse_mm_charge_mode",
    "require_charge_head_for_mode",
]


class MMChargeMode(str, Enum):
    """How ``E_MM`` Coulomb gets its per-atom charges."""

    FIXED = "fixed"
    LATENT = "latent"
    FIXED_PLUS_LATENT = "fixed_plus_latent"


def parse_mm_charge_mode(value: str | MMChargeMode | None) -> MMChargeMode:
    """Parse a CLI/config string into :class:`MMChargeMode`."""
    if value is None:
        return MMChargeMode.FIXED
    if isinstance(value, MMChargeMode):
        return value
    key = str(value).strip().lower().replace("-", "_").replace("+", "_plus_")
    aliases = {
        "fixed": MMChargeMode.FIXED,
        "a": MMChargeMode.FIXED,
        "mode_a": MMChargeMode.FIXED,
        "latent": MMChargeMode.LATENT,
        "b": MMChargeMode.LATENT,
        "mode_b": MMChargeMode.LATENT,
        "fixed_plus_latent": MMChargeMode.FIXED_PLUS_LATENT,
        "fixed_latent": MMChargeMode.FIXED_PLUS_LATENT,
        "c": MMChargeMode.FIXED_PLUS_LATENT,
        "mode_c": MMChargeMode.FIXED_PLUS_LATENT,
        "charge_correction": MMChargeMode.FIXED_PLUS_LATENT,
        "mm_charge_correction": MMChargeMode.FIXED_PLUS_LATENT,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown mm_charge_mode={value!r}. "
            f"Expected one of: {', '.join(m.value for m in MMChargeMode)}."
        )
    return aliases[key]


def mm_charge_mode_from_charge_correction(charge_correction: bool) -> MMChargeMode:
    """Map the training bool flag onto the taxonomy."""
    return (
        MMChargeMode.FIXED_PLUS_LATENT if charge_correction else MMChargeMode.FIXED
    )


def require_charge_head_for_mode(mode: MMChargeMode, *, has_charges: bool) -> None:
    """Raise if the mode needs a charge head and the model has none."""
    if mode in (MMChargeMode.LATENT, MMChargeMode.FIXED_PLUS_LATENT) and not has_charges:
        raise ValueError(
            f"mm_charge_mode={mode.value} requires a model built with charges=True "
            "(the charge head is absent, so there is nothing to correct with)."
        )


def apply_mm_charge_mode(
    mode: str | MMChargeMode,
    q_cgenff: Array,
    q_ml: Array | None,
    mol_id: Array,
    *,
    n_monomers: int = 2,
) -> Array:
    """Return per-atom charges for ``E_MM`` Coulomb under the given mode.

    Parameters
    ----------
    mode
        ``fixed``, ``latent``, or ``fixed_plus_latent``.
    q_cgenff
        Fixed CGenFF / PSF charges, shape ``(..., n_atoms)``.
    q_ml
        Charge-head output (same shape as ``q_cgenff``).  Required for
        ``latent`` and ``fixed_plus_latent``; ignored for ``fixed``.
    mol_id
        Per-atom monomer id (``< 0`` = padding), same trailing shape as charges.
    n_monomers
        Number of monomers for the net-zero projection (training dimers use 2).
    """
    mode = parse_mm_charge_mode(mode)
    q_cgenff = jnp.asarray(q_cgenff)
    mol_id = jnp.asarray(mol_id)

    if mode is MMChargeMode.FIXED:
        return q_cgenff

    if q_ml is None:
        require_charge_head_for_mode(mode, has_charges=False)

    dq = jnp.asarray(q_ml)
    if dq.shape != q_cgenff.shape:
        dq = dq.reshape(q_cgenff.shape)

    def _project(dq_one, mid_one):
        return neutralize_per_monomer(dq_one, mid_one, n_monomers=n_monomers)

    # Support both single-structure ``(n_atoms,)`` and batched ``(B, n_atoms)``.
    if q_cgenff.ndim == 1:
        dq_proj = _project(dq, mol_id)
    else:
        import jax

        dq_proj = jax.vmap(_project)(dq, mol_id)

    if mode is MMChargeMode.LATENT:
        raise NotImplementedError(
            "mm_charge_mode=latent (Mode B) is deferred: hybrid training has no "
            "flag for q_MM = neutralize(q_ML), and MD must not invent one. "
            "Use fixed or fixed_plus_latent."
        )

    # Mode C
    return q_cgenff + dq_proj


def hybrid_mm_metadata_dict(hybrid_mm: Any) -> dict[str, Any]:
    """Serialize hybrid MM charge-mode metadata for checkpoints / sidecars."""
    if hybrid_mm is None:
        return {
            "hybrid_mm": False,
            "charge_correction": False,
            "mm_charge_mode": MMChargeMode.FIXED.value,
        }
    charge_correction = bool(getattr(hybrid_mm, "charge_correction", False))
    mode = mm_charge_mode_from_charge_correction(charge_correction)
    return {
        "hybrid_mm": True,
        "charge_correction": charge_correction,
        "mm_charge_mode": mode.value,
        "mm_switch_on": float(getattr(hybrid_mm, "mm_switch_on", 8.0)),
        "mm_switch_width": float(getattr(hybrid_mm, "mm_switch_width", 5.0)),
        "ml_switch_width": float(getattr(hybrid_mm, "ml_switch_width", 1.5)),
        "complementary_handoff": bool(
            getattr(hybrid_mm, "complementary_handoff", True)
        ),
    }

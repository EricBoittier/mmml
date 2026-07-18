"""MM electrostatic charge modes for hybrid ML/MM (train + MD).

These modes control **only** the per-atom ``q`` that enters intermolecular
``E_MM`` Coulomb.  They are orthogonal to:

* ``--charges`` / the PhysNet charge head (may exist for dipoles even in Mode A)
* ``include_electrostatics`` (Coulomb from ``q_ML`` **inside** ``E_ML``)

Modes
-----
**fixed** (Mode A)
    ``q_MM = q_CGenFF``.  Default hybrid train and MD calculator.

**latent** (Mode B)
    ``q_MM = neutralize_per_monomer(q_ML)``.  Dimer-only (train + MD); liquids
    deferred (no multi-neighbor ``q_ML`` context).

**fixed_plus_latent** (Mode C)
    ``q_MM = q_CGenFF + neutralize_per_monomer(q_ML)``.
    Train: ``--mm-charge-mode fixed_plus_latent`` or ``--mm-charge-correction``.
    ``q_ML`` comes from the **AB dimer** forward.  MD v1 is dimer-only
    (``n_monomers == 2``); liquid ``q_ML`` context is undefined.

**latent_mean** (Mode D, MD-only)
    ``q_MM = tile(mean_over_dataset(neutralize_per_monomer(q_ML)))``.  A fixed,
    precomputed per-atom charge *template* -- one monomer's worth of latent
    charges, averaged over many training-set dimer forwards offline (see
    ``scripts/compute_latent_monomer_charges.py``) -- tiled across every
    monomer in a homogeneous liquid box.  Unlike Modes B/C it needs **no**
    live ``q_ML`` at MD time (no AB-dimer forward, no ``n_monomers == 2``
    restriction), so it is the liquid-compatible answer to the L1/L2 question
    below: it is Wallace & Boittier's L1 (monomer-context charges), computed
    once and frozen.  Not selectable in training (``hybrid_forward`` /
    ``apply_mm_charge_mode`` reject it) -- it is a static array the MD
    calculator loads via ``mm_latent_charge_template`` and injects directly,
    bypassing this module's per-step composition entirely.

Liquid follow-up for *live*, geometry-dependent latent charges (still not
implemented): L2 aggregate active-dimer charges, or L3 liquid-aware training
before ``md-system`` deployment. ``latent_mean`` sidesteps this by freezing
the charges instead of recomputing them per step.
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
    "hybrid_mm_metadata_dict",
    "mm_charge_mode_from_charge_correction",
    "mm_charge_mode_is_static_template",
    "mm_charge_mode_needs_q_ml",
    "parse_mm_charge_mode",
    "require_charge_head_for_mode",
    "resolve_hybrid_mm_charge_mode",
]


class MMChargeMode(str, Enum):
    """How ``E_MM`` Coulomb gets its per-atom charges."""

    FIXED = "fixed"
    LATENT = "latent"
    FIXED_PLUS_LATENT = "fixed_plus_latent"
    LATENT_MEAN = "latent_mean"


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
        "latent_mean": MMChargeMode.LATENT_MEAN,
        "latent-mean": MMChargeMode.LATENT_MEAN,
        "d": MMChargeMode.LATENT_MEAN,
        "mode_d": MMChargeMode.LATENT_MEAN,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown mm_charge_mode={value!r}. "
            f"Expected one of: {', '.join(m.value for m in MMChargeMode)}."
        )
    return aliases[key]


def mm_charge_mode_from_charge_correction(charge_correction: bool) -> MMChargeMode:
    """Map the legacy training bool flag onto the taxonomy."""
    return (
        MMChargeMode.FIXED_PLUS_LATENT if charge_correction else MMChargeMode.FIXED
    )


def mm_charge_mode_needs_q_ml(mode: str | MMChargeMode) -> bool:
    """True when the mode reads the ML charge head for ``E_MM`` Coulomb."""
    mode = parse_mm_charge_mode(mode)
    return mode in (MMChargeMode.LATENT, MMChargeMode.FIXED_PLUS_LATENT)


def mm_charge_mode_is_static_template(mode: str | MMChargeMode) -> bool:
    """True when the mode's charges are a precomputed, per-step-static array.

    ``latent_mean`` needs neither a live ``q_ML`` forward nor the dimer-only
    gate: its charges were fixed offline (see
    ``mmml.models.latent_charge_template``) and are injected once.
    """
    return parse_mm_charge_mode(mode) is MMChargeMode.LATENT_MEAN


def resolve_hybrid_mm_charge_mode(
    *,
    mm_charge_mode: str | MMChargeMode | None = None,
    charge_correction: bool = False,
) -> MMChargeMode:
    """Resolve ``mm_charge_mode`` / legacy ``charge_correction`` to one mode.

    Explicit ``mm_charge_mode`` wins.  ``charge_correction=True`` alone selects
    Mode C.  Conflicting pairs (e.g. correction + ``fixed`` / ``latent``) raise.
    """
    if mm_charge_mode is not None:
        mode = parse_mm_charge_mode(mm_charge_mode)
        if charge_correction and mode is not MMChargeMode.FIXED_PLUS_LATENT:
            raise ValueError(
                "Conflicting MM charge settings: --mm-charge-correction with "
                f"mm_charge_mode={mode.value}."
            )
        return mode
    return mm_charge_mode_from_charge_correction(bool(charge_correction))


def require_charge_head_for_mode(mode: MMChargeMode, *, has_charges: bool) -> None:
    """Raise if the mode needs a charge head and the model has none."""
    if mm_charge_mode_needs_q_ml(mode) and not has_charges:
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

    if mode is MMChargeMode.LATENT_MEAN:
        raise ValueError(
            "mm_charge_mode=latent_mean has no per-step composition -- it is a "
            "precomputed template injected directly by the MD calculator via "
            "mm_latent_charge_template, not a mode this function (training or "
            "live q_ML composition) can apply."
        )

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
        return dq_proj

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
    mode = resolve_hybrid_mm_charge_mode(
        mm_charge_mode=getattr(hybrid_mm, "mm_charge_mode", None),
        charge_correction=bool(getattr(hybrid_mm, "charge_correction", False)),
    )
    return {
        "hybrid_mm": True,
        "charge_correction": mode is MMChargeMode.FIXED_PLUS_LATENT,
        "mm_charge_mode": mode.value,
        "mm_switch_on": float(getattr(hybrid_mm, "mm_switch_on", 8.0)),
        "mm_switch_width": float(getattr(hybrid_mm, "mm_switch_width", 5.0)),
        "ml_switch_width": float(getattr(hybrid_mm, "ml_switch_width", 1.5)),
        "complementary_handoff": bool(
            getattr(hybrid_mm, "complementary_handoff", True)
        ),
    }

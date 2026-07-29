"""MM electrostatic charge modes for hybrid ML/MM (train + MD).

These modes control **only** the per-atom ``q`` that enters intermolecular
``E_MM`` Coulomb.  They are orthogonal to:

* ``--charges`` / the PhysNet charge head (may exist for dipoles even in Mode A)
* ``include_electrostatics`` (Coulomb from ``q_ML`` **inside** ``E_ML``)

Perturbative nomenclature
-------------------------
Hybrid ML energy already follows an E(AB)−E(A)−E(B) split (monomers = E⁰,
switched dimer interaction ≈ E¹).  MM charges use the same language:

* **Q⁰** (``q0``) — unperturbed monomer charges from *isolated* monomer
  forwards (train: ``out_a``/``out_b``; MD: monomer slots in the PhysNet
  batch).  Liquid-capable; train and MD share the same operator.
* **Q¹** (``q1`` / ``latent``) — partner-perturbed charges from the AB dimer
  forward.  Dimer-only (the AB context is undefined for N>2).

Modes
-----
**fixed** (Mode A)
    ``q_MM = q_CGenFF``.  Default hybrid train and MD calculator.

**q0** (Q⁰; unperturbed monomers)
    ``q_MM = neutralize_per_monomer(Q⁰)``.  Train + MD; any ``n_monomers``.

**latent** / **q1** (Mode B / Q¹)
    ``q_MM = neutralize_per_monomer(Q¹)`` with Q¹ from the AB dimer forward.
    Dimer-only (train + MD).

**fixed_plus_latent** (Mode C)
    ``q_MM = q_CGenFF + neutralize_per_monomer(Q¹)``.  AB-context Q¹.
    Dimer-only.

**latent_mean** (Mode D, MD-only)
    Frozen offline mean of neutralize(q_ML) over training homo-dimers, tiled
    across the box.  See ``scripts/compute_latent_monomer_charges.py``.

**latent_dynamic** (Mode E, MD-only)
    Live weighted mean of Q¹ over active ML-dimer slots (L2).  Heuristic;
    not train-identical to Mode B.
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
    "assemble_q0_from_monomer_forwards",
    "hybrid_mm_metadata_dict",
    "mm_charge_mode_from_charge_correction",
    "mm_charge_mode_is_dynamic_liquid",
    "mm_charge_mode_is_q0",
    "mm_charge_mode_is_static_template",
    "mm_charge_mode_needs_q_ml",
    "parse_mm_charge_mode",
    "require_charge_head_for_mode",
    "resolve_hybrid_mm_charge_mode",
]


class MMChargeMode(str, Enum):
    """How ``E_MM`` Coulomb gets its per-atom charges."""

    FIXED = "fixed"
    Q0 = "q0"
    LATENT = "latent"  # Q¹ (AB-perturbed); keep value for checkpoint compat
    FIXED_PLUS_LATENT = "fixed_plus_latent"
    LATENT_MEAN = "latent_mean"
    LATENT_DYNAMIC = "latent_dynamic"


def parse_mm_charge_mode(value: str | MMChargeMode | None) -> MMChargeMode:
    """Parse a CLI/config string into :class:`MMChargeMode`."""
    if value is None:
        return MMChargeMode.FIXED
    if isinstance(value, MMChargeMode):
        return value
    key = str(value).strip().lower().replace("-", "_").replace("+", "_plus_")
    # Unicode / ASCII superscripts → plain digits for Q⁰ / Q¹ aliases.
    key = key.replace("⁰", "0").replace("¹", "1").replace("q⁰", "q0").replace("q¹", "q1")
    aliases = {
        "fixed": MMChargeMode.FIXED,
        "a": MMChargeMode.FIXED,
        "mode_a": MMChargeMode.FIXED,
        # Q⁰ — unperturbed monomer charges (train + liquid MD).
        "q0": MMChargeMode.Q0,
        "q_0": MMChargeMode.Q0,
        "latent_q0": MMChargeMode.Q0,
        "unperturbed": MMChargeMode.Q0,
        "monomer": MMChargeMode.Q0,
        "monomer_charges": MMChargeMode.Q0,
        # Q¹ — AB-perturbed (Mode B); ``latent`` is the historical name.
        "latent": MMChargeMode.LATENT,
        "q1": MMChargeMode.LATENT,
        "q_1": MMChargeMode.LATENT,
        "latent_q1": MMChargeMode.LATENT,
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
        "latent_dynamic": MMChargeMode.LATENT_DYNAMIC,
        "latent-dynamic": MMChargeMode.LATENT_DYNAMIC,
        "dynamic": MMChargeMode.LATENT_DYNAMIC,
        "e": MMChargeMode.LATENT_DYNAMIC,
        "mode_e": MMChargeMode.LATENT_DYNAMIC,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown mm_charge_mode={value!r}. "
            f"Expected one of: {', '.join(m.value for m in MMChargeMode)} "
            "(aliases: q0/q1/latent/...)."
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
    return mode in (
        MMChargeMode.Q0,
        MMChargeMode.LATENT,
        MMChargeMode.FIXED_PLUS_LATENT,
        MMChargeMode.LATENT_DYNAMIC,
    )


def mm_charge_mode_is_q0(mode: str | MMChargeMode) -> bool:
    """True for Q⁰ — unperturbed monomer charges (train + liquid MD)."""
    return parse_mm_charge_mode(mode) is MMChargeMode.Q0


def mm_charge_mode_is_dynamic_liquid(mode: str | MMChargeMode) -> bool:
    """True for ``latent_dynamic`` -- live, per-step aggregation over active
    ML-dimer partners (see ``mmml_calculator.calculate_ml_contributions``),
    as opposed to ``latent``/``fixed_plus_latent``'s single AB-dimer forward
    (dimer-only) or ``latent_mean``'s frozen offline template.
    """
    return parse_mm_charge_mode(mode) is MMChargeMode.LATENT_DYNAMIC


def mm_charge_mode_is_static_template(mode: str | MMChargeMode) -> bool:
    """True when the mode's charges are a precomputed, per-step-static array.

    ``latent_mean`` needs neither a live ``q_ML`` forward nor the dimer-only
    gate: its charges were fixed offline (see
    ``mmml.models.latent_charge_template``) and are injected once.
    """
    return parse_mm_charge_mode(mode) is MMChargeMode.LATENT_MEAN


def assemble_q0_from_monomer_forwards(
    q_a: Array,
    q_b: Array,
    mol_id: Array,
    *,
    batch_size: int,
    n_atoms: int,
) -> Array:
    """Build Q⁰ on the padded dimer layout from isolated A/B charge heads.

    Training evaluates ``out_a`` / ``out_b`` with monomer-restricted masks
    (same forwards as E⁰).  Atoms with ``mol_id == 0`` take ``q_a``; ``== 1``
    take ``q_b``; padding (``mol_id < 0``) is zero.
    """
    mid = jnp.asarray(mol_id).reshape(batch_size, n_atoms)
    qa = jnp.asarray(q_a).reshape(batch_size, n_atoms)
    qb = jnp.asarray(q_b).reshape(batch_size, n_atoms)
    return jnp.where(mid == 0, qa, jnp.where(mid == 1, qb, jnp.zeros_like(qa)))


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
        ``fixed``, ``q0``, ``latent``/``q1``, or ``fixed_plus_latent``.
    q_cgenff
        Fixed CGenFF / PSF charges, shape ``(..., n_atoms)``.
    q_ml
        Charge-head output (same shape as ``q_cgenff``).  Required for
        ``q0`` / ``latent`` / ``fixed_plus_latent``; ignored for ``fixed``.
        For ``q0`` this must already be assembled from isolated monomer
        forwards (:func:`assemble_q0_from_monomer_forwards`).
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

    if mode in (
        MMChargeMode.Q0,
        MMChargeMode.LATENT,
        MMChargeMode.LATENT_DYNAMIC,
    ):
        # Replace CGenFF with neutralize(q_ML).  Modes differ only in how q_ML
        # was obtained upstream (Q⁰ monomers / Q¹ AB / multi-dimer avg).
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
            "learn_mm_lj_scales": False,
        }
    mode = resolve_hybrid_mm_charge_mode(
        mm_charge_mode=getattr(hybrid_mm, "mm_charge_mode", None),
        charge_correction=bool(getattr(hybrid_mm, "charge_correction", False)),
    )
    out = {
        "hybrid_mm": True,
        "charge_correction": mode is MMChargeMode.FIXED_PLUS_LATENT,
        "mm_charge_mode": mode.value,
        "mm_switch_on": float(getattr(hybrid_mm, "mm_switch_on", 8.0)),
        "mm_switch_width": float(getattr(hybrid_mm, "mm_switch_width", 5.0)),
        "ml_switch_width": float(getattr(hybrid_mm, "ml_switch_width", 1.5)),
        "complementary_handoff": bool(
            getattr(hybrid_mm, "complementary_handoff", True)
        ),
        "learn_mm_lj_scales": bool(getattr(hybrid_mm, "learn_mm_lj_scales", False)),
        "include_lj": bool(getattr(hybrid_mm, "include_lj", True)),
        "lr_solver": str(getattr(hybrid_mm, "lr_solver", "mic")),
    }
    type_names = getattr(hybrid_mm, "cgenff_type_names", None)
    if type_names is not None:
        out["cgenff_type_names"] = [str(n) for n in type_names]
    return out

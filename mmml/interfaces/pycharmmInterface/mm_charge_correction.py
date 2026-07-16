"""MD-side helpers for hybrid MM charge Modes B/C (dimer-only).

See ``docs/hybrid-mm-charges.md`` and :mod:`mmml.models.mm_charge_mode`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp

from mmml.models.mm_charge_mode import (
    MMChargeMode,
    apply_mm_charge_mode,
    hybrid_mm_metadata_dict,
    mm_charge_mode_needs_q_ml,
    parse_mm_charge_mode,
    require_charge_head_for_mode,
    resolve_hybrid_mm_charge_mode,
)

Array = jnp.ndarray

__all__ = [
    "assert_mm_charge_mode_dimer_supported",
    "assert_mode_c_dimer_supported",
    "hybrid_mm_metadata_dict",
    "load_hybrid_mm_metadata",
    "map_padded_fragment_charges_to_global",
    "mm_charges_fixed_plus_latent",
    "resolve_mm_charge_mode_arg",
    "warn_mm_charge_mode_mismatch",
]


def resolve_mm_charge_mode_arg(
    *,
    mm_charge_correction: bool = False,
    mm_charge_mode: str | MMChargeMode | None = None,
) -> MMChargeMode:
    """Resolve CLI/setup kwargs to a mode (bool aliases Mode C)."""
    return resolve_hybrid_mm_charge_mode(
        mm_charge_mode=mm_charge_mode,
        charge_correction=bool(mm_charge_correction),
    )


def assert_mm_charge_mode_dimer_supported(
    mode: MMChargeMode,
    *,
    n_monomers: int,
    has_charges: bool,
    lr_solver: str | None,
    doML: bool,
    doML_dimer: bool,
) -> None:
    """Raise if Mode B/C cannot be applied under the current calculator settings."""
    mode = parse_mm_charge_mode(mode)
    if mode is MMChargeMode.FIXED:
        return
    require_charge_head_for_mode(mode, has_charges=has_charges)
    if int(n_monomers) != 2:
        raise ValueError(
            f"mm_charge_mode={mode.value} MD support is dimer-only "
            f"(n_monomers==2); got n_monomers={n_monomers}. Liquid q_ML context "
            "is undefined — see docs/hybrid-mm-charges.md."
        )
    if not doML or not doML_dimer:
        raise ValueError(
            f"mm_charge_mode={mode.value} requires doML and doML_dimer so the "
            "AB forward can supply q_ML (same as hybrid training)."
        )
    lr = (lr_solver or "").strip().lower()
    if lr in {"jax_pme", "jax-pme", "pme"}:
        raise ValueError(
            f"mm_charge_mode={mode.value} is not wired through JAX-PME yet; "
            "refuse half-applied long-range. Use lr_solver=None / jax_mic for "
            "dimer Mode B/C parity."
        )


# Backward-compatible alias (Mode C was the first non-fixed mode).
assert_mode_c_dimer_supported = assert_mm_charge_mode_dimer_supported


def map_padded_fragment_charges_to_global(
    q_fragment: Array,
    global_idx: Array,
    atom_mask: Array,
    n_atoms: int,
) -> Array:
    """Scatter padded fragment charges onto a full-system vector (mask-safe)."""
    q = jnp.asarray(q_fragment).reshape(-1)
    idx = jnp.asarray(global_idx).reshape(-1).astype(jnp.int32)
    mask = jnp.asarray(atom_mask).reshape(-1).astype(q.dtype)
    n = int(n_atoms)
    # Avoid writing padding filler indices (often idx[0]) when mask is False.
    safe_idx = jnp.where(mask > 0, idx, jnp.int32(n))
    q_pad = jnp.concatenate([jnp.zeros((n,), dtype=q.dtype), jnp.zeros((1,), dtype=q.dtype)])
    q_pad = q_pad.at[safe_idx].add(jnp.where(mask > 0, q[: safe_idx.shape[0]], 0.0))
    return q_pad[:n]


def mm_charges_fixed_plus_latent(
    q_cgenff: Array,
    q_ml_global: Array,
    monomer_id: Array,
    *,
    n_monomers: int = 2,
) -> Array:
    """``q_eff = q_CGenFF + neutralize(q_ML)`` for Mode C."""
    return apply_mm_charge_mode(
        MMChargeMode.FIXED_PLUS_LATENT,
        q_cgenff,
        q_ml_global,
        monomer_id,
        n_monomers=n_monomers,
    )


def load_hybrid_mm_metadata(restart_path: str | Path | None) -> dict[str, Any] | None:
    """Load ``hybrid_mm.json`` sidecar next to a checkpoint, if present."""
    if restart_path is None:
        return None
    path = Path(restart_path)
    candidates = []
    if path.is_file():
        candidates.append(path.parent / "hybrid_mm.json")
    else:
        candidates.append(path / "hybrid_mm.json")
        candidates.append(path.parent / "hybrid_mm.json")
    import json

    for cand in candidates:
        if cand.is_file():
            with open(cand) as f:
                return json.load(f)
    return None


def warn_mm_charge_mode_mismatch(
    mode: MMChargeMode,
    meta: dict[str, Any] | None,
    *,
    stream=None,
) -> None:
    """Warn when CLI mode disagrees with checkpoint hybrid_mm metadata."""
    if not meta:
        return
    trained = parse_mm_charge_mode(meta.get("mm_charge_mode", "fixed"))
    if trained is mode:
        return
    msg = (
        f"WARNING: mm_charge_mode={mode.value} but checkpoint hybrid_mm.json "
        f"records mm_charge_mode={trained.value}. Mode B/C are opt-in on MD; "
        "parity requires the same mode on both sides."
    )
    print(msg, file=stream, flush=True)


# Re-export for callers that check whether ML charges are needed.
needs_q_ml = mm_charge_mode_needs_q_ml

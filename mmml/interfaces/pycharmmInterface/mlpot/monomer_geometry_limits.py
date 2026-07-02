"""Bond- and reference-geometry limits for monomer extent / intra contacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mmml.utils.geometry_checks import (
    build_bond_exclusion_pairs,
    monomer_axis_extent,
    normalize_atom_pair,
)

# Defaults when auto limits cannot be computed (legacy behaviour).
DEFAULT_MAX_MONOMER_EXTENT_A = 12.0
DEFAULT_INTRA_MIN_DISTANCE_A = 0.5
DEFAULT_INTER_MIN_DISTANCE_A = 1.5

EXTENT_MARGIN = 1.30
EXTENT_ABS_BUFFER_A = 0.35
EXTENT_MIN_BOND_FACTOR = 2.20
INTRA_REFERENCE_FRACTION = 0.80
INTRA_FLOOR_A = 0.45
INTER_VDW_FRACTION = 0.88


@dataclass(frozen=True)
class MonomerGeometryLimits:
    """Conservative geometry thresholds derived from reference bonded geometry."""

    max_monomer_extent_A: float
    intra_min_distance_A: float
    inter_min_distance_A: float
    reference_max_extent_A: float
    reference_intra_min_A: float
    min_bond_length_A: float
    max_bond_length_A: float
    notes: str = ""


def psf_bond_pairs_0based(*, exclude_1_3: bool = False) -> list[tuple[int, int]]:
    """PSF 1–2 bond pairs as 0-based atom index pairs."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.psf as psf

    nbond = int(psf.get_nbond())
    if nbond <= 0:
        return []
    raw_ib_jb = psf.get_ib_jb()
    if isinstance(raw_ib_jb, tuple) and len(raw_ib_jb) == 2:
        ib, jb = raw_ib_jb
    else:
        return []
    if exclude_1_3:
        pairs = build_bond_exclusion_pairs(ib, jb, exclude_1_3=True)
        return [tuple(int(x) for x in p) for p in pairs]
    out: list[tuple[int, int]] = []
    for i, j in zip(ib, jb, strict=False):
        a, b = int(i) - 1, int(j) - 1
        if a > b:
            a, b = b, a
        out.append((a, b))
    return out


def _bond_lengths_in_monomer(
    positions: np.ndarray,
    offsets: np.ndarray,
    monomer: int,
    bond_pairs: list[tuple[int, int]],
) -> list[float]:
    pos = np.asarray(positions, dtype=np.float64)
    si, ei = int(offsets[monomer]), int(offsets[monomer + 1])
    lengths: list[float] = []
    for a, b in bond_pairs:
        if si <= a < ei and si <= b < ei:
            lengths.append(float(np.linalg.norm(pos[b] - pos[a])))
    return lengths


def _min_nonexcluded_intra_distance(
    positions: np.ndarray,
    offsets: np.ndarray,
    monomer: int,
    excluded_pairs: frozenset[tuple[int, int]],
) -> float | None:
    pos = np.asarray(positions, dtype=np.float64)
    si, ei = int(offsets[monomer]), int(offsets[monomer + 1])
    best = float("inf")
    for gi in range(si, ei):
        for gj in range(gi + 1, ei):
            pair = normalize_atom_pair(gi, gj)
            if pair in excluded_pairs:
                continue
            dist = float(np.linalg.norm(pos[gj] - pos[gi]))
            if dist < best:
                best = dist
    return None if not np.isfinite(best) else float(best)


def compute_monomer_geometry_limits(
    reference_positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    bond_pairs_12: list[tuple[int, int]] | None = None,
    excluded_pairs: frozenset[tuple[int, int]] | None = None,
    atomic_numbers: np.ndarray | None = None,
    extent_margin: float = EXTENT_MARGIN,
    intra_fraction: float = INTRA_REFERENCE_FRACTION,
    inter_vdw_fraction: float = INTER_VDW_FRACTION,
    default_inter_min_A: float = DEFAULT_INTER_MIN_DISTANCE_A,
) -> MonomerGeometryLimits | None:
    """Derive conservative extent / intra / inter limits from reference geometry."""
    pos = np.asarray(reference_positions, dtype=np.float64)
    offsets = np.asarray(monomer_offsets, dtype=int)
    if pos.ndim != 2 or pos.shape[1] != 3 or offsets.size < 2:
        return None
    if int(offsets[-1]) != int(pos.shape[0]):
        return None
    if not np.all(np.isfinite(pos)):
        return None

    n_monomers = int(len(offsets) - 1)
    if n_monomers <= 0:
        return None

    bonds_12 = list(bond_pairs_12 or [])
    excluded = excluded_pairs if excluded_pairs is not None else frozenset()

    ref_extents: list[float] = []
    ref_intra_mins: list[float] = []
    all_bond_lengths: list[float] = []

    for mi in range(n_monomers):
        ref_extents.append(float(monomer_axis_extent(pos, offsets, mi)))
        intra_min = _min_nonexcluded_intra_distance(pos, offsets, mi, excluded)
        if intra_min is not None:
            ref_intra_mins.append(intra_min)
        if bonds_12:
            all_bond_lengths.extend(_bond_lengths_in_monomer(pos, offsets, mi, bonds_12))

    if not ref_extents:
        return None

    ref_max_extent = float(np.max(ref_extents))
    max_bond = float(np.max(all_bond_lengths)) if all_bond_lengths else 0.0
    min_bond = float(np.min(all_bond_lengths)) if all_bond_lengths else 0.0

    max_extent = ref_max_extent * float(extent_margin) + float(EXTENT_ABS_BUFFER_A)
    if max_bond > 0.0:
        max_extent = max(max_extent, float(EXTENT_MIN_BOND_FACTOR) * max_bond)
    max_extent = float(np.clip(max_extent, max(2.5, max_bond * 1.5 if max_bond else 2.5), 30.0))

    if ref_intra_mins:
        ref_intra_min = float(np.min(ref_intra_mins))
        intra_min = max(float(INTRA_FLOOR_A), ref_intra_min * float(intra_fraction))
        if min_bond > 0.0:
            intra_min = min(intra_min, 0.92 * min_bond)
    elif min_bond > 0.0:
        ref_intra_min = 0.0
        intra_min = max(float(INTRA_FLOOR_A), 0.65 * min_bond)
    else:
        ref_intra_min = float(DEFAULT_INTRA_MIN_DISTANCE_A)
        intra_min = float(DEFAULT_INTRA_MIN_DISTANCE_A)

    inter_min = float(default_inter_min_A)
    if atomic_numbers is not None:
        z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
        from mmml.utils.intermonomer_geometry import vdw_contact_hint_A, _element_symbol

        hints: list[float] = []
        for si, ei in zip(offsets[:-1], offsets[1:], strict=False):
            elems = {_element_symbol(int(z[a])) for a in range(int(si), int(ei))}
            for ei_a in elems:
                for ei_b in elems:
                    hint = vdw_contact_hint_A(ei_a, ei_b)
                    if hint is not None:
                        hints.append(float(hint))
        if hints:
            inter_min = max(
                inter_min,
                float(inter_vdw_fraction) * float(np.min(hints)),
            )

    notes = (
        f"ref_extent={ref_max_extent:.2f} Å, ref_intra_min="
        f"{ref_intra_min:.2f} Å"
        + (
            f", bonds [{min_bond:.2f}, {max_bond:.2f}] Å"
            if all_bond_lengths
            else ""
        )
    )
    return MonomerGeometryLimits(
        max_monomer_extent_A=max_extent,
        intra_min_distance_A=intra_min,
        inter_min_distance_A=inter_min,
        reference_max_extent_A=ref_max_extent,
        reference_intra_min_A=ref_intra_min,
        min_bond_length_A=min_bond,
        max_bond_length_A=max_bond,
        notes=notes,
    )


def resolve_monomer_offsets_for_limits(
    mlpot_ctx: Any,
    *,
    n_monomers: int,
    n_atoms: int,
) -> np.ndarray | None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        resolve_monomer_offsets_for_ctx,
    )

    return resolve_monomer_offsets_for_ctx(
        mlpot_ctx, n_monomers=int(n_monomers), n_atoms=int(n_atoms)
    )


def resolve_reference_positions_for_limits(mlpot_ctx: Any) -> np.ndarray | None:
    """Reference geometry for limit derivation (baseline preferred)."""
    for attr in ("geometry_baseline_positions", "geometry_mini_positions"):
        pos = getattr(mlpot_ctx, attr, None)
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float64)
        if arr.size and np.all(np.isfinite(arr)):
            return arr
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

        pos = get_charmm_positions_array()
        if pos is not None:
            arr = np.asarray(pos, dtype=np.float64)
            if arr.size and np.all(np.isfinite(arr)):
                return arr
    except Exception:
        pass
    return None


def compute_geometry_limits_from_mlpot_ctx(
    mlpot_ctx: Any,
    *,
    n_monomers: int,
    default_inter_min_A: float = DEFAULT_INTER_MIN_DISTANCE_A,
) -> MonomerGeometryLimits | None:
    """Build limits from live PSF + stored reference coordinates."""
    ref = resolve_reference_positions_for_limits(mlpot_ctx)
    if ref is None:
        return None
    offsets = resolve_monomer_offsets_for_limits(
        mlpot_ctx, n_monomers=int(n_monomers), n_atoms=int(ref.shape[0])
    )
    if offsets is None:
        return None

    bonds_12: list[tuple[int, int]] = []
    excluded = frozenset()
    try:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm.psf as psf

        nbond = int(psf.get_nbond())
        if nbond > 0:
            raw_ib_jb = psf.get_ib_jb()
            if isinstance(raw_ib_jb, tuple) and len(raw_ib_jb) == 2:
                ib, jb = raw_ib_jb
                excluded = build_bond_exclusion_pairs(ib, jb, exclude_1_3=True)
                for i, j in zip(ib, jb, strict=False):
                    a, b = int(i) - 1, int(j) - 1
                    if a > b:
                        a, b = b, a
                    bonds_12.append((a, b))
    except Exception:
        pass

    z = getattr(mlpot_ctx, "ml_Z", None)
    return compute_monomer_geometry_limits(
        ref,
        offsets,
        bond_pairs_12=bonds_12,
        excluded_pairs=excluded,
        atomic_numbers=np.asarray(z, dtype=int) if z is not None else None,
        default_inter_min_A=float(default_inter_min_A),
    )


def geometry_limits_auto_enabled(args: Any | None) -> bool:
    if args is None:
        return True
    return not bool(getattr(args, "no_dynamics_geometry_limits_auto", False))


def apply_geometry_limits_to_overlap_config(
    overlap: Any,
    mlpot_ctx: Any | None,
    args: Any | None = None,
    *,
    verbose: bool = False,
) -> Any:
    """Replace overlap extent/intra/inter thresholds with bond-derived limits."""
    if mlpot_ctx is None or not geometry_limits_auto_enabled(args):
        return overlap

    explicit_extent = getattr(args, "dynamics_max_monomer_extent", None)
    explicit_intra = getattr(args, "dynamics_intra_min_distance", None)
    explicit_inter = getattr(args, "dynamics_overlap_min_distance", None)

    limits = compute_geometry_limits_from_mlpot_ctx(
        mlpot_ctx,
        n_monomers=int(getattr(overlap, "n_monomers", 1) or 1),
        default_inter_min_A=float(
            explicit_inter if explicit_inter is not None else DEFAULT_INTER_MIN_DISTANCE_A
        ),
    )
    if limits is None:
        return overlap

    setattr(mlpot_ctx, "_monomer_geometry_limits", limits)

    max_extent = (
        float(explicit_extent)
        if explicit_extent is not None
        else float(limits.max_monomer_extent_A)
    )
    intra_min = (
        float(explicit_intra)
        if explicit_intra is not None
        else float(limits.intra_min_distance_A)
    )
    inter_min = (
        float(explicit_inter)
        if explicit_inter is not None
        else float(limits.inter_min_distance_A)
    )

    updated = replace(
        overlap,
        max_monomer_extent_A=max_extent,
        intra_min_distance_A=intra_min,
        min_distance_A=inter_min,
    )
    if verbose or (args is not None and not getattr(args, "quiet", False)):
        print(
            "Monomer geometry limits (bond/reference auto): "
            f"max_extent={max_extent:.2f} Å "
            f"(default was {DEFAULT_MAX_MONOMER_EXTENT_A:.1f}), "
            f"intra_min={intra_min:.2f} Å "
            f"(default was {DEFAULT_INTRA_MIN_DISTANCE_A:.1f}), "
            f"inter_min={inter_min:.2f} Å "
            f"({limits.notes})",
            flush=True,
        )
    return updated


def restore_monomer_from_template_for_violation(
    mlpot_ctx: Any,
    monomer_index: int,
    *,
    context: str,
    restart_path: Any | None = None,
) -> bool:
    """Template-restore one monomer before bonded / JAX recovery."""
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        restore_flagged_monomers_from_template,
    )

    return restore_flagged_monomers_from_template(
        mlpot_ctx,
        (int(monomer_index),),
        context=context,
        restart_path=restart_path,
        verbose=True,
    )

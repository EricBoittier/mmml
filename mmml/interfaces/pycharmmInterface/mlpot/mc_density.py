"""Post-build Monte Carlo density equalization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
    SOLVENT_BULK_PROPS,
    parse_composition_dict,
    resolve_target_density_g_cm3,
    total_mass_g_for_composition,
)
from mmml.utils.geometry_checks import (
    find_worst_intermonomer_overlap,
    wrap_monomers_primary_cell,
)


@dataclass(frozen=True)
class McDensityResult:
    """Outcome of a default post-build density equalization attempt."""

    enabled: bool
    ran: bool
    reason: str
    initial_box_A: float | None
    final_box_A: float | None
    target_density_g_cm3: float | None
    initial_density_g_cm3: float | None
    final_density_g_cm3: float | None
    accepted_moves: int = 0
    attempted_moves: int = 0
    best_objective: float | None = None
    min_intermonomer_distance_A: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def monomer_offsets_from_atoms_per(atoms_per_list: list[int] | tuple[int, ...]) -> np.ndarray:
    """Return atom offsets from per-monomer atom counts."""
    counts = np.asarray(list(atoms_per_list), dtype=int)
    if counts.ndim != 1 or counts.size < 1 or np.any(counts <= 0):
        raise ValueError("atoms_per_list must contain positive per-monomer atom counts")
    offsets = np.zeros(counts.size + 1, dtype=int)
    offsets[1:] = np.cumsum(counts)
    return offsets


def density_g_cm3_for_box(composition: dict[str, int], box_side_A: float) -> float:
    """Mass density for a cubic box side in Angstrom."""
    side = float(box_side_A)
    if side <= 0.0:
        raise ValueError(f"box_side_A must be positive, got {box_side_A}")
    mass_g = total_mass_g_for_composition(composition)
    return float(mass_g) / (side**3 * 1.0e-24)


def resolve_mc_density_target_g_cm3(
    args: Any,
    composition: dict[str, int] | None,
) -> tuple[float | None, str]:
    """Resolve the default MC density target, returning ``(target, source)``."""
    if composition is None:
        return None, "no_composition"
    explicit = getattr(args, "mc_density_target_g_cm3", None)
    if explicit is not None:
        rho = float(explicit)
        if rho <= 0.0:
            raise ValueError(f"--mc-density-target-g-cm3 must be positive, got {rho}")
        return rho, "mc_density_target"
    if getattr(args, "target_density_g_cm3", None) is not None:
        return resolve_target_density_g_cm3(args, composition), "target_density"
    if getattr(args, "bulk_density_fraction", None) is not None:
        return resolve_target_density_g_cm3(args, composition), "bulk_density_fraction"
    if len(composition) == 1:
        residue = next(iter(composition)).upper()
        props = SOLVENT_BULK_PROPS.get(residue)
        if props is not None:
            return float(props["rho_g_cm3"]), "bulk_density_table"
    return None, "no_density_target"


def _scale_molecule_coms(
    positions: np.ndarray,
    offsets: np.ndarray,
    *,
    old_box_A: float,
    new_box_A: float,
) -> np.ndarray:
    """Scale molecule COMs with the cubic box while preserving intramolecular geometry."""
    pos = np.asarray(positions, dtype=float)
    old_L = float(old_box_A)
    new_L = float(new_box_A)
    old_center = np.full(3, 0.5 * old_L, dtype=float)
    new_center = np.full(3, 0.5 * new_L, dtype=float)
    scale = new_L / old_L
    out = pos.copy()
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        chunk = pos[s:e]
        com = chunk.mean(axis=0)
        new_com = new_center + (com - old_center) * scale
        out[s:e] = chunk + (new_com - com)
    return wrap_monomers_primary_cell(out, offsets, np.diag([new_L, new_L, new_L]))


def _density_objective(box_side_A: float, *, total_mass_g: float, target_density_g_cm3: float) -> float:
    rho = float(total_mass_g) / (float(box_side_A) ** 3 * 1.0e-24)
    return abs(np.log(rho / float(target_density_g_cm3)))


def apply_mc_density_equalization(
    args: Any,
    positions: np.ndarray,
    *,
    atoms_per_list: list[int] | tuple[int, ...],
    composition: dict[str, int] | None = None,
    box_side_A: float | None,
    use_pbc: bool,
    handoff_present: bool = False,
    min_intermonomer_distance_A: float | None = None,
) -> tuple[np.ndarray, float | None, McDensityResult]:
    """Apply default post-build MC density equalization when policy allows it."""
    enabled = bool(getattr(args, "mc_density_equalize", True))
    pos = np.asarray(positions, dtype=float)
    comp = composition
    if comp is None:
        comp = parse_composition_dict(getattr(args, "composition", None))
    if not enabled:
        return pos, box_side_A, McDensityResult(
            enabled=False,
            ran=False,
            reason="disabled",
            initial_box_A=box_side_A,
            final_box_A=box_side_A,
            target_density_g_cm3=None,
            initial_density_g_cm3=None,
            final_density_g_cm3=None,
        )
    if not use_pbc:
        return pos, box_side_A, McDensityResult(
            enabled=True,
            ran=False,
            reason="not_pbc",
            initial_box_A=box_side_A,
            final_box_A=box_side_A,
            target_density_g_cm3=None,
            initial_density_g_cm3=None,
            final_density_g_cm3=None,
        )
    if handoff_present:
        return pos, box_side_A, McDensityResult(
            enabled=True,
            ran=False,
            reason="handoff",
            initial_box_A=box_side_A,
            final_box_A=box_side_A,
            target_density_g_cm3=None,
            initial_density_g_cm3=None,
            final_density_g_cm3=None,
        )
    if getattr(args, "box_size", None) is not None:
        return pos, box_side_A, McDensityResult(
            enabled=True,
            ran=False,
            reason="fixed_box",
            initial_box_A=box_side_A,
            final_box_A=box_side_A,
            target_density_g_cm3=None,
            initial_density_g_cm3=None,
            final_density_g_cm3=None,
        )
    if comp is None:
        return pos, box_side_A, McDensityResult(
            enabled=True,
            ran=False,
            reason="no_composition",
            initial_box_A=box_side_A,
            final_box_A=box_side_A,
            target_density_g_cm3=None,
            initial_density_g_cm3=None,
            final_density_g_cm3=None,
        )
    if box_side_A is None or float(box_side_A) <= 0.0:
        return pos, box_side_A, McDensityResult(
            enabled=True,
            ran=False,
            reason="no_box",
            initial_box_A=box_side_A,
            final_box_A=box_side_A,
            target_density_g_cm3=None,
            initial_density_g_cm3=None,
            final_density_g_cm3=None,
        )

    target, target_source = resolve_mc_density_target_g_cm3(args, comp)
    if target is None:
        try:
            unresolved_density = density_g_cm3_for_box(comp, float(box_side_A))
        except ValueError:
            unresolved_density = None
        return pos, box_side_A, McDensityResult(
            enabled=True,
            ran=False,
            reason=target_source,
            initial_box_A=float(box_side_A),
            final_box_A=float(box_side_A),
            target_density_g_cm3=None,
            initial_density_g_cm3=unresolved_density,
            final_density_g_cm3=unresolved_density,
        )

    steps = int(getattr(args, "mc_density_steps", 64) or 0)
    if steps <= 0:
        rho = density_g_cm3_for_box(comp, float(box_side_A))
        return pos, float(box_side_A), McDensityResult(
            enabled=True,
            ran=False,
            reason="zero_steps",
            initial_box_A=float(box_side_A),
            final_box_A=float(box_side_A),
            target_density_g_cm3=float(target),
            initial_density_g_cm3=rho,
            final_density_g_cm3=rho,
            attempted_moves=0,
        )

    try:
        total_mass_g = total_mass_g_for_composition(comp)
    except ValueError:
        return pos, box_side_A, McDensityResult(
            enabled=True,
            ran=False,
            reason="no_mass_metadata",
            initial_box_A=float(box_side_A),
            final_box_A=float(box_side_A),
            target_density_g_cm3=float(target),
            initial_density_g_cm3=None,
            final_density_g_cm3=None,
        )
    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    if int(offsets[-1]) != int(pos.shape[0]):
        raise ValueError(
            f"atoms_per_list sums to {int(offsets[-1])}, but positions has {pos.shape[0]} rows"
        )
    initial_L = float(box_side_A)
    initial_rho = float(total_mass_g) / (initial_L**3 * 1.0e-24)
    target_L = (float(total_mass_g) / (float(target) * 1.0e-24)) ** (1.0 / 3.0)
    min_scale = float(getattr(args, "mc_density_min_scale", 0.75) or 0.75)
    max_scale = float(getattr(args, "mc_density_max_scale", 1.50) or 1.50)
    if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
        raise ValueError("--mc-density-min-scale and --mc-density-max-scale must be positive with min <= max")
    min_L = initial_L * min_scale
    max_L = initial_L * max_scale
    target_L = float(np.clip(target_L, min_L, max_L))
    seed = getattr(args, "mc_density_seed", None)
    if seed is None:
        seed = getattr(args, "seed", 123)
    rng = np.random.default_rng(int(seed))
    step_scale = max(0.0, float(getattr(args, "mc_density_step_scale", 0.04) or 0.0))
    temperature = max(1.0e-12, float(getattr(args, "mc_density_temperature", 0.02) or 0.0))
    min_contact = (
        float(min_intermonomer_distance_A)
        if min_intermonomer_distance_A is not None
        else float(getattr(args, "min_intermonomer_atom_distance", 0.1) or 0.1)
    )

    current_L = initial_L
    current_pos = pos.copy()
    current_obj = _density_objective(current_L, total_mass_g=total_mass_g, target_density_g_cm3=target)
    best_L = current_L
    best_pos = current_pos.copy()
    best_obj = current_obj
    accepted = 0
    best_contact = None
    cell = np.diag([current_L, current_L, current_L])
    if len(offsets) > 2:
        best_contact, _ = find_worst_intermonomer_overlap(current_pos, offsets, cell=cell)

    log_target_L = np.log(target_L)
    for _ in range(steps):
        drift = 0.35 * (log_target_L - np.log(current_L))
        proposal_log_L = np.log(current_L) + drift
        if step_scale > 0.0:
            proposal_log_L += float(rng.normal(0.0, step_scale))
        proposal_L = float(np.clip(np.exp(proposal_log_L), min_L, max_L))
        proposal_pos = _scale_molecule_coms(
            current_pos,
            offsets,
            old_box_A=current_L,
            new_box_A=proposal_L,
        )
        if len(offsets) > 2 and min_contact > 0.0:
            contact, _ = find_worst_intermonomer_overlap(
                proposal_pos,
                offsets,
                cell=np.diag([proposal_L, proposal_L, proposal_L]),
            )
            if contact < min_contact:
                continue
        else:
            contact = float("inf")
        proposal_obj = _density_objective(
            proposal_L,
            total_mass_g=total_mass_g,
            target_density_g_cm3=target,
        )
        delta = proposal_obj - current_obj
        if delta <= 0.0 or float(rng.random()) < float(np.exp(-delta / temperature)):
            current_L = proposal_L
            current_pos = proposal_pos
            current_obj = proposal_obj
            accepted += 1
        if proposal_obj < best_obj:
            best_L = proposal_L
            best_pos = proposal_pos
            best_obj = proposal_obj
            best_contact = contact

    final_rho = density_g_cm3_for_box(comp, best_L)
    return best_pos, best_L, McDensityResult(
        enabled=True,
        ran=True,
        reason=target_source,
        initial_box_A=initial_L,
        final_box_A=best_L,
        target_density_g_cm3=float(target),
        initial_density_g_cm3=initial_rho,
        final_density_g_cm3=final_rho,
        accepted_moves=accepted,
        attempted_moves=steps,
        best_objective=best_obj,
        min_intermonomer_distance_A=None if best_contact is None else float(best_contact),
    )

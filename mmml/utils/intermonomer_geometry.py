"""Inter-monomer contact thresholds and human-readable prep-gate reporting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

# Prep gate: ML-safe inter-monomer MIC floor before USER potential is enabled.
DEFAULT_PRE_MLPOT_OVERLAP_MIN_A = 2.3

# Element-pair prep floors for dense halogenated liquids (e.g. DCM).
DEFAULT_PRE_MLPOT_H_HEAVY_MIN_A = 2.4
DEFAULT_PRE_MLPOT_HEAVY_HEAVY_MIN_A = 2.9
DEFAULT_PRE_MLPOT_H_H_MIN_A = 2.3

# Hard abort before MLpot SD when hybrid forces are already catastrophic.
DEFAULT_MLPOT_REGISTRATION_MAX_GRMS_KCALMOL_A = 50.0

# CHARMM <MKIMAT2> group Min-Distance before MLpot USER (tighter than MIC prep floor).
# Dense all-ML PBC liquids can pass MIC ~3.0 Å yet still hit GRMS >2000 when MKIMAT2 ≈3.0.
DEFAULT_CHARMM_IMAGE_MLPOT_MIN_A = 3.5
# 52-monomer DCM @ L≈28 Å: MKIMAT2 ≈4.9 Å still yields hybrid GRMS >2000 at registration.
DEFAULT_CHARMM_IMAGE_MLPOT_DENSE_DCM_MIN_A = 5.0
DENSE_DCM_MLPOT_MONOMER_COUNT = 40
# When MKIMAT2 cannot be captured, MIC must clear the IMAGE floor plus slack.
DEFAULT_MIC_MKIMAT2_REGISTRATION_SLACK_A = 0.5

# Dynamics overlap guard default (CHARMM close-contact scale); looser than vdW sums.
DYNAMICS_OVERLAP_REFERENCE_A = 1.5

# Representative vdW contact sums (Å) for log context — not hard thresholds.
_VDW_CONTACT_HINT_A: dict[tuple[str, str], float] = {
    ("H", "H"): 2.4,
    ("C", "C"): 3.4,
    ("C", "H"): 2.9,
    ("Cl", "H"): 2.9,
    ("Cl", "Cl"): 3.6,
    ("Cl", "C"): 3.5,
    ("O", "H"): 2.6,
    ("N", "H"): 2.9,
}


def _element_symbol(atomic_number: int) -> str:
    zi = int(atomic_number)
    try:
        from ase.data import chemical_symbols

        if 1 <= zi < len(chemical_symbols) and chemical_symbols[zi]:
            return str(chemical_symbols[zi])
    except ImportError:
        pass
    return {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
        53: "I",
    }.get(zi, "?")


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def vdw_contact_hint_A(label_i: str, label_j: str) -> float | None:
    """Typical equilibrium vdW contact distance for an element pair (Å)."""
    key = _ordered_pair(label_i, label_j)
    if key in _VDW_CONTACT_HINT_A:
        return float(_VDW_CONTACT_HINT_A[key])
    # Fallback: sum of generic radii (C/N/O ~1.7, H ~1.2, Cl ~1.8)
    radii = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75, "Br": 1.85}
    ri = radii.get(label_i)
    rj = radii.get(label_j)
    if ri is not None and rj is not None:
        return ri + rj
    return None


def _is_hydrogen_symbol(label: str) -> bool:
    return str(label).strip().upper() == "H"


def _is_heavy_symbol(label: str) -> bool:
    return not _is_hydrogen_symbol(label) and str(label).strip() not in ("", "?")


def is_dcm_like_prep(args: argparse.Namespace | None) -> bool:
    """True when workflow targets chlorinated dense liquids (stricter pair floors)."""
    if args is None:
        return False
    solvents = getattr(args, "solvents", None) or []
    for raw in solvents:
        if str(raw).strip().upper() == "DCM":
            return True
    for attr in ("composition", "_cluster_composition_summary"):
        comp = getattr(args, attr, None)
        if comp is None:
            continue
        if isinstance(comp, dict):
            for key in comp:
                if str(key).strip().upper() == "DCM":
                    return True
        else:
            text = str(comp).upper()
            if text.startswith("DCM:") or ",DCM:" in text or " DCM:" in text:
                return True
    return False


def resolve_pre_mlpot_element_pair_min_distance(
    label_i: str,
    label_j: str,
    *,
    args: argparse.Namespace | None = None,
) -> float:
    """Minimum MIC distance (Å) for an element pair before MLpot registration."""
    global_floor = (
        resolve_pre_mlpot_overlap_min_distance(args)
        if args is not None
        else float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A)
    )
    explicit_hh = getattr(args, "pre_mlpot_h_heavy_min_distance", None) if args else None
    explicit_heavy = (
        getattr(args, "pre_mlpot_heavy_heavy_min_distance", None) if args else None
    )
    h_heavy = (
        float(explicit_hh)
        if explicit_hh is not None and float(explicit_hh) > 0.0
        else float(DEFAULT_PRE_MLPOT_H_HEAVY_MIN_A)
    )
    heavy_heavy = (
        float(explicit_heavy)
        if explicit_heavy is not None and float(explicit_heavy) > 0.0
        else float(DEFAULT_PRE_MLPOT_HEAVY_HEAVY_MIN_A)
    )
    h_h = float(getattr(args, "pre_mlpot_h_h_min_distance", DEFAULT_PRE_MLPOT_H_H_MIN_A) or DEFAULT_PRE_MLPOT_H_H_MIN_A)

    li, lj = str(label_i).strip(), str(label_j).strip()
    if "?" in (li, lj):
        # Missing element typing: never apply heavy–heavy floor by mistake.
        return float(global_floor)

    hi = _is_hydrogen_symbol(li)
    hj = _is_hydrogen_symbol(lj)
    if hi and hj:
        pair_floor = max(global_floor, h_h)
    elif hi ^ hj:
        pair_floor = max(global_floor, h_heavy)
    else:
        pair_floor = max(global_floor, heavy_heavy)

    if is_dcm_like_prep(args):
        return float(pair_floor)
    # Non-DCM liquid prep still uses the global MIC floor; pair floors apply when set explicitly.
    if explicit_hh is not None or explicit_heavy is not None:
        return float(pair_floor)
    return float(global_floor)


def resolve_mlpot_registration_max_grms(args: argparse.Namespace | None) -> float:
    explicit = getattr(args, "mlpot_registration_max_grms", None) if args is not None else None
    if explicit is not None and float(explicit) > 0.0:
        return float(explicit)
    return float(DEFAULT_MLPOT_REGISTRATION_MAX_GRMS_KCALMOL_A)


def resolve_pre_mlpot_overlap_min_distance(args: argparse.Namespace) -> float:
    """Minimum inter-monomer MIC distance for the pre-MLpot geometry gate (Å).

    Intentionally **not** tied to ``--dynamics-overlap-min-distance`` (default 1.5 Å).
    Structures must be ML-safe before the USER potential is enabled; sub-2.3 Å MIC
    contacts routinely explode hybrid GRMS at registration.
    """
    explicit = getattr(args, "pre_mlpot_overlap_min_distance", None)
    if explicit is not None:
        val = float(explicit)
        if val > 0.0:
            return val
        return float("inf")

    build = getattr(args, "min_intermonomer_atom_distance", None)
    if build is not None and float(build) >= float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A) - 1.0e-9:
        return float(build)

    return float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A)


def resolve_dynamics_overlap_reference_A(args: argparse.Namespace | None) -> float:
    if args is None:
        return float(DYNAMICS_OVERLAP_REFERENCE_A)
    dyn = getattr(args, "dynamics_overlap_min_distance", None)
    if dyn is not None and float(dyn) > 0.0:
        return float(dyn)
    return float(DYNAMICS_OVERLAP_REFERENCE_A)


def resolve_overlap_last_chance_separation_A(args: argparse.Namespace) -> float:
    """Separation target (Å) for prep-gate ``overlap_last_chance`` repack/separate.

    Opens contacts to the ML-safe prep floor (H–heavy / heavy–heavy when DCM-like),
    not the dynamics overlap guard (1.5 Å).
    """
    prep_floor = resolve_pre_mlpot_overlap_min_distance(args)
    pair_target = resolve_pre_mlpot_element_pair_min_distance("H", "C", args=args)
    if is_dcm_like_prep(args):
        return max(float(prep_floor), float(pair_target))
    return max(float(prep_floor), float(DEFAULT_PRE_MLPOT_H_HEAVY_MIN_A))


def resolve_mc_min_intermonomer_distance_A(args: argparse.Namespace) -> float:
    """Minimum contact distance for MC / box-compression moves (Å).

    Under liquid prep, use the same floor as pre-MLpot certification (default 1.0 Å)
    so volume moves do not leave sub-floor contacts that only MD cleanup would fix.
    """
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
            liquid_prep_enabled,
        )

        if liquid_prep_enabled(args):
            return resolve_pre_mlpot_overlap_min_distance(args)
    except ImportError:
        pass
    build = getattr(args, "min_intermonomer_atom_distance", None)
    if build is not None:
        return float(build)
    return 0.1


@dataclass(frozen=True)
class IntermonomerContactSummary:
    distance_A: float
    threshold_A: float
    monomer_i: int
    monomer_j: int
    atom_i: int
    atom_j: int
    label_i: str
    label_j: str
    dynamics_reference_A: float = DYNAMICS_OVERLAP_REFERENCE_A

    def format_log_line(self) -> str:
        d = float(self.distance_A)
        floor = float(self.threshold_A)
        dyn = float(self.dynamics_reference_A)
        pair = f"{self.label_i}–{self.label_j}"
        head = (
            f"worst inter-monomer contact {d:.3f} Å "
            f"(monomers {self.monomer_i}/{self.monomer_j}, atoms {pair}; "
            f"prep floor {floor:.2f} Å"
        )
        vdw = vdw_contact_hint_A(self.label_i, self.label_j)
        if vdw is not None:
            head += f", typical vdW {vdw:.1f} Å"
        head += ")"

        if d < floor:
            status = "FAIL: below prep MIC floor — repack/expand box before MLpot"
        elif d >= dyn:
            status = f"OK: above dynamics guard ({dyn:.1f} Å)"
        else:
            status = (
                f"tight for dynamics ({dyn:.1f} Å) but passes prep MIC floor — "
                "verify element-pair floors before MLpot registration"
            )

        chem = _chemical_note(self.label_i, self.label_j, d, vdw)
        return f"{head}; {status}; {chem}"


def _chemical_note(label_i: str, label_j: str, distance_A: float, vdw: float | None) -> str:
    if "H" in (label_i, label_j):
        other = label_j if label_i == "H" else label_i
        if other in ("Cl", "C", "O", "N"):
            return (
                "H–heavy contact: equilibrium liquid distances are usually ≥2.5 Å; "
                "short prep contacts often involve rotatable methylenes"
            )
    if label_i == "H" and label_j == "H":
        return "H–H: equilibrium ~2.4 Å; sub-2 Å in prep is strained but usually relaxes in mini"
    if vdw is not None and distance_A < 0.85 * vdw:
        return "substantially inside summed vdW radii — worth watching through pre-SD mini"
    if vdw is not None and distance_A < vdw:
        return "inside typical vdW contact — acceptable at prep if GRMS mini succeeds"
    return "contact spacing plausible for dense liquid prep"


def summarize_worst_intermonomer_contact(
    positions: np.ndarray,
    atoms_per_list: list[int],
    *,
    box_side: float | None,
    use_pbc: bool,
    threshold_A: float,
    atomic_numbers: np.ndarray | list[int] | None = None,
    dynamics_reference_A: float = DYNAMICS_OVERLAP_REFERENCE_A,
) -> IntermonomerContactSummary:
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.utils.geometry_checks import find_worst_intermonomer_overlap

    pos = np.asarray(positions, dtype=float)
    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    cell: Any | None = None
    if use_pbc and box_side is not None:
        cell = np.diag([float(box_side), float(box_side), float(box_side)])
    dist, violation = find_worst_intermonomer_overlap(pos, offsets, cell=cell)
    if violation is None:
        return IntermonomerContactSummary(
            distance_A=float("inf"),
            threshold_A=float(threshold_A),
            monomer_i=-1,
            monomer_j=-1,
            atom_i=-1,
            atom_j=-1,
            label_i="?",
            label_j="?",
            dynamics_reference_A=float(dynamics_reference_A),
        )

    z: np.ndarray | None = None
    if atomic_numbers is not None:
        z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    li = _element_symbol(z[violation.atom_i]) if z is not None else "?"
    lj = _element_symbol(z[violation.atom_j]) if z is not None else "?"

    return IntermonomerContactSummary(
        distance_A=float(dist),
        threshold_A=float(threshold_A),
        monomer_i=int(violation.monomer_i),
        monomer_j=int(violation.monomer_j),
        atom_i=int(violation.atom_i),
        atom_j=int(violation.atom_j),
        label_i=li,
        label_j=lj,
        dynamics_reference_A=float(dynamics_reference_A),
    )


@dataclass(frozen=True)
class PreMlpotMicViolation:
    distance_A: float
    required_A: float
    monomer_i: int
    monomer_j: int
    atom_i: int
    atom_j: int
    label_i: str
    label_j: str

    def format_message(self) -> str:
        return (
            f"monomers {self.monomer_i + 1}/{self.monomer_j + 1} (1-based), "
            f"atoms {self.label_i}–{self.label_j}, "
            f"MIC distance {self.distance_A:.3f} Å < required {self.required_A:.2f} Å"
        )


def find_worst_pre_mlpot_mic_violation(
    positions: np.ndarray,
    atoms_per_list: list[int],
    *,
    box_side: float | None,
    use_pbc: bool,
    args: argparse.Namespace | None = None,
    atomic_numbers: np.ndarray | list[int] | None = None,
) -> PreMlpotMicViolation | None:
    """Return the tightest MIC contact that violates prep element-pair floors."""
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.utils.geometry_checks import find_worst_intermonomer_overlap

    pos = np.asarray(positions, dtype=float)
    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    cell: Any | None = None
    if use_pbc and box_side is not None:
        cell = np.diag([float(box_side), float(box_side), float(box_side)])

    z: np.ndarray | None = None
    if atomic_numbers is not None:
        z = np.asarray(atomic_numbers, dtype=int).reshape(-1)

    global_floor = (
        resolve_pre_mlpot_overlap_min_distance(args)
        if args is not None
        else float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A)
    )

    worst: PreMlpotMicViolation | None = None
    n_monomers = int(len(offsets) - 1)
    for mi in range(n_monomers):
        si, ei = int(offsets[mi]), int(offsets[mi + 1])
        for mj in range(mi + 1, n_monomers):
            sj, ej = int(offsets[mj]), int(offsets[mj + 1])
            for gi in range(si, ei):
                for gj in range(sj, ej):
                    from mmml.utils.geometry_checks import _mic_displacement

                    disp = _mic_displacement(pos[gi], pos[gj], cell)
                    dist = float(np.linalg.norm(disp))
                    li = _element_symbol(int(z[gi])) if z is not None else "?"
                    lj = _element_symbol(int(z[gj])) if z is not None else "?"
                    required = resolve_pre_mlpot_element_pair_min_distance(
                        li, lj, args=args
                    )
                    required = max(required, global_floor)
                    if dist + 1.0e-9 < required:
                        if worst is None or dist < worst.distance_A:
                            worst = PreMlpotMicViolation(
                                distance_A=dist,
                                required_A=float(required),
                                monomer_i=mi,
                                monomer_j=mj,
                                atom_i=gi,
                                atom_j=gj,
                                label_i=li,
                                label_j=lj,
                            )
    if worst is not None:
        return worst

    # Fast path when no pair scan violations: still honour global MIC floor.
    dist, violation = find_worst_intermonomer_overlap(pos, offsets, cell=cell)
    if violation is None or dist + 1.0e-9 >= global_floor:
        return None
    li = _element_symbol(int(z[violation.atom_i])) if z is not None else "?"
    lj = _element_symbol(int(z[violation.atom_j])) if z is not None else "?"
    required = max(
        global_floor,
        resolve_pre_mlpot_element_pair_min_distance(li, lj, args=args),
    )
    if dist + 1.0e-9 >= required:
        return None
    return PreMlpotMicViolation(
        distance_A=float(dist),
        required_A=float(required),
        monomer_i=int(violation.monomer_i),
        monomer_j=int(violation.monomer_j),
        atom_i=int(violation.atom_i),
        atom_j=int(violation.atom_j),
        label_i=li,
        label_j=lj,
    )


def assert_pre_mlpot_mic_geometry(
    positions: np.ndarray,
    atoms_per_list: list[int],
    *,
    box_side: float | None,
    use_pbc: bool,
    args: argparse.Namespace | None = None,
    atomic_numbers: np.ndarray | list[int] | None = None,
    context: str = "Pre-MLpot MIC geometry",
) -> float:
    """Abort when post-wrap MIC contacts violate ML-safe prep floors."""
    violation = find_worst_pre_mlpot_mic_violation(
        positions,
        atoms_per_list,
        box_side=box_side,
        use_pbc=use_pbc,
        args=args,
        atomic_numbers=atomic_numbers,
    )
    if violation is not None:
        raise RuntimeError(
            f"{context}: {violation.format_message()}. "
            "Repack at lower density, expand the box, or abort — do not enable MLpot."
        )
    from mmml.interfaces.pycharmmInterface.mlpot.liquid_box_build import (
        measure_worst_intermonomer_A,
    )

    return float(
        measure_worst_intermonomer_A(
            positions,
            atoms_per_list,
            box_side=box_side,
            use_pbc=use_pbc,
        )
    )

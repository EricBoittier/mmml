"""Inter-monomer contact thresholds and human-readable prep-gate reporting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

# Prep gate: reject true cross-monomer clashes, not equilibrium liquid contacts.
DEFAULT_PRE_MLPOT_OVERLAP_MIN_A = 1.0

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


def resolve_pre_mlpot_overlap_min_distance(args: argparse.Namespace) -> float:
    """Minimum inter-monomer atom distance for the pre-MLpot geometry gate (Å).

  Intentionally **not** tied to ``--dynamics-overlap-min-distance`` (default 1.5 Å).
  Dense liquid prep in a periodic box often leaves transient contacts between
  1.0–1.5 Å that hybrid minimization and overlap rescue relax before dynamics.
  """
    explicit = getattr(args, "pre_mlpot_overlap_min_distance", None)
    if explicit is not None:
        val = float(explicit)
        if val > 0.0:
            return val
        return float("inf")

    build = getattr(args, "min_intermonomer_atom_distance", None)
    if build is not None and float(build) > 0.1 + 1.0e-9:
        return float(build)

    return float(DEFAULT_PRE_MLPOT_OVERLAP_MIN_A)


def resolve_dynamics_overlap_reference_A(args: argparse.Namespace | None) -> float:
    if args is None:
        return float(DYNAMICS_OVERLAP_REFERENCE_A)
    dyn = getattr(args, "dynamics_overlap_min_distance", None)
    if dyn is not None and float(dyn) > 0.0:
        return float(dyn)
    return float(DYNAMICS_OVERLAP_REFERENCE_A)


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
            status = "FAIL: below prep floor — true cross-monomer clash"
        elif d >= dyn:
            status = f"OK: above dynamics guard ({dyn:.1f} Å)"
        else:
            status = (
                f"tight for dynamics ({dyn:.1f} Å) but passes prep — "
                "hybrid mini / overlap rescue expected to open this before MD"
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

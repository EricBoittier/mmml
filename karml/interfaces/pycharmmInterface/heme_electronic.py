"""Charge, spin, counterions, and the propionate ML/MM cut for RESI HEME.

``RESI HEME`` is charge −2 (two deprotonated propionates). The residue contains
the porphyrin and the iron and no axial ligand, so the iron is four-coordinate
Fe(II). That state is an intermediate-spin triplet: spin multiplicity 3
(``2S+1``), which is the value PET-OMOL reads from ``atoms.info["spin"]``.

The CHARMM comment "6-liganded planar heme" names the parameter set (a planar
porphyrin). It does not put histidine or CO into the PSF.

Metatomic's ASE calculator otherwise sends charge 0 and multiplicity 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from karml.interfaces.calculators.link_atoms import LinkAtom

HEME_FORMAL_CHARGE = -2
# Four-coordinate Fe(II) porphyrin, no axial ligand in the residue.
HEME_SPIN_MULTIPLICITY = 3

# CHARMM toppar_water_ions.str residue charges.
PROTEIN_ION_CHARGE: dict[str, int] = {
    "LIT": 1,
    "SOD": 1,
    "MG": 2,
    "POT": 1,
    "CAL": 2,
    "RUB": 1,
    "CES": 1,
    "BAR": 2,
    "ZN2": 2,
    "CD2": 2,
    "CLA": -1,
}

# Atoms past the CAA–CBA and CAD–CBD cuts. Each tail sums to formal charge −1.
PROPIONATE_MM_NAMES = frozenset(
    {
        "CBA",
        "HBA1",
        "HBA2",
        "CGA",
        "O1A",
        "O2A",
        "CBD",
        "HBD1",
        "HBD2",
        "CGD",
        "O1D",
        "O2D",
    }
)
PROPIONATE_CUTS = (("CAA", "CBA"), ("CAD", "CBD"))
CARBOXYLATE_SITES = (("CGA", "O1A", "O2A"), ("CGD", "O1D", "O2D"))
SODIUM_OXYGEN_DISTANCE_A = 2.35


def is_protein_ion(name: str) -> bool:
    return str(name).strip().upper() in PROTEIN_ION_CHARGE


def residue_formal_charge(name: str) -> int:
    key = str(name).strip().upper()
    if key == "HEME":
        return HEME_FORMAL_CHARGE
    if key in PROTEIN_ION_CHARGE:
        return PROTEIN_ION_CHARGE[key]
    raise ValueError(
        f"no formal charge tabulated for residue {key!r}. "
        "HEME is −2; LIT/SOD/MG/POT/CAL/RUB/CES/BAR/ZN2/CD2/CLA are the "
        "protein-ion residues. Set --charge for anything else."
    )


def composition_pairs(args: object) -> list[tuple[str, int]]:
    """``[(RES, count), ...]`` from ``--composition`` or ``--residue`` × n."""
    from karml.interfaces.pycharmmInterface.heme_library import residues_from_cluster_args
    from karml.interfaces.pycharmmInterface.mlpot.composition_spec import (
        parse_composition_entries,
    )

    composition = getattr(args, "composition", None)
    if composition:
        return [
            (entry.residue, int(entry.count))
            for entry in parse_composition_entries(
                str(composition), validate_cgenff=False
            )
        ]
    names = residues_from_cluster_args(args)
    if not names:
        return []
    n_mol = int(getattr(args, "n_molecules", 1) or 1)
    if len(names) == 1:
        return [(names[0], n_mol)]
    return [(name, 1) for name in names]


def _solute_pairs(pairs: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    return [(name, count) for name, count in pairs if not is_protein_ion(name)]


def expand_counterions(args: object) -> None:
    """Append neutralizing protein ions to ``args.composition``.

    ``--counterions SOD`` on one HEME writes ``HEME:1,SOD:2``. An ion count
    that is already present is left alone.
    """
    kind = getattr(args, "counterions", None)
    if kind is None or str(kind).strip().lower() in {"", "none"}:
        return
    from karml.interfaces.pycharmmInterface.myoglobin import is_myoglobin_args

    if is_myoglobin_args(args):
        raise ValueError(
            "--counterions does not apply to MBCO. The crystal CRD is loaded "
            "as written, with sulfate omitted. Use --charge to override the "
            "metatomic charge."
        )
    ion = str(kind).strip().upper()
    if ion not in PROTEIN_ION_CHARGE:
        known = ", ".join(sorted(PROTEIN_ION_CHARGE))
        raise ValueError(f"--counterions {ion!r} is not a protein ion ({known})")
    ion_q = PROTEIN_ION_CHARGE[ion]
    pairs = composition_pairs(args)
    solute = _solute_pairs(pairs)
    if not solute:
        raise ValueError("--counterions needs a solute residue (for example HEME)")
    solute_q = sum(residue_formal_charge(name) * count for name, count in solute)
    if solute_q == 0:
        return
    if solute_q * ion_q > 0:
        raise ValueError(
            f"{ion} charge {ion_q:+d} has the same sign as the solute "
            f"({solute_q:+d}); it cannot neutralize it"
        )
    if abs(solute_q) % abs(ion_q) != 0:
        raise ValueError(
            f"solute charge {solute_q:+d} is not a multiple of {ion} charge {ion_q:+d}"
        )
    n_needed = abs(solute_q) // abs(ion_q)
    present = sum(count for name, count in pairs if name == ion)
    if present == n_needed:
        return
    if present:
        raise ValueError(
            f"composition already has {present} {ion}; neutralizing this solute "
            f"needs {n_needed}"
        )
    solute_spec = ",".join(f"{name}:{count}" for name, count in solute)
    setattr(args, "composition", f"{solute_spec},{ion}:{n_needed}")


@dataclass(frozen=True, slots=True)
class MetatomicElectronicState:
    """What PET-OMOL reads: total charge and spin multiplicity (2S+1)."""

    charge: int | None
    spin_multiplicity: int | None
    monomer_charges: tuple[int, ...] | None
    monomer_spins: tuple[int, ...] | None
    reason: str


def _monomer_state(name: str, *, ml_core_only: bool) -> tuple[int, int]:
    key = str(name).strip().upper()
    if key == "HEME" and ml_core_only:
        return 0, HEME_SPIN_MULTIPLICITY
    if key == "HEME":
        return HEME_FORMAL_CHARGE, HEME_SPIN_MULTIPLICITY
    if key in PROTEIN_ION_CHARGE:
        return PROTEIN_ION_CHARGE[key], 1
    return 0, 1


def resolve_metatomic_electronic_state(args: object | None) -> MetatomicElectronicState:
    """Charge and spin for the metatomic system.

    HEME defaults to charge −2 and multiplicity 3. With neutralizing counterions
    in the same whole-system evaluation the charge is 0 and the multiplicity
    stays 3. ``--mm-region propionates`` evaluates the porphyrin core only
    (formal charge 0, multiplicity 3); the carboxylate charge stays on the MM
    tails.
    """
    if args is None:
        return MetatomicElectronicState(None, None, None, None, "")
    from karml.interfaces.pycharmmInterface.ml_cut import ml_cut_spec_from_args

    cut_spec = ml_cut_spec_from_args(args)
    if cut_spec is not None:
        charge = int(cut_spec.charge)
        spin = int(cut_spec.spin_multiplicity)
        explicit_charge = getattr(args, "charge", None)
        explicit_spin = getattr(args, "spin_multiplicity", None)
        if explicit_charge is not None:
            charge = int(explicit_charge)
        if explicit_spin is not None:
            spin = int(explicit_spin)
        return MetatomicElectronicState(
            charge,
            spin,
            None,
            None,
            f"ml_cut {cut_spec.path.name}",
        )
    from karml.interfaces.pycharmmInterface.myoglobin import (
        is_myoglobin_args,
        myoglobin_electronic_state,
    )

    if is_myoglobin_args(args):
        return myoglobin_electronic_state(args)
    explicit_charge = getattr(args, "charge", None)
    explicit_spin = getattr(args, "spin_multiplicity", None)
    mm_region = str(getattr(args, "mm_region", None) or "none").strip().lower()
    ml_core_only = mm_region == "propionates"
    try:
        pairs = composition_pairs(args)
    except (TypeError, ValueError):
        pairs = []
    heme_copies = sum(count for name, count in pairs if str(name).upper() == "HEME")
    if not pairs or heme_copies == 0:
        charge = None if explicit_charge is None else int(explicit_charge)
        spin = None if explicit_spin is None else int(explicit_spin)
        reason = "CLI charge/spin" if charge is not None or spin is not None else ""
        return MetatomicElectronicState(charge, spin, None, None, reason)

    per_monomer: list[tuple[int, int]] = []
    for name, count in pairs:
        q, s = _monomer_state(name, ml_core_only=ml_core_only)
        per_monomer.extend([(q, s)] * int(count))
    if ml_core_only:
        # Propionate tails and ions are MM. Each heme core is charge 0, multiplicity 3.
        ml_states = [state for state in per_monomer if state[1] != 1]
        if not ml_states:
            ml_states = [(0, HEME_SPIN_MULTIPLICITY)]
        total_q = sum(q for q, _s in ml_states)
        spins = [s for _q, s in ml_states]
    else:
        total_q = sum(q for q, _s in per_monomer)
        spins = [s for _q, s in per_monomer if s != 1]
    if explicit_charge is not None:
        total_q = int(explicit_charge)
    if explicit_spin is not None:
        total_spin = int(explicit_spin)
    elif len(spins) == 1:
        total_spin = int(spins[0])
    elif not spins:
        total_spin = 1
    else:
        raise ValueError(
            f"{len(spins)} open-shell HEME copies in one metatomic system; "
            "set --spin-multiplicity (2S+1) for the coupled spin"
        )
    if ml_core_only:
        reason = (
            "propionate tails are MM; PET sees the Fe(II) porphyrin core "
            f"(charge {total_q}, multiplicity {total_spin})"
        )
    elif total_q == 0 and heme_copies:
        reason = (
            "HEME charge −2 plus neutralizing counterions; "
            f"multiplicity {total_spin} (Fe(II), no axial ligand)"
        )
    else:
        reason = (
            "RESI HEME formal charge −2; Fe(II) with no axial ligand in the "
            f"residue, multiplicity {total_spin}"
        )
    if explicit_charge is not None or explicit_spin is not None:
        reason = f"CLI override; {reason}"
    return MetatomicElectronicState(
        int(total_q),
        int(total_spin),
        tuple(q for q, _s in per_monomer),
        tuple(s for _q, s in per_monomer),
        reason,
    )


def _name_index(names: Sequence[str], wanted: str) -> int:
    key = wanted.upper()
    for i, name in enumerate(names):
        if str(name).strip().upper() == key:
            return i
    raise ValueError(f"heme atom {wanted!r} is not in {list(names)}")


def propionate_partition(
    atom_names: Sequence[str],
) -> tuple[np.ndarray, tuple[LinkAtom, ...]]:
    """ML atom indices and the two ghost-hydrogen cuts for one HEME residue."""
    names = [str(name).strip() for name in atom_names]
    missing = [name for name in sorted(PROPIONATE_MM_NAMES) if name not in {n.upper() for n in names}]
    if missing:
        raise ValueError(f"propionate MM region is missing atoms {missing}")
    mm = {n.upper() for n in names if n.upper() in PROPIONATE_MM_NAMES}
    ml = np.asarray(
        [i for i, name in enumerate(names) if name.upper() not in mm],
        dtype=int,
    )
    links = tuple(
        LinkAtom(
            qm_index=_name_index(names, qm_name),
            mm_index=_name_index(names, mm_name),
        )
        for qm_name, mm_name in PROPIONATE_CUTS
    )
    return ml, links


def partition_system(
    atom_names: Sequence[str],
    residue_labels: Sequence[str] | None,
    atoms_per_monomer: Sequence[int] | None,
) -> tuple[np.ndarray, tuple[LinkAtom, ...]]:
    """ML core indices and link atoms for every HEME in a cluster.

    Protein ions and the propionate tails stay in the MM region.
    """
    names = [str(name).strip() for name in atom_names]
    if residue_labels is None or atoms_per_monomer is None:
        ml, links = propionate_partition(names)
        return ml, links
    if len(residue_labels) != len(atoms_per_monomer):
        raise ValueError("residue labels and atoms_per_monomer have different lengths")
    if sum(int(n) for n in atoms_per_monomer) != len(names):
        raise ValueError("atoms_per_monomer does not cover atom_names")
    ml_all: list[int] = []
    links_all: list[LinkAtom] = []
    offset = 0
    for label, count in zip(residue_labels, atoms_per_monomer):
        n = int(count)
        if str(label).strip().upper() == "HEME":
            local_ml, local_links = propionate_partition(names[offset : offset + n])
            ml_all.extend(int(i) + offset for i in local_ml)
            links_all.extend(
                LinkAtom(
                    qm_index=link.qm_index + offset,
                    mm_index=link.mm_index + offset,
                    bond_length_A=link.bond_length_A,
                )
                for link in local_links
            )
        offset += n
    if not ml_all:
        raise ValueError("mm-region propionates found no HEME residue")
    return np.asarray(ml_all, dtype=int), tuple(links_all)


def carboxylate_site(
    positions: np.ndarray,
    atom_names: Sequence[str],
    carbon: str,
    oxygen_a: str,
    oxygen_b: str,
    *,
    oxygen_distance_A: float = SODIUM_OXYGEN_DISTANCE_A,
) -> np.ndarray:
    """Point on the carboxylate bisector, ``oxygen_distance_A`` from each oxygen."""
    names = [str(name).strip().upper() for name in atom_names]
    pos = np.asarray(positions, dtype=np.float64)
    c = pos[_name_index(names, carbon)]
    o1 = pos[_name_index(names, oxygen_a)]
    o2 = pos[_name_index(names, oxygen_b)]
    mid = 0.5 * (o1 + o2)
    oo = o2 - o1
    oo_norm = float(np.linalg.norm(oo))
    half = 0.5 * oo_norm
    if oo_norm < 1.0e-8 or oxygen_distance_A <= half:
        raise ValueError(f"carboxylate {carbon} oxygens are too far apart for an ion")
    plane_normal = np.cross(oo, c - mid)
    plane_norm = float(np.linalg.norm(plane_normal))
    if plane_norm < 1.0e-8:
        raise ValueError(f"carboxylate {carbon} is colinear")
    outward = np.cross(plane_normal, oo)
    outward = outward / float(np.linalg.norm(outward))
    # Point the bisector away from the carbon.
    if float(np.dot(outward, mid - c)) < 0.0:
        outward = -outward
    reach = (oxygen_distance_A**2 - half**2) ** 0.5
    return mid + reach * outward


def seat_heme_counterions(
    positions: np.ndarray,
    atom_names: Sequence[str],
    residue_labels: Sequence[str],
    atoms_per_monomer: Sequence[int],
) -> np.ndarray:
    """Place one cation on each heme carboxylate, in residue order.

    Returns a copy. Non-ion monomers and extra ions keep their coordinates.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    names = [str(name).strip() for name in atom_names]
    if len(residue_labels) != len(atoms_per_monomer):
        return pos
    if sum(int(n) for n in atoms_per_monomer) != len(names):
        return pos
    offsets = np.cumsum([0, *[int(n) for n in atoms_per_monomer]])
    ion_slots: list[int] = []
    heme_spans: list[tuple[int, int]] = []
    for i, label in enumerate(residue_labels):
        key = str(label).strip().upper()
        start, stop = int(offsets[i]), int(offsets[i + 1])
        if key == "HEME":
            heme_spans.append((start, stop))
        elif key in PROTEIN_ION_CHARGE and PROTEIN_ION_CHARGE[key] > 0 and stop - start == 1:
            ion_slots.append(start)
    cursor = 0
    for start, stop in heme_spans:
        block_names = names[start:stop]
        block = pos[start:stop]
        try:
            sites = [
                carboxylate_site(block, block_names, carbon, o1, o2)
                for carbon, o1, o2 in CARBOXYLATE_SITES
            ]
        except ValueError:
            continue
        for site in sites:
            if cursor >= len(ion_slots):
                return pos
            pos[ion_slots[cursor]] = site
            cursor += 1
    return pos

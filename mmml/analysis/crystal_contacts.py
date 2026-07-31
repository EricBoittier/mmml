"""Intermolecular contacts in a periodic crystal.

Reproducing the contact distances a crystallographic paper quotes is the
sharpest cheap check that a cell was built and expanded correctly: lattice
parameters and molecule counts can all be right while a symmetry operator was
misapplied or a molecule was left broken across a cell face. The distances
catch that; the cell metrics do not.

Contacts here are always *intermolecular* and always searched over lattice
images, so a contact between a molecule and a periodic image of itself counts,
as it must in a crystal.

:func:`normalize_hydrogen_positions` is the companion structure-prep step. X-ray
scattering comes from electron density, which for hydrogen sits between the
nuclei, so refined X-H distances are systematically 0.1-0.2 A short and carry
uncertainties of the same size. Any comparison involving hydrogen -- between two
X-ray structures, against a neutron structure, or against a force field whose
equilibrium bond lengths are nuclear -- has to fix that first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

__all__ = [
    "Contact",
    "collapse_equivalent",
    "element_pair_contacts",
    "molecular_frames",
    "normalize_hydrogen_positions",
    "NEUTRON_CH_DISTANCE_A",
]

# Mean neutron-derived C(sp3)-H distance; Allen, Kennard, Watson, Brammer, Orpen
# & Taylor, J. Chem. Soc. Perkin Trans. 2, S1 (1987).
NEUTRON_CH_DISTANCE_A: float = 1.089


@dataclass(frozen=True)
class Contact:
    """One intermolecular contact between molecules ``mol_i`` and ``mol_j``."""

    distance_A: float
    mol_i: int
    mol_j: int
    atom_i: int
    atom_j: int
    angle_deg: float | None = None
    motif: str | None = None


def molecular_frames(atoms: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap a cell into whole molecules; return ``(mol_id, positions, cell)``."""
    from mmml.analysis.lattice_energy import unwrap_molecules

    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    mol_id, positions = unwrap_molecules(
        np.asarray(atoms.get_positions(), dtype=np.float64),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
        cell,
    )
    return mol_id, positions, cell


def normalize_hydrogen_positions(
    atoms: Any,
    target_A: float = NEUTRON_CH_DISTANCE_A,
) -> Any:
    """Move every hydrogen along its bond to a standard heavy-atom distance.

    Returns a copy. Each hydrogen keeps its refined *direction*, which X-ray data
    determines reasonably well, and takes a standard *distance*, which it does
    not. This is the normalisation crystallographic papers apply before quoting
    an H...A contact, and it is a precondition for comparing two X-ray
    structures with each other: the two deposited dichloromethane structures
    refined C-H to 1.01(10) and 1.13(12) A, a spread that is pure noise and
    larger than the compression between them.

    Molecules are unwrapped first, so a hydrogen whose carbon sits across a cell
    face is still moved along the real bond rather than across the cell.
    """
    mol_id, positions, cell = molecular_frames(atoms)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    hydrogens = np.flatnonzero(z == 1)
    heavy = np.flatnonzero(z != 1)
    if not len(hydrogens) or not len(heavy):
        return atoms.copy()

    moved = np.array(positions, dtype=np.float64, copy=True)
    for h in hydrogens:
        same = heavy[mol_id[heavy] == mol_id[h]]
        candidates = same if len(same) else heavy
        vectors = positions[candidates] - positions[h]
        norms = np.linalg.norm(vectors, axis=1)
        nearest = int(np.argmin(norms))
        anchor = positions[candidates[nearest]]
        direction = (positions[h] - anchor) / norms[nearest]
        moved[h] = anchor + direction * float(target_A)

    out = atoms.copy()
    out.set_positions(moved)
    out.set_cell(cell)
    out.set_pbc(True)
    return out


def collapse_equivalent(
    contacts: Iterable[Contact], tolerance_A: float
) -> list[Contact]:
    """Keep one representative per symmetry-equivalent distance, shortest first.

    A crystal generates each contact once per symmetry operator, so an
    uncollapsed list is dominated by repeats of the same physical distance. The
    collapsed list reads like the distances quoted in a paper.
    """
    out: list[Contact] = []
    for contact in sorted(contacts, key=lambda c: c.distance_A):
        if any(abs(contact.distance_A - kept.distance_A) < tolerance_A for kept in out):
            continue
        out.append(contact)
    return out


def element_pair_contacts(
    atoms: Any,
    element_i: str,
    element_j: str,
    *,
    max_distance_A: float = 4.0,
    tolerance_A: float = 5e-3,
    collapse: bool = True,
) -> list[Contact]:
    """Intermolecular ``element_i...element_j`` contacts, shortest first.

    ``mol_i`` / ``mol_j`` identify the molecules in the home cell; the second is
    reached through whichever lattice image made the contact.
    """
    from ase.data import atomic_numbers

    from mmml.analysis.lattice_energy import lattice_shift_vectors, molecular_reach_A

    mol_id, positions, cell = molecular_frames(atoms)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    idx_i = np.flatnonzero(z == atomic_numbers[element_i])
    idx_j = np.flatnonzero(z == atomic_numbers[element_j])
    if not len(idx_i) or not len(idx_j):
        return []

    reach = molecular_reach_A(positions, mol_id)
    shifts = lattice_shift_vectors(cell, max_distance_A, reach_A=reach)

    found: list[Contact] = []
    for shift in shifts:
        is_home = not np.any(shift)
        delta = (positions[idx_j] + shift)[None, :, :] - positions[idx_i][:, None, :]
        dist = np.linalg.norm(delta, axis=-1)
        keep = dist < max_distance_A
        # Within the home cell an atom must not contact its own molecule; in any
        # other image it may, including its own periodic image.
        if is_home:
            keep &= mol_id[idx_i][:, None] != mol_id[idx_j][None, :]
        for i, j in zip(*np.nonzero(keep)):
            found.append(
                Contact(
                    distance_A=float(dist[i, j]),
                    mol_i=int(mol_id[idx_i[i]]),
                    mol_j=int(mol_id[idx_j[j]]),
                    atom_i=int(idx_i[i]),
                    atom_j=int(idx_j[j]),
                )
            )
    return collapse_equivalent(found, tolerance_A) if collapse else sorted(
        found, key=lambda c: c.distance_A
    )

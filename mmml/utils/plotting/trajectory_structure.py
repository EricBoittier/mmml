"""Structural distributions for periodic ASE trajectories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class InternalCoordinates:
    bonds: dict[str, np.ndarray]
    angles: dict[str, np.ndarray]
    dihedrals: dict[str, np.ndarray]


def element_pair_rdfs(
    frames: Sequence[Atoms], *, r_max: float = 8.0, bins: int = 160
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Calculate periodic RDFs for all element pairs in a fixed-topology trajectory."""
    if not frames or not all(np.all(frame.pbc) for frame in frames):
        raise ValueError("periodic frames are required for normalized RDFs")
    symbols = np.asarray(frames[0].get_chemical_symbols())
    elements = sorted(set(symbols))
    edges = np.linspace(0.0, r_max, bins + 1)
    shell_volumes = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    counts = {f"{a}-{b}": np.zeros(bins) for a in elements for b in elements if a <= b}
    expected = {key: np.zeros(bins) for key in counts}
    populations = {element: int(np.sum(symbols == element)) for element in elements}
    for frame in frames:
        first, second, distances = neighbor_list("ijd", frame, r_max, self_interaction=False)
        unique = first < second
        first = first[unique]
        second = second[unique]
        distances = distances[unique]
        for key in counts:
            element_a, element_b = key.split("-")
            if element_a == element_b:
                mask = (symbols[first] == element_a) & (symbols[second] == element_b)
            else:
                mask = ((symbols[first] == element_a) & (symbols[second] == element_b)) | (
                    (symbols[first] == element_b) & (symbols[second] == element_a)
                )
            counts[key] += np.histogram(distances[mask], bins=edges)[0]
        volume = frame.get_volume()
        for key in expected:
            element_a, element_b = key.split("-")
            if element_a == element_b:
                pairs = populations[element_a] * (populations[element_a] - 1) / 2
            else:
                pairs = populations[element_a] * populations[element_b]
            expected[key] += pairs * shell_volumes / volume
    radii = 0.5 * (edges[:-1] + edges[1:])
    return radii, {
        key: np.divide(counts[key], expected[key], out=np.zeros(bins), where=expected[key] > 0)
        for key in counts
    }


def infer_bonds(atoms: Atoms, indices: Sequence[int], *, scale: float = 1.2) -> list[tuple[int, int]]:
    """Infer covalent bonds within an atom subset using covalent radii."""
    selected = set(indices)
    cutoffs = scale * covalent_radii[atoms.numbers]
    first, second = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
    return sorted({(int(i), int(j)) for i, j in zip(first, second) if i < j and i in selected and j in selected})


def internal_coordinate_distributions(
    frames: Sequence[Atoms], indices: Sequence[int]
) -> InternalCoordinates:
    """Collect bonded distances, angles, and proper dihedrals for an atom subset."""
    bonds = infer_bonds(frames[0], indices)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for atom_a, atom_b in bonds:
        neighbors[atom_a].add(atom_b)
        neighbors[atom_b].add(atom_a)
    angles = sorted(
        (atom_a, center, atom_c)
        for center, bonded in neighbors.items()
        for atom_a, atom_c in combinations(sorted(bonded), 2)
    )
    dihedrals = sorted({
        (outer_a, atom_a, atom_b, outer_b)
        for atom_a, atom_b in bonds
        for outer_a in neighbors[atom_a] - {atom_b}
        for outer_b in neighbors[atom_b] - {atom_a}
        if outer_a != outer_b
    })
    symbols = frames[0].get_chemical_symbols()

    def label(atom_indices: tuple[int, ...]) -> str:
        return "-".join(symbols[index] for index in atom_indices)

    values = lambda items, getter: {
        f"{label(item)} ({'-'.join(map(str, item))})": np.asarray([getter(frame, item) for frame in frames])
        for item in items
    }
    return InternalCoordinates(
        bonds=values(bonds, lambda frame, item: frame.get_distance(*item, mic=True)),
        angles=values(angles, lambda frame, item: frame.get_angle(*item, mic=True)),
        dihedrals=values(dihedrals, lambda frame, item: frame.get_dihedral(*item, mic=True)),
    )


def water_tetrahedrality(
    frames: Sequence[Atoms], *, peptide_indices: Sequence[int], near_cutoff: float = 5.0,
    bulk_cutoff: float = 8.0,
) -> dict[str, np.ndarray]:
    """Calculate water oxygen tetrahedral order, split by peptide proximity.

    ``q = 1 - 3/8 sum_(j<k) (cos(psi_jk) + 1/3)^2`` for each oxygen's four
    nearest oxygen neighbors.
    """
    symbols = np.asarray(frames[0].get_chemical_symbols())
    oxygen_indices = np.flatnonzero(symbols == "O")
    peptide_heavy = np.asarray([i for i in peptide_indices if symbols[i] != "H"])
    output: dict[str, list[float]] = {"near": [], "bulk": []}
    for frame in frames:
        lengths = frame.cell.lengths()
        if not np.all(frame.pbc) or not np.allclose(frame.cell.angles(), 90.0):
            raise ValueError("tetrahedrality currently requires an orthorhombic periodic cell")
        wrapped = frame.get_positions(wrap=True)
        oxygen_positions = wrapped[oxygen_indices]
        oxygen_tree = cKDTree(oxygen_positions, boxsize=lengths)
        peptide_tree = cKDTree(wrapped[peptide_heavy], boxsize=lengths)
        _, neighbor_rows = oxygen_tree.query(oxygen_positions, k=5)
        peptide_distances, _ = peptide_tree.query(oxygen_positions, k=1)
        for water_index, neighbors in enumerate(neighbor_rows[:, 1:]):
            vectors = oxygen_positions[neighbors] - oxygen_positions[water_index]
            vectors -= lengths * np.rint(vectors / lengths)
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
            cosine_values = [np.dot(vectors[a], vectors[b]) for a, b in combinations(range(4), 2)]
            q_value = 1.0 - 3.0 / 8.0 * np.sum((np.asarray(cosine_values) + 1.0 / 3.0) ** 2)
            distance = peptide_distances[water_index]
            if distance <= near_cutoff:
                output["near"].append(float(q_value))
            elif distance >= bulk_cutoff:
                output["bulk"].append(float(q_value))
    return {key: np.asarray(values) for key, values in output.items()}

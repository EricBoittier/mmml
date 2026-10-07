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


@dataclass(frozen=True)
class HydrogenBondAnalysis:
    """Hydrogen-bond counts and edge occupancies across a trajectory."""

    total_counts: np.ndarray
    peptide_peptide_counts: np.ndarray
    peptide_water_counts: np.ndarray
    water_water_counts: np.ndarray
    edge_occupancy: dict[tuple[str, str], float]
    node_kinds: dict[str, str]
    representative_frame: int
    representative_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DiffusionAnalysis:
    """Radius of gyration and translational diffusion observables."""

    time_ps: np.ndarray
    radius_of_gyration: np.ndarray
    end_to_end_distance: np.ndarray
    end_to_end_indices: tuple[int, int]
    water_oxygen_msd: np.ndarray
    peptide_com_msd: np.ndarray
    diffusion_angstrom2_per_ps: float
    diffusion_cm2_per_s: float
    peptide_diffusion_angstrom2_per_ps: float
    peptide_diffusion_cm2_per_s: float
    fit_start_frame: int


def unwrap_trajectory_positions(frames: Sequence[Atoms]) -> np.ndarray:
    """Reconstruct continuous Cartesian positions using fractional MIC steps."""
    if not frames or not all(np.all(frame.pbc) for frame in frames):
        raise ValueError("unwrapping requires a non-empty fully periodic trajectory")
    atom_count = len(frames[0])
    fractional = []
    for frame_index, frame in enumerate(frames):
        if len(frame) != atom_count:
            raise ValueError(f"atom count changes in frame {frame_index}")
        fractional.append(frame.get_scaled_positions(wrap=False))
    fractional = np.asarray(fractional)
    unwrapped_fractional = np.empty_like(fractional)
    unwrapped_fractional[0] = fractional[0]
    for frame_index in range(1, len(frames)):
        step = fractional[frame_index] - fractional[frame_index - 1]
        step -= np.rint(step)
        if np.max(np.abs(step)) >= 0.5 - 1e-8:
            raise ValueError(
                f"frame {frame_index} contains a displacement at the half-cell ambiguity limit"
            )
        unwrapped_fractional[frame_index] = unwrapped_fractional[frame_index - 1] + step
    return np.asarray([
        unwrapped_fractional[index] @ frames[index].cell.array for index in range(len(frames))
    ])


def radius_of_gyration_and_diffusion(
    frames: Sequence[Atoms], *, peptide_indices: Sequence[int], timestep_ps: float,
    fit_start_fraction: float = 0.5, end_to_end_indices: tuple[int, int] | None = None,
) -> DiffusionAnalysis:
    """Calculate peptide radius of gyration and water-oxygen Einstein MSD/D."""
    if timestep_ps <= 0:
        raise ValueError("timestep_ps must be positive")
    if not 0 <= fit_start_fraction < 1:
        raise ValueError("fit_start_fraction must be in [0, 1)")
    positions = unwrap_trajectory_positions(frames)
    peptide = np.asarray(list(peptide_indices), dtype=int)
    masses = frames[0].get_masses()[peptide]
    peptide_positions = positions[:, peptide]
    centers = np.sum(peptide_positions * masses[None, :, None], axis=1) / masses.sum()
    radius = np.sqrt(np.sum(
        masses[None, :] * np.sum((peptide_positions - centers[:, None, :]) ** 2, axis=2), axis=1
    ) / masses.sum())
    if end_to_end_indices is None:
        peptide_symbols = np.asarray(frames[0].get_chemical_symbols())[peptide]
        heavy = peptide[peptide_symbols != "H"]
        end_to_end_indices = (int(heavy[0]), int(heavy[-1]))
    start_index, end_index = end_to_end_indices
    if start_index not in set(peptide) or end_index not in set(peptide):
        raise ValueError("end-to-end atom indices must belong to the peptide selection")
    end_to_end = np.linalg.norm(positions[:, end_index] - positions[:, start_index], axis=1)
    symbols = np.asarray(frames[0].get_chemical_symbols())
    oxygen_indices = np.asarray([
        index for index in np.flatnonzero(symbols == "O") if index not in set(peptide)
    ])
    oxygen_positions = positions[:, oxygen_indices]

    def time_origin_averaged_msd(coordinates: np.ndarray) -> np.ndarray:
        result = np.zeros(len(coordinates), dtype=float)
        for lag in range(1, len(coordinates)):
            displacement = coordinates[lag:] - coordinates[:-lag]
            result[lag] = np.mean(np.sum(displacement**2, axis=-1))
        return result

    msd = time_origin_averaged_msd(oxygen_positions)
    peptide_com_msd = time_origin_averaged_msd(centers[:, None, :])
    time = np.arange(len(frames), dtype=float) * timestep_ps
    fit_start = max(1, int(len(frames) * fit_start_fraction))
    slope, _ = np.polyfit(time[fit_start:], msd[fit_start:], 1)
    diffusion = max(0.0, float(slope) / 6.0)
    peptide_slope, _ = np.polyfit(time[fit_start:], peptide_com_msd[fit_start:], 1)
    peptide_diffusion = max(0.0, float(peptide_slope) / 6.0)
    return DiffusionAnalysis(
        time_ps=time, radius_of_gyration=radius, water_oxygen_msd=msd,
        end_to_end_distance=end_to_end, end_to_end_indices=end_to_end_indices,
        peptide_com_msd=peptide_com_msd,
        diffusion_angstrom2_per_ps=diffusion,
        diffusion_cm2_per_s=diffusion * 1e-4,
        peptide_diffusion_angstrom2_per_ps=peptide_diffusion,
        peptide_diffusion_cm2_per_s=peptide_diffusion * 1e-4,
        fit_start_frame=fit_start,
    )


def heavy_atom_pair_distance_umap(
    frames: Sequence[Atoms], *, peptide_indices: Sequence[int], n_neighbors: int = 20,
    min_dist: float = 0.1, random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed peptide conformations from rotation-invariant heavy-atom distances."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("heavy_atom_pair_distance_umap requires umap-learn") from exc
    peptide = np.asarray(list(peptide_indices), dtype=int)
    symbols = np.asarray(frames[0].get_chemical_symbols())
    heavy = peptide[symbols[peptide] != "H"]
    pairs = np.asarray(list(combinations(heavy, 2)), dtype=int)
    features = np.asarray([
        [frame.get_distance(int(first), int(second), mic=True) for first, second in pairs]
        for frame in frames
    ])
    scales = features.std(axis=0)
    standardized = (features - features.mean(axis=0)) / np.where(scales > 1e-12, scales, 1.0)
    embedding = umap.UMAP(
        n_components=2, n_neighbors=min(n_neighbors, len(frames) - 1), min_dist=min_dist,
        metric="euclidean", random_state=random_state,
    ).fit_transform(standardized)
    return embedding, features, pairs


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
        if not np.allclose(frame.cell.angles(), 90.0):
            raise ValueError("RDF calculation currently requires an orthorhombic cell")
        lengths = frame.cell.lengths()
        positions = frame.get_positions(wrap=True)
        pairs = cKDTree(positions, boxsize=lengths).query_pairs(r_max, output_type="ndarray")
        first, second = pairs[:, 0], pairs[:, 1]
        vectors = positions[first] - positions[second]
        vectors -= lengths * np.rint(vectors / lengths)
        distances = np.linalg.norm(vectors, axis=1)
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


def hydrogen_bond_analysis(
    frames: Sequence[Atoms], *, peptide_indices: Sequence[int],
    distance_cutoff: float = 3.8, angle_cutoff_degrees: float = 135.0,
) -> HydrogenBondAnalysis:
    """Find D-H...A hydrogen bonds and aggregate an interaction network.

    Nitrogen and oxygen atoms are treated as donors when covalently bound to a
    hydrogen and as acceptors. Non-peptide atoms are expected to be contiguous
    three-atom water molecules, as in the documented ``test4.pdb`` trajectory.
    """
    if not frames:
        raise ValueError("trajectory is empty")
    peptide = set(int(index) for index in peptide_indices)
    solute_count = max(peptide) + 1
    symbols = np.asarray(frames[0].get_chemical_symbols())
    acceptors = np.flatnonzero(np.isin(symbols, ("N", "O")))
    hydrogens = np.flatnonzero(symbols == "H")
    first = frames[0]
    lengths = first.cell.lengths()
    positions = first.get_positions(wrap=True)
    heavy_tree = cKDTree(positions[acceptors], boxsize=lengths)
    donor_pairs: list[tuple[int, int]] = []
    for hydrogen in hydrogens:
        distances, rows = heavy_tree.query(positions[hydrogen], k=min(2, len(acceptors)))
        for distance, row in zip(np.atleast_1d(distances), np.atleast_1d(rows)):
            donor = int(acceptors[row])
            if distance <= 1.25:
                donor_pairs.append((donor, int(hydrogen)))
                break

    categories = np.zeros((len(frames), 3), dtype=int)
    edge_frames: dict[tuple[str, str], int] = defaultdict(int)
    node_kinds: dict[str, str] = {}
    atom_edges_by_frame: list[tuple[tuple[int, int], ...]] = []

    def node_name(atom_index: int) -> str:
        if atom_index in peptide:
            name = f"{symbols[atom_index]}{atom_index}"
            node_kinds[name] = "peptide"
            return name
        water_number = (atom_index - solute_count) // 3 + 1
        name = f"W{water_number}"
        node_kinds[name] = "water"
        return name

    for frame_index, frame in enumerate(frames):
        if not np.all(frame.pbc) or not np.allclose(frame.cell.angles(), 90.0):
            raise ValueError("hydrogen-bond analysis requires an orthorhombic periodic cell")
        lengths = frame.cell.lengths()
        positions = frame.get_positions(wrap=True)
        acceptor_tree = cKDTree(positions[acceptors], boxsize=lengths)
        frame_edges: set[tuple[str, str]] = set()
        frame_atom_edges: set[tuple[int, int]] = set()
        for donor, hydrogen in donor_pairs:
            candidate_rows = acceptor_tree.query_ball_point(positions[donor], distance_cutoff)
            donor_hydrogen = positions[donor] - positions[hydrogen]
            donor_hydrogen -= lengths * np.rint(donor_hydrogen / lengths)
            for row in candidate_rows:
                acceptor = int(acceptors[row])
                if acceptor == donor:
                    continue
                acceptor_hydrogen = positions[acceptor] - positions[hydrogen]
                acceptor_hydrogen -= lengths * np.rint(acceptor_hydrogen / lengths)
                cosine = np.dot(donor_hydrogen, acceptor_hydrogen) / (
                    np.linalg.norm(donor_hydrogen) * np.linalg.norm(acceptor_hydrogen)
                )
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                if angle < angle_cutoff_degrees:
                    continue
                donor_peptide = donor in peptide
                acceptor_peptide = acceptor in peptide
                if donor_peptide and acceptor_peptide:
                    categories[frame_index, 0] += 1
                elif donor_peptide or acceptor_peptide:
                    categories[frame_index, 1] += 1
                else:
                    categories[frame_index, 2] += 1
                frame_edges.add((node_name(donor), node_name(acceptor)))
                frame_atom_edges.add((donor, acceptor))
        for edge in frame_edges:
            edge_frames[edge] += 1
        atom_edges_by_frame.append(tuple(sorted(frame_atom_edges)))
    occupancy = {edge: count / len(frames) for edge, count in edge_frames.items()}
    representative_frame = int(np.argmax(categories.sum(axis=1)))
    return HydrogenBondAnalysis(
        total_counts=categories.sum(axis=1), peptide_peptide_counts=categories[:, 0],
        peptide_water_counts=categories[:, 1], water_water_counts=categories[:, 2],
        edge_occupancy=occupancy, node_kinds=node_kinds,
        representative_frame=representative_frame,
        representative_edges=atom_edges_by_frame[representative_frame],
    )

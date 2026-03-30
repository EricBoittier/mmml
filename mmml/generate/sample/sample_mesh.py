"""
create a 2D representation of the vdw surface of two (2) molecules - called a "mesh"

use symmetry to reduce the number of points in the mesh

using the cartesian product of the two "meshes" - determine the geometry of the dimer 
by using the normal of the surface at the point of contact


"""

from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3
from mmml.generate.sample import sample_cc

patch_chemcoord_for_pandas3()
import chemcoord as cc
import ase
from ase.data import vdw_radii
import numpy as np

def load_molecule_xyz(filename: str) -> cc.Cartesian:
    return cc.Cartesian.read_xyz(filename)


def symmetrize_molecule(
    molecule: cc.Cartesian,
    *,
    max_n: int = 25,
    tolerance: float = 0.3,
    epsilon: float = 1e-5,
):
    return molecule.symmetrize(max_n=max_n, tolerance=tolerance, epsilon=epsilon)


def vdw_radii_for_cartesian(cart: cc.Cartesian) -> np.ndarray:
    atomic_numbers = np.array([ase.data.atomic_numbers[a] for a in cart["atom"]])
    return vdw_radii[atomic_numbers]


def point_inside_any_other_atom(
    point: np.ndarray,
    *,
    index_to_exclude: int,
    positions: np.ndarray,
    radii: np.ndarray,
) -> bool:
    for index, (position, radius) in enumerate(zip(positions, radii)):
        if index == index_to_exclude:
            continue
        if np.linalg.norm(point - position) < radius:
            return True
    return False


def mesh_points_for_atoms(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    n_radial: int = 10,
    n_angular: int = 10,
    radii_scale: float = 1.0,
) -> list[list[np.ndarray]]:
    """
    Returns a per-atom list of surface points (each point is a 3-vector).
    Points that fall inside any other atom vdW sphere are discarded.
    """
    meshes: list[list[np.ndarray]] = []
    for atom_index, (position, radius) in enumerate(zip(positions, radii)):
        mesh: list[np.ndarray] = []
        r = float(radius) * float(radii_scale)
        for i in range(n_radial):
            for j in range(n_angular):
                theta = j * 2 * np.pi / n_angular
                phi = i * np.pi / n_radial
                x = position[0] + r * np.sin(theta) * np.cos(phi)
                y = position[1] + r * np.sin(theta) * np.sin(phi)
                z = position[2] + r * np.cos(theta)
                p = np.array([x, y, z], dtype=float)
                if not point_inside_any_other_atom(
                    p, index_to_exclude=atom_index, positions=positions, radii=radii
                ):
                    mesh.append(p)
        meshes.append(mesh)
    return meshes


def unique_mesh_points_from_symmetry(eq, atom_meshes: list[list[np.ndarray]]) -> np.ndarray:
    """
    Select one representative atom from each equivalence class in eq['eq_sets'].
    """
    unique_meshes: list[list[np.ndarray]] = []
    for unique_atom_index in eq["eq_sets"].keys():
        unique_meshes.append(atom_meshes[int(unique_atom_index)])
    if not unique_meshes:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate([np.array(mesh, dtype=float) for mesh in unique_meshes], axis=0)


def normals_from_nearest_neighbors(mesh_points: np.ndarray) -> np.ndarray:
    """
    Approximate a surface normal at each mesh point using the cross product of
    vectors to its two nearest neighbor points.
    """
    if len(mesh_points) == 0:
        return np.zeros((0, 3), dtype=float)

    nearest_neighbors = []
    for point in mesh_points:
        distances = np.linalg.norm(mesh_points - point, axis=1)
        nn_idx = distances.argsort()[1:3]  # skip itself at index 0
        nearest_neighbors.append(nn_idx)
    nearest_neighbors = np.array(nearest_neighbors, dtype=int)

    normals = []
    for i, point in enumerate(mesh_points):
        j, k = nearest_neighbors[i]
        v1 = mesh_points[j] - point
        v2 = mesh_points[k] - point
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / norm
        normals.append(n)
    return np.array(normals, dtype=float)


def rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return a 3x3 rotation matrix that rotates vector a onto vector b.
    a and b are 3-element numpy arrays.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, 1.0):
        # vectors are already aligned
        return np.eye(3)
    if np.isclose(c, -1.0):
        # opposite directions: rotate 180 degrees around any perpendicular axis
        # pick an arbitrary axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(a, axis):
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        x, y, z = v
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + 2 * K @ K
    s = np.linalg.norm(v)
    vx, vy, vz = v
    K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R


def has_inter_monomer_overlap(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    radii: np.ndarray,
    *,
    overlap_tolerance: float = 0.1,
) -> bool:
    """
    Returns True if any atom pair (A_i, B_j) is closer than the sum of vdW radii
    minus overlap_tolerance (in Angstrom).
    """
    # pairwise distances (n_atoms x n_atoms)
    diffs = coords_A[:, None, :] - coords_B[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    thresholds = (radii[:, None] + radii[None, :]) - float(overlap_tolerance)
    return bool(np.any(dists < thresholds))


def candidate_point_pairs(n_points: int, *, rng: np.random.Generator, max_pairs: int) -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(n_points) for j in range(n_points) if i != j]
    if len(pairs) <= max_pairs:
        return pairs
    chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[int(k)] for k in chosen]


def generate_dimers(
    mesh_points: np.ndarray,
    normals: np.ndarray,
    mol_cart: cc.Cartesian,
    *,
    max_dimers: int = 100,
    output_xyz: str = "dimers.xyz",
    overlap_tolerance: float = 0.1,
    seed: int = 0,
):
    """
    Wrapper that delegates to the shared mesh-based dimer generator in
    sample_cc so that sampling criteria and random noise are consistent.
    """
    sample_cc.generate_dimers_mesh(
        mesh_points=mesh_points,
        normals=normals,
        mol_cart=mol_cart,
        max_dimers=max_dimers,
        output_xyz=output_xyz,
        overlap_tolerance=overlap_tolerance,
        seed=seed,
    )


def main():
    filename = "old/meoh.xyz"

    molecule = load_molecule_xyz(filename)
    eq = symmetrize_molecule(molecule, max_n=25, tolerance=0.3, epsilon=1e-5)

    sym_mol = eq["sym_mol"]
    positions = sym_mol[["x", "y", "z"]].to_numpy()
    radii = vdw_radii_for_cartesian(sym_mol)

    atom_meshes = mesh_points_for_atoms(positions, radii, n_radial=10, n_angular=10, radii_scale=1.0)
    mesh_points = unique_mesh_points_from_symmetry(eq, atom_meshes)
    normals = normals_from_nearest_neighbors(mesh_points)

    generate_dimers(
        mesh_points,
        normals,
        sym_mol,
        max_dimers=10000,
        output_xyz="meoh_dimers_mesh_sampled.xyz",
        overlap_tolerance=0.1,
        seed=0,
    )


if __name__ == "__main__":
    main()


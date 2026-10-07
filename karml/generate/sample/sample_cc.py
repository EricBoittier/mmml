from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3

patch_chemcoord_for_pandas3()

import chemcoord as cc
import ase
from ase.data import vdw_radii
import numpy as np
import pandas as pd


DEFAULT_NOISE_SCALE = 0.01


def make_rng(seed: int | None = 0) -> np.random.Generator:
    """
    Create a numpy random number generator with an optional seed.
    """
    return np.random.default_rng(seed)


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


def candidate_point_pairs(
    n_points: int, *, rng: np.random.Generator, max_pairs: int
) -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(n_points) for j in range(n_points) if i != j]
    if len(pairs) <= max_pairs:
        return pairs
    chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[int(k)] for k in chosen]


def generate_dimers_mesh(
    mesh_points: np.ndarray,
    normals: np.ndarray,
    mol_cart: cc.Cartesian,
    *,
    max_dimers: int = 100,
    output_xyz: str = "dimers.xyz",
    overlap_tolerance: float = 0.1,
    seed: int | None = 0,
    noise_scale: float = DEFAULT_NOISE_SCALE,
):
    """
    Generate approximate dimer geometries by placing two copies of the molecule
    so that a pair of mesh points touch and their surface normals are antiparallel.
    Results are written to a multi-structure XYZ file.
    """
    positions = mol_cart[["x", "y", "z"]].to_numpy()
    atoms = mol_cart["atom"].to_numpy()
    radii = vdw_radii_for_cartesian(mol_cart)

    n_points = len(mesh_points)
    if n_points == 0:
        return

    rng = make_rng(seed)
    candidate_indices = candidate_point_pairs(n_points, rng=rng, max_pairs=max_dimers)

    with open(output_xyz, "w") as f:
        for idx, (i, j) in enumerate(candidate_indices):
            p1 = mesh_points[i]
            n1 = normals[i]
            p2 = mesh_points[j]
            n2 = normals[j]

            # rotate copy B such that its normal at p2 points opposite to n1
            R = rotation_between_vectors(n2, -n1)

            # translation so that the rotated mesh point at p2 lands on p1
            t = p1 - R @ p2

            # coordinates of monomer A: as-is
            coords_A = positions.copy()
            # coordinates of monomer B
            coords_B = (R @ positions.T).T + t

            if has_inter_monomer_overlap(
                coords_A, coords_B, radii, overlap_tolerance=overlap_tolerance
            ):
                continue

            all_coords = np.vstack([coords_A, coords_B])
            all_atoms = np.concatenate([atoms, atoms])

            # add small random noise consistent with internal-coordinate sampler
            if noise_scale and noise_scale > 0.0:
                all_coords = all_coords + rng.normal(
                    scale=noise_scale, size=all_coords.shape
                )

            f.write(f"{len(all_atoms)}\n")
            f.write(f"dimer {idx} from mesh points {i} and {j}\n")
            for sym, (x, y, z) in zip(all_atoms, all_coords):
                f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def sample_dimer_cc(
    xyz_file: str,
    mol_r_scale: float = 1.0,
    *,
    seed: int | None = 0,
    noise_scale: float = DEFAULT_NOISE_SCALE,
):
    """
    Chemcoord-based sampling of dimers based on internal coordinates.
    Returns a list of XYZ pandas.DataFrame objects.
    """
    import sympy

    rng = make_rng(seed)

    cc_mol_xyz = cc.Cartesian.read_xyz(xyz_file)

    mol_r = (
        cc_mol_xyz[["x", "y", "z"]].max().max()
        - cc_mol_xyz[["x", "y", "z"]].min().min()
    )
    mol_r = mol_r / 2 * mol_r_scale
    print("mol_r", mol_r)
    fragments = cc_mol_xyz.fragmentate()

    sympy.init_printing()
    ba = sympy.Symbol("ba")
    bb = sympy.Symbol("bb")
    aa = sympy.Symbol("aa")
    ab = sympy.Symbol("ab")
    da = sympy.Symbol("da")
    db = sympy.Symbol("db")

    ba_val = 5
    bb_val = 5
    aa_val = 90
    ab_val = -90
    da_val = 0
    db_val = 0

    zmat1 = fragments[0].to_zmat()
    zmat2 = zmat1.copy()

    zmat1.safe_loc[zmat1.index[0], "bond"] = ba
    zmat1.safe_loc[zmat1.index[0], "angle"] = aa
    zmat1.safe_loc[zmat1.index[0], "dihedral"] = da

    zmat2.safe_loc[zmat2.index[0], "bond"] = bb
    zmat2.safe_loc[zmat2.index[0], "angle"] = ab
    zmat2.safe_loc[zmat2.index[0], "dihedral"] = db

    ba_vals = np.arange(mol_r, mol_r + 3, 2)
    bb_vals = np.arange(mol_r, mol_r + 3, 2)
    aa_vals = np.arange(0, 90, 33)
    ab_vals = np.arange(-90, 0, 33)
    da_vals = np.arange(0, 180, 33)
    db_vals = np.arange(-181, 0, 33)

    def make_conf(ba_val, bb_val, aa_val, ab_val, da_val, db_val):
        a = (
            zmat1.subs(ba, ba_val + rng.normal() / 100.0)
            .subs(aa, aa_val + rng.normal())
            .subs(da, da_val + rng.normal())
            .get_cartesian()[["x", "y", "z"]]
            .sort_index()
        )
        a = a.to_numpy()

        b = (
            zmat2.subs(bb, bb_val)
            .subs(ab, ab_val)
            .subs(db, db_val)
            .get_cartesian()[["x", "y", "z"]]
            .sort_index()
        )
        b = b.to_numpy()

        combined = np.concatenate([a, b])
        if noise_scale and noise_scale > 0.0:
            combined = combined + rng.normal(
                scale=noise_scale, size=combined.shape
            )

        XYZ = pd.DataFrame(combined, columns=["x", "y", "z"])
        return XYZ

    xyzs = []

    for ba_val in ba_vals:
        for bb_val in bb_vals:
            for aa_val in aa_vals:
                for ab_val in ab_vals:
                    for da_val in da_vals:
                        for db_val in db_vals:
                            XYZ = make_conf(
                                ba_val, bb_val, aa_val, ab_val, da_val, db_val
                            )
                            # center the dimer
                            XYZ = XYZ - XYZ.mean()
                            xyzs.append(XYZ)

    return xyzs


"""
create a 2D representation of the vdw surface of two (2) molecules - called a "mesh"

use symmetry to reduce the number of points in the mesh

using the cartesian product of the two "meshes" - determine the geometry of the dimer 
by using the normal of the surface at the point of contact


"""

from mmml.interfaces.chemcoordInterface import interface
from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3
patch_chemcoord_for_pandas3()
import chemcoord as cc
import ase
from ase.data import vdw_radii
import numpy as np

filename = "old/meoh.xyz"

molecule = cc.Cartesian.read_xyz(filename)
print(molecule)

eq = molecule.symmetrize(max_n=25, tolerance=0.3, epsilon=1e-5)



pointgroup = eq["sym_mol"].get_pointgroup(tolerance=0.1)
print(pointgroup)
print(eq)

# Cs
# {'sym_mol':   atom         x         y         z
# 0    C  0.140666 -0.178732 -0.693882
# 1    O -0.171845  0.129798  0.661857
# 2    H  0.389774  0.868777  0.938465
# 3    H -0.042119  0.704023 -1.348376
# 4    H  1.206773 -0.488750 -0.787630
# 5    H -0.502864 -1.014594 -1.040000, 'eq_sets': {0: {0}, 1: {1}, 2: {2}, 3: {3, 4}, 5: {5}}, 'sym_ops': {0: {0: array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])}, 1: {1: array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])}, 2: {2: array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])}, 3: {4: array([[ 0.05381204,  0.90367084, -0.42483324],
#        [ 0.90367084,  0.13693577,  0.40574328],
#        [-0.42483324,  0.40574328,  0.80925219]]), 3: array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])}, 4: {3: array([[ 0.05381204,  0.90367084, -0.42483324],
#        [ 0.90367084,  0.13693577,  0.40574328],
#        [-0.42483324,  0.40574328,  0.80925219]]), 4: array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])}, 5: {5: array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])}}}

center = eq["sym_mol"][["x", "y", "z"]].mean()
print(center)

atomic_numbers = [ase.data.atomic_numbers[atom] for atom in eq["sym_mol"]["atom"]]
print(atomic_numbers)

radii = vdw_radii[atomic_numbers]
print(radii)

def point_inside_mesh(point, index_to_exclude, positions, radii):
    for index, (position, radius) in enumerate(zip(positions, radii)):
        if index == index_to_exclude:
            continue
        elif np.linalg.norm(np.array(point) - np.array(position)) < radius:
            return True
    return False


def to_mesh(positions, radii, n_radial = 10, n_angular = 10):
    meshes = []
    for index, (position, radius) in enumerate(zip(positions, radii)):
        mesh = []
        for i in range(n_radial):
            for j in range(n_angular):
                theta = j * 2 * np.pi / n_angular
                phi = i * np.pi / n_radial
                x = position[0] + radius * np.sin(theta) * np.cos(phi)
                y = position[1] + radius * np.sin(theta) * np.sin(phi)
                z = position[2] + radius * np.cos(theta)
                if not point_inside_mesh((x, y, z), index, positions, radii):      
                    mesh.append((x, y, z))
        meshes.append(mesh)
    return meshes



all_atom_mesh = to_mesh(eq["sym_mol"][["x", "y", "z"]].to_numpy(), radii)


all_unique_mesh = []
for unique in eq["eq_sets"].keys():
    unique_mesh = all_atom_mesh[unique]
    all_unique_mesh.append(unique_mesh)

all_unique_mesh = np.concatenate([np.array(mesh) for mesh in all_unique_mesh])
print(all_unique_mesh)
print(all_unique_mesh.shape)


# find each point's two nearest neighbors (excluding itself)
nearest_neighbors = []
for point in all_unique_mesh:
    distances = np.linalg.norm(all_unique_mesh - point, axis=1)
    # first index is the point itself (distance 0); skip it
    nn_idx = distances.argsort()[1:3]
    nearest_neighbors.append(nn_idx)
nearest_neighbors = np.array(nearest_neighbors, dtype=int)

# create a normal vector for each mesh point
normals = []
for i, point in enumerate(all_unique_mesh):
    j, k = nearest_neighbors[i]
    v1 = all_unique_mesh[j] - point
    v2 = all_unique_mesh[k] - point
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm == 0.0:
        # degenerate; just pick some arbitrary but consistent vector
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / norm
    normals.append(n)
normals = np.array(normals)


def rotation_between_vectors(a, b):
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


def generate_dimers(mesh_points, normals, mol_cart, max_dimers=100, output_xyz="dimers.xyz"):
    """
    Generate approximate dimer geometries by placing two copies of the molecule
    so that a pair of mesh points touch and their surface normals are antiparallel.
    Results are written to a multi-structure XYZ file.
    """
    positions = mol_cart[["x", "y", "z"]].to_numpy()
    atoms = mol_cart["atom"].to_numpy()
    atomic_numbers = np.array([ase.data.atomic_numbers[a] for a in atoms])
    radii = vdw_radii[atomic_numbers]

    n_points = len(mesh_points)
    if n_points == 0:
        return

    # sample a subset of point pairs to avoid combinatorial explosion
    rng = np.random.default_rng(0)
    candidate_indices = [(i, j) for i in range(n_points) for j in range(n_points) if i != j]
    if len(candidate_indices) > max_dimers:
        candidate_indices = rng.choice(len(candidate_indices), size=max_dimers, replace=False)
        candidate_indices = [((idx // n_points), (idx % n_points)) for idx in candidate_indices]

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

            # coordinates of monomer A: as-is, but shifted so that its center is at origin already
            coords_A = positions.copy()
            # coordinates of monomer B
            coords_B = (R @ positions.T).T + t

            # reject dimers with overlapping atoms (inter-monomer distances
            # smaller than sum of vdW radii minus a small tolerance)
            overlap = False
            for ia in range(len(positions)):
                for ib in range(len(positions)):
                    d = np.linalg.norm(coords_A[ia] - coords_B[ib])
                    if d < 0.5*(radii[ia] + radii[ib]):
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                continue

            all_coords = np.vstack([coords_A, coords_B])
            all_atoms = np.concatenate([atoms, atoms])

            f.write(f"{len(all_atoms)}\n")
            f.write(f"dimer {idx} from mesh points {i} and {j}\n")
            for sym, (x, y, z) in zip(all_atoms, all_coords):
                f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


if __name__ == "__main__":
    # generate a modest number of dimers and write them to dimers.xyz
    generate_dimers(all_unique_mesh, normals, eq["sym_mol"], max_dimers=10000, output_xyz="dimers.xyz")


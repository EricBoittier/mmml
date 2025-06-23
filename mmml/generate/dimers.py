from pyxtal import pyxtal

import numpy as np
from ase import Atoms
from xtb_ase import XTB
from pyxtal.molecule import pyxtal_molecule
from mmml.chemcoordInterface.interface import ase_to_chemcord, sym_to_ase

def generate_from_crystal_syms(fn, n_repeats=1, dimensions=(1,2,3), space_group_numbers=list(range(1,171))):
    m = pyxtal_molecule(fn)

    confs = {}
    def com_dist(x,n):
        return np.linalg.norm(x[:n].get_center_of_mass() - x[n:].get_center_of_mass())

    for r in range(n_repeats):
        for d in dimensions:
            for i in space_group_numbers:
                c = pyxtal(molecular=True)
                failed = True
                try:
                    c.from_random(d, i, [m], [2])
                    failed = False
                except:
                    pass
                if not failed:
                    catoms = c.to_ase()
                    ccc, ccz = ase_to_chemcord(catoms)
                    eq = ccc.symmetrize()
                    pgs = eq["sym_mol"].get_pointgroup(tolerance=0.1)
                    if str(pgs) not in confs.keys():
                        confs[str(pgs)] = []
                    confs[str(pgs)].append((catoms, eq))

    return confs


def get_sym_scans(min_confs, distance=7.0):
    sym_scans = {}

    for c in min_confs.keys():
        C2_test_atoms_A, C2_test_atoms_B = separate_monomers_by_distance(min_confs[c])
        scan1  = make_rigid_com_scan(C2_test_atoms_A, C2_test_atoms_B, maxdist=distance-0.5, mindist=distance+0.5, n_points=6)
        sym_scans[c] = [make_rigid_angle_scan(
            _[:15], _[15:], n_points=12, distance=com_dist(_,15))
                        for _ in scan1]
    
    return sym_scans


def make_rigid_com_scan(A, B, n_points=20, maxdist=20.0, mindist=2.0):
    """
    Generate a series of rigid dimer configurations by translating B along the COM vector from A.
    
    Parameters:
        A (ase.Atoms): First molecule.
        B (ase.Atoms): Second molecule.
        n_points (int): Number of translation steps.
        maxdist (float): Maximum separation between COMs.
        mindist (float): Minimum separation between COMs.
        
    Returns:
        list[ase.Atoms]: List of combined dimer structures.
    """
    com_A = A.get_center_of_mass()
    com_B = B.get_center_of_mass()
    
    # Vector pointing from B to A
    disp_vec = com_A - com_B
    norm = np.linalg.norm(disp_vec)
    if norm == 0:
        raise ValueError("A and B have the same center of mass.")
    
    disp_unit = disp_vec / norm

    # Distances at which to place B
    distances = np.linspace(mindist, maxdist, n_points)
    structures = []

    for d in distances:
        B_translated = B.copy()
        new_com_B = com_A - disp_unit * d
        shift_vec = new_com_B - com_B
        B_translated.translate(shift_vec)

        combined = A + B_translated
        structures.append(combined)

    return structures


# from ase.geometry import rotate

def make_rigid_angle_scan(A, B, n_points=20, distance=3.0, axis='z'):
    """
    Rotate B around its COM while keeping it at a fixed COM distance from A.
    
    Parameters:
        A (ase.Atoms): First molecule.
        B (ase.Atoms): Second molecule.
        n_points (int): Number of angular steps.
        distance (float): Distance between COMs.
        axis (str or array-like): Axis around which B is rotated.
        
    Returns:
        list[ase.Atoms]: List of rotated and combined dimer structures.
    """
    com_A = A.get_center_of_mass()
    com_B = B.get_center_of_mass()
    
    # Translate B so its COM is 'distance' away from A along x-axis
    initial_direction = np.array([1.0, 0.0, 0.0])
    target_com_B = com_A + initial_direction * distance
    shift_vec = target_com_B - com_B

    structures = []
    angles = np.linspace(0, 360, n_points, endpoint=False)

    for angle in angles:
        B_rotated = B.copy()
        B_rotated.rotate(angle, axis, center=com_B)  # rotate around B's COM
        B_rotated.translate(shift_vec)                    # place at correct distance
        combined = A + B_rotated
        structures.append(combined)

    return structures


# In[101]:


import numpy as np
from ase import Atoms
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def separate_monomers_by_distance(system: Atoms, threshold: float = 1.8):
    """
    Separates a system of atoms into monomers based on distance threshold.
    
    Parameters:
        system (ase.Atoms): The full atomic system (e.g., dimer or multimer).
        threshold (float): Maximum distance to consider atoms as bonded.
        
    Returns:
        list[ase.Atoms]: List of separate monomer Atoms objects.
    """
    positions = system.get_positions()
    dists = squareform(pdist(positions))
    
    # Build adjacency matrix where atoms within threshold are connected
    adjacency = (dists < threshold).astype(int)
    np.fill_diagonal(adjacency, 0)  # remove self-loops
    
    # Use connected components to group atoms
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False)
    
    monomers = []
    for i in range(n_components):
        indices = np.where(labels == i)[0]
        monomer = system[indices]
        monomers.append(monomer)
    
    return monomers

import numpy as np
from ase import Atoms

def get_mean_plane(atoms: Atoms):
    """
    Computes the best-fit (mean) plane for a set of atoms.

    Parameters:
        atoms (ase.Atoms): Atoms object to fit the plane to.

    Returns:
        centroid (np.ndarray): A point on the plane (mean position).
        normal (np.ndarray): Unit vector normal to the best-fit plane.
    """
    positions = atoms.get_positions()
    centroid = np.mean(positions, axis=0)

    # Subtract centroid to center the points
    centered = positions - centroid

    # Singular Value Decomposition
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # Last row corresponds to smallest singular value (normal direction)

    return centroid, normal


import numpy as np

def angle_between_planes(normal1: np.ndarray, normal2: np.ndarray, degrees=True):
    """
    Computes the angle between two planes from their normal vectors.

    Parameters:
        normal1 (np.ndarray): Normal vector of the first plane.
        normal2 (np.ndarray): Normal vector of the second plane.
        degrees (bool): If True, returns angle in degrees. Otherwise, in radians.

    Returns:
        float: Angle between the two planes.
    """
    # Normalize normals
    n1 = normal1 / np.linalg.norm(normal1)
    n2 = normal2 / np.linalg.norm(normal2)

    # Compute angle between normals
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if degrees:
        angle = np.degrees(angle)
    
    return angle


def get_rog(a):
    mi = a.get_masses()
    ri = a.get_positions()
    Rg = np.sqrt(np.sum(mi[:,np.newaxis] * ri**2) / np.sum(mi))
    return Rg


import itertools

# def get_all_atoms(sym_scans):
#     all_atoms = []
#     for _ in list(sym_scans.values()):
#         for __ in _:
#             for ___ in __:
#                 all_atoms.append(___)
#     return all_atoms

def com_dist(x,n):
    return np.linalg.norm(x[:n].get_center_of_mass() - x[n:].get_center_of_mass())

def make_rotations(A, B, distance = 8):
    
    scan_r  = make_rigid_com_scan(A, B, maxdist=distance-.01, mindist=distance+.01, n_points=5)
    scan_r_theta_1 = [make_rigid_angle_scan(
        _[:15], _[15:], n_points=15, distance=com_dist(_,15), axis=get_mean_plane(_)[1])
                    for _ in scan_r] + [make_rigid_angle_scan(
        _[:15], _[15:], n_points=15, distance=com_dist(_,15), axis=-get_mean_plane(_)[1])
                    for _ in scan_r] + [make_rigid_angle_scan(
        _[:15], _[15:], n_points=15, distance=com_dist(_,15), axis=get_mean_plane(_)[0])
                    for _ in scan_r] + [make_rigid_angle_scan(
        _[:15], _[15:], n_points=15, distance=com_dist(_,15), axis=-get_mean_plane(_)[0])
                    for _ in scan_r]
    scan_r_theta_2 = [make_rigid_angle_scan(
        _[:15], _[15:], n_points=30, distance=com_dist(_,15))
                    for _ in scan_r]
    import itertools
    all_atoms = list(itertools.chain.from_iterable(scan_r_theta_1 + scan_r_theta_2))
    # add noise
    for _ in all_atoms:
        _R = _.get_positions()
        _R += ((np.random.normal(size=np.prod(_R.shape)).reshape(_R.shape))*0.01)
        _.set_positions(_R)
    return all_atoms



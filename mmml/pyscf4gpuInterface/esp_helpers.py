import pyscf
import numpy as np
from pyscf import gto
from gpu4pyscf.dft import rks, uks
from pyscf.geomopt.geometric_solver import optimize

import ase
from ase.io import read

import numpy as np
from scipy.spatial import distance_matrix

import cupy
from pyscf import gto
from pyscf.data import radii
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import dist_matrix

import time
from ase.data import chemical_symbols
#modified_Bondi = radii.VDW.copy()
#modified_Bondi[1] = 1.1/radii.BOHR      # modified version

# Van der Waals radii (in angstrom) are taken from GAMESS.
R_VDW = 1.0/radii.BOHR * np.asarray([
    -1,
    1.20, # H
    1.20, # He
    1.37, # Li
    1.45, # Be
    1.45, # B
    1.50, # C
    1.50, # N,
    1.40, # O
    1.35, # F,
    1.30, # Ne,
    1.57, # Na,
    1.36, # Mg
    1.24, # Al,
    1.17, # Si,
    1.80, # P,
    1.75, # S,
    1.70]) # Cl

def unit_surface(n):
    '''
    Generate spherical harmonics grid points on unit sphere
    The number of generated points is less than n in general.
    '''
    ux = []
    uy = []
    uz = []

    eps = 1e-10
    nequat = int(np.sqrt(np.pi*n))
    nvert = int(nequat/2)
    for i in range(nvert+1):
        fi = np.pi*i/nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = int(nequat*xy+eps)
        if nhor < 1:
            nhor = 1
        
        fj = 2.0 * np.pi * np.arange(nhor) / nhor
        x = np.cos(fj) * xy
        y = np.sin(fj) * xy

        ux.append(x)
        uy.append(y)
        uz.append(z*np.ones_like(x))
    
    ux = np.concatenate(ux)
    uy = np.concatenate(uy)
    uz = np.concatenate(uz)

    return np.array([ux[:n], uy[:n], uz[:n]]).T

def vdw_surface(mol, scales=[1.0], density=4*radii.BOHR**2, rad=R_VDW):
    '''
    Generate vdw surface of molecules, in Bohr
    '''
    coords = mol.atom_coords(unit='B')
    charges = mol.atom_charges()
    atom_radii = rad[charges]

    surface_points = []
    for scale in scales:
        scaled_radii = atom_radii * scale
        for i, coord in enumerate(coords):
            r = scaled_radii[i]
            # nd is an indicator of density, not exactly the same as number of points
            nd = int(density * 4.0 * np.pi * r**2)
            points = coord + r * unit_surface(nd)
            points = points +  np.random.normal(0,0.1,np.prod(points.shape)).reshape(points.shape)
            dist = distance_matrix(points, coords) + 1e-10
            included = np.all(dist >= scaled_radii, axis=1)
            surface_points.append(points[included])
    points = np.concatenate(surface_points) 
    points = points +  np.random.normal(0,0.5,np.prod(points.shape)).reshape(points.shape)
    return points


def calculate_dipole(moment_array, positions):
    # Calculate the dipole moment by assuming the array values are charges
    # and positions are their respective positions (1D or 3D vectors)
    return np.sum(moment_array[:, None] * positions, axis=0)

def calculate_quadrupole(moment_array, positions):
    # Calculate the quadrupole moment (diagonal components of the quadrupole tensor)
    Q = np.zeros(3)
    for i in range(len(moment_array)):
        r = positions[i]
        q = moment_array[i]
        for j in range(3):
            for k in range(3):
                if j == k:
                    Q[j] += q * (3 * r[j] ** 2 - np.linalg.norm(r) ** 2)
                else:
                    Q[j] += q * (3 * r[j] * r[k])
    return Q

def balance_array(q, sorted_idxs, positions, ref_dipole, ref_quadrupole, N=None):
    """
    Balance the array by finding the best subset of points to use for the ESP calculation.
    If N is not None, the function will balance the array to the Nth point, otherwise it will balance the array to the 10th point.
    The function returns the balanced array and the indices of the balanced array.
    """
    if N is None:
        N = 10
    a = N
    b = len(q) - N

    print(len(sorted_idxs[a:b]))
    incr = len(sorted_idxs[a:b])//100 + 1  # Step size for adjusting the middle section
    best_alignment = -float('inf')  # Start with a very low alignment score
    best_quadrupole_alignment = -float('inf')
    best_s = float("inf")
    best_subset = None
    s = q[sorted_idxs[a:b]].sum()
    while((a == N) & (b == (len(q) - N)) & len(sorted_idxs[a:b]) > 8000):
        # Compute the sum of the middle section 
        current_subset = q[sorted_idxs[a:b]]
        current_positions = positions[sorted_idxs[a:b]]
        s = current_subset.sum()
        
        # Compute the dipole moment of the current subset
        dipole_moment = calculate_dipole(current_subset, current_positions)
        
        # Compute the quadrupole moment of the current subset
        quadrupole_moment = calculate_quadrupole(current_subset, current_positions)
        
        # Compute the alignment between the current dipole and the reference dipole
        # The alignment should be a scalar value between -1 and 1
        dipole_alignment = np.dot(dipole_moment, ref_dipole) / (np.linalg.norm(dipole_moment) * np.linalg.norm(ref_dipole)) if np.linalg.norm(dipole_moment) > 0 else 0
        
        # Compute the alignment between the current quadrupole and the reference quadrupole
        # The alignment should be a scalar value between -1 and 1
        quadrupole_alignment = np.dot(quadrupole_moment, ref_quadrupole) / (np.linalg.norm(quadrupole_moment) * np.linalg.norm(ref_quadrupole)) if np.linalg.norm(quadrupole_moment) > 0 else 0
        quadrupole_alignment = quadrupole_alignment.sum()
        # print(i, a, b, incr, s, dipole_alignment, quadrupole_alignment)

        # Adjust the step size based on the current sum
        # incr = 1
        # print(best_alignment, best_quadrupole_alignment, best_s)
        
        # If both dipole and quadrupole alignments are good and sum is close to zero, update best
        if dipole_alignment > best_alignment-0.05 and quadrupole_alignment > best_quadrupole_alignment-0.05 and abs(s) < best_s:
            best_alignment = dipole_alignment.sum()
            best_quadrupole_alignment = quadrupole_alignment.sum()
            best_subset = (sorted_idxs[:N], sorted_idxs[a:b], sorted_idxs[-N:])
            best_s = abs(s)
        
        # If the sum is positive, shrink the middle range
        if s > 0:
            b -= incr
        else:
            a += incr

    print(best_alignment, best_quadrupole_alignment, best_s)
    # Return the best subset if we found one that meets the condition
    if best_subset:
        a_idx, b_idx, c_idx = best_subset
        print("returning best subset")
        print(q[np.concatenate([a_idx, b_idx, c_idx])].shape)
        print(np.concatenate([a_idx, b_idx, c_idx]).shape)
        return q[np.concatenate([a_idx, b_idx, c_idx])], np.concatenate([a_idx, b_idx, c_idx])
    
    print("failed...", s)
    return q[sorted_idxs[:N]].tolist() + q[sorted_idxs[a:b]].tolist() + q[sorted_idxs[-N:]].tolist(), sorted_idxs[:N].tolist() + \
        sorted_idxs[a:b].tolist() + sorted_idxs[-N:].tolist()

# Example Usage:
# q = loaded["esp"]
# sorted_idxs = np.argsort(q)
# # Call the function
# balanced_array, balanced_indices = balance_array(q, sorted_idxs, loaded["esp_grid"], loaded["dipole"], loaded["quadrupole"])
# # quadrupole
# print("Balanced array:", balanced_array)
# print("Balanced indices:", balanced_indices)
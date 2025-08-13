import ase
import numpy as np
from ase import data
from scipy.spatial.distance import cdist


def cut_vdw(grid, xyz, elements, vdw_scale=1.4):
    """ """
    if isinstance(elements[0], str):
        elements = [ase.data.atomic_numbers[s] for s in elements]
    vdw_radii = [ase.data.vdw_radii[s] for s in elements]
    vdw_radii = np.array(vdw_radii) * vdw_scale
    distances = cdist(grid, xyz)
    mask = distances < vdw_radii
    closest_atom = np.argmin(distances, axis=1)
    closest_atom_type = elements[closest_atom]
    mask = ~mask.any(axis=1)
    return mask, closest_atom_type, closest_atom

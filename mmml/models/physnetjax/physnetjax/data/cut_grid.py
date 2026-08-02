import ase.data
import numpy as np
from scipy.spatial.distance import cdist


def cut_vdw(grid, xyz, elements, vdw_scale=1.4):
    """Mask grid points that fall inside the scaled van der Waals surface.

    Parameters
    ----------
    grid : array_like
        Grid point coordinates, shape (N, 3)
    xyz : array_like
        Atomic coordinates, shape (M, 3)
    elements : array_like
        Atomic numbers or element symbols, shape (M,)
    vdw_scale : float, optional
        Scaling factor for van der Waals radii, by default 1.4

    Returns
    -------
    tuple
        (mask, closest_atom_type, closest_atom)
    """
    # Must be an array, not a list: ``elements[closest_atom]`` below indexes
    # with a numpy array, which raises "only integer scalar arrays can be
    # converted to a scalar index" on a plain list. The symbol path documented
    # above therefore used to fail outright. This is the same bug that was
    # fixed in the DCMNet copy (``dcmnet/data.py``); the two implementations
    # must stay in step.
    if isinstance(elements[0], str):
        elements = np.array([ase.data.atomic_numbers[s] for s in elements])
    else:
        elements = np.asarray(elements)
    vdw_radii = [ase.data.vdw_radii[s] for s in elements]
    vdw_radii = np.array(vdw_radii) * vdw_scale
    distances = cdist(grid, xyz)
    mask = distances < vdw_radii
    closest_atom = np.argmin(distances, axis=1)
    closest_atom_type = elements[closest_atom]
    mask = ~mask.any(axis=1)
    return mask, closest_atom_type, closest_atom

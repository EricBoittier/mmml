from pathlib import Path
from typing import Dict, List, Tuple, Union

import ase
import numpy as np
from ase.units import Bohr, Hartree, kcal
from numpy.typing import NDArray
from tqdm import tqdm

from physnetjax.utils.enums import KEY_TRANSLATION, MolecularData

# Constants
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
MAX_N_ATOMS = 37
MAX_GRID_POINTS = 10000
BOHR_TO_ANGSTROM = 0.529177

from physnetjax.data.data import ATOM_ENERGIES_HARTREE


def pad_array(
    arr: NDArray, max_size: int, axis: int = 0, pad_value: float = 0.0
) -> NDArray:
    """
    Pad a numpy array along specified axis to a maximum size.

    Args:
        arr: Input array to pad
        max_size: Maximum size to pad to
        axis: Axis along which to pad (default: 0)
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        Padded array
    """
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, max_size - arr.shape[axis])
    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)


def pad_coordinates(coords: NDArray, max_atoms: int) -> NDArray:
    """
    Pad coordinates array to maximum number of atoms.

    Args:
        coords: Array of atomic coordinates with shape (n_atoms, 3)
        max_atoms: Maximum number of atoms to pad to

    Returns:
        Padded coordinates array with shape (max_atoms, 3)
    """
    if len(coords.shape) == 3:
        return pad_array(coords, max_atoms, axis=1)
    return pad_array(coords, max_atoms)


def pad_forces(forces: NDArray, max_atoms: int) -> NDArray:
    """
    Pad and convert forces array from Hartree/Bohr to eV/Angstrom.

    Args:
        forces: Array of forces in Hartree/Bohr with shape (n_atoms, 3)
        n_atoms: Number of atoms
        max_atoms: Maximum number of atoms to pad to

    Returns:
        Padded and converted forces array with shape (max_atoms, 3)
    """
    if len(forces.shape) == 3:
        return pad_array(forces, max_atoms, axis=1)
    return pad_array(forces, max_atoms)


def pad_atomic_numbers(atomic_numbers: NDArray, max_atoms: int) -> NDArray:
    """
    Pad atomic numbers array to maximum number of atoms.

    Args:
        atomic_numbers: Array of atomic numbers
        max_atoms: Maximum number of atoms to pad to

    Returns:
        Padded atomic numbers array
    """
    return pad_array(atomic_numbers, max_atoms, axis=1, pad_value=0)

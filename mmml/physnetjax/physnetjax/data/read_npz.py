from pathlib import Path
from typing import Dict, List, Tuple, Union

import ase
import numpy as np
from ase.units import Bohr, Hartree, kcal
from numpy.ma.core import nonzero
from numpy.typing import NDArray
from tqdm import tqdm

from physnetjax.utils.enums import (
    check_keys,
    Z_KEYS,
    R_KEYS,
    F_KEYS,
    D_KEYS,
    E_KEYS,
    COM_KEYS,
    ESP_GRID_KEYS,
    ESP_KEYS,
    Q_KEYS,
)
from physnetjax.utils.enums import KEY_TRANSLATION, MolecularData

# Constants
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
MAX_N_ATOMS = 37
MAX_GRID_POINTS = 10000
BOHR_TO_ANGSTROM = 0.529177

from physnetjax.data.data import ATOM_ENERGIES_HARTREE


def process_npz_file(filepath: Path) -> Tuple[Union[dict, None], int]:
    """
    Process a single NPZ file and extract relevant data.

    Args:
        filepath: Path to NPZ file

    Returns:
        Tuple of (processed data dict or None, number of atoms)
    """
    with np.load(filepath) as load:
        if load is None:
            return None, 0

        data_keys = load.keys()

        zkey = check_keys(Z_KEYS, data_keys)
        if zkey is None:
            return None, 0
        rkey = check_keys(R_KEYS, data_keys)
        if rkey is None:
            return None, 0

        R = load[rkey]
        Z = load[zkey]
        n_atoms = Z.shape[1]

        output = {
            MolecularData.COORDINATES.value: R,
            MolecularData.ATOMIC_NUMBERS.value: Z,
        }

        fkey = check_keys(F_KEYS, data_keys)
        if fkey is not None:
            output[MolecularData.FORCES.value] = load[fkey]

        ekey = check_keys(E_KEYS, data_keys)
        if ekey is not None:
            atom_energies = np.take(ATOM_ENERGIES_HARTREE, Z)
            output[MolecularData.ENERGY.value] = load[ekey] - np.sum(atom_energies)

        dipkey = check_keys(D_KEYS, data_keys)
        if dipkey is not None:
            output[MolecularData.DIPOLE.value] = load[dipkey]

        qkey = check_keys(Q_KEYS, data_keys)
        if qkey is not None:
            output[MolecularData.QUADRUPOLE.value] = load[qkey]

        espkey = check_keys(ESP_KEYS, data_keys)
        if espkey is not None:
            output[MolecularData.ESP.value] = load[espkey]

        espgridkey = check_keys(ESP_GRID_KEYS, data_keys)
        if espgridkey is not None:
            output[MolecularData.ESP_GRID.value] = load[espgridkey]

        comkey = check_keys(COM_KEYS, data_keys)
        if comkey is not None:
            asemol = ase.Atoms(Z, R)
            output[MolecularData.CENTER_OF_MASS.value] = asemol.get_center_of_mass()

        return output, n_atoms

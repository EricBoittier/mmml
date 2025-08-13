from pathlib import Path
from typing import Dict, List, Tuple, Union

import ase
import numpy as np
from ase.units import Bohr, Hartree, kcal
from numpy.typing import NDArray
from tqdm import tqdm

from physnetjax.data.datasets import process_dataset
from physnetjax.utils.enums import KEY_TRANSLATION, MolecularData

# Constants
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
MAX_N_ATOMS = 37
MAX_GRID_POINTS = 10000
BOHR_TO_ANGSTROM = 0.529177

from physnetjax.data.data import ATOM_ENERGIES_HARTREE


def sort_func(filepath: Path) -> int:
    """
    Extract and return sorting number from filepath.

    Args:
        filepath: Path object containing the file path

    Returns:
        Integer for sorting the file paths
    """
    x = str(filepath)
    spl = x.split("/")
    x = spl[-1]
    spl = x.split("xyz")
    spl = spl[1].split("_")
    spl = [_ for _ in spl if len(_) > 0]
    return abs(int(spl[0]))


def get_input_files(data_path: str) -> List[Path]:
    """
    Get list of NPZ files from the given path.

    Args:
        data_path: Path to directory containing NPZ files

    Returns:
        Sorted list of Path objects
    """
    files = list(Path(data_path).glob("*npz"))
    files.sort(key=sort_func)
    return files


if __name__ == "__main__":
    from physnetjax.directories import MAIN_DIR

    files = list(Path(MAIN_DIR / "/data/basepairs").glob("*"))
    print(files)
    process_dataset(files)

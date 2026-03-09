from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class MolecularData(Enum):
    """Types of data that can be present in molecular datasets"""

    COORDINATES = "coordinates"
    ATOMIC_NUMBERS = "atomic_numbers"
    FORCES = "forces"
    ENERGY = "energy"
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"
    ESP = "esp"
    ESP_GRID = "esp_grid"
    CENTER_OF_MASS = "com"
    NUMBER_OF_ATOMS = "N"


KEY_TRANSLATION = {
    MolecularData.COORDINATES: "R",
    MolecularData.ATOMIC_NUMBERS: "Z",
    MolecularData.ENERGY: "E",
    MolecularData.FORCES: "F",
    MolecularData.DIPOLE: "D",
    MolecularData.NUMBER_OF_ATOMS: "N",
}


# rename the dataset keys to match the enum:
Z_KEYS = ["atomic_numbers", "Z"]
R_KEYS = ["coordinates", "positions", "R"]
F_KEYS = ["forces", "F"]
E_KEYS = ["energy", "energies", "E"]
D_KEYS = ["dipole", "d", "dipoles"]
Q_KEYS = ["quadrupole", "q"]
ESP_KEYS = ["esp", "ESP"]
ESP_GRID_KEYS = ["esp_grid", "ESP_GRID"]
COM_KEYS = ["com", "center_of_mass"]
N_KEYS = ["number_of_atoms", "N"]


def check_keys(keys: List, data_keys: List) -> Optional[str]:
    """Check if any of the keys are present in the data_keys."""
    for key in keys:
        if key in data_keys:
            return key
    return None

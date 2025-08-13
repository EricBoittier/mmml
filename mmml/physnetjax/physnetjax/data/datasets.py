import ase
import numpy as np
from ase.units import Bohr, Hartree, kcal
from numpy.typing import NDArray
from tqdm import tqdm

from typing import Dict, List, Tuple, Union
from pathlib import Path

from physnetjax.data.data import ATOM_ENERGIES_HARTREE
from physnetjax.data.full_padding import pad_atomic_numbers, pad_coordinates, pad_forces
from physnetjax.data.read_npz import process_npz_file
from physnetjax.utils.pretty_printer import print_dict_as_table
from physnetjax.utils.enums import (
    check_keys,
    Z_KEYS,
    R_KEYS,
    D_KEYS,
    E_KEYS,
    COM_KEYS,
    ESP_GRID_KEYS,
    ESP_KEYS,
    Q_KEYS,
    F_KEYS,
)
from physnetjax.utils.enums import KEY_TRANSLATION, MolecularData

# Constants
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
MAX_N_ATOMS = 37
MAX_GRID_POINTS = 10000
BOHR_TO_ANGSTROM = 0.529177
NUM_ESP_CLIP = 1000  # ESP clip limit

# Constants
OUTPUT_FILE_PATTERN = "processed_data_batch_{}.npz"


def process_dataset(
    npz_files: List[Path], batch_index: int = 0, MAX_N_ATOMS=37, name=None
) -> Dict[MolecularData, NDArray]:
    """
    Process a batch of NPZ files and combine their data.

    Args:
        npz_files: List of NPZ files to process.
        batch_index: Index of the batch being processed.

    Returns:
        Dictionary containing combined and processed data, keyed by MolecularData enum.
    """

    def initialize_raw_data() -> Dict[MolecularData, List]:
        """Initialize a raw data dictionary keyed by MolecularData."""
        return {data_type: [] for data_type in MolecularData}

    def extract_molecule_id(file_path: Path) -> str:
        """Extract the molecule ID from a given file path."""
        return str(file_path).split("/")[-2]

    def read_and_filter(filepath: Path) -> Union[Tuple[Dict, int], None]:
        """Read and validate NPZ file data based on atomic count."""
        result, n_atoms = process_npz_file(filepath)
        if result is not None and n_atoms < MAX_N_ATOMS:
            return result, n_atoms
        print(f"Skipping file: {filepath}")
        return None

    raw_data = initialize_raw_data()
    molecule_ids = []

    # Load and collect data
    for filepath in tqdm(npz_files, desc="Processing Files"):
        filtered_data = read_and_filter(filepath)
        if not filtered_data:
            continue
        result, num_atoms = filtered_data
        if num_atoms > MAX_N_ATOMS:
            raise ValueError(f"Number of atoms exceeds maximum limit: {num_atoms}")

        molecule_ids.append(extract_molecule_id(filepath))  # Extract molecule ID
        for data_type in MolecularData:
            if data_type.value in result:
                raw_data[data_type].append(result[data_type.value])

    def pad_data_by_key(data_key: MolecularData, pad_function, *pad_args):
        """Pad raw data of a single type to ensure uniform sizes."""
        out = []
        for item in raw_data[data_key]:
            p = pad_function(item, *pad_args)
            out.append(p)

        # stack the arrays to create a single array
        output_array = np.vstack(out)

        return output_array

    processed_data = {
        MolecularData.ATOMIC_NUMBERS: pad_data_by_key(
            MolecularData.ATOMIC_NUMBERS, pad_atomic_numbers, MAX_N_ATOMS
        ),
        MolecularData.COORDINATES: pad_data_by_key(
            MolecularData.COORDINATES, pad_coordinates, MAX_N_ATOMS
        ),
    }

    # Pad conditional data (forces and energy)
    if raw_data[MolecularData.FORCES]:
        processed_data[MolecularData.FORCES] = pad_data_by_key(
            MolecularData.FORCES, pad_forces, MAX_N_ATOMS
        )
    if raw_data[MolecularData.ENERGY]:
        processed_data[MolecularData.ENERGY] = np.concatenate(
            raw_data[MolecularData.ENERGY]
        )

    # Add unpadded data types
    for data_type in [MolecularData.DIPOLE, MolecularData.CENTER_OF_MASS]:
        if raw_data[data_type]:
            processed_data[data_type] = np.concatenate(raw_data[data_type])

    # Save processed data
    save_dict = {key.value: processed_data[key] for key in processed_data}
    # save_dict["molecule_ids"] = np.array(molecule_ids)
    if name is not None and isinstance(name, str):
        output_path = name + "-" + OUTPUT_FILE_PATTERN.format(batch_index)
        np.savez(output_path, **save_dict)

    return processed_data


def process_in_memory(
    data: List[Dict] | Dict, max_atoms=None, openqdc=False
) -> Dict[MolecularData, NDArray]:
    """
    Process a list of dictionaries containing data.
    """
    if max_atoms is not None:
        MAX_N_ATOMS = max_atoms
    if max_atoms is None:
        MAX_N_ATOMS = 0
    output = {}
    data_keys = list(data[0].keys()) if isinstance(data, list) else list(data.keys())

    # atomic numbers
    _ = check_keys(Z_KEYS, data_keys)
    if _ is not None:
        Z = [np.array([z[_]]) for z in data]
        output[MolecularData.ATOMIC_NUMBERS] = np.array(
            [pad_atomic_numbers(Z[i], MAX_N_ATOMS) for i in range(len(Z))]
        ).squeeze()
        output[MolecularData.NUMBER_OF_ATOMS] = np.array([[_.shape[1]] for _ in Z])
    # coordinates
    _ = check_keys(R_KEYS, data_keys)
    if _ is not None:
        # print(data[0][_])
        output[MolecularData.COORDINATES] = np.array(
            [pad_coordinates(d[_], MAX_N_ATOMS) for d in data]
        )
    # print("output[MolecularData.COORDINATES]", output[MolecularData.COORDINATES].shape)
    _ = check_keys(F_KEYS, data_keys)
    if _ is not None:
        # print(data[0][_])
        output[MolecularData.FORCES] = np.array(
            [
                pad_forces(
                    d[_].squeeze(),
                    MAX_N_ATOMS,
                )
                for d in data
            ]
        )
    # print("output[MolecularData.FORCES].shape", output[MolecularData.FORCES].shape)

    _ = check_keys(E_KEYS, data_keys)
    if _ is not None:
        # do the conversion from hartree to eV...
        # t0d0 check if this is correct, subject to changes
        # in the openqdc library...
        if openqdc:
            output[MolecularData.ENERGY] = np.array(
                [d[_] - float(d["e0"].sum() * 0.0367492929) for d in data]
            )
        else:
            output[MolecularData.ENERGY] = np.array([d[_] for d in data])

    _ = check_keys(D_KEYS, data_keys)
    if _ is not None:
        output[MolecularData.DIPOLE] = np.array([d[_] for d in data])

    _ = check_keys(Q_KEYS, data_keys)
    if _ is not None:
        output[MolecularData.QUADRUPOLE] = np.array([d[_] for d in data])

    _ = check_keys(ESP_KEYS, data_keys)
    if _ is not None:
        output[MolecularData.ESP] = np.array([d[_] for d in data])

    _ = check_keys(ESP_GRID_KEYS, data_keys)
    if _ is not None:
        output[MolecularData.ESP_GRID] = np.array([d[_] for d in data])

    _ = check_keys(COM_KEYS, data_keys)
    if _ is not None:
        output[MolecularData.CENTER_OF_MASS] = np.array

    keys = list(output.keys())
    for k_old in keys:
        k_old = MolecularData[str(k_old).split(".")[-1]]
        k_new = KEY_TRANSLATION[k_old]
        output[k_new] = output.pop(k_old)

    return output



def process_dataset_key(
    data_key, datasets, shape, natoms, not_failed, reshape_dims=None
):
    """Helper to process a key across datasets and apply reshaping."""
    data_array = np.concatenate([ds[data_key] for ds in datasets])
    if reshape_dims:
        data_array = data_array.reshape(shape[0], *reshape_dims)
    data_array = data_array[not_failed]  # Filter failed rows
    return data_array.squeeze()


def clip_or_default_data(datasets, data_key, not_failed, shape, clip=False):
    """Helper function to process ESP data with optional clipping."""
    if clip:
        return np.concatenate(
            [dataset[data_key][:NUM_ESP_CLIP] for dataset in datasets]
        )[not_failed].reshape(shape[0], NUM_ESP_CLIP)
    else:
        return np.concatenate([dataset[data_key] for dataset in datasets])[not_failed]


def prepare_multiple_datasets(
    key,
    train_size=0,
    valid_size=0,
    filename=None,
    verbose=False,
    esp_mask=False,
    clip_esp=False,
    natoms=60,
    subtract_atom_energies=False,
    subtract_mean=False,
):
    """
    Prepare multiple datasets for training and validation.

    Args:
        key: Random key for dataset shuffling.
        train_size (int): Number of training samples.
        valid_size (int): Number of validation samples.
        filename (list): List of filenames to load datasets from.
    Returns:
        tuple: A tuple containing the prepared data and keys.
    """
    # Load datasets
    datasets = [np.load(f) for f in filename]

    if verbose:
        for i, dataset in enumerate(datasets):
            data_shape = {k: v.shape for k, v in dataset.items()}
            print_dict_as_table(data_shape, title=Path(filename[i]).name, plot=True)

    # Validate datasets and initialize variables
    data_ids = (
        np.concatenate([ds["id"] for ds in datasets]) if "id" in datasets[0] else None
    )
    shape = np.concatenate([ds["R"] for ds in datasets]).reshape(-1, natoms, 3).shape
    not_failed = np.arange(shape[0])  # Default: no failed rows filtered

    # Collect processed data
    data = []
    keys = []

    # Handle individual dataset keys
    if "id" in datasets[0]:
        data.append(data_ids[not_failed])
        keys.append("id")

    if "R" in datasets[0]:
        positions = process_dataset_key(
            "R", datasets, shape, natoms, not_failed, reshape_dims=(natoms, 3)
        )
        data.append(positions)
        keys.append("R")

    if "Z" in datasets[0]:
        atomic_numbers = process_dataset_key(
            "Z", datasets, shape, natoms, not_failed, reshape_dims=(natoms,)
        )
        data.append(atomic_numbers)
        keys.append("Z")

    if "F" in datasets[0]:
        forces = process_dataset_key(
            "F", datasets, shape, natoms, not_failed, reshape_dims=(natoms, 3)
        )
        data.append(forces)
        keys.append("F")

    if "E" in datasets[0]:
        energies = np.concatenate([ds["E"] for ds in datasets])[not_failed]
        if subtract_atom_energies:
            tmp_ae = ATOM_ENERGIES_HARTREE[atomic_numbers].sum(axis=1) * 27.2114
            energies -= tmp_ae
        if subtract_mean:
            energies -= np.mean(energies)
        data.append(energies.reshape(-1, 1))
        keys.append("E")

    if "esp" in datasets[0]:
        esp_data = clip_or_default_data(datasets, "esp", not_failed, shape, clip_esp)
        data.append(esp_data)
        keys.append("esp")

    # Additional processing for other keys can follow the same pattern

    return data, keys

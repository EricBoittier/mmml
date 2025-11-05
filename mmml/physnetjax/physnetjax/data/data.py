from pathlib import Path

import ase.data
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mmml.data.atomic_references import (
    DEFAULT_CHARGE_STATE,
    DEFAULT_REFERENCE_LEVEL,
    get_atomic_reference_array,
)
from mmml.physnetjax.physnetjax.utils.pretty_printer import print_dict_as_table

# Atomic energies in Hartree sourced from reference table
ATOM_ENERGIES_HARTREE = get_atomic_reference_array(
    level=DEFAULT_REFERENCE_LEVEL,
    charge_state=DEFAULT_CHARGE_STATE,
    unit="hartree",
)


def prepare_multiple_datasets(
    key,
    train_size=0,
    valid_size=0,
    filename=None,
    clean=False,
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
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.
        filename (list): List of filenames to load datasets from.

    Returns:
        tuple: A tuple containing the prepared data and keys.
    """
    # Load the datasets
    datasets = [np.load(f, allow_pickle=True) for f in filename]
    # datasets_keys() = datasets.keys()
    if verbose:
        for i, dataset in enumerate(datasets):
            data_shape = {}
            for k, v in dataset.items():
                data_shape[k] = v.shape
            print_dict_as_table(data_shape, title=Path(filename[i]).name, plot=True)

    if "id" in datasets[0].keys():
        dataid = np.concatenate([dataset["id"] for dataset in datasets])
    not_failed = None  # np.array(range().reshape(-1,6,3).shape[0]))
    data = []
    keys = []

    if clean:
        failed = pd.read_csv(
            "/pchem-data/meuwly/boittier/home/jaxeq/data/qm9-fails.csv"
        )
        failed = list(failed["0"])
        not_failed = [i for i in range(len(dataid)) if str(dataid[i]) not in failed]
        n_failed = len(dataid) - len(not_failed)
        print("n_failed:", n_failed)
        num_train = int(max([0, num_train - n_failed]))
        if num_train == 0:
            num_valid = int(max([0, num_valid - n_failed]))
        print(num_train, num_valid)

    shape = (
        np.concatenate([dataset["R"] for dataset in datasets])
        .reshape(-1, natoms, 3)
        .shape
    )
    not_failed = np.array(range(shape[0]))
    # print("shape", shape, "not failed", not_failed)

    if "id" in datasets[0].keys():
        dataid = dataid[not_failed]
        data.append(dataid)
        keys.append("id")
    if "R" in datasets[0].keys():
        dataR = np.concatenate([dataset["R"] for dataset in datasets])
        print("dataR", dataR.shape)
        dataR = dataR.reshape(shape[0], natoms, 3)[not_failed]
        data.append(dataR.squeeze())
        keys.append("R")
    if "Z" in datasets[0].keys():
        dataZ = np.concatenate([dataset["Z"] for dataset in datasets]).reshape(
            shape[0], natoms
        )[not_failed]
        data.append(dataZ.squeeze())
        keys.append("Z")
    if "F" in datasets[0].keys():
        dataF = np.concatenate([dataset["F"] for dataset in datasets]).reshape(
            shape[0], natoms, 3
        )[not_failed]
        data.append(dataF.squeeze())
        keys.append("F")
    if "E" in datasets[0].keys():
        dataE = np.concatenate([dataset["E"] for dataset in datasets])[not_failed]
        print("dataE", dataE.flatten()[:10])
        if subtract_atom_energies:
            tmp_ae = ATOM_ENERGIES_HARTREE[dataZ].sum(axis=1) * 27.2114
            dataE = dataE - tmp_ae
        if subtract_mean:
            dataE = dataE - np.mean(dataE)
        print("dataE", dataE.flatten()[:10])
        data.append(dataE.reshape(shape[0], 1))
        keys.append("E")
    if "N" in datasets[0].keys():
        dataN = np.concatenate([dataset["N"] for dataset in datasets])[not_failed]
        data.append(dataN.reshape(shape[0], 1))
        keys.append("N")
    if "mono" in datasets[0].keys():
        dataMono = np.concatenate([dataset["mono"] for dataset in datasets])[not_failed]
        data.append(dataMono.reshape(shape[0], natoms))
        keys.append("mono")
    if "esp" in datasets[0].keys():
        if clip_esp:
            dataEsp = np.concatenate([dataset["esp"][:1000] for dataset in datasets])[
                not_failed
            ].reshape(shape[0], 1000)
        else:
            dataEsp = np.concatenate([dataset["esp"] for dataset in datasets])[
                not_failed
            ]
        data.append(dataEsp)
        keys.append("esp")
    if "vdw_surface" in datasets[0].keys():
        dataVDW = np.concatenate([dataset["vdw_surface"] for dataset in datasets])[
            not_failed
        ]
        data.append(dataVDW)
        keys.append("vdw_surface")
    if "n_grid" in datasets[0].keys():
        dataNgrid = np.concatenate([dataset["n_grid"] for dataset in datasets])[
            not_failed
        ]
        data.append(dataNgrid)
        keys.append("n_grid")
    if "D" in datasets[0].keys():
        dataD = np.concatenate([dataset["D"] for dataset in datasets])[not_failed]
        print("D", dataD.shape)
        try:
            data.append(dataD.reshape(shape[0], 1))
        except:
            try:
                data.append(dataD.reshape(shape[0], natoms, 3))
            except Exception:
                data.append(dataD.reshape(shape[0], 3))
        keys.append("D")
    if "dipole" in datasets[0].keys():
        dipole = np.concatenate([dataset["dipole"] for dataset in datasets]).reshape(
            shape[0], 3
        )
        # [not_failed]
        print("dipole.shape", dipole.shape)
        data.append(dipole)
        keys.append("dipole")
    if "Dxyz" in datasets[0].keys():
        dataDxyz = np.concatenate([dataset["Dxyz"] for dataset in datasets])[not_failed]
        data.append(dataDxyz)
        keys.append("Dxyz")
    if "com" in datasets[0].keys():
        dataCOM = np.concatenate([dataset["com"] for dataset in datasets])[not_failed]
        data.append(dataCOM)
        keys.append("com")
    if "polar" in datasets[0].keys():
        polar = np.concatenate([dataset["polar"] for dataset in datasets]).reshape(
            shape[0], 3, 3
        )
        print("polar", polar.shape)
        polar = polar  # [not_failed,:,:]
        # print(polar)
        data.append(polar)
        keys.append("polar")
    if "esp_grid" in datasets[0].keys():
        esp_grid = np.concatenate(
            [dataset["esp_grid"][:1000] for dataset in datasets]
        ).reshape(-1, 1000, 3)
        esp_grid = esp_grid  # [not_failed,:,:]
        # print(polar)
        data.append(esp_grid)
        keys.append("esp_grid")
    if "quadrupole" in datasets[0].keys():
        quadrupole = np.concatenate(
            [dataset["quadrupole"] for dataset in datasets]
        ).reshape(-1, 3, 3)
        quadrupole = quadrupole  # [not_failed,:,:]
        # print(polar)
        data.append(quadrupole)
        keys.append("quadrupole")

    for k in datasets[0].keys():
        if k not in keys:
            print(
                k,
                len(datasets[0][k].shape),
                datasets[0][k].shape,
                datasets[0][k].shape[0],
            )
            _ = np.concatenate([dataset[k] for dataset in datasets])
            # [not_failed]
            # if
            print(k, _.shape)
            data.append(_)
            keys.append(k)

    if esp_mask:
        if verbose:
            print("creating_mask")
        dataESPmask = np.array(
            [cut_vdw(dataVDW[i], dataR[i], dataZ[i])[0] for i in range(len(dataZ))]
        )
        data.append(dataESPmask)
        keys.append("espMask")

    num_points = len(dataR.squeeze())
    assert_dataset_size(num_points, train_size, valid_size)

    return (
        data,
        keys,
        train_size,
        valid_size,
    )


def prepare_datasets(
    key,
    train_size=0,
    valid_size=0,
    files=None,
    clean=False,
    esp_mask=False,
    clip_esp=False,
    natoms=60,
    verbose=False,
    subtract_atom_energies=False,
    subtract_mean=False,
):
    """
    Prepare datasets for training and validation.

    Args:
        key: Random key for dataset shuffling.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.
        filename (str or list): Filename(s) to load datasets from.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    # Load the datasets
    if isinstance(files, str):
        filename = [files]
    elif isinstance(files, list):
        filename = files
    elif files is None:
        # exit and Warning
        raise ValueError("No filename(s) provided")

    data, keys, num_train, num_valid = prepare_multiple_datasets(
        key,
        train_size=train_size,
        valid_size=valid_size,
        filename=filename,
        clean=clean,
        natoms=natoms,
        clip_esp=clip_esp,
        esp_mask=esp_mask,
        verbose=verbose,
        # dataset_keys
        subtract_atom_energies=subtract_atom_energies,
        subtract_mean=subtract_mean,
    )

    train_choice, valid_choice = get_choices(
        key, len(data[0]), int(num_train), int(num_valid)
    )

    train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)

    return train_data, valid_data


def assert_dataset_size(num_data, num_train, num_valid):
    """
    Assert that the dataset contains enough entries for training and validation.

    Args:
        num_data (int): Total number of data points.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.

    Raises:
        AssertionError: If the dataset doesn't contain enough entries.
    """
    assert num_train >= 0
    assert num_valid >= 0
    # Make sure that the dataset contains enough entries.
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise AssertionError(
            f"datasets only contains {num_data} points, "
            f"requested num_train={num_train}, num_valid={num_valid}"
        )


def get_choices(key, num_data, num_train, num_valid):
    """
    Randomly draw train and validation sets from the dataset.

    Args:
        key: Random key for shuffling.
        num_data (int): Total number of data points.
        num_train (int): Number of training samples.
        num_valid (int): Number of validation samples.

    Returns:
        tuple: A tuple containing train_choice and valid_choice arrays.
    """
    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_data,), replace=False)
    )
    train_choice = choice[:num_train]
    valid_choice = choice[num_train : num_train + num_valid]
    return train_choice, valid_choice


def make_dicts(data, keys, train_choice, valid_choice):
    """
    Create dictionaries for train and validation data.

    Args:
        data (list): List of data arrays.
        keys (list): List of keys for the data arrays.
        train_choice (array): Indices for training data.
        valid_choice (array): Indices for validation data.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    train_data, valid_data = dict(), dict()

    for i, k in enumerate(keys):
        # print(i, k, len(data[i]), data[i].shape)
        train_data[k] = data[i][train_choice]
        valid_data[k] = data[i][valid_choice]

    return train_data, valid_data


def print_shapes(dict, name="Data Shapes"):
    """
    Print the shapes of train and validation data.

    Args:
        dict (dict): Dictionary containing training data.

    Returns:
        tuple: A tuple containing train_data and valid_data dictionaries.
    """
    shapes_dict = {}
    for k, v in dict.items():
        shapes_dict[k] = v.shape

    print_dict_as_table(shapes_dict, title=name, plot=True)

import ase.data
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def cut_vdw(grid, xyz, elements, vdw_scale=1.4):
    """
    Create mask to exclude grid points inside van der Waals radii.
    
    Creates a boolean mask that excludes grid points that are too close
    to atoms based on scaled van der Waals radii. This is useful for
    ESP fitting to avoid sampling inside the molecular core.
    
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
        (mask, closest_atom_type, closest_atom) where:
        - mask: Boolean array indicating valid grid points
        - closest_atom_type: Atomic numbers of closest atoms
        - closest_atom: Indices of closest atoms
    """
    if type(elements[0]) == str:
        elements = [ase.data.atomic_numbers[s] for s in elements]
    vdw_radii = [ase.data.vdw_radii[s] for s in elements]
    vdw_radii = np.array(vdw_radii) * vdw_scale
    distances = cdist(grid, xyz)
    mask = distances < vdw_radii
    closest_atom = np.argmin(distances, axis=1)
    closest_atom_type = elements[closest_atom]
    mask = ~mask.any(axis=1)
    return mask, closest_atom_type, closest_atom


def prepare_multiple_datasets(
    key,
    num_train,
    num_valid,
    filename=["esp2000.npz"],
    clean=False,
    verbose=False,
    esp_mask=False,
    clip_esp=False,
    natoms=60,
):
    """
    Prepare multiple datasets for training and validation.

    Loads and combines multiple NPZ files containing ESP data, handles
    data cleaning, and creates training/validation splits. Supports
    various data formats and optional preprocessing steps.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for dataset shuffling
    num_train : int
        Number of training samples
    num_valid : int
        Number of validation samples
    filename : list, optional
        List of filenames to load datasets from, by default ["esp2000.npz"]
    clean : bool, optional
        Whether to filter failed calculations, by default False
    verbose : bool, optional
        Whether to print dataset information, by default False
    esp_mask : bool, optional
        Whether to create ESP masks using VDW radii, by default False
    clip_esp : bool, optional
        Whether to clip ESP to first 1000 points, by default False
    natoms : int, optional
        Maximum number of atoms per system, by default 60
        
    Returns
    -------
    tuple
        (data, keys, num_train, num_valid) where:
        - data: List of data arrays
        - keys: List of corresponding keys
        - num_train: Adjusted number of training samples
        - num_valid: Adjusted number of validation samples
        
    Notes
    -----
    The function automatically handles datasets with different keys and
    ensures all arrays have compatible shapes. Failed calculations are
    filtered if clean=True.
    """
    # Load the datasets
    datasets = [np.load(f, mmap_mode='r') for f in filename]
    # datasets_keys() = datasets.keys()
    if verbose:
        for dataset in datasets:
            for k, v in dataset.items():
                print(k, v.shape)

    if "id" in datasets[0].keys():
        dataid = np.concatenate([dataset["id"] for dataset in datasets])
    # By default, select all entries. Using slice(None) avoids accidental
    # dimension expansion that occurs with indexing by None.
    not_failed = slice(None)
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
    print("shape", shape)

    if "id" in datasets[0].keys():
        dataid = dataid[not_failed]
        data.append(dataid)
        keys.append("id")
    if "R" in datasets[0].keys():
        dataR = np.concatenate([dataset["R"] for dataset in datasets]).reshape(
            -1, natoms, 3
        )[not_failed]
        data.append(dataR.squeeze())
        keys.append("R")
    if "Z" in datasets[0].keys():
        dataZ = np.concatenate([dataset["Z"] for dataset in datasets]).reshape(
            -1, natoms
        )[not_failed]
        data.append(dataZ.squeeze())
        keys.append("Z")
    if "F" in datasets[0].keys():
        dataF = np.concatenate([dataset["F"] for dataset in datasets]).reshape(
            -1, natoms, 3
        )[not_failed]
        data.append(dataF.squeeze())
        keys.append("F")
    if "E" in datasets[0].keys():
        dataE = np.concatenate([dataset["E"] for dataset in datasets])[not_failed]
        data.append(dataE.reshape(-1, 1))
        keys.append("E")
    if "N" in datasets[0].keys():
        dataN = np.concatenate([dataset["N"] for dataset in datasets])[not_failed]
        data.append(dataN.reshape(-1, 1))
        keys.append("N")
    if "mono" in datasets[0].keys():
        dataMono = np.concatenate([dataset["mono"] for dataset in datasets])[not_failed]
        data.append(dataMono.reshape(-1, natoms))
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
            data.append(dataD.reshape(-1, 1))
        except:
            data.append(dataD.reshape(shape[0], natoms, 3))
        keys.append("D")
    if "dipole" in datasets[0].keys():
        dipole = np.concatenate([dataset["dipole"] for dataset in datasets]).reshape(
            -1, 3
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
            -1, 3, 3
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

    print("R", dataR.shape)

    assert_dataset_size(dataR.squeeze(), num_train, num_valid)

    return data, keys, num_train, num_valid


def prepare_datasets(
    key,
    num_train,
    num_valid,
    filename,
    clean=False,
    esp_mask=False,
    clip_esp=False,
    natoms=60,
):
    """
    Prepare datasets for training and validation.

    Wrapper function that calls prepare_multiple_datasets and then
    creates train/validation splits and dictionaries.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for dataset shuffling
    num_train : int
        Number of training samples
    num_valid : int
        Number of validation samples
    filename : str or list
        Filename(s) to load datasets from
    clean : bool, optional
        Whether to filter failed calculations, by default False
    esp_mask : bool, optional
        Whether to create ESP masks, by default False
    clip_esp : bool, optional
        Whether to clip ESP to first 1000 points, by default False
    natoms : int, optional
        Maximum number of atoms per system, by default 60

    Returns
    -------
    tuple
        A tuple containing train_data and valid_data dictionaries
    """
    # Load the datasets
    if isinstance(filename, str):
        filename = [filename]

    data, keys, num_train, num_valid = prepare_multiple_datasets(
        key,
        num_train,
        num_valid,
        filename,
        clean=clean,
        natoms=natoms,
        clip_esp=clip_esp,
        esp_mask=esp_mask,
        # dataset_keys
    )
    print(data[0].shape)
    print(keys)
    print(len(data[0]))
    train_choice, valid_choice = get_choices(
        key, len(data[0]), int(num_train), int(num_valid)
    )

    train_data, valid_data = make_dicts(data, keys, train_choice, valid_choice)

    return train_data, valid_data


def assert_dataset_size(dataR, num_train, num_valid):
    """
    Assert that the dataset contains enough entries for training and validation.

    Parameters
    ----------
    dataR : array_like
        The dataset to check
    num_train : int
        Number of training samples
    num_valid : int
        Number of validation samples

    Raises
    ------
    RuntimeError
        If the dataset doesn't contain enough entries
    """
    assert num_train >= 0
    assert num_valid >= 0
    # Make sure that the dataset contains enough entries.
    num_data = len(dataR)
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f"datasets only contains {num_data} points, "
            f"requested num_train={num_train}, num_valid={num_valid}"
        )


def get_choices(key, num_data, num_train, num_valid):
    """
    Randomly draw train and validation sets from the dataset.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for shuffling
    num_data : int
        Total number of data points
    num_train : int
        Number of training samples
    num_valid : int
        Number of validation samples

    Returns
    -------
    tuple
        A tuple containing train_choice and valid_choice arrays
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

    Parameters
    ----------
    data : list
        List of data arrays
    keys : list
        List of keys for the data arrays
    train_choice : array_like
        Indices for training data
    valid_choice : array_like
        Indices for validation data

    Returns
    -------
    tuple
        A tuple containing train_data and valid_data dictionaries
    """
    train_data, valid_data = dict(), dict()

    for i, k in enumerate(keys):
        print(i, k, len(data[i]), data[i].shape)
        train_data[k] = data[i][train_choice]
        valid_data[k] = data[i][valid_choice]

    return train_data, valid_data


def print_shapes(train_data, valid_data):
    """
    Print the shapes of train and validation data.

    Parameters
    ----------
    train_data : dict
        Dictionary containing training data
    valid_data : dict
        Dictionary containing validation data

    Returns
    -------
    tuple
        A tuple containing train_data and valid_data dictionaries
    """
    print("...")
    print("...")
    for k, v in train_data.items():
        print(k, v.shape)
    print("...")
    for k, v in valid_data.items():
        print(k, v.shape)

    return train_data, valid_data


def prepare_batches(
    key, data, batch_size, 
    include_id=False, data_keys=None, num_atoms=60,
    dst_idx=None, src_idx=None
) -> list:
    """
    Prepare batches for training.

    Creates batches from the dataset for training. Handles message passing
    indices and batch segmentation for equivariant operations.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for shuffling
    data : dict
        Dictionary containing the dataset
    batch_size : int
        Size of each batch
    include_id : bool, optional
        Whether to include ID in the output, by default False
    data_keys : list, optional
        List of keys to include in the output, by default None
    num_atoms : int, optional
        Number of atoms per system, by default 60
    dst_idx : array_like, optional
        Destination indices for message passing, by default None
    src_idx : array_like, optional
        Source indices for message passing, by default None

    Returns
    -------
    list
        A list of dictionaries, each representing a batch
    """
    # Determine the number of training steps per epoch.

    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size

    # Draw random permutations for fetching batches from the train data.
    perms = jax.random.permutation(key, data_size)
    perms = perms[
        : steps_per_epoch * batch_size
    ]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.

    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    if dst_idx is None and src_idx is None:
        # print("sparse_pairwise_indices")
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
        dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
        src_idx = (src_idx + offsets[:, None]).reshape(-1)
    elif dst_idx is not None and src_idx is not None:
        dst_idx = data["dst_idx"]
        src_idx = data["src_idx"]

    # print(len(dst_idx))

    output = []
    if data_keys is None:
        data_keys = [
            "R",
            "Z",
            "N",
            "mono",
            "esp",
            "vdw_surface",
            "n_grid",
            "D",
            "Dxyz",
            "espMask",
            "com",
        ]
    # if include_id:
    #     data_keys.append("id")

    for perm in perms:
        # print(perm)
        dict_ = dict()
        for k, v in data.items():
            # print(k, v)
            if k in data_keys:
                if k == "R":
                    dict_[k] = v[perm].reshape(-1, 3)
                elif k == "Z":
                    dict_[k] = v[perm].reshape(-1)
                elif k == "mono":
                    dict_[k] = v[perm].reshape(-1)
                else:
                    dict_[k] = v[perm]

        if len(dst_idx.shape) > 1:
            dict_["dst_idx"] = dst_idx[perm[0]]
            # print(dict_["dst_idx"])
        else:
            dict_["dst_idx"] = dst_idx
        if len(dst_idx.shape) > 1:
            dict_["src_idx"] = src_idx[perm[0]]
        else:
            dict_["src_idx"] = src_idx
        dict_["batch_segments"] = batch_segments
        output.append(dict_)

    return output

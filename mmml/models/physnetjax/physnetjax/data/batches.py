from typing import Dict, Iterable, List, Optional

import e3x.ops
import jax
import jax.numpy as jnp
import numpy as np
from ase.units import Bohr, Hartree
from mmml.utils.rotations import rotate_batched_vectors, sample_random_rotations

# Constants
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
MAX_N_ATOMS = 37
MAX_GRID_POINTS = 10000
BOHR_TO_ANGSTROM = 0.529177


def determine_max_nb_length(num_atoms: int | Iterable, batch_size: int) -> int:
    """Determine the maximum number of neighbors for a given number of atoms."""
    if isinstance(num_atoms, int):
        return num_atoms * (num_atoms - 1) * batch_size
    length = 0
    # sort descending
    num_atoms.sort()
    for n in num_atoms:
        length += n * (n - 1)
    return int(length)


def prepare_batches_one(
    key,
    data,
    batch_size,
    include_id=False,
    data_keys=None,
    num_atoms=60,
    dst_idx=None,
    src_idx=None,
    rot_augment: bool = False,
    rot_perturbation: float = 1.0,
) -> list:
    """
    Prepare batches for training.

    Args:
        key: Random key for shuffling.
        data (dict): Dictionary containing the dataset.
        batch_size (int): Size of each batch.
        include_id (bool): Whether to include ID in the output.
        data_keys (list): List of keys to include in the output.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """
    # Determine the number of training steps per epoch.
    # print(batch_size)
    data_size = len(data["R"])
    # print("data_size", data_size)
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
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = dst_idx + offsets[:, None]  # .reshape(-1)  # * good_indices
    src_idx = src_idx + offsets[:, None]  # .reshape(-1)  # * good_indices

    output = []
    for perm in perms:
        # print(perm)
        dict_ = dict()
        for k, v in data.items():
            if k in data_keys:
                # print(k, v.shape)
                if k == "R":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms, 3)
                    # print(dict_[k].
                elif k == "F":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms, 3)
                elif k == "E":
                    dict_[k] = v[perm].reshape(batch_size, 1)
                elif k == "Z":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms)
                elif k == "mono":
                    dict_[k] = v[perm].reshape(batch_size * num_atoms)
                else:
                    dict_[k] = v[perm]

        if rot_augment:
            rot_key = jax.random.fold_in(key, int(perm[0]))
            rotations = sample_random_rotations(
                rot_key, batch_size, perturbation=rot_perturbation
            )
            if "R" in dict_:
                dict_["R"] = rotate_batched_vectors(
                    dict_["R"].reshape(batch_size, num_atoms, 3), rotations
                ).reshape(batch_size * num_atoms, 3)
            if "F" in dict_:
                dict_["F"] = rotate_batched_vectors(
                    dict_["F"].reshape(batch_size, num_atoms, 3), rotations
                ).reshape(batch_size * num_atoms, 3)
            if "D" in dict_ and dict_["D"].ndim == 2 and dict_["D"].shape[-1] == 3:
                dict_["D"] = rotate_batched_vectors(dict_["D"], rotations)

        good_indices = []
        for i, nat in enumerate(dict_["N"]):
            # print("nat", nat)
            cond = (dst_idx[i] < (nat + i * num_atoms)) * (
                src_idx[i] < (nat + i * num_atoms)
            )
            good_indices.append(
                jnp.where(
                    cond,
                    1,
                    0,
                )
            )
        good_indices = jnp.concatenate(good_indices).flatten()
        dict_["dst_idx"] = dst_idx.flatten()
        dict_["src_idx"] = src_idx.flatten()
        dict_["batch_mask"] = good_indices  # .reshape(-1)
        dict_["batch_segments"] = batch_segments.reshape(-1)
        dict_["atom_mask"] = jnp.where(dict_["Z"] > 0, 1, 0).reshape(-1)
        output.append(dict_)
    # print(output)
    return output


def prepare_batches_jit(
    key,
    data: Dict[str, jnp.ndarray],
    batch_size: int,
    data_keys: Optional[List[str]] = None,
    num_atoms: int = 60,
    dst_idx: Optional[jnp.ndarray] = None,
    src_idx: Optional[jnp.ndarray] = None,
    include_id: bool = False,
    debug_mode: bool = False,
    rot_augment: bool = False,
    rot_perturbation: float = 1.0,
) -> List[Dict[str, jnp.ndarray]]:
    """
    Efficiently prepare batches for training.

    Args:
        key: JAX random key for shuffling.
        data (dict): Dictionary containing the dataset.
            Expected keys: 'R', 'N', 'Z', 'F', 'E', and optionally others.
        batch_size (int): Size of each batch.
        data_keys (list, optional): List of keys to include in the output.
            If None, all keys in `data` are included.
        num_atoms (int, optional): Number of atoms per example. Default is 60.
        dst_idx (jax.numpy.ndarray, optional): Precomputed destination indices for atom pairs.
        src_idx (jax.numpy.ndarray, optional): Precomputed source indices for atom pairs.
        include_id (bool, optional): Whether to include 'id' key if present in data.
        debug_mode (bool, optional): If True, run assertions and extra checks.

    Returns:
        list: A list of dictionaries, each representing a batch.
    """

    # -------------------------------------------------------------------------
    # Validation and Setup
    # -------------------------------------------------------------------------

    # Check for mandatory keys
    required_keys = ["R", "N", "Z"]
    for req_key in required_keys:
        if req_key not in data:
            raise ValueError(f"Data dictionary must contain '{req_key}' key.")

    # Default to all keys in data if none provided
    if data_keys is None:
        data_keys = list(data.keys())

    # Verify data sizes
    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "Batch size is larger than the dataset size or no full batch available."
        )

    # -------------------------------------------------------------------------
    # Compute Random Permutation for Batches
    # -------------------------------------------------------------------------
    perms = jax.random.permutation(key, data_size)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # -------------------------------------------------------------------------
    # Precompute Batch Segments and Indices
    # -------------------------------------------------------------------------
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms

    # Compute pairwise indices only if not provided
    # E3x: e3x.ops.sparse_pairwise_indices(num_atoms) -> returns (dst_idx, src_idx)
    if dst_idx is None or src_idx is None:
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    # Adjust indices for batching
    dst_idx = dst_idx + offsets[:, None]
    src_idx = src_idx + offsets[:, None]

    # Centralize reshape logic
    # For keys not listed here, we default to their original shape after indexing.
    reshape_rules = {
        "R": (batch_size * num_atoms, 3),
        "F": (batch_size * num_atoms, 3),
        "E": (batch_size, 1),
        "Z": (batch_size * num_atoms,),
        "D": (batch_size,3),
        "N": (batch_size,),
        "mono": (batch_size * num_atoms,),
    }

    output = []

    # -------------------------------------------------------------------------
    # Batch Preparation Loop
    # -------------------------------------------------------------------------
    for perm in perms:
        # Build the batch dictionary
        batch = {}
        for k in data_keys:
            if k not in data:
                continue
            v = data[k][perm]
            new_shape = reshape_rules.get(k, None)
            if new_shape is not None:
                batch[k] = v.reshape(new_shape)
            else:
                # Default to just attaching the permuted data without reshape
                batch[k] = v

        # Optionally include 'id' if requested and present
        if include_id and "id" in data and "id" in data_keys:
            batch["id"] = data["id"][perm]

        if rot_augment:
            rot_key = jax.random.fold_in(key, int(perm[0]))
            rotations = sample_random_rotations(
                rot_key, batch_size, perturbation=rot_perturbation
            )
            if "R" in batch:
                batch["R"] = rotate_batched_vectors(
                    batch["R"].reshape(batch_size, num_atoms, 3), rotations
                ).reshape(batch_size * num_atoms, 3)
            if "F" in batch:
                batch["F"] = rotate_batched_vectors(
                    batch["F"].reshape(batch_size, num_atoms, 3), rotations
                ).reshape(batch_size * num_atoms, 3)
            if "D" in batch and batch["D"].ndim == 2 and batch["D"].shape[-1] == 3:
                batch["D"] = rotate_batched_vectors(batch["D"], rotations)

        # Compute good_indices (mask for valid atom pairs)
        # Vectorized approach: We know N is shape (batch_size,)
        # Expand N to compare with dst_idx/src_idx
        # dst_idx[i], src_idx[i] range over atom pairs within the ith example
        # Condition: (dst_idx[i] < N[i]+i*num_atoms) & (src_idx[i] < N[i]+i*num_atoms)
        # We'll compute this for all i and concatenate.
        N = batch["N"]
        # Expand N and offsets for comparison
        expanded_n = N[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_n
        valid_src = src_idx < expanded_n
        good_pairs = (valid_dst & valid_src).astype(jnp.int32)
        good_indices = good_pairs.reshape(-1)

        # Add metadata to the batch
        atom_mask = jnp.where(batch["Z"] > 0, 1, 0)
        batch.update(
            {
                "dst_idx": dst_idx.flatten(),
                "src_idx": src_idx.flatten(),
                "batch_mask": good_indices,
                "batch_segments": batch_segments,
                "atom_mask": atom_mask,
            }
        )

        # Debug checks
        if debug_mode:
            # Check expected shapes
            assert batch["R"].shape == (
                batch_size * num_atoms,
                3,
            ), f"R shape mismatch: {batch['R'].shape}"
            assert batch["F"].shape == (
                batch_size * num_atoms,
                3,
            ), f"F shape mismatch: {batch['F'].shape}"
            assert batch["E"].shape == (
                batch_size,
                1,
            ), f"E shape mismatch: {batch['E'].shape}"
            assert batch["Z"].shape == (
                batch_size * num_atoms,
            ), f"Z shape mismatch: {batch['Z'].shape}"
            assert batch["N"].shape == (
                batch_size,
            ), f"N shape mismatch: {batch['N'].shape}"
            # Optional: print or log if needed

        output.append(batch)

    return output




def compute_dst_src_lookup(data):
    """Pre-compute destination-source indices for all unique numbers of atoms."""
    dst_src_lookup = {}
    for atom_count in np.unique(data["N"]):
        dst, src = e3x.ops.sparse_pairwise_indices(atom_count)
        dst_src_lookup[atom_count] = (dst, src)
    return dst_src_lookup


def create_batch(
    perm,
    dst_src_lookup,
    data,
    data_keys,
    batch_shape,
    batch_nbl_len,
    num_atoms,
    batch_size,
):
    """Create a single batch based on a given permutation of indices."""
    PADDING_VALUE = batch_shape + 1  # Padding value for unfilled batch elements
    batch = {
        "dst_idx": np.full(batch_nbl_len, PADDING_VALUE),
        "src_idx": np.full(batch_nbl_len, PADDING_VALUE),
        "batch_mask": np.zeros(batch_nbl_len, dtype=int),
    }
    n = data["N"][perm]
    # Determine stopping indices for padding
    cum_sum_n = np.cumsum(n)
    stop_idx = np.nonzero(cum_sum_n > batch_shape)[0]
    excluded_indices = set(stop_idx[stop_idx > 0] - 1)
    n[stop_idx] = 0

    """
    Loop over the atom-wise properties and fill the batch dictionary.
    """
    # Fill `dst_idx` and `src_idx` arrays
    idx_counter = 0
    an_counter = 0
    for i, n_atoms in enumerate(n):
        n_atoms = int(n_atoms)
        if n_atoms == 0 or n_atoms > batch_shape:
            raise ValueError(f"Invalid number of atoms: {n_atoms}")
        tmp_dst, tmp_src = dst_src_lookup[int(n_atoms)]
        len_current_nbl = int(n_atoms) * (int(n_atoms) - 1)
        if idx_counter + len_current_nbl > batch_nbl_len:
            n[i] = 0
            break
        batch["batch_mask"][idx_counter : idx_counter + len_current_nbl] = np.ones_like(
            tmp_dst
        )
        batch["dst_idx"][idx_counter : idx_counter + len_current_nbl] = (
            tmp_dst + an_counter
        )
        batch["src_idx"][idx_counter : idx_counter + len_current_nbl] = (
            tmp_src + an_counter
        )
        idx_counter += len_current_nbl
        an_counter += n_atoms

    """
    Loop Over the data keys and fill the batch dictionary.
    """
    # Handle additional batch data
    for key in data_keys:
        if key in data:
            if key in {"N", "E"}:
                shape = (batch_size,)
            elif key in {"dst_idx", "src_idx"}:
                break
            elif key == "D":
                shape = (batch_size, 3)
            elif key in {"R", "F"}:
                shape = (batch_shape, 3)
            elif key == "Z":
                shape = (batch_shape,)
            else:
                raise ValueError(f"Invalid key: {key}")

            batch[key] = np.zeros(shape)
            idx_counter = 0

            """
            Subloop over the permutation indices and fill the batch dictionary.
            """

            for i, permutation_index in enumerate(perm):
                if i not in excluded_indices:
                    start = int(0)
                    stop = int(n[i])

                    val = data[key][permutation_index]

                    if key in {"R", "F"}:
                        # print(i, key, val.shape, val)
                        val = val[start:stop, :].reshape(int(n[i]), 3)
                    elif key in {"D"}:
                        val = val.reshape(1, 3)
                    elif key in {"E", "N"}:
                        val = val.flatten()
                        # pad with zeros to make the batch size
                        # val = np.pad(val, (0, batch_size - len(val)))
                    elif key in {"Z"}:
                        val = val[start:stop].reshape(int(n[i]))
                    else:
                        break

                    if idx_counter + int(n[i]) > batch_shape:
                        break
                    # print(key, val.shape)
                    if key in {"R", "F"}:
                        batch[key][idx_counter : idx_counter + int(n[i])] = val
                    if key in {"Z"}:
                        batch[key][idx_counter : idx_counter + int(n[i])] = val
                    if key in {"E"}:
                        batch[key][i] = val

                    idx_counter += int(n[i])

    # mask for atoms
    atom_mask = jnp.where(batch["Z"] > 0, 1, 0)
    batch["atom_mask"] = atom_mask
    # mask for batches (atom wise)
    batch["N"] = np.array(n, dtype=np.int32).reshape(-1)
    batch["Z"] = np.array(batch["Z"], dtype=np.int32).reshape(-1)
    batch["E"] = np.pad(batch["E"], (0, batch_size - len(batch["E"])))

    batch_mask_atoms = np.concatenate(
        [np.ones(int(x)) * i for i, x in enumerate(batch["N"])]
    )
    padded_batch_segs = np.pad(
        batch_mask_atoms, (0, batch_shape - len(batch_mask_atoms))
    )
    batch["batch_segments"] = np.array(padded_batch_segs, dtype=np.int32)
    return batch


def prepare_batches_advanced_minibatching(
    key,
    data,
    batch_size,
    batch_shape,
    batch_nbl_len,
    data_keys=None,
    num_atoms=60,
    rot_augment: bool = False,
    rot_perturbation: float = 1.0,
) -> list:
    """
    Prepare batches for training.
    """
    assert data is not None, "Data cannot be None"

    # Dataset statistics
    data_size = len(data["R"])
    dst_src_lookup = compute_dst_src_lookup(data)
    steps_per_epoch = max(data_size // batch_size, 1)

    # Generate and reshape permutations
    perms = jax.random.permutation(key, data_size)[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Build batches
    output = []
    for perm in perms:
        batch = create_batch(
            perm,
            dst_src_lookup,
            data,
            data_keys,
            batch_shape,
            batch_nbl_len,
            num_atoms,
            batch_size,
        )
        if rot_augment:
            rot_key = jax.random.fold_in(key, int(perm[0]))
            rotations = sample_random_rotations(
                rot_key, batch_size, perturbation=rot_perturbation
            )
            if "R" in batch:
                batch["R"] = np.asarray(
                    rotate_batched_vectors(
                        jnp.asarray(batch["R"]).reshape(batch_size, -1, 3), rotations
                    ).reshape(-1, 3)
                )
            if "F" in batch:
                batch["F"] = np.asarray(
                    rotate_batched_vectors(
                        jnp.asarray(batch["F"]).reshape(batch_size, -1, 3), rotations
                    ).reshape(-1, 3)
                )
            if "D" in batch:
                d_arr = jnp.asarray(batch["D"])
                if d_arr.ndim == 2 and d_arr.shape[-1] == 3:
                    batch["D"] = np.asarray(rotate_batched_vectors(d_arr, rotations))
        output.append(batch)

    return output


_prepare_batches = prepare_batches_jit #jax.jit(prepare_batches_jit, static_argnames=("batch_size", "num_atoms", "data_keys"))


def _as_numpy_dict(data: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    """Host-side arrays for fast fancy indexing during batch prep."""
    return {k: np.asarray(v) for k, v in data.items()}


def _pair_indices(
    num_atoms: int,
    batch_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute batched pair indices and segment ids (constant across epochs)."""
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_2d = dst_idx + offsets[:, None]
    src_2d = src_idx + offsets[:, None]
    return (
        batch_segments,
        offsets,
        dst_2d,
        src_2d,
        dst_2d.reshape(-1),
        src_2d.reshape(-1),
    )


def prepare_batches_fast(
    key,
    data: Dict[str, jnp.ndarray],
    batch_size: int,
    data_keys: Optional[List[str]] = None,
    num_atoms: int = 60,
    dst_idx: Optional[jnp.ndarray] = None,
    src_idx: Optional[jnp.ndarray] = None,
    include_id: bool = False,
    debug_mode: bool = False,
    rot_augment: bool = False,
    rot_perturbation: float = 1.0,
    pair_cache: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> List[Dict[str, jnp.ndarray]]:
    """Vectorized batch preparation (same outputs as ``prepare_batches_jit``).

    Gathers all epoch batches with NumPy fancy indexing, then materializes a
    list of per-step dicts.  Pair indices and segment metadata are computed once
    and reused across epochs via ``pair_cache``.
    """
    required_keys = ["R", "N", "Z"]
    for req_key in required_keys:
        if req_key not in data:
            raise ValueError(f"Data dictionary must contain '{req_key}' key.")

    if data_keys is None:
        data_keys = list(data.keys())

    host = _as_numpy_dict(data)
    data_size = int(host["R"].shape[0])
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "Batch size is larger than the dataset size or no full batch available."
        )

    perms = np.asarray(
        jax.random.permutation(key, data_size)[: steps_per_epoch * batch_size],
        dtype=np.int64,
    )
    perms = perms.reshape((steps_per_epoch, batch_size))

    if pair_cache is None:
        pair_cache = _pair_indices(num_atoms, batch_size)
    batch_segments, offsets, dst_2d, src_2d, dst_flat, src_flat = pair_cache
    offsets_np = np.asarray(offsets, dtype=np.int64)
    dst_2d_np = np.asarray(dst_2d)
    src_2d_np = np.asarray(src_2d)

    reshape_rules = {
        "R": (batch_size * num_atoms, 3),
        "F": (batch_size * num_atoms, 3),
        "E": (batch_size, 1),
        "Z": (batch_size * num_atoms,),
        "D": (batch_size, 3),
        "N": (batch_size,),
        "mono": (batch_size * num_atoms,),
    }

    stacked: Dict[str, np.ndarray] = {}
    for k in data_keys:
        if k not in host:
            continue
        gathered = host[k][perms]
        new_shape = reshape_rules.get(k)
        if new_shape is not None:
            stacked[k] = gathered.reshape((steps_per_epoch, *new_shape))
        else:
            stacked[k] = gathered

    if rot_augment:
        for step in range(steps_per_epoch):
            rot_key = jax.random.fold_in(key, int(perms[step, 0]))
            rotations = sample_random_rotations(
                rot_key, batch_size, perturbation=rot_perturbation
            )
            if "R" in stacked:
                r = stacked["R"][step].reshape(batch_size, num_atoms, 3)
                stacked["R"][step] = np.asarray(
                    rotate_batched_vectors(r, rotations)
                ).reshape(batch_size * num_atoms, 3)
            if "F" in stacked:
                f = stacked["F"][step].reshape(batch_size, num_atoms, 3)
                stacked["F"][step] = np.asarray(
                    rotate_batched_vectors(f, rotations)
                ).reshape(batch_size * num_atoms, 3)
            if "D" in stacked:
                d = stacked["D"][step]
                if d.ndim == 2 and d.shape[-1] == 3:
                    stacked["D"][step] = np.asarray(
                        rotate_batched_vectors(d, rotations)
                    )

    n_batch = stacked["N"]
    expanded_n = n_batch[:, :, None] + offsets_np[None, :, None]
    valid = (dst_2d_np[None, :, :] < expanded_n) & (src_2d_np[None, :, :] < expanded_n)
    good_indices = valid.reshape(steps_per_epoch, -1).astype(np.int32)

    atom_mask = (stacked["Z"] > 0).astype(np.int32)

    output: List[Dict[str, jnp.ndarray]] = []
    for step in range(steps_per_epoch):
        batch: Dict[str, jnp.ndarray] = {}
        for k in data_keys:
            if k in stacked:
                batch[k] = jnp.asarray(stacked[k][step])
        if include_id and "id" in host and "id" in data_keys:
            batch["id"] = jnp.asarray(host["id"][perms[step]])
        batch["dst_idx"] = dst_flat
        batch["src_idx"] = src_flat
        batch["batch_mask"] = jnp.asarray(good_indices[step])
        batch["batch_segments"] = batch_segments
        batch["atom_mask"] = jnp.asarray(atom_mask[step].reshape(-1))
        if debug_mode:
            assert batch["R"].shape == (batch_size * num_atoms, 3)
        output.append(batch)
    return output


_prepare_batches = prepare_batches_fast

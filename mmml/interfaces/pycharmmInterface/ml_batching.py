"""ML batch preparation utilities for hybrid MM/ML calculator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

try:
    import e3x  # type: ignore[import-not-found]
except ModuleNotFoundError:
    e3x = None  # type: ignore[assignment]


def prepare_batch_structure(
    batch_size: int,
    num_atoms: int,
    dst_idx: Optional[jnp.ndarray] = None,
    src_idx: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Precompute static batch indices for JIT-friendly reuse.

    Returns dst_idx, src_idx, batch_segments (flattened). Data-dependent
    batch_mask and atom_mask must be computed per-call from R, Z, N.
    """
    if e3x is None:
        raise ModuleNotFoundError("e3x is required for prepare_batch_structure")

    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms

    if dst_idx is None or src_idx is None:
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    dst_idx = dst_idx + offsets[:, None]
    src_idx = src_idx + offsets[:, None]

    return {
        "dst_idx": dst_idx.flatten(),
        "src_idx": src_idx.flatten(),
        "batch_segments": batch_segments,
    }


def prepare_batches_md(
    data: Dict[str, Any],
    batch_size: int,
    data_keys: Optional[List[str]] = None,
    num_atoms: int = 60,
    dst_idx: Optional[jnp.ndarray] = None,
    src_idx: Optional[jnp.ndarray] = None,
    cached_structure: Optional[Dict[str, jnp.ndarray]] = None,
    include_id: bool = False,
    debug_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Efficiently prepare batches for training.

    Args:
        data: Dictionary containing the dataset.
            Expected keys: 'R', 'N', 'Z', 'F', 'E', and optionally others.
        batch_size: Size of each batch.
        data_keys: List of keys to include in the output.
            If None, all keys in `data` are included.
        num_atoms: Number of atoms per example. Default is 60.
        dst_idx: Precomputed destination indices for atom pairs.
        src_idx: Precomputed source indices for atom pairs.
        include_id: Whether to include 'id' key if present in data.
        debug_mode: If True, run assertions and extra checks.

    Returns:
        A list of dictionaries, each representing a batch.
    """
    if e3x is None:
        raise ModuleNotFoundError("e3x is required for prepare_batches_md")

    required_keys = ["R", "N", "Z"]
    for req_key in required_keys:
        if req_key not in data:
            raise ValueError(f"Data dictionary must contain '{req_key}' key.")

    if data_keys is None:
        data_keys = list(data.keys())

    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "Batch size is larger than the dataset size or no full batch available."
        )

    perms = jnp.arange(0, data_size)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Use cached structure if provided, else build
    if cached_structure is not None:
        dst_idx = cached_structure["dst_idx"]
        src_idx = cached_structure["src_idx"]
        batch_segments = cached_structure["batch_segments"]
    else:
        batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
        offsets = jnp.arange(batch_size) * num_atoms
        if dst_idx is None or src_idx is None:
            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
        if dst_idx.ndim == 2:
            dst_idx = dst_idx + offsets[:, None]
            src_idx = src_idx + offsets[:, None]
            dst_idx = dst_idx.flatten()
            src_idx = src_idx.flatten()

    reshape_rules = {
        "R": (batch_size * num_atoms, 3),
        "F": (batch_size * num_atoms, 3),
        "E": (batch_size, 1),
        "Z": (batch_size * num_atoms,),
        "D": (batch_size, 3),
        "N": (batch_size,),
        "mono": (batch_size * num_atoms,),
    }

    output = []
    for perm in perms:
        batch = {}
        for k in data_keys:
            if k not in data:
                continue
            v = data[k][jnp.array(perm)]
            new_shape = reshape_rules.get(k, None)
            if new_shape is not None:
                batch[k] = v.reshape(new_shape)
            else:
                batch[k] = v

        if include_id and "id" in data and "id" in data_keys:
            batch["id"] = data["id"][jnp.array(perm)]

        N = batch["N"]
        expanded_n = N[:, None] + offsets[:, None]
        valid_dst = dst_idx < expanded_n
        valid_src = src_idx < expanded_n
        good_pairs = (valid_dst & valid_src).astype(jnp.int32)
        good_indices = good_pairs.reshape(-1)

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

        if debug_mode:
            assert batch["R"].shape == (
                batch_size * num_atoms,
                3,
            ), f"R shape mismatch: {batch['R'].shape}"
            if "F" in batch:
                assert batch["F"].shape == (
                    batch_size * num_atoms,
                    3,
                ), f"F shape mismatch: {batch['F'].shape}"
            if "E" in batch:
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

        output.append(batch)

    return output

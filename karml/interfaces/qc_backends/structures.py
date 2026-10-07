"""Load structures from NPZ or XYZ for cross-check evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read


def load_structures_npz(
    path: Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[list[Atoms], dict[str, np.ndarray]]:
    """Load ASE frames and raw NPZ arrays from an MMML-style NPZ file."""
    data = np.load(path, allow_pickle=True)
    arrays = {k: np.asarray(data[k]) for k in data.files}

    if "R" not in arrays or "Z" not in arrays:
        raise KeyError(f"{path} must contain R and Z arrays.")

    r = np.asarray(arrays["R"], dtype=np.float64)
    z = np.asarray(arrays["Z"])
    if r.ndim == 2:
        r = r[np.newaxis, ...]

    n_total = r.shape[0]
    indices = np.arange(0, n_total, max(1, int(stride)), dtype=int)
    if max_frames is not None and len(indices) > int(max_frames):
        indices = indices[: int(max_frames)]

    frames: list[Atoms] = []
    for idx in indices:
        zi = z[idx] if z.ndim == 2 else z
        ri = r[idx]
        if z.ndim == 2:
            n_atoms = int(arrays["N"][idx]) if "N" in arrays else int(np.sum(zi > 0))
            zi = np.asarray(zi[:n_atoms], dtype=int)
            ri = np.asarray(ri[:n_atoms], dtype=np.float64)
        else:
            n_atoms = len(zi)
        frames.append(Atoms(numbers=zi, positions=ri))

    subset = {
        k: (v[indices] if hasattr(v, "__len__") and len(v) == n_total else v)
        for k, v in arrays.items()
    }
    subset["_frame_indices"] = indices
    return frames, subset


def load_structures_xyz(path: Path) -> tuple[list[Atoms], dict[str, np.ndarray]]:
    """Load one or more structures from an XYZ/extxyz file."""
    atoms_obj = ase_read(str(path), index=":")
    if isinstance(atoms_obj, list):
        frames = atoms_obj
    else:
        frames = [atoms_obj]

    r_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    for atoms in frames:
        z_list.append(np.asarray(atoms.get_atomic_numbers(), dtype=np.int32))
        r_list.append(np.asarray(atoms.get_positions(), dtype=np.float64))

    n_atoms = max(len(z) for z in z_list)
    n_frames = len(frames)
    r_batch = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
    z_batch = np.zeros((n_frames, n_atoms), dtype=np.int32)
    n_arr = np.zeros(n_frames, dtype=np.int32)
    for i, (zi, ri) in enumerate(zip(z_list, r_list)):
        n = len(zi)
        n_arr[i] = n
        z_batch[i, :n] = zi
        r_batch[i, :n] = ri

    arrays = {
        "R": r_batch,
        "Z": z_batch,
        "N": n_arr,
        "_frame_indices": np.arange(n_frames, dtype=int),
    }
    return frames, arrays


def load_structures(
    path: Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[list[Atoms], dict[str, np.ndarray]]:
    """Load structures from NPZ or XYZ."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_structures_npz(path, max_frames=max_frames, stride=stride)
    if suffix in {".xyz", ".extxyz"}:
        frames, arrays = load_structures_xyz(path)
        indices = arrays["_frame_indices"]
        if stride > 1:
            indices = indices[:: int(stride)]
        if max_frames is not None:
            indices = indices[: int(max_frames)]
        if len(indices) != len(frames):
            frames = [frames[i] for i in indices]
            arrays = {
                k: (v[indices] if hasattr(v, "__len__") and len(v) == len(arrays["_frame_indices"]) else v)
                for k, v in arrays.items()
            }
            arrays["_frame_indices"] = indices
        return frames, arrays
    raise ValueError(f"Unsupported structure file type: {path}")

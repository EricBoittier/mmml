"""
Load MD17 / revised MD17 (rMD17) NPZ files into MMML PhysNet format.

rMD17 files use keys ``nuclear_charges``, ``coords``, ``energies``, ``forces``.
MMML expects ``R``, ``Z``, ``E``, ``F``, ``N`` (coordinates in Å; default rMD17
units are kcal/mol and kcal/mol/Å).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .preprocessing import convert_energy_units

KCAL_TO_EV = 0.043364106370


def resolve_rmd17_splits_dir(
    data_path: Union[str, Path],
    splits_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Locate the rMD17 ``splits/`` directory with official index CSV files.

    If ``splits_dir`` is given, it is returned as-is. Otherwise, looks for
    ``<rmd17_root>/splits`` when ``data_path`` lives under ``.../npz_data/``.
    """
    if splits_dir is not None:
        return Path(splits_dir)

    data_path = Path(data_path)
    if data_path.parent.name == "npz_data":
        candidate = data_path.parent.parent / "splits"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find rMD17 splits directory. Set RMD17_SPLITS_DIR to the "
        "folder containing index_train_01.csv and index_test_01.csv "
        "(typically <rmd17>/splits next to npz_data/)."
    )


def load_rmd17_official_splits(
    splits_dir: Union[str, Path],
    split_id: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load official rMD17 train-pool and test indices from Materials Cloud CSVs.

    Files: ``index_train_{split_id:02d}.csv``, ``index_test_{split_id:02d}.csv``
    for ``split_id`` in 1..5 (five predefined folds).

    Returns
    -------
    train_pool_idx, test_idx
        Integer row indices into the NPZ arrays. Hold out ``test_idx`` first;
        draw train/validation only from ``train_pool_idx``.
    """
    if not 1 <= split_id <= 5:
        raise ValueError("split_id must be between 1 and 5 (official rMD17 folds)")

    splits_dir = Path(splits_dir)
    train_path = splits_dir / f"index_train_{split_id:02d}.csv"
    test_path = splits_dir / f"index_test_{split_id:02d}.csv"

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing official split file: {path}")

    train_pool_idx = np.loadtxt(train_path, dtype=np.int64).reshape(-1)
    test_idx = np.loadtxt(test_path, dtype=np.int64).reshape(-1)

    overlap = np.intersect1d(train_pool_idx, test_idx)
    if overlap.size:
        raise ValueError(
            f"Official train and test indices overlap ({overlap.size} entries)"
        )

    return train_pool_idx, test_idx


def load_rmd17_npz(
    file_path: Union[str, Path],
    natoms: Optional[int] = None,
    convert_to_ev: bool = False,
    max_structures: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Load an MD17/rMD17 NPZ file and return MMML-standard arrays.

    Parameters
    ----------
    file_path
        Path to ``*.npz`` (e.g. ``rmd17_aspirin.npz``).
    natoms
        Pad/truncate to this many atoms. Defaults to the molecule size.
    convert_to_ev
        If True, convert energies and forces from kcal/mol (and kcal/mol/Å)
        to eV (and eV/Å). PhysNetJax MAE reporting assumes eV when using the
        default ``train_model`` conversion factors.
    max_structures
        Optionally limit the number of conformations loaded (useful for quick
        tests; rMD17 recommends training on at most ~1000 structures).

    Returns
    -------
    dict
        ``R``, ``Z``, ``E``, ``F``, ``N`` arrays ready for ``train_model`` /
        ``prepare_datasets``.
    """
    file_path = Path(file_path)
    raw = np.load(file_path, allow_pickle=True)

    z_key = _first_key(raw.files, ("nuclear_charges", "Z", "atomic_numbers"))
    r_key = _first_key(raw.files, ("coords", "coordinates", "R", "positions"))
    e_key = _first_key(raw.files, ("energies", "energy", "E"))
    f_key = _first_key(raw.files, ("forces", "F"))

    Z_mol = np.asarray(raw[z_key], dtype=np.int32).reshape(-1)
    R = np.asarray(raw[r_key], dtype=np.float64)
    E = np.asarray(raw[e_key], dtype=np.float64).reshape(-1)
    F = np.asarray(raw[f_key], dtype=np.float64)

    if max_structures is not None:
        R = R[:max_structures]
        E = E[:max_structures]
        F = F[:max_structures]

    n_frames, n_atoms, _ = R.shape
    if Z_mol.shape[0] != n_atoms:
        raise ValueError(
            f"Expected {n_atoms} nuclear charges, got {Z_mol.shape[0]}"
        )

    n_real = int(np.count_nonzero(Z_mol))
    if natoms is None:
        natoms = n_real
    if natoms < n_real:
        raise ValueError(
            f"natoms={natoms} is smaller than molecule size ({n_real} atoms)"
        )

    Z = np.tile(Z_mol.reshape(1, -1), (n_frames, 1))
    R_pad = np.zeros((n_frames, natoms, 3), dtype=np.float64)
    F_pad = np.zeros((n_frames, natoms, 3), dtype=np.float64)
    Z_pad = np.zeros((n_frames, natoms), dtype=np.int32)
    R_pad[:, :n_atoms, :] = R
    F_pad[:, :n_atoms, :] = F
    Z_pad[:, :n_atoms] = Z

    N = np.full((n_frames,), n_real, dtype=np.int32)

    if convert_to_ev:
        E = convert_energy_units(E, from_unit="kcal/mol", to_unit="eV")
        F_pad = F_pad * KCAL_TO_EV

    return {
        "R": R_pad,
        "Z": Z_pad,
        "E": E.reshape(-1, 1),
        "F": F_pad,
        "N": N.reshape(-1, 1),
    }


def _first_key(files, candidates):
    for key in candidates:
        if key in files:
            return key
    raise KeyError(
        f"None of {candidates} found in NPZ (available: {list(files)})"
    )

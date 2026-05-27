"""
Load MD17 / revised MD17 (rMD17) NPZ files into MMML PhysNet format.

rMD17 files use keys ``nuclear_charges``, ``coords``, ``energies``, ``forces``.
MMML expects ``R``, ``Z``, ``E``, ``F``, ``N`` (coordinates in Å; default rMD17
units are kcal/mol and kcal/mol/Å).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .preprocessing import convert_energy_units

KCAL_TO_EV = 0.043364106370


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

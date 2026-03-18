"""
Lightweight electrostatic utilities for dipole and ESP from point charges.

Uses only numpy and ase - no PySCF/CUDA dependencies.
"""

import numpy as np
import ase.data

from mmml.data.units import ANGSTROM_TO_BOHR


def compute_dipole_from_point_charges(
    charges: np.ndarray,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
) -> np.ndarray:
    """
    Compute molecular dipole from atomic point charges.

    μ = Σ q_i * (r_i - r_COM)

    Parameters
    ----------
    charges : np.ndarray
        (n_atoms,) in electron charge units
    positions : np.ndarray
        (n_atoms, 3) in Angstrom
    atomic_numbers : np.ndarray
        (n_atoms,)

    Returns
    -------
    np.ndarray
        Dipole (3,) in e·Å
    """
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    dipole = np.sum(charges[:, None] * (positions - com), axis=0)
    return dipole


def compute_esp_from_point_charges(
    charges: np.ndarray,
    atom_pos: np.ndarray,
    grid_positions: np.ndarray,
    atom_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute ESP at grid points from atomic point charges.

    V = Σ q_i / r_bohr in Hartree/e when r is in Bohr.

    Parameters
    ----------
    charges : np.ndarray
        (n_atoms,) in electron charge units
    atom_pos : np.ndarray
        (n_atoms, 3) in Angstrom
    grid_positions : np.ndarray
        (ngrid, 3) in Angstrom
    atom_mask : np.ndarray, optional
        (n_atoms,) 1 for real atoms, 0 for padding

    Returns
    -------
    np.ndarray
        ESP (ngrid,) in Hartree/e
    """
    charges = np.asarray(charges).flatten()
    atom_pos = np.asarray(atom_pos).reshape(-1, 3)
    grid_positions = np.asarray(grid_positions).reshape(-1, 3)

    if atom_mask is not None:
        atom_mask = np.asarray(atom_mask).flatten()
        charges = charges * atom_mask

    # distances: (ngrid, n_atoms)
    diff = grid_positions[:, None, :] - atom_pos[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    distances = np.where(distances < 1e-10, 1e10, distances)  # avoid div by zero

    r_bohr = distances * ANGSTROM_TO_BOHR
    esp = np.sum(charges[None, :] / r_bohr, axis=1)

    # Mask invalid grid points (convention: 1e6 for invalid)
    valid_grid = np.all(np.abs(grid_positions) < 1e5, axis=1)
    esp = np.where(valid_grid, esp, np.nan)

    return esp

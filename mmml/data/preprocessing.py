"""
Data preprocessing utilities for MMML.

Common preprocessing operations for molecular data including
centering, normalization, and masking.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import ase.data
from scipy.spatial.distance import cdist

from mmml.data.atomic_references import (
    DEFAULT_CHARGE_STATE,
    DEFAULT_REFERENCE_LEVEL,
    get_atomic_reference_array,
    get_atomic_reference_dict,
)

ATOM_ENERGIES_HARTREE = get_atomic_reference_array(
    level=DEFAULT_REFERENCE_LEVEL,
    charge_state=DEFAULT_CHARGE_STATE,
    unit="hartree",
)


def center_coordinates(
    coordinates: np.ndarray,
    n_atoms: Optional[np.ndarray] = None,
    method: str = 'com'
) -> np.ndarray:
    """
    Center molecular coordinates.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Coordinates array of shape (n_structures, n_atoms, 3)
    n_atoms : np.ndarray, optional
        Number of atoms per structure, by default None (use all)
    method : str, optional
        Centering method: 'com' (center of mass) or 'geometric' (geometric center),
        by default 'com'
        
    Returns
    -------
    np.ndarray
        Centered coordinates
    """
    centered = coordinates.copy()
    n_structures = coordinates.shape[0]
    
    for i in range(n_structures):
        if n_atoms is not None:
            n = n_atoms[i]
            coords = coordinates[i, :n, :]
        else:
            coords = coordinates[i]
        
        if method == 'geometric':
            center = np.mean(coords, axis=0)
        elif method == 'com':
            # For COM, would need masses - fall back to geometric for now
            center = np.mean(coords, axis=0)
        else:
            raise ValueError(f"Unknown centering method: {method}")
        
        if n_atoms is not None:
            centered[i, :n, :] -= center
        else:
            centered[i] -= center
    
    return centered


def convert_energy_units(
    energies: np.ndarray,
    from_unit: str = 'hartree',
    to_unit: str = 'eV'
) -> np.ndarray:
    """
    Convert energy units.
    
    Parameters
    ----------
    energies : np.ndarray
        Energy array
    from_unit : str, optional
        Source unit: 'hartree', 'eV', 'kcal/mol', 'kJ/mol', by default 'hartree'
    to_unit : str, optional
        Target unit: 'hartree', 'eV', 'kcal/mol', 'kJ/mol', by default 'eV'
        
    Returns
    -------
    np.ndarray
        Converted energies
    """
    # Conversion factors to eV
    to_ev = {
        'hartree': 27.211386245988,  # Hartree to eV
        'ev': 1.0,
        'kcal/mol': 0.043364106370,  # kcal/mol to eV
        'kj/mol': 0.010364269656,    # kJ/mol to eV
    }
    
    from_unit_lower = from_unit.lower()
    to_unit_lower = to_unit.lower()
    
    if from_unit_lower not in to_ev:
        raise ValueError(f"Unknown source unit: {from_unit}. Choose from {list(to_ev.keys())}")
    if to_unit_lower not in to_ev:
        raise ValueError(f"Unknown target unit: {to_unit}. Choose from {list(to_ev.keys())}")
    
    # Convert to eV first, then to target unit
    energies_ev = energies * to_ev[from_unit_lower]
    converted = energies_ev / to_ev[to_unit_lower]
    
    return converted


def compute_atomic_energies(
    energies: np.ndarray,
    atomic_numbers: np.ndarray,
    n_atoms: np.ndarray,
    method: str = 'linear_regression'
) -> Dict[int, float]:
    """
    Compute atomic energy references from molecular energies.
    
    Parameters
    ----------
    energies : np.ndarray
        Total molecular energies, shape (n_structures,)
    atomic_numbers : np.ndarray
        Atomic numbers, shape (n_structures, max_atoms)
    n_atoms : np.ndarray
        Number of atoms per structure, shape (n_structures,)
    method : str, optional
        Method to compute references: 'linear_regression' or 'mean', by default 'linear_regression'
        
    Returns
    -------
    dict
        Dictionary mapping atomic number to atomic energy reference
    """
    # Get unique atomic numbers
    unique_z = np.unique(atomic_numbers[atomic_numbers > 0])
    
    if method == 'linear_regression':
        # Solve: E_mol = sum_i n_i * E_i
        # Using least squares: X @ atomic_energies = energies
        # where X[mol, z] = count of element z in molecule
        
        from scipy.linalg import lstsq
        
        # Build composition matrix
        n_structures = len(energies)
        X = np.zeros((n_structures, len(unique_z)))
        
        for i, z in enumerate(unique_z):
            for j in range(n_structures):
                n = n_atoms[j]
                X[j, i] = np.sum(atomic_numbers[j, :n] == z)
        
        # Solve least squares
        result = lstsq(X, energies)
        atomic_energies_array = result[0]
        
        # Create dictionary
        atomic_energies = {int(z): float(e) for z, e in zip(unique_z, atomic_energies_array)}
        
    elif method == 'mean':
        # Simple mean per atom type
        atomic_energies = {}
        for z in unique_z:
            # Find molecules containing this element
            mask = np.any(atomic_numbers == z, axis=1)
            if np.sum(mask) > 0:
                avg_energy = np.mean(energies[mask] / n_atoms[mask])
                atomic_energies[int(z)] = float(avg_energy)
            else:
                atomic_energies[int(z)] = 0.0
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'linear_regression' or 'mean'")
    
    return atomic_energies


def subtract_atomic_energies(
    energies: np.ndarray,
    atomic_numbers: np.ndarray,
    n_atoms: np.ndarray,
    atomic_energy_refs: Dict[int, float]
) -> np.ndarray:
    """
    Subtract atomic energy references from molecular energies.
    
    Parameters
    ----------
    energies : np.ndarray
        Total molecular energies, shape (n_structures,)
    atomic_numbers : np.ndarray
        Atomic numbers, shape (n_structures, max_atoms)
    n_atoms : np.ndarray
        Number of atoms per structure, shape (n_structures,)
    atomic_energy_refs : dict
        Dictionary mapping atomic number to atomic energy reference
        
    Returns
    -------
    np.ndarray
        Energies with atomic references subtracted
    """
    corrected = energies.copy()
    
    for i in range(len(energies)):
        n = n_atoms[i]
        atomic_contribution = 0.0
        
        for j in range(n):
            z = int(atomic_numbers[i, j])
            if z in atomic_energy_refs:
                atomic_contribution += atomic_energy_refs[z]
        
        corrected[i] -= atomic_contribution
    
    return corrected


def get_default_atomic_energies(
    unit: str = 'eV',
    reference: str = DEFAULT_REFERENCE_LEVEL,
    charge_state: int = DEFAULT_CHARGE_STATE,
    *,
    fallback_to_neutral: bool = True,
) -> Dict[int, float]:
    """Return default isolated atomic energies from the reference table.

    Parameters
    ----------
    unit
        Target energy unit (``'eV'``, ``'hartree'``, ``'kcal/mol'`` or ``'kJ/mol'``).
    reference
        Level of theory key inside :mod:`mmml.data.atomic_reference_energies`.
    charge_state
        Atomic charge state to select (default: neutral atoms).
    fallback_to_neutral
        If ``True`` and the requested charge is missing for an element, fall back
        to the neutral value.
    """

    return get_atomic_reference_dict(
        level=reference,
        charge_state=charge_state,
        unit=unit,
        fallback_to_neutral=fallback_to_neutral,
    )


def scale_energies_by_atoms(
    energies: np.ndarray,
    n_atoms: np.ndarray
) -> np.ndarray:
    """
    Scale energies by number of atoms (convert to per-atom energies).
    
    Parameters
    ----------
    energies : np.ndarray
        Total molecular energies, shape (n_structures,)
    n_atoms : np.ndarray
        Number of atoms per structure, shape (n_structures,)
        
    Returns
    -------
    np.ndarray
        Per-atom energies
    """
    return energies / n_atoms


def normalize_energies(
    energies: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    per_atom: bool = False,
    n_atoms: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize energies to mean=0, std=1.
    
    Parameters
    ----------
    energies : np.ndarray
        Energy array
    mean : float, optional
        Pre-computed mean (if None, compute from data), by default None
    std : float, optional
        Pre-computed std (if None, compute from data), by default None
    per_atom : bool, optional
        Whether to normalize per-atom energies, by default False
    n_atoms : np.ndarray, optional
        Number of atoms per structure (required if per_atom=True), by default None
        
    Returns
    -------
    tuple
        (normalized_energies, statistics_dict)
    """
    if per_atom:
        if n_atoms is None:
            raise ValueError("n_atoms required for per-atom normalization")
        energies_per_atom = energies / n_atoms
        if mean is None:
            mean = np.mean(energies_per_atom)
        if std is None:
            std = np.std(energies_per_atom)
        normalized = (energies_per_atom - mean) / std * n_atoms
    else:
        if mean is None:
            mean = np.mean(energies)
        if std is None:
            std = np.std(energies)
        normalized = (energies - mean) / std
    
    stats = {
        'mean': float(mean),
        'std': float(std),
        'per_atom': per_atom,
    }
    
    return normalized, stats


def denormalize_energies(
    normalized_energies: np.ndarray,
    stats: Dict,
    n_atoms: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Denormalize energies using stored statistics.
    
    Parameters
    ----------
    normalized_energies : np.ndarray
        Normalized energy array
    stats : dict
        Statistics dictionary from normalize_energies
    n_atoms : np.ndarray, optional
        Number of atoms per structure (required if per_atom=True), by default None
        
    Returns
    -------
    np.ndarray
        Denormalized energies
    """
    mean = stats['mean']
    std = stats['std']
    per_atom = stats.get('per_atom', False)
    
    if per_atom:
        if n_atoms is None:
            raise ValueError("n_atoms required for per-atom denormalization")
        energies_per_atom = normalized_energies / n_atoms
        denormalized = (energies_per_atom * std + mean) * n_atoms
    else:
        denormalized = normalized_energies * std + mean
    
    return denormalized


def create_esp_mask(
    esp_grid: np.ndarray,
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    vdw_scale: float = 1.4
) -> np.ndarray:
    """
    Create mask to exclude ESP grid points inside van der Waals radii.
    
    Parameters
    ----------
    esp_grid : np.ndarray
        ESP grid coordinates of shape (n_structures, n_grid, 3)
    coordinates : np.ndarray
        Atomic coordinates of shape (n_structures, n_atoms, 3)
    atomic_numbers : np.ndarray
        Atomic numbers of shape (n_structures, n_atoms)
    vdw_scale : float, optional
        Scaling factor for VDW radii, by default 1.4
        
    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_structures, n_grid) where True = valid point
    """
    n_structures = esp_grid.shape[0]
    n_grid = esp_grid.shape[1]
    
    masks = np.zeros((n_structures, n_grid), dtype=bool)
    
    for i in range(n_structures):
        grid = esp_grid[i]
        xyz = coordinates[i]
        z = atomic_numbers[i]
        
        # Get VDW radii for atoms
        valid_atoms = z > 0
        z_valid = z[valid_atoms]
        xyz_valid = xyz[valid_atoms]
        
        vdw_radii = np.array([ase.data.vdw_radii[int(zi)] for zi in z_valid]) * vdw_scale
        
        # Compute distances
        distances = cdist(grid, xyz_valid)
        
        # Mask points inside VDW radii
        inside_vdw = distances < vdw_radii
        masks[i] = ~inside_vdw.any(axis=1)
    
    return masks


def compute_pairwise_distances(
    coordinates: np.ndarray,
    n_atoms: Optional[np.ndarray] = None,
    max_distance: Optional[float] = None
) -> np.ndarray:
    """
    Compute pairwise distances between atoms.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Coordinates of shape (n_structures, n_atoms, 3)
    n_atoms : np.ndarray, optional
        Number of atoms per structure, by default None
    max_distance : float, optional
        Maximum distance to include (for cutoff), by default None
        
    Returns
    -------
    np.ndarray
        Distance matrix
    """
    n_structures = coordinates.shape[0]
    n_atoms_max = coordinates.shape[1]
    
    distances = np.zeros((n_structures, n_atoms_max, n_atoms_max))
    
    for i in range(n_structures):
        if n_atoms is not None:
            n = n_atoms[i]
            coords = coordinates[i, :n, :]
        else:
            coords = coordinates[i]
        
        dist = cdist(coords, coords)
        
        if max_distance is not None:
            dist = np.where(dist > max_distance, 0, dist)
        
        if n_atoms is not None:
            distances[i, :n, :n] = dist
        else:
            distances[i] = dist
    
    return distances


def pad_arrays(
    data: Dict[str, np.ndarray],
    n_atoms_target: int
) -> Dict[str, np.ndarray]:
    """
    Pad arrays to a target number of atoms.
    
    Parameters
    ----------
    data : dict
        Data dictionary
    n_atoms_target : int
        Target number of atoms
        
    Returns
    -------
    dict
        Padded data dictionary
    """
    padded = data.copy()
    
    if 'R' in data:
        n_structures, n_atoms, _ = data['R'].shape
        if n_atoms < n_atoms_target:
            R_padded = np.zeros((n_structures, n_atoms_target, 3))
            R_padded[:, :n_atoms, :] = data['R']
            padded['R'] = R_padded
    
    if 'Z' in data:
        n_structures, n_atoms = data['Z'].shape
        if n_atoms < n_atoms_target:
            Z_padded = np.zeros((n_structures, n_atoms_target), dtype=np.int32)
            Z_padded[:, :n_atoms] = data['Z']
            padded['Z'] = Z_padded
    
    if 'F' in data:
        n_structures, n_atoms, _ = data['F'].shape
        if n_atoms < n_atoms_target:
            F_padded = np.zeros((n_structures, n_atoms_target, 3))
            F_padded[:, :n_atoms, :] = data['F']
            padded['F'] = F_padded
    
    if 'mono' in data:
        n_structures, n_atoms = data['mono'].shape
        if n_atoms < n_atoms_target:
            mono_padded = np.zeros((n_structures, n_atoms_target))
            mono_padded[:, :n_atoms] = data['mono']
            padded['mono'] = mono_padded
    
    return padded


if __name__ == '__main__':
    # Test preprocessing functions
    import sys
    
    # Create test data
    coords = np.random.randn(10, 5, 3)
    n_atoms = np.array([5] * 10)
    
    # Test centering
    centered = center_coordinates(coords, n_atoms)
    print(f"Original center: {np.mean(coords[0, :5, :], axis=0)}")
    print(f"Centered center: {np.mean(centered[0, :5, :], axis=0)}")
    
    # Test normalization
    energies = np.random.randn(10) * 100 + 500
    normalized, stats = normalize_energies(energies)
    print(f"\nOriginal energies: mean={np.mean(energies):.2f}, std={np.std(energies):.2f}")
    print(f"Normalized energies: mean={np.mean(normalized):.2f}, std={np.std(normalized):.2f}")
    print(f"Stats: {stats}")
    
    # Test denormalization
    denorm = denormalize_energies(normalized, stats)
    print(f"Denormalized matches original: {np.allclose(energies, denorm)}")


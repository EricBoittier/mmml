"""
Data preprocessing utilities for MMML.

Common preprocessing operations for molecular data including
centering, normalization, and masking.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import ase.data
from scipy.spatial.distance import cdist


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


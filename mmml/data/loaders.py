"""
Data loading utilities for MMML.

Provides functions to load NPZ files, validate them, and prepare them
for training with DCMNet, PhysNetJAX, or other models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field

from .npz_schema import validate_npz, NPZSchema


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Attributes
    ----------
    batch_size : int
        Batch size for training
    targets : List[str]
        List of target properties to train on (e.g., ['energy', 'forces'])
    num_atoms : int
        Maximum number of atoms per structure (for padding)
    center_coordinates : bool
        Whether to center coordinates at origin
    normalize_energy : bool
        Whether to normalize energies to mean=0, std=1
    esp_mask_vdw : bool
        Whether to apply VDW masking to ESP data
    vdw_scale : float
        Scaling factor for VDW radii
    shuffle : bool
        Whether to shuffle data when creating batches
    include_metadata : bool
        Whether to include metadata in loaded data
    """
    
    batch_size: int = 32
    targets: List[str] = field(default_factory=lambda: ['energy'])
    num_atoms: int = 60
    center_coordinates: bool = False
    normalize_energy: bool = False
    esp_mask_vdw: bool = False
    vdw_scale: float = 1.4
    shuffle: bool = True
    include_metadata: bool = False


def load_npz(
    file_path: Union[str, Path],
    config: Optional[DataConfig] = None,
    validate: bool = True,
    keys: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Load and optionally validate an NPZ file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to NPZ file
    config : DataConfig, optional
        Configuration for data loading, by default None
    validate : bool, optional
        Whether to validate against schema, by default True
    keys : list, optional
        Specific keys to load (loads all if None), by default None
    verbose : bool, optional
        Whether to print information, by default False
        
    Returns
    -------
    dict
        Dictionary of arrays from NPZ file
        
    Examples
    --------
    >>> data = load_npz('train.npz', validate=True)
    >>> print(f"Loaded {len(data['E'])} structures")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {file_path}")
    
    # Load NPZ file
    try:
        npz = np.load(file_path, allow_pickle=True)
        
        # Select specific keys or load all
        if keys is None:
            data = {k: npz[k] for k in npz.files}
        else:
            data = {k: npz[k] for k in keys if k in npz.files}
            missing = set(keys) - set(npz.files)
            if missing and verbose:
                print(f"Warning: Keys not found in NPZ: {missing}")
        
        npz.close()
        
    except Exception as e:
        raise RuntimeError(f"Error loading NPZ file {file_path}: {e}")
    
    # Validate if requested
    if validate:
        schema = NPZSchema()
        is_valid, errors = schema.validate(data)
        if not is_valid and verbose:
            print(f"Validation warnings for {file_path}:")
            for error in errors:
                print(f"  - {error}")
    
    # Apply config-based preprocessing if provided
    if config is not None:
        data = _apply_config_preprocessing(data, config)
    
    if verbose:
        print(f"Loaded NPZ file: {file_path}")
        print(f"  Structures: {len(data.get('E', []))}")
        print(f"  Keys: {list(data.keys())}")
    
    return data


def load_multiple_npz(
    file_paths: List[Union[str, Path]],
    config: Optional[DataConfig] = None,
    validate: bool = True,
    combine: bool = True,
    verbose: bool = False
) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
    """
    Load multiple NPZ files.
    
    Parameters
    ----------
    file_paths : list
        List of paths to NPZ files
    config : DataConfig, optional
        Configuration for data loading, by default None
    validate : bool, optional
        Whether to validate each file, by default True
    combine : bool, optional
        Whether to combine into single dataset, by default True
    verbose : bool, optional
        Whether to print information, by default False
        
    Returns
    -------
    dict or list
        Combined dictionary if combine=True, otherwise list of dictionaries
    """
    datasets = []
    
    for file_path in file_paths:
        data = load_npz(file_path, config, validate, verbose=verbose)
        datasets.append(data)
    
    if not combine:
        return datasets
    
    # Combine datasets
    if len(datasets) == 1:
        return datasets[0]
    
    combined = {}
    
    # Get all keys present in any dataset
    all_keys = set()
    for ds in datasets:
        all_keys.update(ds.keys())
    
    # Concatenate each property
    for key in all_keys:
        if key == 'metadata':
            # Combine metadata specially
            all_metadata = [ds.get('metadata', [{}])[0] for ds in datasets if 'metadata' in ds]
            if all_metadata:
                combined_metadata = {
                    'combined_from': len(datasets),
                    'source_files': [m.get('source_file', 'unknown') for m in all_metadata],
                }
                combined['metadata'] = np.array([combined_metadata], dtype=object)
        else:
            # Concatenate arrays
            arrays = [ds[key] for ds in datasets if key in ds]
            if arrays:
                try:
                    combined[key] = np.concatenate(arrays, axis=0)
                except ValueError as e:
                    if verbose:
                        print(f"Warning: Could not concatenate '{key}': {e}")
    
    if verbose:
        print(f"\nCombined {len(datasets)} datasets:")
        print(f"  Total structures: {len(combined.get('E', []))}")
    
    return combined


def train_valid_split(
    data: Dict[str, np.ndarray],
    train_fraction: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split data into training and validation sets.
    
    Parameters
    ----------
    data : dict
        Dictionary of arrays
    train_fraction : float, optional
        Fraction of data to use for training, by default 0.8
    shuffle : bool, optional
        Whether to shuffle before splitting, by default True
    seed : int, optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    tuple
        (train_data, valid_data) dictionaries
    """
    # Get number of structures
    n_structures = len(data.get('E', data.get('R', [])))
    n_train = int(n_structures * train_fraction)
    
    # Create indices
    indices = np.arange(n_structures)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:]
    
    # Split data
    train_data = {}
    valid_data = {}
    
    for key, value in data.items():
        if key == 'metadata':
            train_data[key] = value
            valid_data[key] = value
        elif isinstance(value, np.ndarray) and len(value) == n_structures:
            train_data[key] = value[train_idx]
            valid_data[key] = value[valid_idx]
        else:
            # Copy non-indexed arrays to both
            train_data[key] = value
            valid_data[key] = value
    
    return train_data, valid_data


def _apply_config_preprocessing(
    data: Dict[str, np.ndarray],
    config: DataConfig
) -> Dict[str, np.ndarray]:
    """
    Apply preprocessing based on DataConfig.
    
    Parameters
    ----------
    data : dict
        Data dictionary
    config : DataConfig
        Configuration
        
    Returns
    -------
    dict
        Preprocessed data
    """
    # Center coordinates
    if config.center_coordinates and 'R' in data:
        from .preprocessing import center_coordinates
        data['R'] = center_coordinates(data['R'], data.get('N'))
    
    # Normalize energies
    if config.normalize_energy and 'E' in data:
        from .preprocessing import normalize_energies
        data['E'], stats = normalize_energies(data['E'])
        # Store normalization statistics in metadata
        if 'metadata' not in data:
            data['metadata'] = np.array([{}], dtype=object)
        metadata = data['metadata'][0]
        metadata['energy_normalization'] = stats
        data['metadata'] = np.array([metadata], dtype=object)
    
    # Create ESP mask
    if config.esp_mask_vdw and 'esp' in data and 'vdw_surface' in data:
        from .preprocessing import create_esp_mask
        data['espMask'] = create_esp_mask(
            data['vdw_surface'],
            data['R'],
            data['Z'],
            vdw_scale=config.vdw_scale
        )
    
    return data


def get_data_statistics(data: Dict[str, np.ndarray]) -> Dict:
    """
    Compute statistics for a dataset.
    
    Parameters
    ----------
    data : dict
        Data dictionary
        
    Returns
    -------
    dict
        Statistics dictionary
    """
    stats = {
        'n_structures': len(data.get('E', [])),
        'keys': list(data.keys()),
    }
    
    # Coordinate statistics
    if 'R' in data:
        stats['coordinates'] = {
            'shape': data['R'].shape,
            'min': float(np.min(data['R'])),
            'max': float(np.max(data['R'])),
            'mean': float(np.mean(data['R'])),
            'std': float(np.std(data['R'])),
        }
    
    # Energy statistics
    if 'E' in data:
        stats['energy'] = {
            'min': float(np.min(data['E'])),
            'max': float(np.max(data['E'])),
            'mean': float(np.mean(data['E'])),
            'std': float(np.std(data['E'])),
        }
    
    # Force statistics
    if 'F' in data:
        stats['forces'] = {
            'shape': data['F'].shape,
            'min': float(np.min(data['F'])),
            'max': float(np.max(data['F'])),
            'mean': float(np.mean(data['F'])),
            'std': float(np.std(data['F'])),
        }
    
    # Element statistics
    if 'Z' in data:
        unique_elements = np.unique(data['Z'][data['Z'] > 0])
        stats['elements'] = {
            'unique': unique_elements.tolist(),
            'counts': {int(z): int(np.sum(data['Z'] == z)) for z in unique_elements}
        }
    
    return stats


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        data = load_npz(file_path, validate=True, verbose=True)
        stats = get_data_statistics(data)
        
        print("\nDataset Statistics:")
        import json
        print(json.dumps(stats, indent=2))
    else:
        print("Usage: python loaders.py <npz_file>")


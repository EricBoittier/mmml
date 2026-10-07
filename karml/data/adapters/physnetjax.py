"""
PhysNetJAX data adapter.

Converts standardized NPZ format to PhysNetJAX-specific batch format.
"""

import numpy as np
from typing import Dict, List


def prepare_physnet_batches(
    data: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_atoms: int = 60,
    data_keys: List[str] = None,
    shuffle: bool = True,
    seed: int = None
) -> List[Dict]:
    """
    Prepare batches for PhysNetJAX training.
    
    Converts standardized NPZ format into PhysNetJAX batch format
    focusing on energies, forces, and other molecular properties.
    
    Parameters
    ----------
    data : dict
        Standardized NPZ data dictionary
    batch_size : int, optional
        Batch size, by default 32
    num_atoms : int, optional
        Number of atoms per structure, by default 60
    data_keys : list, optional
        Keys to include in batches, by default None (use all available)
    shuffle : bool, optional
        Whether to shuffle data, by default True
    seed : int, optional
        Random seed for shuffling, by default None
        
    Returns
    -------
    list
        List of batch dictionaries
        
    Notes
    -----
    This adapter is based on the prepare_batches function from
    mmml/physnetjax/physnetjax/data/ but adapted to work with the
    standardized NPZ format.
    
    TODO: This is a stub that needs full implementation
    """
    # Get data size
    data_size = len(data["R"])
    steps_per_epoch = data_size // batch_size
    
    # Shuffle if requested
    if shuffle:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(data_size)
    else:
        indices = np.arange(data_size)
    
    indices = indices[:steps_per_epoch * batch_size]
    indices = indices.reshape((steps_per_epoch, batch_size))
    
    # Determine which keys to include
    if data_keys is None:
        data_keys = [
            "R", "Z", "N", "E", "F", "D"
        ]
        # Only include keys that exist in data
        data_keys = [k for k in data_keys if k in data]
    
    # Create batches
    batches = []
    for perm in indices:
        batch_dict = {}
        
        for key in data_keys:
            if key in data:
                batch_dict[key] = data[key][perm]
        
        batches.append(batch_dict)
    
    return batches


# TODO: Implement more sophisticated features from original data.py:
# - Proper handling of variable-size molecules
# - Periodic boundary conditions if needed
# - Neighbor lists
# - Any PhysNet-specific requirements


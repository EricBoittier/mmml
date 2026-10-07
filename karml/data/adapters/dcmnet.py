"""
DCMNet data adapter.

Converts standardized NPZ format to DCMNet-specific batch format.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List
import e3x


def prepare_dcmnet_batches(
    data: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_atoms: int = 60,
    data_keys: List[str] = None,
    shuffle: bool = True,
    seed: int = None
) -> List[Dict]:
    """
    Prepare batches for DCMNet training.
    
    Converts standardized NPZ format into DCMNet batch format with
    message passing indices, batch segments, and ESP data handling.
    
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
    mmml/dcmnet/dcmnet/data.py but adapted to work with the
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
    
    # Prepare batch segments and message passing indices
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    offsets = jnp.arange(batch_size) * num_atoms
    
    # Create message passing indices (all pairs)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)
    
    # Determine which keys to include
    if data_keys is None:
        data_keys = [
            "R", "Z", "N", "E", "F", "D", 
            "mono", "esp", "vdw_surface", "esp_grid",
            "n_grid", "Dxyz", "espMask", "com"
        ]
        # Only include keys that exist in data
        data_keys = [k for k in data_keys if k in data]
    
    # Create batches
    batches = []
    for perm in indices:
        batch_dict = {}
        
        for key in data_keys:
            if key in data:
                if key == "R":
                    # Reshape coordinates to (batch_size * num_atoms, 3)
                    batch_dict[key] = data[key][perm].reshape(-1, 3)
                elif key == "Z":
                    # Reshape atomic numbers to (batch_size * num_atoms,)
                    batch_dict[key] = data[key][perm].reshape(-1)
                elif key == "mono":
                    # Reshape monopoles to (batch_size * num_atoms,)
                    batch_dict[key] = data[key][perm].reshape(-1)
                else:
                    # Keep other arrays as is
                    batch_dict[key] = data[key][perm]
        
        # Add message passing indices and batch segments
        batch_dict["dst_idx"] = dst_idx
        batch_dict["src_idx"] = src_idx
        batch_dict["batch_segments"] = batch_segments
        
        batches.append(batch_dict)
    
    return batches


# TODO: Implement more sophisticated features from original data.py:
# - Variable-size message passing indices
# - ESP mask handling
# - Custom grid handling
# - Per-batch indexing


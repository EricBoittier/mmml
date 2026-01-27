"""Utility functions for DCMNet training."""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from omegaconf import DictConfig

from mmml.dcmnet.dcmnet.data import prepare_datasets
from mmml.dcmnet.dcmnet.modules import MessagePassingModel


def setup_environment(cfg: DictConfig) -> None:
    """Setup JAX environment variables."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.device.mem_fraction)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.cuda_visible_devices)


def get_data_path(paths: List[str]) -> Path:
    """Find the first existing data path from a list of candidates."""
    for path_str in paths:
        path = Path(path_str)
        if path.exists():
            return path
    raise FileNotFoundError(f"No data file found in: {paths}")


def random_sample_esp(
    esp: np.ndarray,
    esp_grid: np.ndarray,
    n_sample: int,
    seed: int = 0,
    clip_min: float = -2.0,
    clip_max: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample ESP points within specified range.
    
    Args:
        esp: ESP values array
        esp_grid: ESP grid points array
        n_sample: Number of points to sample
        seed: Random seed
        clip_min: Minimum ESP value to include
        clip_max: Maximum ESP value to include
    
    Returns:
        Tuple of (sampled_esp, sampled_grid)
    """
    np.random.seed(seed)
    sampled_esp = []
    sampled_grid = []
    
    for i in range(len(esp)):
        # Create mask for valid ESP values
        lessthan = esp[i] < clip_max
        morethan = esp[i] > clip_min
        not_0 = esp[i] != 0.0
        condmask = lessthan * morethan * not_0
        
        _shape = esp[i][condmask].shape[0]
        if _shape < n_sample:
            # If not enough points, take all available
            indices = np.arange(_shape)
        else:
            indices = np.random.choice(_shape, n_sample, replace=False)
        
        sampled_esp.append(np.take(esp[i], condmask.nonzero()[0][indices]))
        sampled_grid.append(np.take(esp_grid[i], condmask.nonzero()[0][indices], axis=0))
    
    return np.array(sampled_esp), np.array(sampled_grid)


def prepare_data(
    cfg: DictConfig,
    data_key: jax.random.PRNGKey,
    bootstrap_idx: int = 0,
) -> Tuple[Dict, Dict]:
    """
    Prepare training and validation datasets.
    
    Args:
        cfg: Configuration object
        data_key: JAX random key
        bootstrap_idx: Bootstrap iteration index
    
    Returns:
        Tuple of (train_data, valid_data) dictionaries
    """
    # Find data file
    data_path = get_data_path(cfg.data.paths)
    
    # Load and prepare datasets
    train_data, valid_data = prepare_datasets(
        data_key,
        num_train=cfg.data.num_train,
        num_valid=cfg.data.num_valid,
        filename=[data_path],
        clean=cfg.data.clean,
        esp_mask=cfg.data.esp_mask,
        natoms=cfg.data.natoms,
        clip_esp=cfg.data.clip_esp,
    )
    
    # Sample ESP points
    n_sample = cfg.data.esp_sample_size
    seed = cfg.seed * bootstrap_idx
    
    train_data["esp"], train_data["esp_grid"] = random_sample_esp(
        train_data["esp"], train_data["esp_grid"], n_sample, seed=seed,
        clip_min=cfg.data.esp_clip_min, clip_max=cfg.data.esp_clip_max,
    )
    valid_data["esp"], valid_data["esp_grid"] = random_sample_esp(
        valid_data["esp"], valid_data["esp_grid"], n_sample, seed=seed,
        clip_min=cfg.data.esp_clip_min, clip_max=cfg.data.esp_clip_max,
    )
    
    # Convert ESP to atomic units
    train_data["esp"] = cfg.data.esp_to_au * train_data["esp"]
    valid_data["esp"] = cfg.data.esp_to_au * valid_data["esp"]
    
    # Set VDW surface
    train_data["vdw_surface"] = train_data["esp_grid"]
    valid_data["vdw_surface"] = valid_data["esp_grid"]
    
    # Set n_grid
    train_data["n_grid"] = np.full(train_data["Z"].shape[0], n_sample)
    valid_data["n_grid"] = np.full(valid_data["Z"].shape[0], n_sample)
    
    # Initialize monopoles based on atomic numbers
    Hs_train = train_data["Z"] == 1.0
    Os_train = train_data["Z"] == 8.0
    Hs_valid = valid_data["Z"] == 1.0
    Os_valid = valid_data["Z"] == 8.0
    
    train_data["mono"] = Hs_train * cfg.data.monopole_H + Os_train * cfg.data.monopole_O
    valid_data["mono"] = Hs_valid * cfg.data.monopole_H + Os_valid * cfg.data.monopole_O
    
    # Fix N (number of atoms per molecule)
    train_data["N"] = np.count_nonzero(train_data["Z"], axis=1)
    valid_data["N"] = np.count_nonzero(valid_data["Z"], axis=1)
    
    return train_data, valid_data


def create_model(cfg: DictConfig) -> MessagePassingModel:
    """Create model from configuration."""
    return MessagePassingModel(
        features=cfg.model.features,
        max_degree=cfg.model.max_degree,
        num_iterations=cfg.model.num_iterations,
        num_basis_functions=cfg.model.num_basis_functions,
        cutoff=cfg.model.cutoff,
        n_dcm=cfg.model.n_dcm,
        include_pseudotensors=cfg.model.include_pseudotensors,
    )


def get_loss_weight(schedule: str, start: float, end: float, iteration: int, total: int) -> float:
    """
    Calculate loss weight based on schedule.
    
    Args:
        schedule: Schedule type ('linear', 'inverse', 'constant')
        start: Starting weight
        end: Ending weight
        iteration: Current iteration (0-indexed)
        total: Total number of iterations
    
    Returns:
        Weight value
    """
    if schedule == "constant":
        return start
    elif schedule == "linear":
        # Linear interpolation from start to end
        alpha = (iteration + 1) / total
        return start + (end - start) * alpha
    elif schedule == "inverse":
        # Inverse schedule: start / (iteration + 1)
        return start / (iteration + 1)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def print_batch_info(batch: Dict, model: MessagePassingModel) -> None:
    """Print information about a batch for debugging."""
    print("\nBatch information:")
    for key in ['mono', 'esp', 'vdw_surface', 'n_grid', 'N', 'R', 'Z']:
        if key in batch:
            print(f"  {key}: {batch[key].shape}")
    
    if 'mono' in batch and 'N' in batch:
        print(f"\n  mono values (first molecule): {batch['mono'][0]}")
        print(f"  N values: {batch['N']}")
        print(f"  n_grid values: {batch['n_grid']}")


def setup_wandb(cfg: DictConfig, run_name: Optional[str] = None):
    """Setup Weights & Biases logging if enabled."""
    if not cfg.wandb.enabled:
        return None
    
    try:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=dict(cfg),
            name=run_name,
        )
        return wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping experiment tracking.")
        return None


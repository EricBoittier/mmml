"""
Helper functions for training PhysNetJAX models in Jupyter notebooks.

Provides notebook-friendly wrappers around the training pipeline that accept
dictionaries or dataclasses instead of argparse.Namespace objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import jax
import numpy as np

from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.training import train_model
from mmml.physnetjax.physnetjax.data.data import prepare_datasets


def to_jsonable(obj: Any):
    """Recursively convert JAX/NumPy objects to JSON-serializable types."""
    try:
        import jax
        jax_array_cls = getattr(jax, "Array", None)
        jax_array_types = (jax_array_cls,) if jax_array_cls is not None else tuple()
    except Exception:
        jax_array_types = tuple()

    if isinstance(obj, (np.ndarray,)) or (jax_array_types and isinstance(obj, jax_array_types)):
        return np.asarray(obj).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass
class TrainingConfig:
    """Configuration for training PhysNetJAX models."""
    # Data
    data: Union[str, list]  # Path(s) to NPZ file(s)
    n_train: int = 1000
    n_valid: int = 100
    num_atoms: int = 20
    
    # Model
    model: Optional[str] = None  # Path to model JSON file, or None to create new
    features: int = 64
    max_degree: int = 0
    num_basis_functions: int = 32
    num_iterations: int = 2
    n_res: int = 2
    cutoff: float = 8.0
    max_atomic_number: int = 28
    zbl: bool = False
    efa: bool = False
    
    # Training
    seed: int = 42
    batch_size: int = 1
    num_epochs: int = 100
    learning_rate: float = 0.001
    energy_weight: float = 1.0
    forces_weight: float = 52.91
    dipole_weight: float = 27.21
    charges_weight: float = 14.39
    objective: str = "valid_loss"
    restart: Optional[str] = None
    
    # Output
    tag: str = "run"
    ckpt_dir: Optional[Path] = None
    save_model: bool = True
    save_params: bool = True
    
    # Data preprocessing
    clean: bool = False
    esp_mask: bool = False
    clip_esp: bool = False
    subtract_atom_energies: bool = False
    subtract_mean: bool = False
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> TrainingConfig:
        """Create from dictionary."""
        # Handle Path objects
        if 'ckpt_dir' in config_dict and isinstance(config_dict['ckpt_dir'], str):
            config_dict['ckpt_dir'] = Path(config_dict['ckpt_dir'])
        return cls(**config_dict)


def load_model_from_json(model_file: Union[str, Path]) -> EF:
    """Load a model from a JSON file."""
    with open(model_file, 'r') as f:
        model_attrs = json.load(f)
    return EF(**model_attrs)


def prepare_training(
    config: Union[TrainingConfig, Dict[str, Any]],
    return_model: bool = True,
    return_data: bool = True,
) -> Dict[str, Any]:
    """
    Prepare training setup from configuration.
    
    This function processes the training configuration and prepares:
    - Random keys for data shuffling and training
    - Training and validation datasets
    - Model (either loaded from file or created new)
    
    Parameters
    ----------
    config : TrainingConfig or dict
        Training configuration. If dict, will be converted to TrainingConfig.
    return_model : bool, default=True
        Whether to return the model in the result dictionary.
    return_data : bool, default=True
        Whether to return train/valid data in the result dictionary.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'data_key': JAX random key for data shuffling
        - 'train_key': JAX random key for training
        - 'train_data': Training data dictionary (if return_data=True)
        - 'valid_data': Validation data dictionary (if return_data=True)
        - 'model': EF model instance (if return_model=True)
        - 'config': TrainingConfig object
        - 'model_attrs': Model attributes dictionary (if return_model=True)
        
    Examples
    --------
    >>> config = TrainingConfig(
    ...     data="path/to/data.npz",
    ...     n_train=1000,
    ...     n_valid=100,
    ...     num_atoms=20
    ... )
    >>> setup = prepare_training(config)
    >>> train_data = setup['train_data']
    >>> model = setup['model']
    
    Or with a dictionary:
    >>> config_dict = {
    ...     "data": "path/to/data.npz",
    ...     "n_train": 1000,
    ...     "n_valid": 100,
    ...     "num_atoms": 20
    ... }
    >>> setup = prepare_training(config_dict)
    """
    # Convert dict to TrainingConfig if needed
    if isinstance(config, dict):
        config = TrainingConfig.from_dict(config)
    
    # Generate random keys
    seed = config.seed
    data_key, train_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    
    result = {
        'data_key': data_key,
        'train_key': train_key,
        'config': config,
    }
    
    # Prepare datasets
    if return_data:
        files = [config.data] if isinstance(config.data, str) else config.data
        train_data, valid_data = prepare_datasets(
            data_key,
            train_size=config.n_train,
            valid_size=config.n_valid,
            files=files,
            natoms=config.num_atoms,
            clean=config.clean,
            esp_mask=config.esp_mask,
            clip_esp=config.clip_esp,
            verbose=config.verbose,
            subtract_atom_energies=config.subtract_atom_energies,
            subtract_mean=config.subtract_mean,
        )
        result['train_data'] = train_data
        result['valid_data'] = valid_data
    
    # Load or create model
    if return_model:
        if config.model is not None:
            model = load_model_from_json(config.model)
        else:
            model = EF(
                features=config.features,
                max_degree=config.max_degree,
                num_basis_functions=config.num_basis_functions,
                num_iterations=config.num_iterations,
                n_res=config.n_res,
                cutoff=config.cutoff,
                max_atomic_number=config.max_atomic_number,
                zbl=config.zbl,
                efa=config.efa,
            )
        
        result['model'] = model
        result['model_attrs'] = model.return_attributes()
        
        # Optionally save model
        if config.save_model and config.model is None:
            model_file = f"{config.tag}_model.json"
            with open(model_file, 'w') as f:
                json.dump(result['model_attrs'], f, default=to_jsonable, indent=2)
            result['model_file'] = model_file
    
    return result


def run_training(
    config: Union[TrainingConfig, Dict[str, Any]],
    setup: Optional[Dict[str, Any]] = None,
    **override_kwargs
) -> Dict[str, Any]:
    """
    Run training with the given configuration.
    
    This function prepares the training setup (if not provided) and runs
    the training loop. Returns the final EMA parameters and training results.
    
    Parameters
    ----------
    config : TrainingConfig or dict
        Training configuration.
    setup : dict, optional
        Pre-computed setup from prepare_training(). If None, will be computed.
    **override_kwargs
        Additional keyword arguments to override config values.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'params': Final EMA parameters from training
        - 'setup': Training setup dictionary
        - 'config': TrainingConfig object
        
    Examples
    --------
    >>> config = TrainingConfig(
    ...     data="path/to/data.npz",
    ...     n_train=1000,
    ...     num_epochs=10
    ... )
    >>> results = run_training(config)
    >>> final_params = results['params']
    
    Or prepare setup first, then train:
    >>> setup = prepare_training(config)
    >>> results = run_training(config, setup=setup)
    """
    # Convert dict to TrainingConfig if needed
    if isinstance(config, dict):
        config = TrainingConfig.from_dict(config)
    
    # Override config with kwargs
    if override_kwargs:
        config_dict = config.to_dict()
        config_dict.update(override_kwargs)
        config = TrainingConfig.from_dict(config_dict)
    
    # Prepare setup if not provided
    if setup is None:
        setup = prepare_training(config, return_model=True, return_data=True)
    
    # Extract components
    train_key = setup['train_key']
    model = setup['model']
    train_data = setup['train_data']
    valid_data = setup['valid_data']
    
    # Set checkpoint directory
    ckpt_dir = config.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = Path("checkpoints")
    
    # Run training
    params_out = train_model(
        train_key,
        model,
        train_data,
        valid_data,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_atoms=config.num_atoms,
        energy_weight=config.energy_weight,
        forces_weight=config.forces_weight,
        dipole_weight=config.dipole_weight,
        charges_weight=config.charges_weight,
        restart=config.restart,
        conversion={'energy': 1, 'forces': 1},
        print_freq=1,
        name=config.tag,
        best=False,
        optimizer=None,
        transform=None,
        schedule_fn=None,
        objective=config.objective,
        ckpt_dir=ckpt_dir,
        log_tb=False,
        batch_method="default",
        batch_args_dict=None,
        data_keys=('R', 'Z', 'F', 'N', 'E', 'D', 'batch_segments'),
        num_epochs=config.num_epochs,
    )
    
    result = {
        'params': params_out,
        'setup': setup,
        'config': config,
    }
    
    # Optionally save parameters
    if config.save_params:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        params_file = f"{config.tag}_params_{now}.json"
        with open(params_file, 'w') as f:
            json.dump(to_jsonable(params_out), f, indent=2)
        result['params_file'] = params_file
    
    return result


"""
Model checkpointing utilities for saving and loading model parameters and configurations.

This module provides functions to save model parameters (JAX PyTrees) and configurations
as JSON files, enabling easy model persistence and reloading.

Example Usage:
    ```python
    from mmml.utils.model_checkpoint import save_model_checkpoint, load_model_checkpoint
    from mmml.physnetjax.physnetjax.models.model import EF
    
    # Save a model checkpoint
    model = EF(features=64, cutoff=8.0)
    params = model.init(key, R, Z)
    
    save_model_checkpoint(
        params=params,
        model=model,
        save_dir="my_checkpoint",
        metadata={'epoch': 100, 'loss': 0.001}
    )
    
    # Load a model checkpoint
    checkpoint = load_model_checkpoint("my_checkpoint")
    params = checkpoint['params']
    config = checkpoint['config']
    
    # Create model from checkpoint
    from mmml.utils.model_checkpoint import create_model_from_checkpoint
    model, params, config = create_model_from_checkpoint(
        checkpoint_dir="my_checkpoint",
        model_class=EF
    )
    ```
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import pickle

import jax
import numpy as np


def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert JAX/NumPy objects to JSON-serializable types.
    
    Parameters
    ----------
    obj : Any
        Object to convert (may contain JAX arrays, NumPy arrays, etc.)
        
    Returns
    -------
    Any
        JSON-serializable representation
    """
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


def extract_model_config(model: Any) -> Dict[str, Any]:
    """
    Extract configuration from a model object.
    
    Supports multiple model types:
    - Models with `return_attributes()` method (PhysNet EF models)
    - Models with config attributes (DCMNet, Joint models)
    - Dataclass-based configs with `to_dict()` method
    
    Parameters
    ----------
    model : Any
        Model object to extract config from
        
    Returns
    -------
    Dict[str, Any]
        Model configuration dictionary
    """
    config = {}
    
    # Try return_attributes() method (PhysNet EF models)
    if hasattr(model, 'return_attributes'):
        try:
            attrs = model.return_attributes()
            if isinstance(attrs, dict):
                config.update(attrs)
            else:
                # If it's a dataclass or similar, try to convert
                if hasattr(attrs, '__dict__'):
                    config.update(attrs.__dict__)
        except Exception:
            pass
    
    # Try config attributes (DCMNet, Joint models)
    for attr_name in ['physnet_config', 'dcmnet_config', 'noneq_config', 'config']:
        if hasattr(model, attr_name):
            attr = getattr(model, attr_name)
            if isinstance(attr, dict):
                config[attr_name] = attr
            elif hasattr(attr, 'to_dict'):
                config[attr_name] = attr.to_dict()
            elif hasattr(attr, '__dict__'):
                config[attr_name] = attr.__dict__
    
    # Try other common attributes
    for attr_name in ['mix_coulomb_energy', 'cutoff', 'features', 'max_degree', 
                      'num_iterations', 'num_basis_functions', 'max_atomic_number']:
        if hasattr(model, attr_name):
            value = getattr(model, attr_name)
            if value is not None:
                config[attr_name] = to_jsonable(value)
    
    # If model is a dataclass or has to_dict method
    if hasattr(model, 'to_dict'):
        try:
            model_dict = model.to_dict()
            if isinstance(model_dict, dict):
                config.update(model_dict)
        except Exception:
            pass
    
    # Fallback: try __dict__
    if not config and hasattr(model, '__dict__'):
        config = {k: to_jsonable(v) for k, v in model.__dict__.items() 
                 if not k.startswith('_')}
    
    return config


def save_model_checkpoint(
    params: Any,
    model: Any,
    save_dir: Union[str, Path],
    config: Optional[Union[Dict[str, Any], Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_orbax: bool = False,
) -> Dict[str, Path]:
    """
    Save model parameters and configuration to disk.
    
    Parameters
    ----------
    params : Any
        Model parameters (JAX PyTree)
    model : Any
        Model object (used to extract config if not provided)
    save_dir : Union[str, Path]
        Directory to save checkpoint files
    config : Optional[Union[Dict[str, Any], Any]], optional
        Model configuration. If None, will be extracted from model.
        Can be a dict or an object with to_dict() method.
    metadata : Optional[Dict[str, Any]], optional
        Additional metadata to save (e.g., epoch, loss, etc.)
    use_orbax : bool, default=False
        If True, use orbax for parameter saving (better for large models).
        If False, use pickle (more compatible).
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping file types to saved paths:
        - 'params': Path to parameters file
        - 'config': Path to config JSON file
        - 'metadata': Path to metadata JSON file (if provided)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Extract or prepare config
    if config is None:
        config_dict = extract_model_config(model)
    elif hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        # Try to convert config object
        config_dict = extract_model_config(config)
    
    # Add any additional model attributes
    if config_dict is None or not config_dict:
        config_dict = extract_model_config(model)
    
    # Save parameters
    if use_orbax:
        try:
            from orbax.checkpoint import Checkpointer, PyTreeCheckpointer
            checkpointer = PyTreeCheckpointer()
            params_path = save_dir / "params"
            checkpointer.save(params_path, params)
            saved_paths['params'] = params_path
        except ImportError:
            print("Warning: orbax not available, falling back to pickle")
            use_orbax = False
    
    if not use_orbax:
        params_path = save_dir / "params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        saved_paths['params'] = params_path
    
    # Save config as JSON
    config_path = save_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(to_jsonable(config_dict), f, indent=2)
    saved_paths['config'] = config_path
    
    # Save metadata if provided
    if metadata:
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(to_jsonable(metadata), f, indent=2)
        saved_paths['metadata'] = metadata_path
    
    print(f"✓ Saved model checkpoint to {save_dir}")
    print(f"  - Parameters: {saved_paths['params']}")
    print(f"  - Config: {saved_paths['config']}")
    if metadata:
        print(f"  - Metadata: {saved_paths['metadata']}")
    
    return saved_paths


def load_model_checkpoint(
    checkpoint_dir: Union[str, Path],
    load_params: bool = True,
    load_config: bool = True,
    load_metadata: bool = True,
    use_orbax: bool = False,
) -> Dict[str, Any]:
    """
    Load model checkpoint from disk.
    
    Parameters
    ----------
    checkpoint_dir : Union[str, Path]
        Directory containing checkpoint files
    load_params : bool, default=True
        Whether to load parameters
    load_config : bool, default=True
        Whether to load configuration
    load_metadata : bool, default=True
        Whether to load metadata
    use_orbax : bool, default=False
        If True, use orbax for parameter loading
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'params': Model parameters (if load_params=True)
        - 'config': Model configuration dict (if load_config=True)
        - 'metadata': Metadata dict (if load_metadata=True and exists)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    result = {}
    
    # Load parameters
    if load_params:
        if use_orbax:
            try:
                from orbax.checkpoint import PyTreeCheckpointer
                checkpointer = PyTreeCheckpointer()
                params_path = checkpoint_dir / "params"
                if params_path.exists():
                    result['params'] = checkpointer.restore(params_path)
                else:
                    raise FileNotFoundError(f"Parameters not found at {params_path}")
            except ImportError:
                print("Warning: orbax not available, trying pickle")
                use_orbax = False
        
        if not use_orbax:
            # Try JSON file first (preferred for portability)
            json_params_path = checkpoint_dir / "params.json"
            if json_params_path.exists():
                with open(json_params_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Extract params if checkpoint is a dict
                if isinstance(checkpoint_data, dict):
                    if 'params' in checkpoint_data:
                        result['params'] = checkpoint_data['params']
                    elif 'ema_params' in checkpoint_data:
                        result['params'] = checkpoint_data['ema_params']
                    else:
                        result['params'] = checkpoint_data
                else:
                    result['params'] = checkpoint_data
            else:
                # Fall back to pickle file
                params_path = checkpoint_dir / "params.pkl"
                if not params_path.exists():
                    # Try common alternative names
                    alternatives = [
                        checkpoint_dir / "best_params.pkl",
                        checkpoint_dir / "checkpoint_latest.pkl",
                        checkpoint_dir / "checkpoint_best.pkl",
                    ]
                    for alt in alternatives:
                        if alt.exists():
                            params_path = alt
                            break
                    else:
                        raise FileNotFoundError(
                            f"Parameters not found in {checkpoint_dir}. "
                            f"Tried: params.json, params.pkl, and alternatives."
                        )
                
                with open(params_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    
                # Extract params if checkpoint is a dict
                if isinstance(checkpoint_data, dict):
                    if 'params' in checkpoint_data:
                        result['params'] = checkpoint_data['params']
                    elif 'ema_params' in checkpoint_data:
                        result['params'] = checkpoint_data['ema_params']
                    else:
                        result['params'] = checkpoint_data
                else:
                    result['params'] = checkpoint_data
    
    # Load config
    if load_config:
        config_path = checkpoint_dir / "model_config.json"
        if not config_path.exists():
            # Try pickle version
            config_pkl_path = checkpoint_dir / "model_config.pkl"
            if config_pkl_path.exists():
                with open(config_pkl_path, 'rb') as f:
                    result['config'] = pickle.load(f)
            else:
                print(f"Warning: Config not found at {config_path}")
        else:
            with open(config_path, 'r') as f:
                result['config'] = json.load(f)
    
    # Load metadata
    if load_metadata:
        metadata_path = checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                result['metadata'] = json.load(f)
    
    print(f"✓ Loaded checkpoint from {checkpoint_dir}")
    return result


def create_model_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    model_class: Any,
    **model_kwargs
) -> tuple:
    """
    Create a model instance and load parameters from checkpoint.
    
    Parameters
    ----------
    checkpoint_dir : Union[str, Path]
        Directory containing checkpoint files
    model_class : Any
        Model class to instantiate (e.g., EF, JointPhysNetDCMNet)
    **model_kwargs
        Additional keyword arguments to pass to model constructor.
        If config is loaded from checkpoint, it will override these.
        
    Returns
    -------
    tuple
        (model_instance, params, config_dict)
    """
    checkpoint = load_model_checkpoint(checkpoint_dir)
    
    # Get config
    config_dict = checkpoint.get('config', {})
    
    # Merge config with provided kwargs (kwargs take precedence)
    model_config = {**config_dict, **model_kwargs}
    
    # Create model instance
    # Filter out non-model attributes
    model_attrs = ['features', 'max_degree', 'num_iterations', 'num_basis_functions',
                   'cutoff', 'max_atomic_number', 'n_res', 'zbl', 'efa', 'charges',
                   'natoms', 'total_charge', 'n_dcm', 'include_pseudotensors']
    
    filtered_config = {k: v for k, v in model_config.items() 
                      if k in model_attrs or k.endswith('_config')}
    
    model = model_class(**filtered_config)
    
    params = checkpoint.get('params')
    
    return model, params, config_dict


# Convenience function for quick save
def quick_save(
    params: Any,
    model: Any,
    save_path: Union[str, Path],
    **kwargs
) -> Dict[str, Path]:
    """
    Quick save function with sensible defaults.
    
    Parameters
    ----------
    params : Any
        Model parameters
    model : Any
        Model object
    save_path : Union[str, Path]
        Path to save directory or file
    **kwargs
        Additional arguments passed to save_model_checkpoint
        
    Returns
    -------
    Dict[str, Path]
        Dictionary of saved file paths
    """
    save_path = Path(save_path)
    
    # If it's a file path, use parent directory
    if save_path.suffix:
        save_dir = save_path.parent
    else:
        save_dir = save_path
    
    return save_model_checkpoint(params, model, save_dir, **kwargs)


# Convenience function for quick load
def quick_load(
    checkpoint_path: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Quick load function with sensible defaults.
    
    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Path to checkpoint directory or file
    **kwargs
        Additional arguments passed to load_model_checkpoint
        
    Returns
    -------
    Dict[str, Any]
        Loaded checkpoint dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    
    # If it's a file, use parent directory
    if checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = checkpoint_path
    
    return load_model_checkpoint(checkpoint_dir, **kwargs)


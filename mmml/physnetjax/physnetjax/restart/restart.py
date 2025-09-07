"""
Training restart utilities for PhysNetJax.

This module provides functions for saving and loading training checkpoints,
allowing training to be resumed from previous states.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import orbax
import orbax.checkpoint

from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.utils.pretty_printer import print_dict_as_table
from mmml.physnetjax.physnetjax.utils.utils import get_files

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def get_last(path: str) -> Path:
    """
    Get the last checkpoint directory.
    
    Parameters
    ----------
    path : str
        Path to checkpoint directory
        
    Returns
    -------
    Path
        Path to the most recent checkpoint directory
    """
    dirs = get_files(path)
    if "tmp" in str(dirs[-1]):
        dirs.pop()
    return dirs[-1]


def get_params_model(restart: str, natoms: int = None):
    """
    Load parameters and model from checkpoint.
    
    Parameters
    ----------
    restart : str
        Path to checkpoint directory
    natoms : int, optional
        Number of atoms to set in model, by default None
        
    Returns
    -------
    tuple
        Tuple of (parameters, model)
    """
    restored = orbax_checkpointer.restore(restart)
    # print(f"Restoring from {restart}")
    modification_time = os.path.getmtime(restart)
    modification_date = datetime.fromtimestamp(modification_time)

    params = restored["params"]
    print(restored["model"].keys())
    if "model_attributes" not in restored.keys():
        return params, None

    # kwargs = _process_model_attributes(restored["model_attributes"], natoms)
    kwargs = restored["model_attributes"]
    # print(kwargs)
    model = EF(**kwargs)
    model.natoms = natoms
    model.zbl = bool(kwargs["zbl"]) if "zbl" in kwargs.keys() else False

    print_dict_as_table(kwargs, title="Model Attributes", plot=True)
    restart_dict = {
        "Checkpoint": restart,
        "name": Path(restart).name,
        "epoch": restored["epoch"],
        "best_loss": restored["best_loss"],
        "Save Time": modification_date,
    }
    print_dict_as_table(restart_dict, title="Last Checkpoint", plot=True)

    # print(model)
    return params, model


def restart_training(restart: str, transform, optimizer, num_atoms: int):
    """
    Restart training from a previous checkpoint.

    Loads model parameters, optimizer state, and training configuration
    from a checkpoint to resume training.

    Parameters
    ----------
    restart : str
        Path to the checkpoint directory
    transform : optax.GradientTransformation
        Transform for learning rate scaling
    optimizer : optax.GradientTransformation
        Optimizer to use
    num_atoms : int
        Number of atoms in the system
        
    Returns
    -------
    tuple
        Tuple containing:
        - ema_params: EMA parameters
        - model: Model instance
        - opt_state: Optimizer state
        - params: Model parameters
        - transform_state: Transform state
        - step: Current training step
        - best_loss: Best loss achieved
        - CKPT_DIR: Checkpoint directory
        - state: Training state
    """
    restart = get_last(restart)
    _, _model = get_params_model(restart, num_atoms)
    if _model is not None:
        model = _model

    restored = orbax_checkpointer.restore(restart)
    print("Restoring from", restart)
    print("Restored keys:", restored.keys())
    state = restored["model"]
    # print(state)
    params = _  # restored["params"]
    ema_params = restored["ema_params"]
    # opt_state = restored["opt_state"]
    # print("Opt state", opt_state)
    transform_state = transform.init(params)
    # transform_state = restored["transform_state"]
    # Validate and reinitialize states if necessary
    opt_state = optimizer.init(params)
    # update mu
    # o_a, o_b = opt_state
    # from optax import ScaleByAmsgradState
    # _ = ScaleByAmsgradState(
    #     mu=opt_state[1][0]["mu"],
    #     nu=opt_state[1][0]["nu"],
    #     nu_max=opt_state[1][0]["nu_max"],
    #     count=opt_state[1][0]["count"],
    # )
    # # opt_state = (o_a, (_, o_b[1]))
    # Set training variables
    step = restored["epoch"] + 1
    best_loss = restored["best_loss"]
    print(f"Training resumed from step {step - 1}, best_loss {best_loss}")
    CKPT_DIR = Path(restart).parent
    return (
        ema_params,
        model,
        opt_state,
        params,
        transform_state,
        step,
        best_loss,
        CKPT_DIR,
        state,
    )


def get_params_model_with_ase(pkl_path, model_path, atoms):
    """
    Load parameters and model from pickle files with ASE atoms.
    
    Parameters
    ----------
    pkl_path : str
        Path to parameters pickle file
    model_path : str
        Path to model configuration pickle file
    atoms : ase.Atoms
        ASE atoms object
        
    Returns
    -------
    tuple
        Tuple of (parameters, model)
    """
    import pandas as pd

    from physnetjax.models.model import EF
    from physnetjax.utils.utils import _process_model_attributes

    params = pd.read_pickle(pkl_path)
    model_kwargs = pd.read_pickle(model_path)
    print(model_kwargs)
    model_kwargs = _process_model_attributes(model_kwargs)
    model_kwargs["natoms"] = len(atoms)
    model = EF(**model_kwargs)
    print(model)
    return params, model

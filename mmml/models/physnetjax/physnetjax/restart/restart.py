"""
Training restart utilities for PhysNetJax.

This module provides functions for saving and loading training checkpoints,
allowing training to be resumed from previous states.
"""

import os
from datetime import datetime
from pathlib import Path

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import orbax
import orbax.checkpoint

from mmml.models.physnetjax.physnetjax.utils.pretty_printer import print_dict_as_table
from mmml.models.physnetjax.physnetjax.utils.utils import get_files

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def _safe_float(val, default=float("inf")):
    if val is None:
        return default
    try:
        return float(np.asarray(val))
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    if val is None:
        return default
    try:
        return int(np.asarray(val))
    except (ValueError, TypeError):
        return default


def save_training_checkpoint(
    path: Path | str,
    ckpt: dict,
    *,
    checkpointer: orbax.checkpoint.PyTreeCheckpointer | None = None,
) -> None:
    """Save one training epoch checkpoint.

    Uses ``force=True`` when supported so a retry or duplicate save to the same
  epoch path does not fail with "Destination ... already exists".
    """
    import shutil

    from flax.training import orbax_utils

    ckp = Path(path)
    ckpt_checkpointer = checkpointer or orbax_checkpointer
    save_args = orbax_utils.save_args_from_target(ckpt)
    try:
        ckpt_checkpointer.save(ckp, ckpt, save_args=save_args, force=True)
    except TypeError:
        # Older orbax without force= on PyTreeCheckpointer.save
        if ckp.exists():
            shutil.rmtree(ckp)
        ckpt_checkpointer.save(ckp, ckpt, save_args=save_args)


def _merge_params(init_params, loaded_params):
    """
    Merge loaded params with init params, filling in any keys missing from loaded.
    Used when loading checkpoints that lack newer submodules (e.g. repulsion).
    Prefers loaded values when both exist (e.g. trained repulsion params).
    """
    if not isinstance(loaded_params, dict):
        return loaded_params  # leaf: prefer loaded (checkpoint) values
    if not isinstance(init_params, dict):
        return loaded_params
    result = {}
    for k in init_params:
        if k not in loaded_params:
            result[k] = init_params[k]
        else:
            result[k] = _merge_params(init_params[k], loaded_params[k])
    return result


def _is_params_json(path: Path) -> bool:
    """True for a portable PhysNet params JSON file (``params.json`` / ``params_*.json``)."""
    return path.is_file() and path.suffix == ".json"


def _find_latest_params_json(directory: Path) -> Path | None:
    """Pick newest ``params.json`` / ``params_*.json`` under *directory*."""
    if not directory.is_dir():
        return None
    candidates = [
        p
        for p in directory.glob("params*.json")
        if p.is_file() and (p.name == "params.json" or p.name.startswith("params_"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def get_last(path: str) -> Path:
    """
    Resolve a restart path to a concrete checkpoint.

    ``path`` may be:

    * an experiment root with ``epoch-*/`` Orbax dirs (latest by name);
    * a flat Orbax checkpoint directory (``manifest.ocdbt`` / metadata);
    * a portable ``params.json`` / ``params_*.json`` file;
    * a directory containing only such JSON files (newest by mtime).

    Parameters
    ----------
    path : str
        Path to checkpoint directory or portable JSON file

    Returns
    -------
    Path
        Path to the most recent checkpoint directory or JSON file
    """
    p = Path(path).expanduser()
    if _is_params_json(p):
        return p.resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint path not found: '{path}'. "
            "Pass an Orbax run dir (with epoch-*/), a flat Orbax checkpoint, "
            "or a portable params_*.json file."
        )
    if (p / "manifest.ocdbt").exists() or (p / "_CHECKPOINT_METADATA").exists():
        return p
    dirs = get_files(str(p))
    # get_files already drops names containing "tmp"; keep a name-only
    # guard here for callers that pass a pre-filtered list edge case.
    while dirs and "tmp" in dirs[-1].name:
        dirs.pop()
    if dirs:
        return dirs[-1]
    json_ckpt = _find_latest_params_json(p)
    if json_ckpt is not None:
        return json_ckpt.resolve()
    raise FileNotFoundError(
        f"No checkpoint epochs (epoch-*/) or portable params*.json found in '{path}'. "
        "Cannot restart training without an existing checkpoint. "
        "Pass --restart path/to/params_TAG_TIMESTAMP.json to resume from a "
        "portable JSON exported at the end of training."
    )


def _restore_json_checkpoint(restart: Path) -> dict:
    """Load a portable ``params_*.json`` into an Orbax-like restored dict."""
    from mmml.utils.model_checkpoint import (
        json_to_params,
        normalize_flax_params_for_apply,
    )

    loaded = json_to_params(restart, backend="jax")
    raw_params = loaded.get("params")
    if raw_params is None:
        raise ValueError(f"JSON checkpoint missing 'params' key: {restart}")
    params = normalize_flax_params_for_apply(raw_params, backend="jax")
    config = loaded.get("config")
    if not isinstance(config, dict):
        config = loaded.get("model_attributes")
    if not isinstance(config, dict) or not config:
        raise ValueError(
            f"JSON checkpoint {restart} has no 'config' / 'model_attributes'. "
            "Portable exports from mmml make-training include config; "
            "re-export with orbax_to_json(..., config=...) if needed."
        )
    meta = loaded.get("metadata") if isinstance(loaded.get("metadata"), dict) else {}
    return {
        "params": params,
        "ema_params": params,
        "model_attributes": dict(config),
        "epoch": meta.get("epoch", 0),
        "best_loss": meta.get("best_loss"),
        "metadata": meta,
        "_checkpoint_format": "json",
        "_json_path": str(restart.resolve()),
    }


def get_params_model(
    restart: str,
    natoms: int = None,
    return_everything: bool = False,
    *,
    quiet: bool = False,
    return_meta: bool = False,
    prefer_ema: bool = True,
):
    """
    Load parameters and model from checkpoint.

    Parameters
    ----------
    restart : str
        Path to Orbax checkpoint directory or portable ``params_*.json``
    natoms : int, optional
        Number of atoms to set in model, by default None
    return_everything : bool, optional
        Whether to return everything from the checkpoint, by default False
    prefer_ema : bool, optional
        Use the checkpoint's ``ema_params`` instead of the live ``params``,
        by default True. Live params can swing several-fold between adjacent
        epochs in extrapolation regions the loss never visits; EMA smooths
        that out. Falls back to ``params`` when ``ema_params`` is absent.
        Portable JSON exports already store EMA weights under ``params``.

    Returns
    -------
    tuple
        Tuple of (parameters, model)
    """
    from mmml.utils.model_checkpoint import _restore_pytree_cpu_safe

    restart_path = Path(restart)
    if _is_params_json(restart_path):
        restored = _restore_json_checkpoint(restart_path)
    else:
        restored = _restore_pytree_cpu_safe(orbax_checkpointer, str(restart))
    # print(f"Restoring from {restart}")
    modification_time = os.path.getmtime(restart)
    modification_date = datetime.fromtimestamp(modification_time)

    params = None
    if prefer_ema:
        params = restored.get("ema_params")

    if params is None:
        params = restored.get("params")
    if params is None and "model" in restored:
        model_state = restored["model"]
        if hasattr(model_state, "params"):
            params = model_state.params
        elif isinstance(model_state, dict) and "params" in model_state:
            params = model_state["params"]

    if params is None:
        params = restored.get("ema_params")

    if not quiet and "model" in restored:
        model_state = restored["model"]
        if hasattr(model_state, "keys"):
            print(model_state.keys())
        elif isinstance(model_state, dict):
            print(model_state.keys())

    if "model_attributes" not in restored.keys():
        if return_everything:
            return params, None, restored
        if return_meta:
            return params, None, None
        return params, None

    # kwargs = _process_model_attributes(restored["model_attributes"], natoms)
    kwargs = restored["model_attributes"]
    from mmml.utils.model_checkpoint import build_physnet_from_config

    model = build_physnet_from_config(
        kwargs,
        max_padded_atoms=natoms if natoms is not None else kwargs.get("natoms"),
    )
    if natoms is not None:
        model.max_padded_atoms = natoms
    model.zbl = bool(kwargs["zbl"]) if "zbl" in kwargs.keys() else False

    checkpoint_meta = {
        "Checkpoint": str(restart),
        "name": Path(restart).name,
        "epoch": _safe_int(restored.get("epoch")),
        "best_loss": _safe_float(restored.get("best_loss")),
        "Save Time": modification_date,
        "format": restored.get("_checkpoint_format", "orbax"),
    }
    if not quiet:
        print_dict_as_table(kwargs, title="Model Attributes", plot=True)
        print_dict_as_table(checkpoint_meta, title="Last Checkpoint", plot=True)

    # Fill missing params (e.g. repulsion) from old checkpoints that lack newer submodules
    if model.zbl:
        n = natoms if natoms is not None else getattr(model, "natoms", 10) or 10
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n)
        init_kwargs = dict(
            atomic_numbers=jnp.ones(n, dtype=jnp.int32),
            positions=jnp.zeros((n, 3)),
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        if type(model).__name__ == "SpookyPhysNet":
            # SpookyPhysNet.__call__ additionally requires charges/spins;
            # values are irrelevant here (only used to discover init_params'
            # tree structure for _merge_params), so a neutral singlet dummy
            # matches the shape convention used at real inference call sites.
            init_kwargs["charges"] = jnp.zeros((n, 1))
            init_kwargs["spins"] = jnp.ones((n, 1))
        init_params = model.init(jax.random.PRNGKey(0), **init_kwargs)
        params = _merge_params(init_params, params)

    if return_everything:
        return params, model, restored
    if return_meta:
        return params, model, checkpoint_meta
    # print(model)
    return params, model


def restart_training(restart: str, transform, optimizer, num_atoms: int):
    """
    Restart training from a previous checkpoint.

    Loads model parameters, optimizer state, and training configuration
    from a checkpoint to resume training. Accepts Orbax ``epoch-*/`` run
    directories and portable ``params_*.json`` files (optimizer state is
    re-initialized for JSON / when Orbax opt state is incompatible).

    Parameters
    ----------
    restart : str
        Path to the checkpoint directory or portable JSON file
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
    params, model, restored = get_params_model(
        restart, num_atoms, return_everything=True, quiet=True
    )
    if model is None:
        raise ValueError(
            f"Could not rebuild model from checkpoint {restart}. "
            "Orbax checkpoints need model_attributes; JSON needs a 'config' key."
        )

    print("Restoring from", restart)
    print("Restored keys:", list(restored.keys()))
    state = restored.get("model")
    ema_params = restored.get("ema_params", params)
    if ema_params is None:
        ema_params = params
    transform_state = transform.init(params)
    # Validate and reinitialize states if necessary
    opt_state = optimizer.init(params)
    # Set training variables
    step = _safe_int(restored.get("epoch")) + 1
    if step < 1:
        step = 1
    best_loss = _safe_float(restored.get("best_loss"))
    print(f"Training resumed from step {step - 1}, best_loss {best_loss:.6f}")
    # New Orbax epoch-* dirs go next to the JSON / under the run root.
    CKPT_DIR = Path(restart).parent.resolve()
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

    from physnetjax.utils.utils import _process_model_attributes

    from mmml.utils.model_checkpoint import build_physnet_from_config

    params = pd.read_pickle(pkl_path)
    model_kwargs = pd.read_pickle(model_path)
    print(model_kwargs)
    model_kwargs = _process_model_attributes(model_kwargs)
    model = build_physnet_from_config(model_kwargs, max_padded_atoms=len(atoms))
    print(model)
    return params, model

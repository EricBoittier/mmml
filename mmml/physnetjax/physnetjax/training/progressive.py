"""
Progressive model growth training for PhysNetJAX.

Trains a sequence of increasingly large models, transplanting learned
parameters from each stage into the next.  Growth order: features first,
then max_degree.  Convergence at each stage is detected via plateau
(no improvement for ``growth_patience`` epochs).

Usage
-----
>>> from mmml.physnetjax.physnetjax.training.progressive import train_model_progressive
>>> growth_stages = [
...     {"features": 16, "max_degree": 0},
...     {"features": 32, "max_degree": 0},
...     {"features": 64, "max_degree": 1},
...     {"features": 128, "max_degree": 3},
... ]
>>> ema_params, best_loss = train_model_progressive(
...     key, train_data, valid_data,
...     growth_stages=growth_stages,
...     base_model_kwargs={"natoms": 34, "cutoff": 6.0, "zbl": True},
...     growth_patience=50,
...     num_epochs=500,            # per-stage epoch budget
...     batch_size=32,
...     batch_method="default",
... )
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import e3x
import jax
import jax.numpy as jnp
import numpy as np

from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.training import train_model


# ---------------------------------------------------------------------------
# Parameter transplanting
# ---------------------------------------------------------------------------


def transplant_params(
    old_params: dict,
    new_params: dict,
) -> dict:
    """
    Transplant learned parameters from a smaller model into a larger one.

    Recursively walks two Flax parameter pytrees (nested dicts).  For each
    matching leaf:

    * If shapes are identical the old value is copied directly.
    * If the old array is *smaller* along one or more axes the old values are
      copied into the corresponding slice of the new (randomly initialised)
      array.  The remaining entries keep their fresh initialisation, giving
      the new capacity room to learn.
    * Keys present in ``new_params`` but absent in ``old_params`` are kept
      as-is (fresh init).

    Parameters
    ----------
    old_params : dict
        Parameter pytree from the previous (smaller) model.
    new_params : dict
        Parameter pytree from the new (larger) model, already randomly
        initialised.

    Returns
    -------
    dict
        A new pytree with the same structure as *new_params* but with
        values transplanted from *old_params* wherever possible.
    """
    if not isinstance(new_params, dict):
        # Leaf node -- attempt to transplant the array.
        return _transplant_leaf(old_params, new_params)

    result = {}
    for key in new_params:
        if key not in old_params:
            # New key (e.g. a submodule added by a larger model) -- keep init.
            result[key] = new_params[key]
        else:
            result[key] = transplant_params(old_params[key], new_params[key])
    return result


def _transplant_leaf(old_arr, new_arr):
    """
    Copy *old_arr* into the top-left corner of *new_arr*.

    Works for any combination of shapes: if an axis is smaller in
    *old_arr* only the first ``old_size`` elements along that axis are
    overwritten.  If *old_arr* is larger along any axis we truncate
    (should not happen in a grow-only schedule, but handled for safety).

    Non-array leaves (scalars, ints) are returned from old directly when
    shapes match, otherwise the new value is kept.
    """
    old_arr = np.asarray(old_arr)
    new_arr = np.asarray(new_arr)

    if old_arr.shape == new_arr.shape:
        return old_arr

    if old_arr.ndim != new_arr.ndim:
        # Dimensionality mismatch -- keep new init (can happen with
        # fundamentally different layer structures).
        return new_arr

    # Build the slice for each axis: min(old, new) elements.
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_arr.shape, new_arr.shape))
    out = new_arr.copy()
    out[slices] = old_arr[slices]
    return out


# ---------------------------------------------------------------------------
# Progressive training wrapper
# ---------------------------------------------------------------------------


def train_model_progressive(
    key,
    train_data: dict,
    valid_data: dict,
    growth_stages: List[Dict[str, Any]],
    base_model_kwargs: Dict[str, Any],
    growth_patience: int = 50,
    num_epochs: int = 500,
    name: str = "progressive",
    resume_stage: int = 0,
    resume_checkpoint: str = None,
    verbose: bool = True,
    **train_kwargs,
):
    """
    Progressively grow and train a PhysNetJAX model.

    Starting from the smallest configuration in *growth_stages*, this
    function trains each model until the objective plateaus (no
    improvement for *growth_patience* epochs), then creates a larger
    model, transplants the learned parameters, and continues.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    train_data : dict
        Training data (keys R, Z, F, E, N, ...).
    valid_data : dict
        Validation data.
    growth_stages : list of dict
        Each dict overrides entries in *base_model_kwargs* for that stage.
        Example::

            [
                {"features": 16, "max_degree": 0},
                {"features": 32, "max_degree": 0},
                {"features": 64, "max_degree": 1},
                {"features": 128, "max_degree": 3},
            ]

    base_model_kwargs : dict
        Shared model keyword arguments (``natoms``, ``cutoff``,
        ``max_atomic_number``, ``zbl``, ``charges``, etc.).
    growth_patience : int, optional
        Epochs without improvement before growing to the next stage.
        Default 50.
    num_epochs : int, optional
        Maximum epochs *per stage*.  Default 500.
    name : str, optional
        Base experiment name for checkpointing.  Each stage appends a
        suffix like ``-stage0``, ``-stage1``, etc.
    resume_stage : int, optional
        Stage index to resume from.  Stages before this index are
        skipped.  Use together with *resume_checkpoint* to supply
        the learned parameters from the last completed stage.
        Default 0 (start from the beginning).
    resume_checkpoint : str, optional
        Path to an orbax checkpoint directory from a previously
        completed stage.  The ``ema_params`` are loaded and used as
        ``prev_ema_params`` so that the *resume_stage* can transplant
        from them.  Ignored when *resume_stage* is 0.
    verbose : bool, optional
        Print stage transition information.
    **train_kwargs
        Additional keyword arguments forwarded to ``train_model()``
        (``learning_rate``, ``batch_size``, ``batch_method``,
        ``energy_weight``, ``forces_weight``, ``data_keys``, etc.).

    Returns
    -------
    tuple
        ``(ema_params, best_loss)`` from the final stage.
    """
    prev_ema_params = None
    best_loss = float("inf")
    num_atoms = base_model_kwargs.get("natoms", 60)

    # ------------------------------------------------------------------
    # Resume: load params from a completed stage's checkpoint
    # ------------------------------------------------------------------
    if resume_stage > 0 and resume_checkpoint is not None:
        from mmml.physnetjax.physnetjax.restart.restart import (
            get_last,
            get_params_model,
        )
        ckpt_path = get_last(resume_checkpoint)
        params_loaded, _ = get_params_model(ckpt_path, num_atoms)
        prev_ema_params = params_loaded
        if verbose:
            print(f"Resuming from stage {resume_stage}, loaded params from "
                  f"{ckpt_path}")
    elif resume_stage > 0:
        raise ValueError(
            f"resume_stage={resume_stage} but no resume_checkpoint provided. "
            "Pass the checkpoint path from the last completed stage."
        )

    for stage_idx, stage_overrides in enumerate(growth_stages):
        # Skip already-completed stages when resuming
        if stage_idx < resume_stage:
            if verbose:
                print(f"Skipping stage {stage_idx} (already completed)")
            continue

        # -----------------------------------------------------------------
        # Build model for this stage
        # -----------------------------------------------------------------
        model_kwargs = {**base_model_kwargs, **stage_overrides}
        model = EF(**model_kwargs)

        stage_name = f"{name}-stage{stage_idx}"
        if verbose:
            print("=" * 70)
            print(f"PROGRESSIVE STAGE {stage_idx}/{len(growth_stages) - 1}: "
                  f"features={model.features}, max_degree={model.max_degree}, "
                  f"num_iterations={model.num_iterations}")
            print("=" * 70)

        # -----------------------------------------------------------------
        # Initialise parameters for the new model
        # -----------------------------------------------------------------
        key, init_key = jax.random.split(key)
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
        new_params = model.init(
            init_key,
            atomic_numbers=train_data["Z"][0],
            positions=train_data["R"][0],
            dst_idx=dst_idx,
            src_idx=src_idx,
        )

        # -----------------------------------------------------------------
        # Transplant previous parameters into new model
        # -----------------------------------------------------------------
        if prev_ema_params is not None:
            transplanted = transplant_params(prev_ema_params, new_params)
            n_transplanted, n_total = _count_transplanted(
                prev_ema_params, new_params, transplanted
            )
            if verbose:
                print(f"Transplanted {n_transplanted}/{n_total} parameter arrays "
                      f"from previous stage")
        else:
            transplanted = None

        # -----------------------------------------------------------------
        # Train this stage
        # -----------------------------------------------------------------
        key, train_key = jax.random.split(key)

        # Build train_model kwargs; let caller override everything except
        # the progressive-specific ones we control.
        tm_kwargs = dict(train_kwargs)
        # Never forward ``restart`` -- each stage must start fresh (warm-
        # started via init_params when transplanted params are available).
        tm_kwargs.pop("restart", None)
        tm_kwargs.update(
            num_epochs=num_epochs,
            num_atoms=num_atoms,
            name=stage_name,
            early_stop_patience=growth_patience,
        )

        ema_params, stage_best_loss = train_model(
            key=train_key,
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            init_params=transplanted,
            **tm_kwargs,
        )

        if verbose:
            print(f"Stage {stage_idx} finished: best {tm_kwargs.get('objective', 'valid_forces_mae')} "
                  f"= {float(stage_best_loss):.6f}")

        prev_ema_params = ema_params
        best_loss = stage_best_loss

    return ema_params, best_loss



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_transplanted(old_params, new_params, transplanted, _prefix=""):
    """Count how many leaf arrays were transplanted vs total."""
    n_transplanted = 0
    n_total = 0

    if not isinstance(new_params, dict):
        n_total = 1
        old_arr = np.asarray(old_params)
        new_arr = np.asarray(new_params)
        tra_arr = np.asarray(transplanted)
        # If transplanted differs from new_params init, we transplanted something
        if old_arr.shape == new_arr.shape or (old_arr.ndim == new_arr.ndim):
            n_transplanted = 1
        return n_transplanted, n_total

    for key in new_params:
        if key in old_params:
            t, n = _count_transplanted(
                old_params[key], new_params[key], transplanted[key],
                _prefix=f"{_prefix}/{key}"
            )
            n_transplanted += t
            n_total += n
        else:
            # Count leaves in new_params[key] that were NOT transplanted
            n_total += _count_leaves(new_params[key])

    return n_transplanted, n_total


def _count_leaves(pytree):
    """Count leaf arrays in a pytree."""
    if not isinstance(pytree, dict):
        return 1
    return sum(_count_leaves(v) for v in pytree.values())

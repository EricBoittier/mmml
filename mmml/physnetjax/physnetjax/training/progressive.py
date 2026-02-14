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

    for stage_idx, stage_overrides in enumerate(growth_stages):
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
        tm_kwargs.update(
            num_epochs=num_epochs,
            num_atoms=num_atoms,
            name=stage_name,
            early_stop_patience=growth_patience,
        )

        # If we have transplanted params, we need to inject them. Since
        # train_model initialises params internally, we pass the model and
        # let train_model do its own init.  We then rely on the restart /
        # _merge_params path.  However, the cleanest approach is to pass
        # pre-initialised params via a restart-like mechanism.
        #
        # For simplicity we use a direct approach: let train_model init
        # as usual, then the progressive wrapper re-initialises the
        # optimizer with transplanted params.  We achieve this by passing
        # restart=False and doing the transplant inside train_model's
        # param init.  Since we can't do that without modifying train_model
        # further, we use a pragmatic alternative: save transplanted params
        # to a temporary orbax checkpoint and pass it as restart.
        #
        # ACTUALLY the simplest correct approach: since train_model inits
        # params at line 326 and then optionally overwrites via restart,
        # we can monkey-patch the init result.  But that's fragile.
        #
        # Instead we use the most robust approach: call train_model as
        # normal (it will init fresh params) and rely on the fact that
        # _merge_params(init, loaded) prefers loaded.  We accomplish this
        # by wrapping in a thin helper that patches params after init.
        #
        # The cleanest way: pass transplanted as initial params by
        # temporarily setting restart to inject them.  Since this is
        # complex, let's just call train_model with a small wrapper.

        ema_params, stage_best_loss = _train_stage(
            key=train_key,
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            transplanted_params=transplanted,
            **tm_kwargs,
        )

        if verbose:
            print(f"Stage {stage_idx} finished: best {tm_kwargs.get('objective', 'valid_forces_mae')} "
                  f"= {stage_best_loss:.6f}")

        prev_ema_params = ema_params
        best_loss = stage_best_loss

    return ema_params, best_loss


def _train_stage(
    key,
    model,
    train_data,
    valid_data,
    transplanted_params=None,
    **train_kwargs,
):
    """
    Train a single stage, optionally warm-starting from transplanted params.

    This wraps ``train_model`` and, if *transplanted_params* is provided,
    injects them by temporarily saving to an orbax checkpoint that
    ``train_model`` can load via its restart mechanism.

    For the case where transplanted_params is None, it simply calls
    train_model directly.
    """
    import tempfile
    import warnings
    from pathlib import Path

    from flax.training import orbax_utils, train_state
    from mmml.physnetjax.physnetjax.restart.restart import orbax_checkpointer

    if transplanted_params is None:
        return train_model(
            key=key,
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            **train_kwargs,
        )

    # Save transplanted params to a temporary checkpoint so that
    # train_model's restart logic picks them up.
    num_atoms = train_kwargs.get("num_atoms", model.natoms)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    key, init_key = jax.random.split(key)
    init_params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )

    # Use transplanted params as both params and ema_params
    from mmml.physnetjax.physnetjax.training.training import _merge_params
    merged_params = _merge_params(init_params, transplanted_params)

    optimizer_obj = train_kwargs.get("optimizer", None)
    from mmml.physnetjax.physnetjax.training.optimizer import get_optimizer
    opt, tfm, sched, _ = get_optimizer(
        learning_rate=train_kwargs.get("learning_rate", 0.001),
        schedule_fn=train_kwargs.get("schedule_fn", None),
        optimizer=optimizer_obj,
        transform=train_kwargs.get("transform", None),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=merged_params, tx=opt
    )

    ckpt = {
        "model": state,
        "model_attributes": model.return_attributes(),
        "transform_state": tfm.init(merged_params),
        "ema_params": merged_params,
        "params": merged_params,
        "epoch": 0,
        "opt_state": opt.init(merged_params),
        "best_loss": float("inf"),
        "lr_eff": 0.001,
        "objectives": {},
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="progressive_ckpt_"))
    ckpt_path = tmp_dir / "epoch-0"

    save_args = orbax_utils.save_args_from_target(ckpt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        orbax_checkpointer.save(ckpt_path, ckpt, save_args=save_args)

    # Train with restart pointing to the temp checkpoint
    result = train_model(
        key=key,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        restart=str(tmp_dir),
        **train_kwargs,
    )

    # Clean up temp checkpoint
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


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

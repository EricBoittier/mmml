from contextlib import nullcontext
import gc
import logging
import time
import uuid
import warnings
from pathlib import Path

import ase.units
import e3x
import jax
from flax.training import train_state

# Try to enable lovely_jax for better array printing (optional; may fail with lovely-numpy>=0.2.19)
try:
    import lovely_jax as lj
    lj.monkey_patch()
except ImportError:
    lj = None  # type: ignore[assignment]
from rich.console import Console
from rich.live import Live

# Suppress asyncio warnings from Jupyter/IPython kernel and Orbax checkpointing
# These are harmless but noisy when running in Jupyter notebooks
logging.getLogger("asyncio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Task was destroyed.*")

from mmml.models.physnetjax.physnetjax.data.data import print_shapes
from mmml.models.physnetjax.physnetjax.directories import BASE_CKPT_DIR, print_paths
from mmml.models.physnetjax.physnetjax.restart.restart import (
    restart_training,
    save_training_checkpoint,
)
from mmml.data.units import TRAINING_UNITS
from mmml.models.physnetjax.physnetjax.training.distill import parse_distill_targets
from mmml.models.physnetjax.physnetjax.training.evalstep import eval_step
from mmml.models.physnetjax.physnetjax.training.optimizer import (
    base_optimizer,
    base_schedule_fn,
    base_transform,
    get_optimizer,
)
from mmml.models.physnetjax.physnetjax.training.trainstep import train_step
from mmml.models.physnetjax.physnetjax.training.validation import validate_atomic_numbers
from mmml.models.physnetjax.physnetjax.utils.ascii import computer 
from mmml.models.physnetjax.physnetjax.utils.pretty_printer import (
    Printer,
    print_dict_as_table,
)

PROFILE = False
PROFILE_EPOCH_TIMING = False
if PROFILE:
    import jax.profiler

schedule_fn = base_schedule_fn
transform = base_transform
optimizer = base_optimizer

# Energy/force unit conversions
CONVERSION = {
    "energy": 1 / (ase.units.kcal / ase.units.mol),
    "forces": 1 / (ase.units.kcal / ase.units.mol),
}

# Explicit eV -> kcal/mol for the streaming per-epoch line, which must state a
# unit that does not depend on whatever `conversion` the caller passed.
_EV_TO_KCAL_MOL = 1 / (ase.units.kcal / ase.units.mol)

def is_valid_advanced_batch_config(batch_args_dict):
    """
    Check if batch arguments dictionary has valid advanced batching configuration.
    
    Parameters
    ----------
    batch_args_dict : dict
        Dictionary containing batch configuration parameters
        
    Returns
    -------
    bool
        True if the configuration is valid for advanced batching
    """
    return (
        isinstance(batch_args_dict, dict)
        and "batch_shape" in batch_args_dict
        and "batch_nbl_len" in batch_args_dict
    )


def _merge_params(init_params, loaded_params):
    """
    Merge loaded params with init params, filling in any keys missing from loaded.
    Used when restarting from checkpoints that lack newer submodules (e.g. repulsion).
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


def train_model(
    key,
    model,
    train_data,
    valid_data,
    num_epochs=1,
    learning_rate=0.001,
    energy_weight=1.0,
    forces_weight=52.91,
    dipole_weight=27.21,
    charges_weight=14.39,
    batch_size=1,
    num_atoms=60,
    restart=False,
    conversion=CONVERSION,
    print_freq=1,
    name="test",
    best=False,
    optimizer=None,
    transform=None,
    schedule_fn=None,
    # Global-norm gradient clip. None keeps get_optimizer's default (10.0),
    # which is loose enough that most steps pass through unclipped; lower it
    # (~1.0) when the raw params oscillate instead of converging.
    clip_global=None,
    objective="valid_forces_mae",
    ckpt_dir=BASE_CKPT_DIR,
    log_tb=False,
    batch_method=None,
    batch_args_dict=None,
    data_keys=("R", "Z", "F", "E", "N",  "D", "dst_idx", "src_idx", "batch_segments"),
    early_stop_patience=None,
    init_params=None,
    rot_augment: bool = False,
    rot_perturbation: float = 1.0,
    save_every_epoch: bool = True,
    profile_epoch_timing: bool | None = None,
    teacher_params=None,
    distill_alpha: float = 1.0,
    distill_targets=None,
    ema_decay: float = 0.999,
    hybrid_mm=None,
):
    """
    Train a PhysNetJax model with comprehensive logging and checkpointing.
    
    This function implements the main training loop for PhysNetJax models,
    including data batching, optimization, validation, and checkpointing.
    Supports both standard energy/force prediction
    and charge/dipole prediction modes.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for initialization and shuffling
    model : physnetjax.models.model.EF
        PhysNetJax model instance
    train_data : dict
        Training data dictionary
    valid_data : dict
        Validation data dictionary
    num_epochs : int, optional
        Number of training epochs, by default 1
    learning_rate : float, optional
        Learning rate, by default 0.001
    energy_weight : float, optional
        Weight for energy loss, by default 1.0
    forces_weight : float, optional
        Weight for forces loss, by default 52.91
    dipole_weight : float, optional
        Weight for dipole loss, by default 27.21
    charges_weight : float, optional
        Weight for charges loss, by default 14.39
    batch_size : int, optional
        Batch size, by default 1
    num_atoms : int, optional
        Maximum number of atoms per molecule, by default 60
    restart : bool | str, optional
        Whether to restart from checkpoint, by default False
    conversion : dict, optional
        Display-only multipliers for reported energy/force MAE (not applied to
        NPZ data or loss). Keys ``energy`` and ``forces``; default identity.
    print_freq : int, optional
        Frequency of progress printing, by default 1
    name : str, optional
        Experiment name for checkpointing, by default "test"
    best : bool, optional
        Whether to save checkpoints when the objective improves, by default False
    save_every_epoch : bool, optional
        If True, write an orbax checkpoint at the end of every epoch (in addition
        to any best-epoch saves when ``best=True``), by default False
    optimizer : optax.GradientTransformation | str | None, optional
        Optimizer or string identifier, by default None
    transform : optax.GradientTransformation | str | None, optional
        Transform or string identifier, by default None
    schedule_fn : optax.Schedule | str | None, optional
        Learning rate schedule, by default None
    objective : str, optional
        Objective metric for best model selection by early stopping, by default "valid_forces_mae"
        options: "valid_forces_mae", "valid_energy_mae", 
        "valid_loss", "train_forces_mae", "train_energy_mae", "train_loss", "lr"
    ckpt_dir : pathlib.Path, optional
        Checkpoint directory, by default BASE_CKPT_DIR
    log_tb : bool, optional
        Deprecated and ignored. TensorBoard logging has been removed.
    batch_method : str | None, optional
        Batching method ("advanced" or None), by default None
    batch_args_dict : dict | None, optional
        Additional batch arguments, by default None
    data_keys : tuple, optional
        Keys for data dictionary, by default ("R", "Z", "F", "E", "D", "dst_idx", "src_idx", "batch_segments")
    early_stop_patience : int | None, optional
        If set, stop training early when the objective has not improved for
        this many consecutive epochs.  None disables early stopping (default).
    init_params : dict | None, optional
        If provided, use these parameters instead of freshly initialised ones.
        Useful for warm-starting from transplanted parameters (progressive
        training).  The optimizer and EMA are initialised from these params.
        Ignored when ``restart`` is set.
    hybrid_mm : HybridMMConfig | dict | None, optional
        When set, train on the hybrid ML/MM total the MD calculator
        evaluates: ``E = (1 - s) * (E_A + E_B) + s * E_AB + E_MM``, where the
        taper ``s(r_com)`` applies to the dimer *interaction* only.  A
        ``mmml.models.hybrid_energy.HybridMMConfig`` (master LJ tables +
        switching widths); a plain dict is coerced to one.  Requires the CGenFF
        per-atom fields in the batch (see ``HYBRID_MM_BATCH_KEYS``).
    ema_decay : float, optional
        Decay for the exponential moving average of parameters, by default
        0.999.  Validation, checkpointing and restart all use the EMA weights,
        so this affects the saved model.  Set to ``0.0`` to disable EMA
        entirely (``ema_params`` then tracks the raw parameters exactly).

    Returns
    -------
    tuple
        (ema_params, best_loss) -- final EMA parameters and the best
        objective value achieved during training.
        
    Notes
    -----
    The training process includes:
    - Data batching (advanced or default)
    - Model initialization or checkpoint restoration
    - Training loop with gradient updates
    - Validation after each epoch
    - Checkpointing of best models and/or every epoch (``save_every_epoch``)
    - Progress monitoring with rich console output
    """
    _ = log_tb  # Deprecated argument retained for backward compatibility.

    # Freeze the hybrid settings here, outside the jit boundary: they are a
    # static argument (a dict is unhashable and its bools would trace).
    from mmml.models.hybrid_energy import HybridMMConfig

    hybrid_mm = HybridMMConfig.coerce(hybrid_mm)
    if profile_epoch_timing is None:
        import os

        profile_epoch_timing = PROFILE_EPOCH_TIMING or bool(
            os.environ.get("MMML_PHYSNET_PROFILE_EPOCH_TIMING")
        )
    from mmml.models.physnetjax.physnetjax.training.epoch_timing import (
        EpochTiming,
        EpochTimingSummary,
    )

    timing_summary = EpochTimingSummary()
    data_keys = tuple(data_keys)
    validate_atomic_numbers(
        train_data=train_data,
        valid_data=valid_data,
        model_max_atomic_number=model.max_atomic_number,
    )

    print_shapes(train_data, name="Train Data")
    print_shapes(valid_data, name="Validation Data")

    if batch_method is None:
        raise ValueError("batch_method must be specified")

    # Decide batching method
    if batch_method == "advanced" and is_valid_advanced_batch_config(batch_args_dict):
        print("Using append batching method")
        from physnetjax.data.batches import prepare_batches_advanced_minibatching
        def _prepare_batches(x):
            return prepare_batches_advanced_minibatching(
                x["key"],
                x["data"],
                x["batch_size"],
                x["batch_shape"],
                x["batch_nbl_len"],
                num_atoms=x["num_atoms"],
                data_keys=x["data_keys"],
            )
    else:
        print("Using default (fat) batching method")
        import sys
        sys.stdout.flush()  # Flush for SLURM logging
        from mmml.models.physnetjax.physnetjax.data.batches import (
            _pair_indices,
            _prepare_batches,
        )

        fat_pair_cache = _pair_indices(num_atoms, batch_size)

        def _prepare_batches_default(
            shuffle_key,
            *,
            data,
            batch_size,
            num_atoms,
            data_keys,
            rot_augment,
            rot_perturbation,
        ):
            return _prepare_batches(
                shuffle_key,
                data=data,
                batch_size=batch_size,
                num_atoms=num_atoms,
                data_keys=data_keys,
                rot_augment=rot_augment,
                rot_perturbation=rot_perturbation,
                pair_cache=fat_pair_cache,
            )

    # Force terminal output for SLURM environments
    import sys
    console = Console(
        width=250,  # Wide enough for all columns
        force_terminal=True,  # Force color output in SLURM
        force_interactive=False,  # Better for log files
    )
    sys.stdout.flush()  # Ensure console initialization is logged

    if console is not None:
        console.print("Training Routine")
        console.print(computer)
        print_paths()

    start_time = time.time()
    if not isinstance(model.debug, list):
        console.print(
            "Start Time: ", time.strftime("%H:%M:%S", time.gmtime(start_time))
        )

    best_loss = float("inf") if (best or save_every_epoch) else None
    # Snapshot the CLI/constructed model's charge-head flag.  Restart may
    # rebuild ``model`` from checkpoint ``model_attributes``, which can disagree
    # (e.g. YAML ``charges: true`` while restarting a charges=False hybrid run).
    # ``do_charges`` must follow the *restored* model — the loss reads
    # ``output["sum_charges"]``, which is None without a charge head.
    cli_charges = bool(getattr(model, "charges", False))
    do_charges = cli_charges
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)

    optimizer, transform, schedule_fn, optimizer_kwargs = get_optimizer(
        learning_rate=learning_rate,
        schedule_fn=schedule_fn,
        optimizer=optimizer,
        transform=transform,
        **({} if clip_global is None else {"clip_global": float(clip_global)}),
    )

    train_params_dict = {
        "energy_weight": energy_weight,
        "forces_weight": forces_weight,
        "dipole_weight": dipole_weight,
        "charges_weight": charges_weight,
        "batch_size": batch_size,
        "num_atoms": num_atoms,
        "rot_augment": rot_augment,
        "rot_perturbation": rot_perturbation,
        "training_units": dict(TRAINING_UNITS),
    }
    if batch_method == "advanced":
        train_params_dict.update(batch_args_dict)
    training_style_dict = {
        "restart": restart,
        "best": best,
        "save_every_epoch": save_every_epoch,
        "data_keys": data_keys,
        "objective": objective,
        "distill": teacher_params is not None and distill_alpha < 1.0,
        "distill_alpha": distill_alpha,
        "distill_targets": distill_targets,
    }
    distill_energy, distill_forces, distill_dipole = parse_distill_targets(distill_targets)
    do_distill = teacher_params is not None and distill_alpha < 1.0

    if console is not None:
        print_dict_as_table(optimizer_kwargs, title="Optimizer Arguments", plot=True)
        print_dict_as_table(train_params_dict, title="Training Parameters", plot=True)
        print_dict_as_table(training_style_dict, title="Training Style", plot=True)

    # Orbax requires absolute checkpoint paths
    ckpt_dir = Path(ckpt_dir).resolve()
    uuid_ = str(uuid.uuid4())
    CKPT_DIR = ckpt_dir / f"{name}-{uuid_}"
    if not restart:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if hybrid_mm is not None:
        # Persist Mode A/C metadata next to the run so MD can warn on mismatch.
        import json

        from mmml.models.mm_charge_mode import hybrid_mm_metadata_dict
        from mmml.models.mm_lj_scales import (
            cgenff_type_names_from_prm,
            mm_lj_scales_metadata,
        )

        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        _meta = hybrid_mm_metadata_dict(hybrid_mm)
        if bool(getattr(hybrid_mm, "learn_mm_lj_scales", False)):
            try:
                _names = cgenff_type_names_from_prm()
                if len(_names) == len(hybrid_mm.master_sigmas):
                    _meta.update(
                        mm_lj_scales_metadata(
                            learn_mm_lj_scales=True,
                            type_names=_names,
                            sigma_bounds=hybrid_mm.mm_lj_sigma_scale_bounds,
                            epsilon_bounds=hybrid_mm.mm_lj_epsilon_scale_bounds,
                            trainable_mask=hybrid_mm.mm_lj_trainable_mask,
                            type_frame_counts=hybrid_mm.mm_lj_type_frame_counts,
                        )
                    )
            except Exception as exc:  # pragma: no cover - PRM missing in some envs
                print(f"WARNING: could not resolve CGenFF type names: {exc}", flush=True)
        _meta_path = CKPT_DIR / "hybrid_mm.json"
        with open(_meta_path, "w") as _mf:
            json.dump(_meta, _mf, indent=2)
            _mf.write("\n")
        print(f"Wrote hybrid MM metadata to {_meta_path}", flush=True)

    # Batches for the validation set need to be prepared only once.
    key, valid_shuffle_key = jax.random.split(key)
    kwargs = {
        "key": valid_shuffle_key,
        "data": valid_data,
        "batch_size": batch_size,
        "num_atoms": num_atoms,
        "data_keys": data_keys,
        "rot_augment": rot_augment,
        "rot_perturbation": rot_perturbation,
    }
    if batch_method == "advanced":
        kwargs.update(batch_args_dict)
        valid_batches = _prepare_batches(kwargs)
    else:
        valid_batches = _prepare_batches_default(
            valid_shuffle_key,
            data=valid_data,
            batch_size=batch_size,
            num_atoms=num_atoms,
            data_keys=data_keys,
            rot_augment=rot_augment,
            rot_perturbation=rot_perturbation,
        )

    print_shapes(valid_batches[0], name="Validation Batch[0]")

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    fresh_params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    if hybrid_mm is not None and bool(getattr(hybrid_mm, "learn_mm_lj_scales", False)):
        from mmml.models.mm_lj_scales import attach_mm_lj_scales

        fresh_params = attach_mm_lj_scales(fresh_params, len(hybrid_mm.master_sigmas))
    # Use caller-supplied params (e.g. transplanted from a previous stage)
    # when available, falling back to fresh random init.
    if init_params is not None and not restart:
        params = _merge_params(fresh_params, init_params)
    else:
        params = fresh_params

    # load from restart
    if restart:
        (
            ema_params,
            model,
            opt_state,
            params,
            transform_state,
            step,
            best_loss,
            CKPT_DIR,
            state,
        ) = restart_training(restart, transform, optimizer, num_atoms)
        # Fill missing params (e.g. repulsion) from old checkpoints that lack newer submodules
        fresh_restart_params = model.init(
            init_key,
            atomic_numbers=train_data["Z"][0],
            positions=train_data["R"][0],
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        if hybrid_mm is not None and bool(getattr(hybrid_mm, "learn_mm_lj_scales", False)):
            from mmml.models.mm_lj_scales import attach_mm_lj_scales

            fresh_restart_params = attach_mm_lj_scales(
                fresh_restart_params, len(hybrid_mm.master_sigmas)
            )
        params = _merge_params(fresh_restart_params, params)
        ema_params = _merge_params(fresh_restart_params, ema_params)
        do_charges = bool(getattr(model, "charges", False))
        if do_charges != cli_charges:
            print(
                f"WARNING: restart checkpoint has charges={do_charges} but this run "
                f"was constructed with charges={cli_charges}. Using the checkpoint "
                f"architecture (doCharges={do_charges}). For a charge head / "
                f"--mm-charge-correction, start a fresh run (no --restart) with "
                f"charges=true; you cannot graft a charge head onto a "
                f"charges=false checkpoint by flipping the YAML flag.",
                flush=True,
            )
        if hybrid_mm is not None and not do_charges:
            from mmml.models.mm_charge_mode import (
                mm_charge_mode_needs_q_ml,
                resolve_hybrid_mm_charge_mode,
            )

            _mode = resolve_hybrid_mm_charge_mode(
                mm_charge_mode=getattr(hybrid_mm, "mm_charge_mode", None),
                charge_correction=bool(
                    getattr(hybrid_mm, "charge_correction", False)
                ),
            )
            if mm_charge_mode_needs_q_ml(_mode):
                raise ValueError(
                    f"hybrid_mm.mm_charge_mode={_mode.value} requires a model with "
                    "a charge head, but the restart checkpoint has charges=False. "
                    "Start a fresh run with charges=true and the same "
                    "mm_charge_mode (omit --restart / restart: from the "
                    "charges=false hybrid checkpoint)."
                )
    # initialize
    else:
        ema_params = params
        step = 1
        opt_state = optimizer.init(params)
        transform_state = transform.init(params)
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )

    if hybrid_mm is not None and bool(getattr(hybrid_mm, "learn_mm_lj_scales", False)):
        print(
            f"Learnable MM LJ scales enabled ({len(hybrid_mm.master_sigmas)} CGenFF types; "
            f"projected each step to sigma {hybrid_mm.mm_lj_sigma_scale_bounds}, "
            f"epsilon {hybrid_mm.mm_lj_epsilon_scale_bounds}; "
            f"{sum(hybrid_mm.mm_lj_trainable_mask or ())} trainable)",
            flush=True,
        )

    if best_loss is None or restart:
        best_loss = float('inf')

    epochs_without_improvement = 0

    train_time1 = time.time()
    epoch_printer = Printer()
    ckp = None
    save_time = None

    model_attributes = model.return_attributes()
    table = print_dict_as_table(model_attributes, title="Model Attributes")
    if console is not None:
        console.print(table)
    print(f"Training loss will use doCharges={do_charges}", flush=True)


    live_context = Live(auto_refresh=False) if console is not None else nullcontext()
    with live_context as live:
        # Train for 'num_epochs' epochs.
        for epoch in range(step, num_epochs + 1):
            epoch_timing = EpochTiming()
            epoch_t0 = time.perf_counter()

            key, epoch_shuffle_key = jax.random.split(key)

            batch_t0 = time.perf_counter()
            kwargs = {
                "key": epoch_shuffle_key,
                "data": train_data,
                "batch_size": batch_size,
                "num_atoms": num_atoms,
                "data_keys": data_keys,
                "rot_augment": rot_augment,
                "rot_perturbation": rot_perturbation,
            }
            if (
                batch_method == "advanced"
                and isinstance(batch_args_dict, dict)
                and "batch_shape" in batch_args_dict
                and "nb_len" in batch_args_dict
            ):
                kwargs.update(batch_args_dict)

            if batch_method == "advanced":
                train_batches = _prepare_batches(kwargs)
            else:
                train_batches = _prepare_batches_default(
                    epoch_shuffle_key,
                    data=train_data,
                    batch_size=batch_size,
                    num_atoms=num_atoms,
                    data_keys=data_keys,
                    rot_augment=rot_augment,
                    rot_perturbation=rot_perturbation,
                )
            epoch_timing.batch_prep_s = time.perf_counter() - batch_t0

            train_t0 = time.perf_counter()
            # NOTE: train_loss below is measured on the raw `params`, while
            # valid_loss is measured on `ema_params` (see the eval loop). The
            # two are therefore NOT comparable, and a large train/valid ratio
            # means the raw weights are oscillating, not that the model is
            # overfitting. On the DES warm start (job 19360535) this read as a
            # 128x "generalization gap" that was purely params-vs-EMA.
            train_loss = 0.0
            train_energy_mae = 0.0
            train_forces_mae = 0.0
            train_dipoles_mae = 0.0
            for i, batch in enumerate(train_batches):
                (
                    params,
                    ema_params,
                    opt_state,
                    transform_state,
                    loss,
                    energy_mae,
                    forces_mae,
                    dipole_mae,
                ) = train_step(
                    model_apply=model.apply,
                    optimizer_update=optimizer.update,
                    transform_state=transform_state,
                    batch=batch,
                    batch_size=batch_size,
                    energy_weight=energy_weight,
                    forces_weight=forces_weight,
                    dipole_weight=dipole_weight,
                    charges_weight=charges_weight,
                    opt_state=opt_state,
                    doCharges=do_charges,
                    params=params,
                    ema_params=ema_params,
                    ema_decay=ema_decay,
                    hybrid_mm=hybrid_mm,
                    teacher_params=teacher_params,
                    distill_alpha=distill_alpha,
                    doDistill=do_distill,
                    distill_energy=distill_energy,
                    distill_forces=distill_forces,
                    distill_dipole=distill_dipole,
                    debug=True,
                )
                # Block until JAX operations complete to avoid async context issues
                # This prevents RuntimeError: cannot enter context in IPython/Jupyter
                train_loss += (loss - train_loss) / (i + 1)
                train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
                train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)
                train_dipoles_mae += (dipole_mae - train_dipoles_mae) / (i + 1)
            jax.block_until_ready(loss)
            jax.block_until_ready(params)
            epoch_timing.train_s = time.perf_counter() - train_t0

            valid_t0 = time.perf_counter()
            valid_loss = 0.0
            valid_energy_mae = 0.0
            valid_forces_mae = 0.0
            valid_dipoles_mae = 0.0
            for i, batch in enumerate(valid_batches):
                loss, energy_mae, forces_mae, dipole_mae = eval_step(
                    model_apply=model.apply,
                    batch=batch,
                    batch_size=batch_size,
                    energy_weight=energy_weight,
                    forces_weight=forces_weight,
                    dipole_weight=dipole_weight,
                    charges_weight=charges_weight,
                    charges=do_charges,
                    params=ema_params,
                    hybrid_mm=hybrid_mm,
                )
                # Per-batch sync removed; one sync per validation epoch is enough.
                valid_loss += (loss - valid_loss) / (i + 1)
                valid_energy_mae += (energy_mae - valid_energy_mae) / (i + 1)
                valid_forces_mae += (forces_mae - valid_forces_mae) / (i + 1)
                valid_dipoles_mae += (dipole_mae - valid_dipoles_mae) / (i + 1)
            jax.block_until_ready(valid_loss)
            epoch_timing.valid_s = time.perf_counter() - valid_t0

            _, transform_state = transform.update(
                updates=params, state=transform_state, value=valid_loss
            )

            # Raw (pre-conversion) values, kept so the streaming line below can
            # state kcal/mol unambiguously whatever `conversion` happens to be.
            _raw_valid_e_mae = valid_energy_mae
            _raw_valid_f_mae = valid_forces_mae

            # convert statistics to kcal/mol for printing
            # NB: the CLI passes conversion={'energy':1,'forces':1}, so in practice
            # this is a no-op and the table's MAE columns are eV, not kcal/mol.
            valid_energy_mae *= conversion["energy"]
            valid_forces_mae *= conversion["forces"]
            train_energy_mae *= conversion["energy"]
            train_forces_mae *= conversion["forces"]
            scale = transform_state.scale
            slr = schedule_fn(epoch)
            lr_eff = scale * slr

            train_time = time.time()
            epoch_length = train_time - train_time1
            epoch_length = f"{epoch_length:.2f} s"
            train_time1 = train_time

            obj_res = {
                "valid_energy_mae": valid_energy_mae,
                "valid_forces_mae": valid_forces_mae,
                "train_energy_mae": train_energy_mae,
                "train_forces_mae": train_forces_mae,
                "train_dipole_mae": train_dipoles_mae,
                "valid_dipole_mae": valid_dipoles_mae,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "lr": lr_eff,
                "batch_size": batch_size,
                "energy_w": energy_weight,
                "charges_w": charges_weight,
                "dipole_w": dipole_weight,
                "forces_w": forces_weight,
            }

            ckp = CKPT_DIR / f"epoch-{epoch}"
            save_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))

            improved = obj_res[objective] < best_loss
            if improved:
                best_loss = obj_res[objective]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            should_save = save_every_epoch or (best and improved)
            best_ = improved and best

            if should_save:
                ckpt_t0 = time.perf_counter()
                model_attributes = model.return_attributes()
                from mmml.models.mm_charge_mode import hybrid_mm_metadata_dict

                ckpt = {
                    "model": state,
                    "model_attributes": model_attributes,
                    "transform_state": transform_state,
                    "ema_params": ema_params,
                    "params": params,
                    "epoch": epoch,
                    "opt_state": opt_state,
                    "best_loss": best_loss,
                    "lr_eff": lr_eff,
                    "objectives": obj_res,
                    "training_units": dict(TRAINING_UNITS),
                    "hybrid_mm": hybrid_mm_metadata_dict(hybrid_mm),
                }
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    save_training_checkpoint(ckp, ckpt)
                epoch_timing.checkpoint_s = time.perf_counter() - ckpt_t0

            epoch_timing.other_s = max(
                0.0,
                time.perf_counter() - epoch_t0 - epoch_timing.total_s,
            )
            timing_summary.record(epoch_timing)
            if profile_epoch_timing and console is not None and epoch % print_freq == 0:
                console.print(
                    f"Epoch {epoch} timing (s): "
                    f"batch_prep={epoch_timing.batch_prep_s:.3f}, "
                    f"train={epoch_timing.train_s:.3f}, "
                    f"valid={epoch_timing.valid_s:.3f}, "
                    f"ckpt={epoch_timing.checkpoint_s:.3f}"
                )

            if best_ or (epoch % print_freq == 0) and console is not None:
                combined = epoch_printer.update(
                    epoch,
                    train_loss,
                    valid_loss,
                    best_loss,
                    train_energy_mae,
                    valid_energy_mae,
                    train_forces_mae,
                    valid_forces_mae,
                    do_charges,
                    train_dipoles_mae,
                    valid_dipoles_mae,
                    scale,
                    slr,
                    lr_eff,
                    epoch_length,
                    ckp,
                    save_time,
                )
                live.update(combined, refresh=True)
                import sys
                sys.stdout.flush()  # Force output to SLURM log file

            # Plain one-line-per-epoch record. The rich Live table above is a
            # *live* display: it overwrites in place, so a redirected log keeps
            # only the final render and a multi-hour run shows no progress at all
            # until it exits. (TERM=dumb removes the control codes but not this;
            # Live still renders once.) Units are spelled out because the table's
            # MAE columns are eV, which has been misread as kcal/mol.
            if epoch % print_freq == 0:
                print(
                    f"[epoch {epoch}/{num_epochs}] "
                    f"train_loss={train_loss:.6g} valid_loss={valid_loss:.6g} "
                    f"best={best_loss:.6g} "
                    f"valid_E_MAE={_raw_valid_e_mae * _EV_TO_KCAL_MOL:.4f}kcal/mol "
                    f"valid_F_MAE={_raw_valid_f_mae * _EV_TO_KCAL_MOL:.4f}kcal/mol/A "
                    f"lr={lr_eff:.3g} t={epoch_length}",
                    flush=True,
                )
                sys.stderr.flush()  # Flush errors too
                gc.collect()  # Force garbage collection to prevent memory buildup during long training runs
                if PROFILE:
                    jax.profiler.save_device_memory_profile(f"{save_time}-memory-{epoch}.prof")

            # Early stopping check
            if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
                if console is not None:
                    console.print(
                        f"Early stopping: no improvement for {early_stop_patience} epochs "
                        f"(best {objective}={best_loss:.6f})"
                    )
                break

    if profile_epoch_timing and console is not None and timing_summary.epochs > 0:
        console.print(timing_summary.format_means())

    if hybrid_mm is not None and bool(getattr(hybrid_mm, "learn_mm_lj_scales", False)):
        from mmml.models.mm_lj_scales import (
            MM_LJ_EPSILON_SCALE_KEY,
            MM_LJ_SIGMA_SCALE_KEY,
            cgenff_type_names_from_prm,
            write_mm_lj_scales_into_hybrid_mm_json,
        )

        if (
            isinstance(ema_params, dict)
            and MM_LJ_SIGMA_SCALE_KEY in ema_params
            and MM_LJ_EPSILON_SCALE_KEY in ema_params
        ):
            try:
                _names = cgenff_type_names_from_prm()
            except Exception:
                _names = [f"type_{i}" for i in range(len(hybrid_mm.master_sigmas))]
            if len(_names) != len(hybrid_mm.master_sigmas):
                _names = [f"type_{i}" for i in range(len(hybrid_mm.master_sigmas))]
            write_mm_lj_scales_into_hybrid_mm_json(
                CKPT_DIR / "hybrid_mm.json",
                type_names=_names,
                sigma_scale=ema_params[MM_LJ_SIGMA_SCALE_KEY],
                epsilon_scale=ema_params[MM_LJ_EPSILON_SCALE_KEY],
                sigma_bounds=hybrid_mm.mm_lj_sigma_scale_bounds,
                epsilon_bounds=hybrid_mm.mm_lj_epsilon_scale_bounds,
                trainable_mask=hybrid_mm.mm_lj_trainable_mask,
                type_frame_counts=hybrid_mm.mm_lj_type_frame_counts,
            )
            print(
                f"Wrote final MM LJ scales to {CKPT_DIR / 'hybrid_mm.json'}",
                flush=True,
            )

    # Return final model parameters, best objective value, and run checkpoint dir.
    return ema_params, best_loss, CKPT_DIR

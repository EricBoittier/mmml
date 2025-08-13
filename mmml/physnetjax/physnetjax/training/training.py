from contextlib import nullcontext
import gc
import time
import uuid

import ase.units
import e3x
import jax
import lovely_jax as lj
import tensorflow as tf
from flax.training import orbax_utils, train_state
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from rich.console import Console
from rich.live import Live

from physnetjax.data.data import print_shapes
from physnetjax.directories import BASE_CKPT_DIR, print_paths
from physnetjax.logger.tensorboard_logging import write_tb_log
from physnetjax.restart.restart import orbax_checkpointer, restart_training
from physnetjax.training.evalstep import eval_step
from physnetjax.training.optimizer import (
    base_optimizer,
    base_schedule_fn,
    base_transform,
    get_optimizer,
)
from physnetjax.training.trainstep import train_step
from physnetjax.utils.ascii import computer, cubes
from physnetjax.utils.pretty_printer import (
    Printer,
    pretty_print_optimizer,
    print_dict_as_table,
    training_printer,
)

PROFILE = False
if PROFILE:
    import jax.profiler

lj.monkey_patch()

schedule_fn = base_schedule_fn
transform = base_transform
optimizer = base_optimizer

# Energy/force unit conversions
CONVERSION = {
    "energy": 1 / (ase.units.kcal / ase.units.mol),
    "forces": 1 / (ase.units.kcal / ase.units.mol),
}

def is_valid_advanced_batch_config(batch_args_dict):
    return (
        isinstance(batch_args_dict, dict)
        and "batch_shape" in batch_args_dict
        and "batch_nbl_len" in batch_args_dict
    )

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
    objective="valid_forces_mae",
    ckpt_dir=BASE_CKPT_DIR,
    log_tb=True,
    batch_method=None,
    batch_args_dict=None,
    data_keys=("R", "Z", "F", "E", "D", "dst_idx", "src_idx", "batch_segments"),
):
    """Train a model."""
    data_keys = tuple(data_keys)

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
        from physnetjax.data.batches import _prepare_batches

    console = Console(width=200, color_system="auto")

    if console is not None:
        console.print("Training Routine")
        console.print(computer)
        print_paths()

    start_time = time.time()
    if not isinstance(model.debug, list):
        console.print(
            "Start Time: ", time.strftime("%H:%M:%S", time.gmtime(start_time))
        )

    best_loss = float('inf') if best else None
    do_charges = model.charges
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)

    optimizer, transform, schedule_fn, optimizer_kwargs = get_optimizer(
        learning_rate=learning_rate,
        schedule_fn=schedule_fn,
        optimizer=optimizer,
        transform=transform,
    )

    train_params_dict = {
        "energy_weight": energy_weight,
        "forces_weight": forces_weight,
        "dipole_weight": dipole_weight,
        "charges_weight": charges_weight,
        "batch_size": batch_size,
        "num_atoms": num_atoms,
    }
    if batch_method == "advanced":
        train_params_dict.update(batch_args_dict)
    training_style_dict = {
        "restart": restart,
        "best": best,
        "data_keys": data_keys,
        "objective": objective,
    }

    if console is not None:
        print_dict_as_table(optimizer_kwargs, title="Optimizer Arguments", plot=True)
        print_dict_as_table(train_params_dict, title="Training Parameters", plot=True)
        print_dict_as_table(training_style_dict, title="Training Style", plot=True)

    uuid_ = str(uuid.uuid4())
    CKPT_DIR = ckpt_dir / f"{name}-{uuid_}"

    # Batches for the validation set need to be prepared only once.
    key, valid_shuffle_key = jax.random.split(key)
    key, train_shuffle_key = jax.random.split(key)
    kwargs = {
        "key": valid_shuffle_key,
        "data": valid_data,
        "batch_size": batch_size,
        "num_atoms": num_atoms,
        "data_keys": data_keys,
    }
    if batch_method == "advanced":
        kwargs.update(batch_args_dict)
        valid_batches = _prepare_batches(kwargs)
    else:
        valid_batches = _prepare_batches(key,
                                         data=valid_data,
                                         batch_size=batch_size,
                                         num_atoms=num_atoms,
                                         data_keys=data_keys)

    print_shapes(valid_batches[0], name="Validation Batch[0]")
    jax.debug.print("Extra Validation Info:")
    jax.debug.print("Z: {x}", x=valid_data["Z"])
    jax.debug.print("R: {x}", x=valid_data["R"])
    jax.debug.print("E: {x}", x=valid_data["E"])
    jax.debug.print("N: {x}", x=valid_data["N"])
    jax.debug.print("F: {x}", x=valid_data["F"])
    if model.charges:
        jax.debug.print("D: {x}", x=valid_data["D"])

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
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
    # initialize
    else:
        ema_params = params
        step = 1
        opt_state = optimizer.init(params)
        transform_state = transform.init(params)
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )
    if best_loss is None or restart:
        best_loss = float('inf')

    train_time1 = time.time()
    epoch_printer = Printer()
    ckp = None
    save_time = None

    model_attributes = model.return_attributes()
    table = print_dict_as_table(model_attributes, title="Model Attributes")
    if console is not None:
        console.print(table)


    with (Live(auto_refresh=False) if console is not None \
          else nullcontext() as live):
        # Train for 'num_epochs' epochs.
        for epoch in range(step, num_epochs + 1):
            # Prepare batches.

            kwargs = {
                "key": train_shuffle_key,
                "data": train_data,
                "batch_size": batch_size,
                "num_atoms": num_atoms,
                "data_keys": data_keys,
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
                train_batches = _prepare_batches(key,
                                                 data=train_data,
                                                 batch_size=batch_size,
                                                 num_atoms=num_atoms,
                                                 data_keys=data_keys)
            # Loop over train batches.
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
                    debug=True,
                )
                train_loss += (loss - train_loss) / (i + 1)
                train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
                train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)
                train_dipoles_mae += (dipole_mae - train_dipoles_mae) / (i + 1)
            # Evaluate on validation set.
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
                )
                valid_loss += (loss - valid_loss) / (i + 1)
                valid_energy_mae += (energy_mae - valid_energy_mae) / (i + 1)
                valid_forces_mae += (forces_mae - valid_forces_mae) / (i + 1)
                valid_dipoles_mae += (dipole_mae - valid_dipoles_mae) / (i + 1)

            _, transform_state = transform.update(
                updates=params, state=transform_state, value=valid_loss
            )

            # convert statistics to kcal/mol for printing
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

            if log_tb:
                writer = tf.summary.create_file_writer(str(CKPT_DIR / "tfevents"))
                writer.set_as_default()
                write_tb_log(writer, obj_res, epoch)  # Call your logger function here

            best_ = False

            if obj_res[objective] < best_loss:
                save_time = time.time()
                save_time = time.strftime("%H:%M:%S", time.gmtime(save_time))
                # print("Saving checkpoint at", save_time)
                ckp = CKPT_DIR / f"epoch-{epoch}"
                # update best loss
                best_loss = obj_res[objective]
                model_attributes = model.return_attributes()
                # checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=epoch)
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
                }
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(
                    CKPT_DIR / f"epoch-{epoch}", ckpt, save_args=save_args
                )

                best_ = True

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
                gc.collect()  # Force garbage collection to prevent memory buildup during long training runs
                if PROFILE:
                    jax.profiler.save_device_memory_profile(f"{save_time}-memory-{epoch}.prof")

    # Return final model parameters.
    return ema_params

import functools
import pickle

import e3x
import jax
import jax.numpy as jnp
import optax
from dcmnet.loss import esp_mono_loss

from dcmnet.data import prepare_batches, prepare_datasets

from typing import Callable, Any, Optional
from functools import partial


@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "esp_w", "ndcm"),
)
def train_step(
    model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, ndcm
):
    def loss_fn(params):
        mono, dipo = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
        )
        loss = esp_mono_loss(
            dipo_prediction=dipo,
            mono_prediction=mono,
            vdw_surface=batch["vdw_surface"],
            esp_target=batch["esp"],
            mono=batch["mono"],
            ngrid=batch["n_grid"],
            n_atoms=batch["N"],
            batch_size=batch_size,
            esp_w=esp_w,
            n_dcm=ndcm,
        )
        return loss, (mono, dipo)

    (loss, (mono, dipo)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@functools.partial(
    jax.jit, static_argnames=("model_apply", "batch_size", "esp_w", "ndcm")
)
def eval_step(model_apply, batch, batch_size, params, esp_w, ndcm):
    mono, dipo = model_apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    loss = esp_mono_loss(
        dipo_prediction=dipo,
        mono_prediction=mono,
        vdw_surface=batch["vdw_surface"],
        esp_target=batch["esp"],
        mono=batch["mono"],
        ngrid=batch["n_grid"],
        n_atoms=batch["N"],
        batch_size=batch_size,
        esp_w=esp_w,
        n_dcm=ndcm,
    )
    return loss


def train_model(
    key,
    model,
    train_data,
    valid_data,
    num_epochs,
    learning_rate,
    batch_size,
    writer,
    ndcm,
    esp_w=1.0,
    restart_params=None,
    ema_decay=0.999  
):
    best = 10**7
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data["Z"][0]))
    params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    if restart_params is not None:
        params = restart_params

    opt_state = optimizer.init(params)
    # Initialize EMA parameters (a copy of the initial parameters)
    ema_params = initialize_ema_params(params)
    
    print("Preparing batches")
    print("..................")
    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    print("Training")
    print("..................")
    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)
        # Loop over train batches.
        train_loss = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                esp_w=esp_w,
                ndcm=ndcm,
            )

            ema_params = update_ema_params(ema_params, params, ema_decay)
            
            train_loss += (loss - train_loss) / (i + 1)

        # Evaluate on validation set.
        valid_loss = 0.0
        for i, batch in enumerate(valid_batches):
            loss = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=ema_params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            valid_loss += (loss - valid_loss) / (i + 1)

        # Print progress.
        print(f"epoch: {epoch: 3d}      train:   valid:")
        print(f"    loss [a.u.]             {train_loss : 8.3e} {valid_loss : 8.3e}")
        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("RMSE/train", jnp.sqrt(2 * train_loss), epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        # writer.add_scalar("RMSE/valid", jnp.sqrt(2 * valid_loss), epoch)

        if valid_loss < best:
            best = valid_loss
            # open a file, where you want to store the data
            with open(f"{writer.logdir}/best_{esp_w}_params.pkl", "wb") as file:
                pickle.dump(ema_params, file)

    # Return final model parameters.
    return params, valid_loss


def clip_grads_by_global_norm(grads, max_norm):
    """
    Clips gradients by their global norm.
    Args:
    - grads: The gradients to clip.
    - max_norm: The maximum allowed global norm.
    Returns:
    - clipped_grads: The gradients after global norm clipping.
    """
    import jax
    import jax.numpy as jnp
    global_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    clipped_grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    return clipped_grads


def create_adam_optimizer_with_exponential_decay(
    initial_lr, final_lr, transition_steps, total_steps
):
    """
    Create an Adam optimizer with an exponentially decaying learning rate.
    """
    import optax
    import jax.numpy as jnp
    num_cycles = 10
    lr_schedule = optax.join_schedules(schedules=[
        optax.cosine_onecycle_schedule(
            peak_value=0.0005 - 0.00005*i,
            transition_steps=500,
            div_factor=1.1,
            final_div_factor=2
        ) for i in range(num_cycles)], 
        boundaries=jnp.cumsum(jnp.array([500] * num_cycles)))
    optimizer = optax.adamw(learning_rate=lr_schedule)
    return optimizer


def initialize_ema_params(params):
    """
    Initialize EMA parameters. Typically initialized to the same values as the initial model parameters.
    """
    import jax
    return jax.tree_util.tree_map(lambda p: p, params)


def update_ema_params(ema_params, new_params, decay):
    """
    Update EMA parameters using exponential moving average.
    """
    import jax
    return jax.tree_util.tree_map(
        lambda ema, new: decay * ema + (1.0 - decay) * new, ema_params, new_params
    )

# ===================== DIPOLAR TRAINING FUNCTIONS =====================
from dcmnet.loss import dipo_esp_mono_loss

import functools

@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "esp_w", "ndcm"),
)
def train_step_dipo(
    model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, ndcm,
    clip_norm=2.0
):
    def loss_fn(params):
        mono, dipo = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
        )
        esp_l, mono_l, dipo_l = dipo_esp_mono_loss(
            dipo_prediction=dipo,
            mono_prediction=mono,
            vdw_surface=batch["vdw_surface"],
            esp_target=batch["esp"],
            mono=batch["mono"],
            Dxyz=batch["Dxyz"],
            com=batch["com"],
            espMask=batch["espMask"],
            n_atoms=batch["N"],
            batch_size=batch_size,
            esp_w=esp_w,
            n_dcm=ndcm,
        )
        loss = esp_l + mono_l + dipo_l
        return loss, (mono, dipo, esp_l, mono_l, dipo_l)

    (loss, (mono, dipo, esp_l, mono_l, dipo_l)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    clipped_grads = clip_grads_by_global_norm(grad, clip_norm)
    updates, opt_state = optimizer_update(clipped_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, esp_l, mono_l, dipo_l


@functools.partial(
    jax.jit, static_argnames=("model_apply", "batch_size", "esp_w", "ndcm")
)
def eval_step_dipo(model_apply, batch, batch_size, params, esp_w, ndcm):
    mono, dipo = model_apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    esp_l, mono_l, dipo_l = dipo_esp_mono_loss(
        dipo_prediction=dipo,
        mono_prediction=mono,
        vdw_surface=batch["vdw_surface"],
        esp_target=batch["esp"],
        mono=batch["mono"],
        Dxyz=batch["Dxyz"],
        com=batch["com"],
        espMask=batch["espMask"],
        n_atoms=batch["N"],
        batch_size=batch_size,
        esp_w=esp_w,
        n_dcm=ndcm,
    )
    loss = esp_l + mono_l + dipo_l
    return loss, esp_l, mono_l, dipo_l


def train_model_dipo(
    key,
    model,
    train_data,
    valid_data,
    num_epochs,
    learning_rate,
    batch_size,
    writer,
    ndcm,
    esp_w=1.0,
    restart_params=None,
):
    """
    Dipole-specific training loop. See train_model for default training.
    """
    best = 10**7
    key, init_key = jax.random.split(key)
    initial_lr = learning_rate
    final_lr = 1e-6
    transition_steps = 10
    optimizer = create_adam_optimizer_with_exponential_decay(
        initial_lr=initial_lr,
        final_lr=final_lr,
        transition_steps=transition_steps,
        total_steps=num_epochs
    )
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data["Z"][0]))
    params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    if restart_params is not None:
        params = restart_params
    opt_state = optimizer.init(params)
    print("Preparing batches")
    print("..................")
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)
    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)
        train_loss = 0.0
        train_esp_l = 0.0
        train_mono_l = 0.0
        train_dipo_l = 0.0
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, esp_l, mono_l, dipo_l = train_step_dipo(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            train_loss += (loss - train_loss) / (i + 1)
            train_esp_l += (esp_l - train_esp_l) / (i + 1)
            train_mono_l += (mono_l - train_mono_l) / (i + 1)
            train_dipo_l += (dipo_l - train_dipo_l) / (i + 1)
        del train_batches
        valid_loss = 0.0
        valid_esp_l = 0.0
        valid_mono_l = 0.0
        valid_dipo_l = 0.0
        for i, batch in enumerate(valid_batches):
            loss, esp_l, mono_l, dipo_l = eval_step_dipo(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            valid_loss += (loss - valid_loss) / (i + 1)
            valid_esp_l += (esp_l - valid_esp_l) / (i + 1)
            valid_mono_l += (mono_l - valid_mono_l) / (i + 1)
            valid_dipo_l += (dipo_l - valid_dipo_l) / (i + 1)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("esp_l/train", train_esp_l, epoch)
        writer.add_scalar("esp_l/valid", valid_esp_l, epoch)
        writer.add_scalar("mono_l/train", train_mono_l, epoch)
        writer.add_scalar("mono_l/valid", train_mono_l, epoch)
        writer.add_scalar("dipo_l/train", train_dipo_l, epoch)
        writer.add_scalar("dipo_l/valid", valid_dipo_l, epoch)
        if valid_loss < best:
            best = valid_loss
            with open(f"{writer.logdir}/best_{esp_w}_params.pkl", "wb") as file:
                pickle.dump(params, file)
        writer.add_scalar("Loss/bestValid", best, epoch)
    return params, valid_loss


def train_model_general(
    key,
    model,
    train_data,
    valid_data,
    num_epochs,
    learning_rate,
    batch_size,
    writer,
    ndcm,
    esp_w=1.0,
    restart_params=None,
    loss_step_fn: Callable = None,         # (model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, ndcm, ...)
    eval_step_fn: Callable = None,         # (model_apply, batch, batch_size, params, esp_w, ndcm, ...)
    optimizer_fn: Callable = None,         # (learning_rate, num_epochs, ...)
    use_ema: bool = False,
    ema_decay: float = 0.999,
    use_grad_clip: bool = False,
    grad_clip_norm: float = 2.0,
    log_extra_metrics: Optional[Callable] = None,  # (writer, metrics_dict, epoch)
    save_best_params_with_ema: bool = False,
    extra_valid_args: dict = None,
    extra_train_args: dict = None,
):
    """
    Unified training loop for both default and dipole models.
    Pass in the appropriate step functions, optimizer, and logging hooks.
    """
    best = 10**7
    key, init_key = jax.random.split(key)
    optimizer = optimizer_fn(learning_rate, num_epochs) if optimizer_fn else optax.adam(learning_rate)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data["Z"][0]))
    params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    if restart_params is not None:
        params = restart_params
    opt_state = optimizer.init(params)
    if use_ema:
        ema_params = initialize_ema_params(params)
    print("Preparing batches\n..................")
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)
    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)
        # Training metrics
        train_metrics = {}
        for i, batch in enumerate(train_batches):
            step_args = dict(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            if use_grad_clip:
                step_args["clip_norm"] = grad_clip_norm
            if extra_train_args:
                step_args.update(extra_train_args)
            out = loss_step_fn(**step_args)
            # Unpack outputs
            if use_grad_clip:
                params, opt_state, loss, *extras = out
            else:
                params, opt_state, loss, *extras = out
            if use_ema:
                ema_params = update_ema_params(ema_params, params, ema_decay)
            # Accumulate metrics
            if i == 0:
                train_metrics = {"loss": float(loss)}
                if extras:
                    for idx, val in enumerate(extras):
                        train_metrics[f"extra{idx}"] = float(val)
            else:
                train_metrics["loss"] += (float(loss) - train_metrics["loss"]) / (i + 1)
                if extras:
                    for idx, val in enumerate(extras):
                        k = f"extra{idx}"
                        train_metrics[k] += (float(val) - train_metrics[k]) / (i + 1)
        del train_batches
        # Validation metrics
        valid_metrics = {}
        for i, batch in enumerate(valid_batches):
            eval_args = dict(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=ema_params if (use_ema and save_best_params_with_ema) else params,
                esp_w=esp_w,
                ndcm=ndcm,
            )
            if extra_valid_args:
                eval_args.update(extra_valid_args)
            out = eval_step_fn(**eval_args)
            if isinstance(out, tuple):
                loss, *extras = out
            else:
                loss, extras = out, []
            if i == 0:
                valid_metrics = {"loss": float(loss)}
                if extras:
                    for idx, val in enumerate(extras):
                        valid_metrics[f"extra{idx}"] = float(val)
            else:
                valid_metrics["loss"] += (float(loss) - valid_metrics["loss"]) / (i + 1)
                if extras:
                    for idx, val in enumerate(extras):
                        k = f"extra{idx}"
                        valid_metrics[k] += (float(val) - valid_metrics[k]) / (i + 1)
        # Logging
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/valid", valid_metrics["loss"], epoch)
        if log_extra_metrics:
            log_extra_metrics(writer, train_metrics, valid_metrics, epoch)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            # Save best params (EMA or not)
            best_params = ema_params if (use_ema and save_best_params_with_ema) else params
            with open(f"{writer.logdir}/best_{esp_w}_params.pkl", "wb") as file:
                pickle.dump(best_params, file)
        writer.add_scalar("Loss/bestValid", best, epoch)
    return params, valid_metrics["loss"]

# --- Backward-compatible wrappers ---

def _log_extra_metrics_none(writer, train_metrics, valid_metrics, epoch):
    pass

def _log_extra_metrics_dipo(writer, train_metrics, valid_metrics, epoch):
    # Assumes extra0=esp_l, extra1=mono_l, extra2=dipo_l
    writer.add_scalar("esp_l/train", train_metrics.get("extra0", 0.0), epoch)
    writer.add_scalar("esp_l/valid", valid_metrics.get("extra0", 0.0), epoch)
    writer.add_scalar("mono_l/train", train_metrics.get("extra1", 0.0), epoch)
    writer.add_scalar("mono_l/valid", valid_metrics.get("extra1", 0.0), epoch)
    writer.add_scalar("dipo_l/train", train_metrics.get("extra2", 0.0), epoch)
    writer.add_scalar("dipo_l/valid", valid_metrics.get("extra2", 0.0), epoch)

# Default (esp_mono_loss) config
train_model = partial(
    train_model_general,
    loss_step_fn=train_step,
    eval_step_fn=eval_step,
    optimizer_fn=lambda lr, _: optax.adam(lr),
    use_ema=True,
    ema_decay=0.999,
    use_grad_clip=False,
    grad_clip_norm=2.0,
    log_extra_metrics=_log_extra_metrics_none,
    save_best_params_with_ema=True,
)

# Dipole (dipo_esp_mono_loss) config
train_model_dipo = partial(
    train_model_general,
    loss_step_fn=train_step_dipo,
    eval_step_fn=eval_step_dipo,
    optimizer_fn=lambda lr, epochs: create_adam_optimizer_with_exponential_decay(lr, 1e-6, 10, epochs),
    use_ema=False,
    ema_decay=0.999,
    use_grad_clip=True,
    grad_clip_norm=2.0,
    log_extra_metrics=_log_extra_metrics_dipo,
    save_best_params_with_ema=False,
)

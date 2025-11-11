import functools
import pickle

import e3x
import jax
import jax.numpy as jnp
import optax
from .loss import esp_mono_loss

from .data import prepare_batches, prepare_datasets

from typing import Callable, Any, Optional
from functools import partial

# Try to enable lovely_jax for better array printing
try:
    import lovely_jax as lj
    lj.monkey_patch()
    print("lovely_jax enabled for enhanced array visualization")
except ImportError:
    print("lovely_jax not available, using standard JAX array printing")


@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "esp_w", "chg_w", "ndcm"),
)
def train_step(
    model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, chg_w, ndcm, clip_norm=None
):
    """
    Single training step for DCMNet with ESP and monopole losses.
    
    Performs forward pass, computes loss, calculates gradients, and updates
    model parameters using the provided optimizer.
    
    Parameters
    ----------
    model_apply : callable
        Function to apply the model (typically model.apply)
    optimizer_update : callable
        Function to update optimizer state (typically optimizer.update)
    batch : dict
        Batch dictionary containing 'Z', 'R', 'dst_idx', 'src_idx', 'batch_segments',
        'vdw_surface', 'esp', 'mono', 'n_grid', 'N'
    batch_size : int
        Size of the current batch
    opt_state : Any
        Current optimizer state
    params : Any
        Current model parameters
    esp_w : float
        Weight for ESP loss term
    ndcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (updated_params, updated_opt_state, loss_value)
    """
    def loss_fn(params):
        mono, dipo = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
        )
        loss, esp_pred, esp_target, esp_errors = esp_mono_loss(
            dipo_prediction=dipo,
            mono_prediction=mono,
            vdw_surface=batch["vdw_surface"],
            esp_target=batch["esp"],
            mono=batch["mono"],
            ngrid=batch["n_grid"],
            n_atoms=batch["N"],
            batch_size=batch_size,
            esp_w=esp_w,
            chg_w=chg_w,
            n_dcm=ndcm,
        )
        return loss, (mono, dipo, esp_pred, esp_target, esp_errors)

    (loss, (mono, dipo, esp_pred, esp_target, esp_errors)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    if clip_norm is not None:
        grad = clip_grads_by_global_norm(grad, clip_norm)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, mono, dipo, esp_pred, esp_target, esp_errors


def compute_statistics(predictions, targets=None):
    """
    Compute comprehensive statistics for predictions and optionally compare with targets.
    
    Parameters
    ----------
    predictions : array_like
        Predicted values
    targets : array_like, optional
        Target values for comparison
        
    Returns
    -------
    dict
        Dictionary containing statistical measures
    """
    stats = {
        'mean': float(jnp.mean(predictions)),
        'std': float(jnp.std(predictions)),
        'min': float(jnp.min(predictions)),
        'max': float(jnp.max(predictions)),
        'median': float(jnp.median(predictions)),
    }
    
    if targets is not None:
        error = predictions - targets
        stats.update({
            'mae': float(jnp.mean(jnp.abs(error))),
            'rmse': float(jnp.sqrt(jnp.mean(error**2))),
            'target_mean': float(jnp.mean(targets)),
            'target_std': float(jnp.std(targets)),
        })
    
    return stats


def print_statistics_table(train_stats, valid_stats, epoch):
    """
    Print a formatted table comparing training and validation statistics.
    
    Parameters
    ----------
    train_stats : dict
        Training statistics
    valid_stats : dict
        Validation statistics
    epoch : int
        Current epoch number
    """
    print(f"\n{'='*80}")
    print(f"Epoch {epoch:3d} Statistics")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Train':>15} {'Valid':>15} {'Difference':>15}")
    print(f"{'-'*80}")
    
    # Common metrics to compare
    metrics_list = ['loss', 'mono_mae', 'mono_rmse', 'mono_mean', 'mono_std',
                    'esp_mae', 'esp_rmse', 'esp_pred_mean', 'esp_pred_std',
                    'esp_error_mean', 'esp_error_std']
    for key in metrics_list:
        if key in train_stats and key in valid_stats:
            train_val = train_stats[key]
            valid_val = valid_stats[key]
            diff = valid_val - train_val
            print(f"{key:<20} {train_val:>15.6e} {valid_val:>15.6e} {diff:>15.6e}")
    
    print(f"{'-'*80}")
    print(f"Monopole Prediction Statistics:")
    print(f"  Train: mean={train_stats.get('mono_mean', 0):.6e}, "
          f"std={train_stats.get('mono_std', 0):.6e}, "
          f"min={train_stats.get('mono_min', 0):.6e}, "
          f"max={train_stats.get('mono_max', 0):.6e}")
    print(f"  Valid: mean={valid_stats.get('mono_mean', 0):.6e}, "
          f"std={valid_stats.get('mono_std', 0):.6e}, "
          f"min={valid_stats.get('mono_min', 0):.6e}, "
          f"max={valid_stats.get('mono_max', 0):.6e}")
    print(f"{'='*80}\n")


@functools.partial(
    jax.jit, static_argnames=("model_apply", "batch_size", "esp_w", "chg_w", "ndcm")
)
def eval_step(model_apply, batch, batch_size, params, esp_w, chg_w, ndcm):
    """
    Single evaluation step for DCMNet.
    
    Performs forward pass and computes loss without updating parameters.
    
    Parameters
    ----------
    model_apply : callable
        Function to apply the model (typically model.apply)
    batch : dict
        Batch dictionary containing model inputs and targets
    batch_size : int
        Size of the current batch
    params : Any
        Current model parameters
    esp_w : float
        Weight for ESP loss term
    ndcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (loss, mono, dipo, esp_pred, esp_target, esp_errors)
    """
    mono, dipo = model_apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    loss, esp_pred, esp_target, esp_errors = esp_mono_loss(
        dipo_prediction=dipo,
        mono_prediction=mono,
        vdw_surface=batch["vdw_surface"],
        esp_target=batch["esp"],
        mono=batch["mono"],
        ngrid=batch["n_grid"],
        n_atoms=batch["N"],
        batch_size=batch_size,
        esp_w=esp_w,
        chg_w=chg_w,
        n_dcm=ndcm,
    )
    return loss, mono, dipo, esp_pred, esp_target, esp_errors


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
    chg_w=0.01,
    restart_params=None,
    ema_decay=0.999,
    num_atoms=60,
    use_grad_clip=False,
    grad_clip_norm=2.0,
    mono_imputation_fn=None
):
    """
    Train DCMNet model with ESP and monopole losses.
    
    Performs full training loop with validation, logging, and checkpointing.
    Uses exponential moving average (EMA) for parameter smoothing and saves
    best parameters based on validation loss.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for training
    model : MessagePassingModel
        DCMNet model instance
    train_data : dict
        Training dataset dictionary
    valid_data : dict
        Validation dataset dictionary
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimization
    batch_size : int
        Batch size for training
    writer : SummaryWriter
        TensorBoard writer for logging
    ndcm : int
        Number of distributed multipoles per atom
    esp_w : float, optional
        Weight for ESP loss term, by default 1.0
    chg_w : float, optional
        Weight for charge/monopole loss term, by default 0.01
    restart_params : Any, optional
        Parameters to restart from, by default None
    ema_decay : float, optional
        Exponential moving average decay rate, by default 0.999
    num_atoms : int, optional
        Maximum number of atoms for batching, by default 60
    use_grad_clip : bool, optional
        Whether to use gradient clipping, by default False
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping, by default 2.0
    mono_imputation_fn : callable, optional
        Function to impute monopoles if missing from batches. Should take a batch dict
        and return monopoles with shape (batch_size * num_atoms,). By default None
        
    Returns
    -------
    tuple
        (final_params, final_valid_loss)
    """
    if writer is None:
        LOGDIR = "."
    else:
        LOGDIR = writer.logdir
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
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size, num_atoms=num_atoms, mono_imputation_fn=mono_imputation_fn)

    print("Training")
    print("..................")
    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size, num_atoms=num_atoms, mono_imputation_fn=mono_imputation_fn)
        # Loop over train batches.
        train_loss = 0.0
        train_mono_preds = []
        train_mono_targets = []
        train_esp_preds = []
        train_esp_targets = []
        train_esp_errors = []
        
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, mono, dipo, esp_pred, esp_target, esp_error = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
                esp_w=esp_w,
                chg_w=chg_w,
                ndcm=ndcm,
                clip_norm=grad_clip_norm if use_grad_clip else None,
            )

            # Block until JAX operations complete to avoid async context issues
            # This prevents RuntimeError: cannot enter context in IPython/Jupyter
            jax.block_until_ready(loss)
            jax.block_until_ready(params)

            ema_params = update_ema_params(ema_params, params, ema_decay)
            
            train_loss += (loss - train_loss) / (i + 1)
            
            # Collect predictions for statistics
            train_mono_preds.append(mono)
            train_mono_targets.append(batch["mono"])
            train_esp_preds.append(esp_pred)
            train_esp_targets.append(esp_target)
            train_esp_errors.append(esp_error)

        # Concatenate all predictions and targets
        train_mono_preds = jnp.concatenate(train_mono_preds, axis=0)
        train_mono_targets = jnp.concatenate(train_mono_targets, axis=0)
        train_esp_preds = jnp.concatenate([jnp.ravel(e) for e in train_esp_preds])
        train_esp_targets = jnp.concatenate([jnp.ravel(e) for e in train_esp_targets])
        train_esp_errors = jnp.concatenate([jnp.ravel(e) for e in train_esp_errors])
        
        # Compute training statistics
        train_mono_stats = compute_statistics(train_mono_preds.sum(axis=-1), train_mono_targets)
        train_esp_stats = {
            'mae': float(jnp.mean(jnp.abs(train_esp_errors))),
            'rmse': float(jnp.sqrt(jnp.mean(train_esp_errors**2))),
            'pred_mean': float(jnp.mean(train_esp_preds)),
            'pred_std': float(jnp.std(train_esp_preds)),
            'target_mean': float(jnp.mean(train_esp_targets)),
            'error_mean': float(jnp.mean(train_esp_errors)),
            'error_std': float(jnp.std(train_esp_errors)),
        }
        train_stats = {
            'loss': float(train_loss),
            'mono_mae': train_mono_stats['mae'],
            'mono_rmse': train_mono_stats['rmse'],
            'mono_mean': train_mono_stats['mean'],
            'mono_std': train_mono_stats['std'],
            'mono_min': train_mono_stats['min'],
            'mono_max': train_mono_stats['max'],
            'esp_mae': train_esp_stats['mae'],
            'esp_rmse': train_esp_stats['rmse'],
            'esp_pred_mean': train_esp_stats['pred_mean'],
            'esp_pred_std': train_esp_stats['pred_std'],
            'esp_target_mean': train_esp_stats['target_mean'],
            'esp_error_mean': train_esp_stats['error_mean'],
            'esp_error_std': train_esp_stats['error_std'],
        }

        # Evaluate on validation set.
        valid_loss = 0.0
        valid_mono_preds = []
        valid_mono_targets = []
        valid_esp_preds = []
        valid_esp_targets = []
        valid_esp_errors = []
        
        for i, batch in enumerate(valid_batches):
            loss, mono, dipo, esp_pred, esp_target, esp_error = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=ema_params,
                esp_w=esp_w,
                chg_w=chg_w,
                ndcm=ndcm,
            )
            # Block until JAX operations complete to avoid async context issues
            jax.block_until_ready(loss)
            valid_loss += (loss - valid_loss) / (i + 1)
            
            # Collect predictions for statistics
            valid_mono_preds.append(mono)
            valid_mono_targets.append(batch["mono"])
            valid_esp_preds.append(esp_pred)
            valid_esp_targets.append(esp_target)
            valid_esp_errors.append(esp_error)

        # Concatenate all predictions and targets
        valid_mono_preds = jnp.concatenate(valid_mono_preds, axis=0)
        valid_mono_targets = jnp.concatenate(valid_mono_targets, axis=0)
        valid_esp_preds = jnp.concatenate([jnp.ravel(e) for e in valid_esp_preds])
        valid_esp_targets = jnp.concatenate([jnp.ravel(e) for e in valid_esp_targets])
        valid_esp_errors = jnp.concatenate([jnp.ravel(e) for e in valid_esp_errors])
        
        # Compute validation statistics
        valid_mono_stats = compute_statistics(valid_mono_preds.sum(axis=-1), valid_mono_targets)
        valid_esp_stats = {
            'mae': float(jnp.mean(jnp.abs(valid_esp_errors))),
            'rmse': float(jnp.sqrt(jnp.mean(valid_esp_errors**2))),
            'pred_mean': float(jnp.mean(valid_esp_preds)),
            'pred_std': float(jnp.std(valid_esp_preds)),
            'target_mean': float(jnp.mean(valid_esp_targets)),
            'error_mean': float(jnp.mean(valid_esp_errors)),
            'error_std': float(jnp.std(valid_esp_errors)),
        }
        valid_stats = {
            'loss': float(valid_loss),
            'mono_mae': valid_mono_stats['mae'],
            'mono_rmse': valid_mono_stats['rmse'],
            'mono_mean': valid_mono_stats['mean'],
            'mono_std': valid_mono_stats['std'],
            'mono_min': valid_mono_stats['min'],
            'mono_max': valid_mono_stats['max'],
            'esp_mae': valid_esp_stats['mae'],
            'esp_rmse': valid_esp_stats['rmse'],
            'esp_pred_mean': valid_esp_stats['pred_mean'],
            'esp_pred_std': valid_esp_stats['pred_std'],
            'esp_target_mean': valid_esp_stats['target_mean'],
            'esp_error_mean': valid_esp_stats['error_mean'],
            'esp_error_std': valid_esp_stats['error_std'],
        }

        # Print detailed statistics
        print_statistics_table(train_stats, valid_stats, epoch)
        
        if writer is not None:
            # Log loss metrics
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", valid_loss, epoch)
            writer.add_scalar("Loss/bestValid", best, epoch)
            
            # Log monopole statistics
            writer.add_scalar("Monopole/train_mae", train_stats['mono_mae'], epoch)
            writer.add_scalar("Monopole/valid_mae", valid_stats['mono_mae'], epoch)
            writer.add_scalar("Monopole/train_rmse", train_stats['mono_rmse'], epoch)
            writer.add_scalar("Monopole/valid_rmse", valid_stats['mono_rmse'], epoch)
            writer.add_scalar("Monopole/train_mean", train_stats['mono_mean'], epoch)
            writer.add_scalar("Monopole/valid_mean", valid_stats['mono_mean'], epoch)
            writer.add_scalar("Monopole/train_std", train_stats['mono_std'], epoch)
            writer.add_scalar("Monopole/valid_std", valid_stats['mono_std'], epoch)
        if valid_loss < best:
            best = valid_loss
            # open a file, where you want to store the data
            with open(f"{LOGDIR}/best_{esp_w}_params.pkl", "wb") as file:
                pickle.dump(ema_params, file)

    # Return final model parameters.
    return params, valid_loss


def clip_grads_by_global_norm(grads, max_norm):
    """
    Clips gradients by their global norm.
    
    Parameters
    ----------
    grads : Any
        The gradients to clip
    max_norm : float
        The maximum allowed global norm
        
    Returns
    -------
    Any
        The gradients after global norm clipping
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
    
    Parameters
    ----------
    initial_lr : float
        Initial learning rate
    final_lr : float
        Final learning rate
    transition_steps : int
        Number of steps for each learning rate cycle
    total_steps : int
        Total number of training steps
        
    Returns
    -------
    optax.GradientTransformation
        Configured Adam optimizer with learning rate schedule
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
    
    Parameters
    ----------
    params : Any
        Initial model parameters
        
    Returns
    -------
    Any
        Copy of initial parameters for EMA
    """
    import jax
    return jax.tree_util.tree_map(lambda p: p, params)


def update_ema_params(ema_params, new_params, decay):
    """
    Update EMA parameters using exponential moving average.
    
    Parameters
    ----------
    ema_params : Any
        Current EMA parameters
    new_params : Any
        New model parameters
    decay : float
        Decay rate for EMA (between 0 and 1)
        
    Returns
    -------
    Any
        Updated EMA parameters
    """
    import jax
    return jax.tree_util.tree_map(
        lambda ema, new: decay * ema + (1.0 - decay) * new, ema_params, new_params
    )


def create_mono_imputation_fn(model, params):
    """
    Create a monopole imputation function from a trained model.
    
    This function wraps a model to create an imputation function that can be
    passed to prepare_batches. The function predicts monopoles from a batch
    and returns them in the expected format (per-atom atomic monopoles).
    
    Parameters
    ----------
    model : MessagePassingModel
        DCMNet model instance (must have n_dcm attribute)
    params : Any
        Model parameters
        
    Returns
    -------
    callable
        Imputation function that takes a batch dict and returns monopoles
        with shape (batch_size * num_atoms,) - atomic monopoles per atom
    
    Examples
    --------
    Load a pretrained model and use it to impute monopoles:
    
    >>> import pickle
    >>> from mmml.dcmnet.dcmnet.training import create_mono_imputation_fn, train_model
    >>> 
    >>> # Load a pretrained model
    >>> with open('pretrained_model/best_params.pkl', 'rb') as f:
    ...     pretrained_params = pickle.load(f)
    >>> 
    >>> # Create imputation function (n_dcm is inferred from model.n_dcm)
    >>> mono_imputation_fn = create_mono_imputation_fn(
    ...     model=pretrained_model,
    ...     params=pretrained_params
    ... )
    >>> 
    >>> # Use in training
    >>> final_params, valid_loss = train_model(
    ...     key=key,
    ...     model=new_model,
    ...     train_data=train_data,
    ...     valid_data=valid_data,
    ...     num_epochs=5,
    ...     learning_rate=0.0005,
    ...     batch_size=1000,
    ...     writer=writer,
    ...     ndcm=3,
    ...     mono_imputation_fn=mono_imputation_fn  # Pass imputation function
    ... )
    """
    def imputation_fn(batch):
        """
        Impute monopoles for a batch.
        
        Parameters
        ----------
        batch : dict
            Batch dictionary containing 'Z', 'R', 'dst_idx', 'src_idx', 'batch_segments'
            
        Returns
        -------
        jnp.ndarray
            Atomic monopoles with shape (batch_size * num_atoms,)
        """
        # Run model forward pass
        mono_pred, dipo_pred = model.apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
        )
        
        # Sum distributed monopoles to get atomic monopoles
        # mono_pred shape: (batch_size * num_atoms, n_dcm)
        # We want: (batch_size * num_atoms,) - sum over n_dcm dimension to get per-atom charges
        atomic_mono = mono_pred.sum(axis=-1)
        
        return atomic_mono
    
    return imputation_fn

# ===================== DIPOLAR TRAINING FUNCTIONS =====================
from .loss import dipo_esp_mono_loss

import functools

@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "esp_w", "ndcm"),
)
def train_step_dipo(
    model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, ndcm,
    clip_norm=2.0
):
    """
    Single training step for DCMNet with dipole-augmented losses.
    
    Performs forward pass, computes dipole-augmented loss, calculates gradients,
    clips gradients, and updates model parameters.
    
    Parameters
    ----------
    model_apply : callable
        Function to apply the model
    optimizer_update : callable
        Function to update optimizer state
    batch : dict
        Batch dictionary (must include 'Dxyz', 'com', 'espMask')
    batch_size : int
        Size of the current batch
    opt_state : Any
        Current optimizer state
    params : Any
        Current model parameters
    esp_w : float
        Weight for ESP loss term
    ndcm : int
        Number of distributed multipoles per atom
    clip_norm : float, optional
        Maximum gradient norm for clipping, by default 2.0
        
    Returns
    -------
    tuple
        (updated_params, updated_opt_state, total_loss, esp_loss, mono_loss, dipole_loss)
    """
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
    return params, opt_state, loss, esp_l, mono_l, dipo_l, mono, dipo


@functools.partial(
    jax.jit, static_argnames=("model_apply", "batch_size", "esp_w", "ndcm")
)
def eval_step_dipo(model_apply, batch, batch_size, params, esp_w, ndcm):
    """
    Single evaluation step for DCMNet with dipole-augmented losses.
    
    Parameters
    ----------
    model_apply : callable
        Function to apply the model
    batch : dict
        Batch dictionary (must include 'Dxyz', 'com', 'espMask')
    batch_size : int
        Size of the current batch
    params : Any
        Current model parameters
    esp_w : float
        Weight for ESP loss term
    ndcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (total_loss, esp_loss, mono_loss, dipole_loss)
    """
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
    return loss, esp_l, mono_l, dipo_l, mono, dipo


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
    Train DCMNet model with dipole-augmented losses.
    
    Performs full training loop with dipole-specific loss components.
    Uses exponential learning rate decay and gradient clipping.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for training
    model : MessagePassingModel
        DCMNet model instance
    train_data : dict
        Training dataset (must include 'Dxyz', 'com', 'espMask')
    valid_data : dict
        Validation dataset
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Initial learning rate
    batch_size : int
        Batch size for training
    writer : SummaryWriter
        TensorBoard writer for logging
    ndcm : int
        Number of distributed multipoles per atom
    esp_w : float, optional
        Weight for ESP loss term, by default 1.0
    restart_params : Any, optional
        Parameters to restart from, by default None
        
    Returns
    -------
    tuple
        (final_params, final_valid_loss)
    """
    best = 10**7
    if writer is None:
        LOGDIR = "."
    else:
        LOGDIR = writer.logdir
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
        train_mono_preds = []
        train_mono_targets = []
        
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, esp_l, mono_l, dipo_l, mono, dipo = train_step_dipo(
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
            
            # Collect predictions for statistics
            train_mono_preds.append(mono)
            train_mono_targets.append(batch["mono"])
            
        del train_batches
        
        # Concatenate all training predictions and targets
        train_mono_preds = jnp.concatenate(train_mono_preds, axis=0)
        train_mono_targets = jnp.concatenate(train_mono_targets, axis=0)
        
        # Compute training statistics
        train_mono_stats = compute_statistics(train_mono_preds.sum(axis=-1), train_mono_targets)
        train_stats = {
            'loss': float(train_loss),
            'esp_loss': float(train_esp_l),
            'mono_loss': float(train_mono_l),
            'dipo_loss': float(train_dipo_l),
            'mono_mae': train_mono_stats['mae'],
            'mono_rmse': train_mono_stats['rmse'],
            'mono_mean': train_mono_stats['mean'],
            'mono_std': train_mono_stats['std'],
            'mono_min': train_mono_stats['min'],
            'mono_max': train_mono_stats['max'],
        }
        
        valid_loss = 0.0
        valid_esp_l = 0.0
        valid_mono_l = 0.0
        valid_dipo_l = 0.0
        valid_mono_preds = []
        valid_mono_targets = []
        
        for i, batch in enumerate(valid_batches):
            loss, esp_l, mono_l, dipo_l, mono, dipo = eval_step_dipo(
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
            
            # Collect predictions for statistics
            valid_mono_preds.append(mono)
            valid_mono_targets.append(batch["mono"])
        
        # Concatenate all validation predictions and targets
        valid_mono_preds = jnp.concatenate(valid_mono_preds, axis=0)
        valid_mono_targets = jnp.concatenate(valid_mono_targets, axis=0)
        
        # Compute validation statistics
        valid_mono_stats = compute_statistics(valid_mono_preds.sum(axis=-1), valid_mono_targets)
        valid_stats = {
            'loss': float(valid_loss),
            'esp_loss': float(valid_esp_l),
            'mono_loss': float(valid_mono_l),
            'dipo_loss': float(valid_dipo_l),
            'mono_mae': valid_mono_stats['mae'],
            'mono_rmse': valid_mono_stats['rmse'],
            'mono_mean': valid_mono_stats['mean'],
            'mono_std': valid_mono_stats['std'],
            'mono_min': valid_mono_stats['min'],
            'mono_max': valid_mono_stats['max'],
        }
        
        # Print detailed statistics
        print_statistics_table(train_stats, valid_stats, epoch)
        
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", valid_loss, epoch)
            writer.add_scalar("esp_l/train", train_esp_l, epoch)
            writer.add_scalar("esp_l/valid", valid_esp_l, epoch)
            writer.add_scalar("mono_l/train", train_mono_l, epoch)
            writer.add_scalar("mono_l/valid", train_mono_l, epoch)
            writer.add_scalar("dipo_l/train", train_dipo_l, epoch)
            writer.add_scalar("dipo_l/valid", valid_dipo_l, epoch)
            writer.add_scalar("Loss/bestValid", best, epoch)
            
            # Log monopole statistics
            writer.add_scalar("Monopole/train_mae", train_stats['mono_mae'], epoch)
            writer.add_scalar("Monopole/valid_mae", valid_stats['mono_mae'], epoch)
            writer.add_scalar("Monopole/train_rmse", train_stats['mono_rmse'], epoch)
            writer.add_scalar("Monopole/valid_rmse", valid_stats['mono_rmse'], epoch)
            writer.add_scalar("Monopole/train_mean", train_stats['mono_mean'], epoch)
            writer.add_scalar("Monopole/valid_mean", valid_stats['mono_mean'], epoch)
            writer.add_scalar("Monopole/train_std", train_stats['mono_std'], epoch)
            writer.add_scalar("Monopole/valid_std", valid_stats['mono_std'], epoch)     
        
        if valid_loss < best:
            best = valid_loss
            with open(f"{LOGDIR}/best_{esp_w}_params.pkl", "wb") as file:
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
    chg_w=0.01,
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
    
    A flexible training function that can handle different loss functions,
    optimizers, and logging configurations. Pass in the appropriate step
    functions, optimizer, and logging hooks.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for training
    model : MessagePassingModel
        DCMNet model instance
    train_data : dict
        Training dataset dictionary
    valid_data : dict
        Validation dataset dictionary
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimization
    batch_size : int
        Batch size for training
    writer : SummaryWriter
        TensorBoard writer for logging
    ndcm : int
        Number of distributed multipoles per atom
    esp_w : float, optional
        Weight for ESP loss term, by default 1.0
    restart_params : Any, optional
        Parameters to restart from, by default None
    loss_step_fn : callable, optional
        Function for training step, by default None
    eval_step_fn : callable, optional
        Function for evaluation step, by default None
    optimizer_fn : callable, optional
        Function to create optimizer, by default None
    use_ema : bool, optional
        Whether to use exponential moving average, by default False
    ema_decay : float, optional
        EMA decay rate, by default 0.999
    use_grad_clip : bool, optional
        Whether to use gradient clipping, by default False
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping, by default 2.0
    log_extra_metrics : callable, optional
        Function to log additional metrics, by default None
    save_best_params_with_ema : bool, optional
        Whether to save EMA parameters as best, by default False
    extra_valid_args : dict, optional
        Extra arguments for validation step, by default None
    extra_train_args : dict, optional
        Extra arguments for training step, by default None
        
    Returns
    -------
    tuple
        (final_params, final_valid_loss)
    """
    best = 10**7
    if writer is None:
        LOGDIR = "."
    else:
        LOGDIR = writer.logdir
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
                        # Only store scalar values in metrics
                        if jnp.ndim(val) == 0:
                            train_metrics[f"extra{idx}"] = float(val)
            else:
                train_metrics["loss"] += (float(loss) - train_metrics["loss"]) / (i + 1)
                if extras:
                    for idx, val in enumerate(extras):
                        # Only update scalar values in metrics
                        if jnp.ndim(val) == 0:
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
                chg_w=chg_w,
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
                        # Only store scalar values in metrics
                        if jnp.ndim(val) == 0:
                            valid_metrics[f"extra{idx}"] = float(val)
            else:
                valid_metrics["loss"] += (float(loss) - valid_metrics["loss"]) / (i + 1)
                if extras:
                    for idx, val in enumerate(extras):
                        # Only update scalar values in metrics
                        if jnp.ndim(val) == 0:
                            k = f"extra{idx}"
                            valid_metrics[k] += (float(val) - valid_metrics[k]) / (i + 1)
        
        if writer is not None and log_extra_metrics:
            log_extra_metrics(writer, train_metrics, valid_metrics, epoch)
        if valid_metrics["loss"] < best:
            # jax.debug.print("Best loss updated to {x}", x=valid_metrics["loss"])
            best = valid_metrics["loss"]
            # Save best params (EMA or not)
            best_params = ema_params if (use_ema and save_best_params_with_ema) else params
            with open(f"{LOGDIR}/best_{esp_w}_params.pkl", "wb") as file:
                pickle.dump(best_params, file)
        if writer is not None:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/valid", valid_metrics["loss"], epoch)
            writer.add_scalar("Loss/bestValid", best, epoch)


        jax.debug.print("Epoch {x} train: {y}, valid: {z}, best: {w}", 
        x=epoch, y=train_metrics["loss"], 
                 z=valid_metrics["loss"], w=valid_metrics["loss"] <= best)

    return params, valid_metrics["loss"]

# --- Backward-compatible wrappers ---

def _log_extra_metrics_none(writer, train_metrics, valid_metrics, epoch):
    """No-op function for logging extra metrics."""
    pass

def _log_extra_metrics_dipo(writer, train_metrics, valid_metrics, epoch):
    """
    Log dipole-specific metrics to TensorBoard.
    
    Assumes extra0=esp_l, extra1=mono_l, extra2=dipo_l
    """
    # Assumes extra0=esp_l, extra1=mono_l, extra2=dipo_l
    writer.add_scalar("esp_l/train", train_metrics.get("extra0", 0.0), epoch)
    writer.add_scalar("esp_l/valid", valid_metrics.get("extra0", 0.0), epoch)
    writer.add_scalar("mono_l/train", train_metrics.get("extra1", 0.0), epoch)
    writer.add_scalar("mono_l/valid", valid_metrics.get("extra1", 0.0), epoch)
    writer.add_scalar("dipo_l/train", train_metrics.get("extra2", 0.0), epoch)
    writer.add_scalar("dipo_l/valid", valid_metrics.get("extra2", 0.0), epoch)

# Alternative general training interface using train_model_general
# (kept for backward compatibility with code that uses the general interface)
train_model_general_default = partial(
    train_model_general,
    loss_step_fn=train_step,
    eval_step_fn=eval_step,
    optimizer_fn=lambda lr, _: optax.adam(lr),
    use_ema=True,
    ema_decay=0.999,
    use_grad_clip=True,
    grad_clip_norm=2.0,
    log_extra_metrics=_log_extra_metrics_none,
    save_best_params_with_ema=True,
    extra_train_args={"chg_w": 0.01},
)

# Note: train_model is defined earlier in the file with enhanced statistics
# Note: train_model_dipo is defined earlier in the file with enhanced statistics

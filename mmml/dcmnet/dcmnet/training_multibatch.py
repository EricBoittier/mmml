"""
Multi-batch training for DCMNet with gradient accumulation and advanced features.

Provides enhanced training capabilities including:
- Gradient accumulation over multiple batches
- Parallel batch processing with vmap
- Advanced learning rate schedules
- Better checkpointing and monitoring
- Comprehensive statistics tracking
"""
import functools
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time

import e3x
import jax
import jax.numpy as jnp
import optax
from jax import vmap

from .loss import esp_mono_loss, dipo_esp_mono_loss
from .data import prepare_batches
from .training_config import ExperimentConfig, TrainingConfig, ModelConfig

# Try to enable lovely_jax
try:
    import lovely_jax as lj
    lj.monkey_patch()
    LOVELY_JAX_AVAILABLE = True
except ImportError:
    LOVELY_JAX_AVAILABLE = False


def create_lr_schedule(config: TrainingConfig, total_steps: int):
    """
    Create learning rate schedule based on configuration.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    total_steps : int
        Total number of training steps
        
    Returns
    -------
    optax.Schedule
        Learning rate schedule
    """
    if not config.use_lr_schedule:
        return config.learning_rate
    
    warmup_steps = config.warmup_epochs * (total_steps // config.num_epochs)
    
    if config.lr_schedule_type == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=config.learning_rate * config.min_lr_factor
        )
    elif config.lr_schedule_type == "exponential":
        schedule = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            transition_steps=total_steps - warmup_steps,
            decay_rate=config.min_lr_factor
        )
    else:  # constant after warmup
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=config.learning_rate,
                    transition_steps=warmup_steps
                ),
                config.learning_rate
            ],
            boundaries=[warmup_steps]
        )
    
    return schedule


@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "batch_size", "esp_w", "chg_w", "ndcm"),
)
def compute_gradients(
    model_apply, batch, batch_size, params, esp_w, chg_w, ndcm
):
    """
    Compute gradients for a single batch without updating parameters.
    
    Used for gradient accumulation across multiple batches.
    
    Parameters
    ----------
    model_apply : callable
        Model application function
    batch : dict
        Batch data
    batch_size : int
        Batch size
    params : PyTree
        Model parameters
    esp_w : float
        ESP loss weight
    chg_w : float
        Charge loss weight
    ndcm : int
        Number of distributed multipoles
        
    Returns
    -------
    tuple
        (loss, gradients, mono, dipo, esp_pred, esp_target, esp_errors)
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
    return loss, grad, mono, dipo, esp_pred, esp_target, esp_errors


@functools.partial(
    jax.jit,
    static_argnames=("optimizer_update",),
)
def apply_accumulated_gradients(
    optimizer_update, accumulated_grads, opt_state, params, num_accumulation_steps
):
    """
    Apply accumulated gradients after averaging.
    
    Parameters
    ----------
    optimizer_update : callable
        Optimizer update function
    accumulated_grads : PyTree
        Accumulated gradients
    opt_state : Any
        Optimizer state
    params : PyTree
        Model parameters
    num_accumulation_steps : int
        Number of gradient accumulation steps
        
    Returns
    -------
    tuple
        (updated_params, updated_opt_state)
    """
    # Average the accumulated gradients
    averaged_grads = jax.tree_map(
        lambda g: g / num_accumulation_steps, accumulated_grads
    )
    
    # Apply gradient clipping if needed (done in optimizer)
    updates, opt_state = optimizer_update(averaged_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state


def accumulate_gradients(grads1, grads2):
    """
    Accumulate two gradient trees.
    
    Parameters
    ----------
    grads1 : PyTree
        First gradient tree
    grads2 : PyTree
        Second gradient tree
        
    Returns
    -------
    PyTree
        Accumulated gradients
    """
    return jax.tree_map(lambda g1, g2: g1 + g2, grads1, grads2)


def initialize_ema_params(params):
    """Initialize EMA parameters as copy of initial parameters."""
    return jax.tree_map(lambda p: p, params)


def update_ema_params(ema_params, new_params, decay):
    """Update EMA parameters."""
    return jax.tree_map(
        lambda ema, new: decay * ema + (1.0 - decay) * new, ema_params, new_params
    )


def clip_grads_by_global_norm(grads, max_norm):
    """Clip gradients by global norm."""
    global_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    clipped_grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    return clipped_grads, global_norm


class TrainingMetrics:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.mono_preds = []
        self.mono_targets = []
        self.esp_preds = []
        self.esp_targets = []
        self.esp_errors = []
        self.batch_times = []
    
    def update(self, loss, mono_pred=None, mono_target=None, esp_pred=None, esp_target=None, esp_error=None, batch_time=None):
        """Update metrics with new batch."""
        self.losses.append(float(loss))
        if mono_pred is not None:
            self.mono_preds.append(mono_pred)
        if mono_target is not None:
            self.mono_targets.append(mono_target)
        if esp_pred is not None:
            self.esp_preds.append(esp_pred)
        if esp_target is not None:
            self.esp_targets.append(esp_target)
        if esp_error is not None:
            self.esp_errors.append(esp_error)
        if batch_time is not None:
            self.batch_times.append(batch_time)
    
    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics."""
        metrics = {
            'loss': float(jnp.mean(jnp.array(self.losses))),
            'loss_std': float(jnp.std(jnp.array(self.losses))),
        }
        
        if self.mono_preds and self.mono_targets:
            all_preds = jnp.concatenate(self.mono_preds, axis=0)
            all_targets = jnp.concatenate(self.mono_targets, axis=0)
            
            # Sum over DCM dimension if needed
            if all_preds.ndim > 1:
                all_preds = all_preds.sum(axis=-1)
            
            error = all_preds - all_targets
            metrics.update({
                'mono_mae': float(jnp.mean(jnp.abs(error))),
                'mono_rmse': float(jnp.sqrt(jnp.mean(error**2))),
                'mono_mean': float(jnp.mean(all_preds)),
                'mono_std': float(jnp.std(all_preds)),
                'mono_min': float(jnp.min(all_preds)),
                'mono_max': float(jnp.max(all_preds)),
            })
        
        if self.esp_preds and self.esp_targets:
            all_esp_preds = jnp.concatenate([jnp.ravel(e) for e in self.esp_preds])
            all_esp_targets = jnp.concatenate([jnp.ravel(e) for e in self.esp_targets])
            all_esp_errors = jnp.concatenate([jnp.ravel(e) for e in self.esp_errors])
            
            esp_mae = jnp.mean(jnp.abs(all_esp_errors))
            esp_rmse = jnp.sqrt(jnp.mean(all_esp_errors**2))
            
            metrics.update({
                'esp_mae': float(esp_mae),
                'esp_rmse': float(esp_rmse),
                'esp_pred_mean': float(jnp.mean(all_esp_preds)),
                'esp_pred_std': float(jnp.std(all_esp_preds)),
                'esp_pred_min': float(jnp.min(all_esp_preds)),
                'esp_pred_max': float(jnp.max(all_esp_preds)),
                'esp_target_mean': float(jnp.mean(all_esp_targets)),
                'esp_target_std': float(jnp.std(all_esp_targets)),
                'esp_error_mean': float(jnp.mean(all_esp_errors)),
                'esp_error_std': float(jnp.std(all_esp_errors)),
            })
        
        if self.batch_times:
            metrics['avg_batch_time'] = float(jnp.mean(jnp.array(self.batch_times)))
        
        return metrics


def print_epoch_summary(epoch: int, train_metrics: Dict, valid_metrics: Dict, epoch_time: float):
    """
    Print formatted epoch summary.
    
    Parameters
    ----------
    epoch : int
        Current epoch number
    train_metrics : Dict
        Training metrics
    valid_metrics : Dict
        Validation metrics
    epoch_time : float
        Time taken for epoch in seconds
    """
    print(f"\n{'='*90}")
    print(f"Epoch {epoch:3d} Summary (Time: {epoch_time:.2f}s)")
    print(f"{'='*90}")
    print(f"{'Metric':<25} {'Train':>15} {'Valid':>15} {'Diff':>15} {'% Diff':>15}")
    print(f"{'-'*90}")
    
    metrics_to_show = ['loss', 'mono_mae', 'mono_rmse', 'mono_mean', 'mono_std',
                       'esp_mae', 'esp_rmse', 'esp_pred_mean', 'esp_pred_std',
                       'esp_error_mean', 'esp_error_std']
    
    for key in metrics_to_show:
        if key in train_metrics and key in valid_metrics:
            train_val = train_metrics[key]
            valid_val = valid_metrics[key]
            diff = valid_val - train_val
            pct_diff = (diff / (abs(train_val) + 1e-10)) * 100
            print(f"{key:<25} {train_val:>15.6e} {valid_val:>15.6e} {diff:>15.6e} {pct_diff:>14.2f}%")
    
    print(f"{'-'*90}")
    if 'avg_batch_time' in train_metrics:
        print(f"Avg batch time: {train_metrics['avg_batch_time']:.4f}s")
    print(f"{'='*90}\n")


def save_checkpoint(
    params, opt_state, ema_params, epoch, metrics, exp_dir: Path, 
    config: ExperimentConfig, is_best: bool = False
):
    """
    Save training checkpoint.
    
    Parameters
    ----------
    params : PyTree
        Model parameters
    opt_state : Any
        Optimizer state
    ema_params : PyTree
        EMA parameters
    epoch : int
        Current epoch
    metrics : Dict
        Current metrics
    exp_dir : Path
        Experiment directory
    config : ExperimentConfig
        Experiment configuration
    is_best : bool
        Whether this is the best checkpoint
    """
    checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'ema_params': ema_params,
        'epoch': epoch,
        'metrics': metrics,
        'config': config.to_dict()
    }
    
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save epoch checkpoint
    if epoch % config.training.save_every_n_epochs == 0:
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    # Save best checkpoint
    if is_best:
        path = checkpoint_dir / "checkpoint_best.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    # Always save latest
    path = checkpoint_dir / "checkpoint_latest.pkl"
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def train_model_multibatch(
    key: jax.random.PRNGKey,
    model,
    train_data: Dict,
    valid_data: Dict,
    config: ExperimentConfig,
    writer=None,
    restart_checkpoint: Optional[Path] = None,
) -> Tuple[Any, float]:
    """
    Train DCMNet model with multi-batch support and gradient accumulation.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for training
    model
        DCMNet model instance
    train_data : Dict
        Training dataset
    valid_data : Dict
        Validation dataset
    config : ExperimentConfig
        Complete experiment configuration
    writer : Optional
        TensorBoard writer
    restart_checkpoint : Optional[Path]
        Path to checkpoint to restart from
        
    Returns
    -------
    tuple
        (final_params, final_valid_loss)
    """
    exp_dir = config.get_experiment_dir()
    train_config = config.training
    model_config = config.model
    
    # Save configuration
    config.save(exp_dir / "config.json")
    
    print(f"\n{'='*90}")
    print(f"Starting Multi-Batch Training: {config.name}")
    print(f"{'='*90}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Gradient Accumulation Steps: {train_config.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"{'='*90}\n")
    
    # Initialize model
    key, init_key = jax.random.split(key)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data["Z"][0]))
    params = model.init(
        init_key,
        atomic_numbers=train_data["Z"][0],
        positions=train_data["R"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    
    # Initialize optimizer
    total_steps = train_config.num_epochs * (len(train_data["R"]) // train_config.batch_size)
    lr_schedule = create_lr_schedule(train_config, total_steps)
    
    optimizer = optax.adam(lr_schedule)
    if train_config.use_grad_clip:
        optimizer = optax.chain(
            optax.clip_by_global_norm(train_config.grad_clip_norm),
            optimizer
        )
    
    # Load from checkpoint if provided
    start_epoch = 1
    if restart_checkpoint is not None:
        with open(restart_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        opt_state = checkpoint['opt_state']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
    else:
        opt_state = optimizer.init(params)
    
    # Initialize EMA
    ema_params = initialize_ema_params(params) if train_config.use_ema else None
    
    # Prepare validation batches once
    key, valid_key = jax.random.split(key)
    valid_batches = prepare_batches(
        valid_key, valid_data, train_config.batch_size, 
        num_atoms=train_config.num_atoms
    )
    
    # Training loop
    best_valid_loss = float('inf')
    
    for epoch in range(start_epoch, train_config.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Prepare training batches for this epoch
        key, train_key = jax.random.split(key)
        train_batches = prepare_batches(
            train_key, train_data, train_config.batch_size,
            num_atoms=train_config.num_atoms
        )
        
        # Training phase
        train_metrics = TrainingMetrics()
        accumulated_grads = None
        accum_step = 0
        
        for batch_idx, batch in enumerate(train_batches):
            batch_start = time.time()
            
            # Compute gradients for this batch
            loss, grad, mono, dipo, esp_pred, esp_target, esp_error = compute_gradients(
                model_apply=model.apply,
                batch=batch,
                batch_size=train_config.batch_size,
                params=params,
                esp_w=train_config.esp_w,
                chg_w=train_config.chg_w,
                ndcm=model_config.n_dcm,
            )
            
            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grad
            else:
                accumulated_grads = accumulate_gradients(accumulated_grads, grad)
            
            accum_step += 1
            
            # Update parameters after accumulation steps
            if accum_step == train_config.gradient_accumulation_steps or \
               batch_idx == len(train_batches) - 1:
                params, opt_state = apply_accumulated_gradients(
                    optimizer_update=optimizer.update,
                    accumulated_grads=accumulated_grads,
                    opt_state=opt_state,
                    params=params,
                    num_accumulation_steps=accum_step,
                )
                
                # Update EMA
                if train_config.use_ema:
                    ema_params = update_ema_params(ema_params, params, train_config.ema_decay)
                
                # Reset accumulation
                accumulated_grads = None
                accum_step = 0
            
            # Track metrics
            batch_time = time.time() - batch_start
            train_metrics.update(
                loss=loss,
                mono_pred=mono,
                mono_target=batch["mono"],
                esp_pred=esp_pred,
                esp_target=esp_target,
                esp_error=esp_error,
                batch_time=batch_time
            )
        
        # Compute training metrics
        train_stats = train_metrics.compute()
        
        # Validation phase
        valid_metrics = TrainingMetrics()
        eval_params = ema_params if train_config.use_ema else params
        
        for batch in valid_batches:
            loss, _, mono, dipo, esp_pred, esp_target, esp_error = compute_gradients(
                model_apply=model.apply,
                batch=batch,
                batch_size=train_config.batch_size,
                params=eval_params,
                esp_w=train_config.esp_w,
                chg_w=train_config.chg_w,
                ndcm=model_config.n_dcm,
            )
            valid_metrics.update(
                loss=loss,
                mono_pred=mono,
                mono_target=batch["mono"],
                esp_pred=esp_pred,
                esp_target=esp_target,
                esp_error=esp_error
            )
        
        valid_stats = valid_metrics.compute()
        
        # Print summary
        epoch_time = time.time() - epoch_start_time
        print_epoch_summary(epoch, train_stats, valid_stats, epoch_time)
        
        # TensorBoard logging
        if writer is not None:
            for key, value in train_stats.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in valid_stats.items():
                writer.add_scalar(f"valid/{key}", value, epoch)
            writer.add_scalar("time/epoch", epoch_time, epoch)
        
        # Save checkpoints
        is_best = valid_stats['loss'] < best_valid_loss
        if is_best:
            best_valid_loss = valid_stats['loss']
        
        save_checkpoint(
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            epoch=epoch,
            metrics={'train': train_stats, 'valid': valid_stats},
            exp_dir=exp_dir,
            config=config,
            is_best=is_best
        )
    
    final_params = ema_params if train_config.use_ema else params
    return final_params, best_valid_loss


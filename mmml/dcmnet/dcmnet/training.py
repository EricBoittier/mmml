import functools
import pickle
import hashlib
import os
from pathlib import Path

import e3x
import jax
import jax.numpy as jnp
import optax
import numpy as np
from .loss import esp_mono_loss
from .electrostatics import calc_esp

from .data import prepare_batches, prepare_datasets

from typing import Callable, Any, Optional
from functools import partial

# Constants for ESP masking (matching loss.py)
BOHR_TO_ANGSTROM = 0.529177
ANGSTROM_TO_BOHR = 1.88973
RADII_TABLE = jnp.array(ase.data.covalent_radii)

# Try to enable lovely_jax for better array printing
try:
    import lovely_jax as lj
    lj.monkey_patch()
    print("lovely_jax enabled for enhanced array visualization")
except ImportError:
    print("lovely_jax not available, using standard JAX array printing")


@functools.partial(
    jax.jit,
    static_argnames=("model_apply", "optimizer_update", "batch_size", "esp_w", "chg_w", "ndcm", "distance_weighting", "distance_scale", "distance_min", "esp_magnitude_weighting", "charge_conservation_w"),
)
def train_step(
    model_apply, optimizer_update, batch, batch_size, opt_state, params, esp_w, chg_w, ndcm, clip_norm=None,
    distance_weighting=False, distance_scale=2.0, distance_min=0.5, esp_magnitude_weighting=False, charge_conservation_w=1.0
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
        loss, esp_pred, esp_target, esp_errors, loss_components = esp_mono_loss(
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
            atom_positions=batch.get("R"),  # Pass atom positions for distance weighting and masking
            atomic_numbers=batch.get("Z"),  # Pass atomic numbers for atomic radii masking
            atom_mask=batch.get("atom_mask"),  # Pass atom mask for dummy atom handling
            distance_weighting=distance_weighting,  # Use function parameter, not batch value
            distance_scale=distance_scale,  # Use function parameter, not batch value
            distance_min=distance_min,  # Use function parameter, not batch value
            esp_magnitude_weighting=esp_magnitude_weighting,  # Weight by ESP magnitude instead
            use_atomic_radii_mask=True,  # Enable atomic radii masking (critical for reducing ESP errors)
            charge_conservation_w=charge_conservation_w,  # Weight for charge conservation loss
        )
        return loss, (mono, dipo, esp_pred, esp_target, esp_errors, loss_components)

    (loss, (mono, dipo, esp_pred, esp_target, esp_errors, loss_components)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    if clip_norm is not None:
        grad = clip_grads_by_global_norm(grad, clip_norm)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, mono, dipo, esp_pred, esp_target, esp_errors, loss_components


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
    metrics_list = ['loss', 'esp_loss', 'mono_loss', 'charge_conservation_loss',
                    'esp_loss_weighted', 'mono_loss_weighted', 'charge_conservation_loss_weighted',
                    'mono_mae', 'mono_rmse', 'mono_mean', 'mono_std',
                    'esp_mae', 'esp_rmse', 'esp_pred_mean', 'esp_pred_std',
                    'esp_error_mean', 'esp_error_std']
    
    # Conversion factor: Hartree to kcal/mol
    HARTREE_TO_KCAL_MOL = 627.5
    
    for key in metrics_list:
        if key in train_stats and key in valid_stats:
            train_val = train_stats[key]
            valid_val = valid_stats[key]
            diff = valid_val - train_val
            
            # Convert ESP RMSE and MAE to kcal/mol/e
            if key in ['esp_mae', 'esp_rmse']:
                train_val_kcal = train_val * HARTREE_TO_KCAL_MOL
                valid_val_kcal = valid_val * HARTREE_TO_KCAL_MOL
                diff_kcal = diff * HARTREE_TO_KCAL_MOL
                print(f"{key:<20} {train_val:>15.6e} {valid_val:>15.6e} {diff:>15.6e}")
                print(f"{key+'_kcal':<20} {train_val_kcal:>15.6f} {valid_val_kcal:>15.6f} {diff_kcal:>15.6f} (kcal/mol/e)")
            else:
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
    
    # Warnings for concerning metrics
    warnings = []
    
    # Check for overfitting
    train_loss_val = train_stats.get('loss', 0)
    valid_loss_val = valid_stats.get('loss', 0)
    if valid_loss_val > 0 and train_loss_val > 0:
        loss_ratio = valid_loss_val / train_loss_val
        if loss_ratio > 2.0:
            warnings.append(f"⚠️  Severe overfitting: validation loss ({valid_loss_val:.2e}) is {loss_ratio:.1f}x higher than training loss ({train_loss_val:.2e})")
        elif loss_ratio > 1.5:
            warnings.append(f"⚠️  Overfitting detected: validation loss ({valid_loss_val:.2e}) is {loss_ratio:.1f}x higher than training loss ({train_loss_val:.2e})")
    
    # Check for negative R²
    train_r2_unmasked = train_stats.get('esp_r2_unmasked', 0)
    valid_r2_unmasked = valid_stats.get('esp_r2_unmasked', 0)
    train_r2_masked = train_stats.get('esp_r2_masked', 0)
    valid_r2_masked = valid_stats.get('esp_r2_masked', 0)
    
    if train_r2_unmasked < 0:
        warnings.append(f"⚠️  Training ESP R² is negative ({train_r2_unmasked:.4f}) - model performs worse than predicting the mean")
    if valid_r2_unmasked < 0:
        warnings.append(f"⚠️  Validation ESP R² is negative ({valid_r2_unmasked:.4f}) - model performs worse than predicting the mean")
    if train_r2_masked < -5:
        train_r2_unclamped = train_stats.get('r2_masked_unclamped', train_r2_masked)
        train_ss_res = train_stats.get('ss_res_masked', 0)
        train_ss_tot = train_stats.get('ss_tot_masked', 0)
        warnings.append(f"⚠️  Training masked ESP R² is very negative ({train_r2_masked:.4f}, unclamped: {train_r2_unclamped:.4f})")
        warnings.append(f"      SS_res={train_ss_res:.2e}, SS_tot={train_ss_tot:.2e}, ratio={train_ss_res/(train_ss_tot+1e-10):.2f} - check masking logic")
    if valid_r2_masked < -5:
        valid_r2_unclamped = valid_stats.get('r2_masked_unclamped', valid_r2_masked)
        valid_ss_res = valid_stats.get('ss_res_masked', 0)
        valid_ss_tot = valid_stats.get('ss_tot_masked', 0)
        warnings.append(f"⚠️  Validation masked ESP R² is very negative ({valid_r2_masked:.4f}, unclamped: {valid_r2_unclamped:.4f})")
        warnings.append(f"      SS_res={valid_ss_res:.2e}, SS_tot={valid_ss_tot:.2e}, ratio={valid_ss_res/(valid_ss_tot+1e-10):.2f} - check masking logic")
    
    # Check ESP error magnitude
    valid_esp_rmse = valid_stats.get('esp_rmse', 0)
    if valid_esp_rmse > 0.1:  # > 0.1 Ha/e = ~63 kcal/mol/e
        warnings.append(f"⚠️  High validation ESP RMSE ({valid_esp_rmse*627.5:.1f} kcal/mol/e) - model may not be learning ESP well")
    
    # Check monopole errors
    valid_mono_rmse = valid_stats.get('mono_rmse', 0)
    if valid_mono_rmse > 0.2:  # > 0.2 e
        warnings.append(f"⚠️  High validation monopole RMSE ({valid_mono_rmse:.4f} e) - model may not be learning charges well")
    
    # Check mask fraction
    valid_mask_fraction = valid_stats.get('esp_mask_fraction', 0)
    if valid_mask_fraction < 0.1:
        warnings.append(f"⚠️  Very low mask fraction ({valid_mask_fraction:.1%}) - most points are being masked out")
    elif valid_mask_fraction > 0.95:
        warnings.append(f"⚠️  Very high mask fraction ({valid_mask_fraction:.1%}) - masking may not be working")
    
    # Check for scale mismatch
    valid_std_ratio = valid_stats.get('std_ratio', 1.0)
    train_std_ratio = train_stats.get('std_ratio', 1.0)
    if valid_std_ratio < 0.5:
        warnings.append(f"⚠️  Scale mismatch: validation predicted std ({valid_stats.get('esp_pred_std', 0):.6f}) is "
                       f"{valid_std_ratio:.2f}x smaller than target std ({valid_stats.get('esp_target_std', 0):.6f}) - predictions have too little variance")
        if valid_std_ratio < 0.1:
            warnings.append(f"      CRITICAL: Predicted variance is {1/valid_std_ratio:.1f}x too small - model may be collapsing!")
            warnings.append(f"      Possible causes:")
            warnings.append(f"      - ESP loss weight ({train_stats.get('esp_loss_weighted', 0):.2e}) may be too high, causing gradient issues")
            warnings.append(f"      - Learning rate may be too high, causing instability")
            warnings.append(f"      - Model may not be expressive enough for ESP prediction")
            warnings.append(f"      - Check if predictions are nearly constant: pred_mean={valid_stats.get('esp_pred_mean', 0):.6f}, pred_std={valid_stats.get('esp_pred_std', 0):.6f}")
    elif valid_std_ratio > 2.0:
        warnings.append(f"⚠️  Scale mismatch: validation predicted std ({valid_stats.get('esp_pred_std', 0):.6f}) is "
                       f"{valid_std_ratio:.2f}x larger than target std ({valid_stats.get('esp_target_std', 0):.6f}) - predictions have too much variance")
    
    # Check for model collapse (very negative R² combined with scale mismatch)
    valid_r2_unmasked = valid_stats.get('esp_r2_unmasked', 0)
    train_r2_unmasked = train_stats.get('esp_r2_unmasked', 0)
    if valid_r2_unmasked < -0.5 and valid_std_ratio < 0.1:
        warnings.append(f"⚠️  MODEL COLLAPSE DETECTED: R²={valid_r2_unmasked:.4f}, std_ratio={valid_std_ratio:.4f}")
        warnings.append(f"      The model is predicting nearly constant values. Immediate actions:")
        warnings.append(f"      1. Reduce ESP loss weight (currently {train_stats.get('esp_loss_weighted', 0):.2e})")
        warnings.append(f"      2. Check learning rate (may need to reduce)")
        warnings.append(f"      3. Verify model architecture can express ESP variation")
        warnings.append(f"      4. Check if monopole predictions are reasonable (mono_rmse={valid_stats.get('mono_rmse', 0):.4f})")
    
    # Check for systematic bias
    valid_bias_relative = valid_stats.get('bias_relative', 0)
    train_bias_relative = train_stats.get('bias_relative', 0)
    if abs(valid_bias_relative) > 0.5:  # Bias > 0.5 * target_std
        warnings.append(f"⚠️  Systematic bias detected: validation error mean ({valid_stats.get('error_mean', 0):.6f} Ha/e) is "
                       f"{abs(valid_bias_relative):.2f}x the target std - predictions are systematically {'high' if valid_bias_relative > 0 else 'low'}")
    if abs(train_bias_relative) > 0.5:
        warnings.append(f"⚠️  Systematic bias detected: training error mean ({train_stats.get('error_mean', 0):.6f} Ha/e) is "
                       f"{abs(train_bias_relative):.2f}x the target std - predictions are systematically {'high' if train_bias_relative > 0 else 'low'}")
    
    # Print warnings if any
    if warnings:
        print(f"\n{'='*80}")
        print("WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
        print(f"{'='*80}")
    
    print(f"{'='*80}\n")


@functools.partial(
    jax.jit, static_argnames=("model_apply", "batch_size", "esp_w", "chg_w", "ndcm", "charge_conservation_w")
)
def eval_step(model_apply, batch, batch_size, params, esp_w, chg_w, ndcm, charge_conservation_w=1.0):
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
    loss, esp_pred, esp_target, esp_errors, loss_components = esp_mono_loss(
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
        atom_positions=batch.get("R"),
        atomic_numbers=batch.get("Z"),  # Pass atomic numbers for atomic radii masking
        atom_mask=batch.get("atom_mask"),  # Pass atom mask for dummy atom handling
        distance_weighting=False,  # Typically don't weight during evaluation
        distance_scale=2.0,
        distance_min=0.5,
        use_atomic_radii_mask=True,  # Enable atomic radii masking (critical for reducing ESP errors)
        charge_conservation_w=charge_conservation_w,  # Weight for charge conservation loss
    )
    return loss, mono, dipo, esp_pred, esp_target, esp_errors, loss_components


def verify_esp_grid_alignment(train_data, valid_data, num_atoms=60, verbose=True):
    """
    Verify ESP grid alignment and masking.
    
    Checks that ESP grids are properly aligned with molecular centers of mass
    and that masking is working correctly.
    
    Parameters
    ----------
    train_data : dict
        Training data dictionary
    valid_data : dict
        Validation data dictionary
    num_atoms : int
        Number of atoms per system (padded)
    verbose : bool
        Whether to print detailed information
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    
    def check_alignment(data, dataset_name):
        """Check ESP grid alignment for a dataset."""
        if "R" not in data or "vdw_surface" not in data:
            print(f"  ⚠️  {dataset_name}: Missing R or vdw_surface data")
            return
        
        n_samples = len(data["R"])
        if n_samples == 0:
            print(f"  ⚠️  {dataset_name}: No samples")
            return
        
        # Check first few samples
        n_check = min(5, n_samples)
        alignment_errors = []
        
        for i in range(n_check):
            R = np.array(data["R"][i])  # (num_atoms, 3)
            vdw = np.array(data["vdw_surface"][i])  # (n_grid, 3)
            
            # Get actual number of atoms
            if "N" in data:
                n_real = int(data["N"][i])
            else:
                # Infer from non-zero positions
                n_real = np.sum(np.any(R != 0, axis=1))
            
            # Compute center of mass from actual atoms
            R_real = R[:n_real]
            com = np.mean(R_real, axis=0)
            
            # Compute center of ESP grid
            grid_com = np.mean(vdw, axis=0)
            
            # Check alignment (should be close)
            alignment_error = np.linalg.norm(grid_com - com)
            alignment_errors.append(alignment_error)
        
        avg_error = np.mean(alignment_errors)
        max_error = np.max(alignment_errors)
        
        if avg_error > 0.1:  # More than 0.1 Angstrom
            print(f"  ⚠️  {dataset_name}: ESP grids may be misaligned")
            print(f"      Average COM alignment error: {avg_error:.4f} Å")
            print(f"      Max alignment error: {max_error:.4f} Å")
        else:
            print(f"  ✓  {dataset_name}: ESP grids aligned (avg error: {avg_error:.4f} Å)")
        
        # Check masking (if atomic radii masking is used)
        if "Z" in data:
            n_masked_samples = 0
            for i in range(n_check):
                R = np.array(data["R"][i])
                Z = np.array(data["Z"][i])
                vdw = np.array(data["vdw_surface"][i])
                
                if "N" in data:
                    n_real = int(data["N"][i])
                else:
                    n_real = np.sum(Z != 0)
                
                R_real = R[:n_real]
                Z_real = Z[:n_real]
                
                # Compute distances from grid to atoms
                distances = cdist(vdw, R_real)  # (n_grid, n_real)
                min_distances = np.min(distances, axis=1)  # (n_grid,)
                
                # Check if any points are too close (within 2 * covalent radius)
                import ase.data
                radii = np.array([ase.data.covalent_radii[z] if z > 0 else 0 for z in Z_real])
                cutoff = 2.0 * radii.max() if len(radii) > 0 else 1.0
                
                n_too_close = np.sum(min_distances < cutoff)
                if n_too_close > 0:
                    n_masked_samples += 1
            
            if n_masked_samples > 0:
                print(f"  ⚠️  {dataset_name}: {n_masked_samples}/{n_check} samples have ESP points too close to atoms")
                print(f"      (These should be masked by atomic_radii_mask)")
            else:
                print(f"  ✓  {dataset_name}: ESP masking OK (no points too close to atoms)")
    
    print("Checking ESP grid alignment and masking...")
    check_alignment(train_data, "Training")
    check_alignment(valid_data, "Validation")


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
    mono_imputation_fn=None,
    distance_weighting=False,
    distance_scale=2.0,
    distance_min=0.5,
    esp_magnitude_weighting=False,
    charge_conservation_w=1.0,
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
    distance_weighting : bool, optional
        Whether to apply distance-based weighting to ESP loss. Errors further from atoms
        will have HIGHER weight (reversed from typical). By default False
    distance_scale : float, optional
        Scale parameter for distance weighting (in Angstroms). Larger values give slower
        increase with distance. Weight = exp((distance - distance_min) / distance_scale),
        normalized to have mean=1. By default 2.0
    distance_min : float, optional
        Minimum distance for weighting (in Angstroms). Distances below this are clamped
        to avoid singularities. By default 0.5
    esp_magnitude_weighting : bool, optional
        Whether to weight by ESP magnitude instead of distance. Errors at points with
        larger |ESP| values will have LOWER weight. This reduces the impact of points
        where nuclear-electron shielding occurs and ESP approaches singularity (near
        atomic nuclei). By default False
        
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
    
    # Print data statistics before training
    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)
    
    # Training data statistics
    train_n_samples = len(train_data["R"])
    train_n_atoms = len(train_data["Z"][0]) if len(train_data["Z"]) > 0 else 0
    
    # Compute monopole statistics safely
    try:
        train_mono_data = jnp.array(train_data.get("mono", []))
        if train_mono_data.size > 0:
            train_mono_mean = float(jnp.mean(train_mono_data))
            train_mono_std = float(jnp.std(train_mono_data))
        else:
            train_mono_mean = train_mono_std = 0.0
    except:
        train_mono_mean = train_mono_std = 0.0
    
    # Compute ESP statistics safely
    try:
        train_esp_flat = jnp.concatenate([jnp.ravel(e) for e in train_data.get("esp", [])])
        if train_esp_flat.size > 0:
            train_esp_mean = float(jnp.mean(train_esp_flat))
            train_esp_std = float(jnp.std(train_esp_flat))
            train_esp_min = float(jnp.min(train_esp_flat))
            train_esp_max = float(jnp.max(train_esp_flat))
        else:
            train_esp_mean = train_esp_std = train_esp_min = train_esp_max = 0.0
    except:
        train_esp_mean = train_esp_std = train_esp_min = train_esp_max = 0.0
    
    try:
        train_n_grid = int(jnp.mean(jnp.array([len(e) for e in train_data.get("esp", [])]))) if len(train_data.get("esp", [])) > 0 else 0
    except:
        train_n_grid = 0
    
    print(f"\nTraining Data:")
    print(f"  Samples: {train_n_samples}")
    print(f"  Atoms per sample: {train_n_atoms}")
    print(f"  Grid points per sample: {train_n_grid}")
    print(f"  Monopoles: mean={train_mono_mean:.6f} e, std={train_mono_std:.6f} e")
    print(f"  ESP: mean={train_esp_mean:.6f} Ha/e ({train_esp_mean*627.5:.3f} kcal/mol/e)")
    print(f"        std={train_esp_std:.6f} Ha/e ({train_esp_std*627.5:.3f} kcal/mol/e)")
    print(f"        range=[{train_esp_min:.6f}, {train_esp_max:.6f}] Ha/e")
    print(f"        range=[{train_esp_min*627.5:.3f}, {train_esp_max*627.5:.3f}] kcal/mol/e")
    
    # Validation data statistics
    valid_n_samples = len(valid_data["R"])
    valid_n_atoms = len(valid_data["Z"][0]) if len(valid_data["Z"]) > 0 else 0
    
    try:
        valid_mono_data = jnp.array(valid_data.get("mono", []))
        if valid_mono_data.size > 0:
            valid_mono_mean = float(jnp.mean(valid_mono_data))
            valid_mono_std = float(jnp.std(valid_mono_data))
        else:
            valid_mono_mean = valid_mono_std = 0.0
    except:
        valid_mono_mean = valid_mono_std = 0.0
    
    try:
        valid_esp_flat = jnp.concatenate([jnp.ravel(e) for e in valid_data.get("esp", [])])
        if valid_esp_flat.size > 0:
            valid_esp_mean = float(jnp.mean(valid_esp_flat))
            valid_esp_std = float(jnp.std(valid_esp_flat))
            valid_esp_min = float(jnp.min(valid_esp_flat))
            valid_esp_max = float(jnp.max(valid_esp_flat))
        else:
            valid_esp_mean = valid_esp_std = valid_esp_min = valid_esp_max = 0.0
    except:
        valid_esp_mean = valid_esp_std = valid_esp_min = valid_esp_max = 0.0
    
    try:
        valid_n_grid = int(jnp.mean(jnp.array([len(e) for e in valid_data.get("esp", [])]))) if len(valid_data.get("esp", [])) > 0 else 0
    except:
        valid_n_grid = 0
    
    print(f"\nValidation Data:")
    print(f"  Samples: {valid_n_samples}")
    print(f"  Atoms per sample: {valid_n_atoms}")
    print(f"  Grid points per sample: {valid_n_grid}")
    print(f"  Monopoles: mean={valid_mono_mean:.6f} e, std={valid_mono_std:.6f} e")
    print(f"  ESP: mean={valid_esp_mean:.6f} Ha/e ({valid_esp_mean*627.5:.3f} kcal/mol/e)")
    print(f"        std={valid_esp_std:.6f} Ha/e ({valid_esp_std*627.5:.3f} kcal/mol/e)")
    print(f"        range=[{valid_esp_min:.6f}, {valid_esp_max:.6f}] Ha/e")
    print(f"        range=[{valid_esp_min*627.5:.3f}, {valid_esp_max*627.5:.3f}] kcal/mol/e")
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {train_n_samples // batch_size}")
    print(f"  ESP weight: {esp_w}")
    print(f"  Charge weight: {chg_w}")
    print(f"  Distance weighting: {distance_weighting}")
    print(f"  ESP magnitude weighting: {esp_magnitude_weighting}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of DCM per atom: {ndcm}")
    print(f"  Total parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    print("="*80 + "\n")
    
    print("Preparing batches")
    print("..................")
    # Pre-impute monopoles for all data before training (more efficient)
    if mono_imputation_fn is not None:
        print("\nPreprocessing monopoles...")
        train_data = preprocess_monopoles(train_data, mono_imputation_fn, num_atoms=num_atoms, batch_size=batch_size, verbose=True)
        valid_data = preprocess_monopoles(valid_data, mono_imputation_fn, num_atoms=num_atoms, batch_size=batch_size, verbose=True)
        # Set imputation function to None since monopoles are already imputed
        mono_imputation_fn = None
    
    # Verify ESP grid alignment and masking (before training starts)
    print("\n" + "="*80)
    print("ESP Grid Verification")
    print("="*80)
    verify_esp_grid_alignment(train_data, valid_data, num_atoms=num_atoms, verbose=True)
    print("="*80 + "\n")
    
    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size, num_atoms=num_atoms, mono_imputation_fn=mono_imputation_fn)

    print("\nTraining")
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
        train_loss_components = []
        
        for i, batch in enumerate(train_batches):
            params, opt_state, loss, mono, dipo, esp_pred, esp_target, esp_error, loss_components = train_step(
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
                distance_weighting=distance_weighting,
                distance_scale=distance_scale,
                distance_min=distance_min,
                esp_magnitude_weighting=esp_magnitude_weighting,
                charge_conservation_w=charge_conservation_w,
            )

            # Update EMA parameters (non-blocking for better GPU utilization)
            ema_params = update_ema_params(ema_params, params, ema_decay)
            
            # Only block at end of epoch for statistics, not every batch
            
            train_loss += (loss - train_loss) / (i + 1)
            
            # Collect predictions for statistics
            train_mono_preds.append(mono)
            train_mono_targets.append(batch["mono"])
            train_esp_preds.append(esp_pred)
            train_esp_targets.append(esp_target)
            train_esp_errors.append(esp_error)
            train_loss_components.append(loss_components)

        # Concatenate all predictions and targets (block once at end of epoch)
        train_mono_preds = jnp.concatenate(train_mono_preds, axis=0)
        train_mono_targets = jnp.concatenate(train_mono_targets, axis=0)
        train_esp_preds = jnp.concatenate([jnp.ravel(e) for e in train_esp_preds])
        train_esp_targets = jnp.concatenate([jnp.ravel(e) for e in train_esp_targets])
        train_esp_errors = jnp.concatenate([jnp.ravel(e) for e in train_esp_errors])
        train_esp_masks_list = train_esp_masks  # Save list before concatenation
        train_esp_masks = jnp.concatenate([jnp.ravel(e) for e in train_esp_masks]) if len(train_esp_masks_list) > 0 else jnp.ones_like(train_esp_targets)
        
        # Debug: Check mask statistics (only on first epoch to avoid spam)
        if epoch == 1:
            mask_sum = float(jnp.sum(train_esp_masks))
            mask_size = float(train_esp_masks.size)
            mask_mean = float(jnp.mean(train_esp_masks))
            mask_min = float(jnp.min(train_esp_masks))
            mask_max = float(jnp.max(train_esp_masks))
            n_valid = float(jnp.sum(train_esp_masks > 0.5))
            n_total = float(train_esp_masks.size)
            print(f"  DEBUG: train_esp_masks: size={mask_size}, sum={mask_sum}, mean={mask_mean:.6f}, min={mask_min:.6f}, max={mask_max:.6f}")
            print(f"  DEBUG: train_esp_masks: n_valid={n_valid}, n_total={n_total}, fraction={n_valid/n_total:.6f}")
            # Also check first few mask values from first batch
            if len(train_esp_masks_list) > 0:
                first_mask = train_esp_masks_list[0]
                print(f"  DEBUG: first_mask shape={first_mask.shape}, first_mask[:10]={first_mask.ravel()[:10]}")
        
        # Block once for statistics computation (better GPU utilization)
        jax.block_until_ready(train_esp_errors)
        
        # Aggregate loss components
        train_loss_components_agg = {}
        for component_name in train_loss_components[0].keys():
            values = jnp.array([comp[component_name] for comp in train_loss_components])
            train_loss_components_agg[component_name] = float(jnp.mean(values))
        
        # Block once for statistics computation (better GPU utilization)
        jax.block_until_ready(train_esp_errors)
        
        # Aggregate loss components
        train_loss_components_agg = {}
        for component_name in train_loss_components[0].keys():
            values = jnp.array([comp[component_name] for comp in train_loss_components])
            train_loss_components_agg[component_name] = float(jnp.mean(values))
        
        # Compute training statistics
        train_mono_stats = compute_statistics(train_mono_preds.sum(axis=-1), train_mono_targets)
        
        # Compute R² for ESP (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        # SS_res = sum((y_pred - y_true)²)
        # SS_tot = sum((y_true - y_mean)²)
        
        # Unmasked R² (all points)
        train_esp_ss_res_unmasked = jnp.sum((train_esp_preds - train_esp_targets)**2)
        train_esp_target_mean_unmasked = jnp.mean(train_esp_targets)
        train_esp_ss_tot_unmasked = jnp.sum((train_esp_targets - train_esp_target_mean_unmasked)**2)
        # Clamp to avoid division by zero and extreme values
        train_esp_r2_unmasked = 1.0 - (train_esp_ss_res_unmasked / jnp.maximum(train_esp_ss_tot_unmasked, 1e-10))
        # Clamp R² to reasonable range [-10, 1] to avoid extreme values
        train_esp_r2_unmasked = jnp.clip(train_esp_r2_unmasked, -10.0, 1.0)
        
        # Masked R² (only valid points)
        # Ensure masks and predictions have matching shapes for boolean indexing
        # Initialize diagnostic variables
        train_esp_r2_masked_unclamped_val = float(train_esp_r2_unmasked)
        train_esp_ss_res_masked_val = 0.0
        train_esp_ss_tot_masked_val = 0.0
        
        if train_esp_masks.shape != train_esp_preds.shape:
            # Reshape mask to match predictions if needed
            if train_esp_masks.size == train_esp_preds.size:
                train_esp_masks = train_esp_masks.reshape(train_esp_preds.shape)
            else:
                # If sizes don't match, fall back to unmasked R²
                train_esp_r2_masked = train_esp_r2_unmasked
                train_mask_fraction = 0.0
        else:
            mask_bool = train_esp_masks > 0.5
            n_masked_points = float(jnp.sum(mask_bool))
            n_total_points = float(train_esp_masks.size)
            train_mask_fraction = n_masked_points / n_total_points if n_total_points > 0 else 0.0
            if n_masked_points > 10:  # Need at least 10 points for meaningful R²
                train_esp_preds_masked = train_esp_preds[mask_bool]
                train_esp_targets_masked = train_esp_targets[mask_bool]
                train_esp_ss_res_masked = jnp.sum((train_esp_preds_masked - train_esp_targets_masked)**2)
                train_esp_target_mean_masked = jnp.mean(train_esp_targets_masked)
                train_esp_ss_tot_masked = jnp.sum((train_esp_targets_masked - train_esp_target_mean_masked)**2)
                # Clamp to avoid division by zero and extreme values
                train_esp_r2_masked_unclamped = 1.0 - (train_esp_ss_res_masked / jnp.maximum(train_esp_ss_tot_masked, 1e-10))
                # Clamp R² to reasonable range [-10, 1] to avoid extreme values
                train_esp_r2_masked = jnp.clip(train_esp_r2_masked_unclamped, -10.0, 1.0)
                
                # Store diagnostics for warnings
                train_esp_r2_masked_unclamped_val = float(train_esp_r2_masked_unclamped)
                train_esp_ss_res_masked_val = float(train_esp_ss_res_masked)
                train_esp_ss_tot_masked_val = float(train_esp_ss_tot_masked)
            else:
                # Too few masked points - use unmasked R²
                train_esp_r2_masked = train_esp_r2_unmasked
                train_esp_r2_masked_unclamped_val = float(train_esp_r2_unmasked)
                train_esp_ss_res_masked_val = 0.0
                train_esp_ss_tot_masked_val = 0.0
        
        # Check for systematic bias and scale mismatches
        train_esp_pred_mean = float(jnp.mean(train_esp_preds))
        train_esp_target_mean = float(jnp.mean(train_esp_targets))
        train_esp_pred_std = float(jnp.std(train_esp_preds))
        train_esp_target_std = float(jnp.std(train_esp_targets))
        train_esp_error_mean = float(jnp.mean(train_esp_errors))
        
        # Relative bias (as fraction of target std)
        train_esp_bias_relative = train_esp_error_mean / (train_esp_target_std + 1e-10)
        # Scale mismatch (ratio of stds)
        train_esp_std_ratio = train_esp_pred_std / (train_esp_target_std + 1e-10)
        
        train_esp_stats = {
            'mae': float(jnp.mean(jnp.abs(train_esp_errors))),
            'rmse': float(jnp.sqrt(jnp.mean(train_esp_errors**2))),
            'r2_unmasked': float(train_esp_r2_unmasked),
            'r2_masked': float(train_esp_r2_masked),
            'r2_masked_unclamped': train_esp_r2_masked_unclamped_val,
            'ss_res_masked': train_esp_ss_res_masked_val,
            'ss_tot_masked': train_esp_ss_tot_masked_val,
            'mask_fraction': float(train_mask_fraction),  # Fraction of points that passed masking
            'pred_mean': train_esp_pred_mean,
            'pred_std': train_esp_pred_std,
            'target_mean': train_esp_target_mean,
            'error_mean': train_esp_error_mean,
            'error_std': float(jnp.std(train_esp_errors)),
            'bias_relative': train_esp_bias_relative,  # Bias relative to target std
            'std_ratio': train_esp_std_ratio,  # Ratio of pred std to target std
        }
        
        # Verify monopoles are non-zero (after imputation)
        train_mono_sum_abs = float(jnp.mean(jnp.abs(train_mono_targets)))
        if train_mono_sum_abs < 1e-6:
            print(f"⚠️  WARNING: Training monopoles are all zeros (mean_abs={train_mono_sum_abs:.2e})")
        else:
            print(f"✓  Training monopoles OK (mean_abs={train_mono_sum_abs:.6f} e)")
        
        train_stats = {
            'loss': float(train_loss),
            'esp_loss': train_loss_components_agg['esp_loss'],
            'mono_loss': train_loss_components_agg['mono_loss'],
            'charge_conservation_loss': train_loss_components_agg['charge_conservation_loss'],
            'esp_loss_weighted': train_loss_components_agg['esp_loss_weighted'],
            'mono_loss_weighted': train_loss_components_agg['mono_loss_weighted'],
            'charge_conservation_loss_weighted': train_loss_components_agg['charge_conservation_loss_weighted'],
            'mono_mae': train_mono_stats['mae'],
            'mono_rmse': train_mono_stats['rmse'],
            'mono_mean': train_mono_stats['mean'],
            'mono_std': train_mono_stats['std'],
            'mono_min': train_mono_stats['min'],
            'mono_max': train_mono_stats['max'],
            'esp_mae': train_esp_stats['mae'],
            'esp_rmse': train_esp_stats['rmse'],
            'esp_r2_unmasked': train_esp_stats['r2_unmasked'],
            'esp_r2_masked': train_esp_stats['r2_masked'],
            'esp_mask_fraction': train_esp_stats['mask_fraction'],
            'esp_pred_mean': train_esp_stats['pred_mean'],
            'esp_pred_std': train_esp_stats['pred_std'],
            'esp_target_mean': train_esp_stats['target_mean'],
            'esp_target_std': train_esp_target_std,  # Add target std for warnings
            'esp_error_mean': train_esp_stats['error_mean'],
            'esp_error_std': train_esp_stats['error_std'],
            'bias_relative': train_esp_stats['bias_relative'],
            'std_ratio': train_esp_stats['std_ratio'],
        }

        # Evaluate on validation set.
        valid_loss = 0.0
        valid_mono_preds = []
        valid_mono_targets = []
        valid_esp_preds = []
        valid_esp_targets = []
        valid_esp_errors = []
        valid_loss_components = []
        
        for i, batch in enumerate(valid_batches):
            loss, mono, dipo, esp_pred, esp_target, esp_error, loss_components = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=ema_params,
                esp_w=esp_w,
                chg_w=chg_w,
                ndcm=ndcm,
                charge_conservation_w=charge_conservation_w,
            )
            # Accumulate loss (non-blocking for better GPU utilization)
            valid_loss += (loss - valid_loss) / (i + 1)
            
            # Collect predictions for statistics
            valid_mono_preds.append(mono)
            valid_mono_targets.append(batch["mono"])
            valid_esp_preds.append(esp_pred)
            valid_esp_targets.append(esp_target)
            valid_esp_errors.append(esp_error)
            valid_loss_components.append(loss_components)

        # Concatenate all predictions and targets (block once at end of epoch)
        valid_mono_preds = jnp.concatenate(valid_mono_preds, axis=0)
        valid_mono_targets = jnp.concatenate(valid_mono_targets, axis=0)
        valid_esp_preds = jnp.concatenate([jnp.ravel(e) for e in valid_esp_preds])
        valid_esp_targets = jnp.concatenate([jnp.ravel(e) for e in valid_esp_targets])
        valid_esp_errors = jnp.concatenate([jnp.ravel(e) for e in valid_esp_errors])
        valid_esp_masks_list = valid_esp_masks  # Save list before concatenation
        valid_esp_masks = jnp.concatenate([jnp.ravel(e) for e in valid_esp_masks]) if len(valid_esp_masks_list) > 0 else jnp.ones_like(valid_esp_targets)
        
        # Debug: Check mask statistics (only on first epoch to avoid spam)
        if epoch == 1:
            mask_sum = float(jnp.sum(valid_esp_masks))
            mask_size = float(valid_esp_masks.size)
            mask_mean = float(jnp.mean(valid_esp_masks))
            mask_min = float(jnp.min(valid_esp_masks))
            mask_max = float(jnp.max(valid_esp_masks))
            n_valid = float(jnp.sum(valid_esp_masks > 0.5))
            n_total = float(valid_esp_masks.size)
            print(f"  DEBUG: valid_esp_masks: size={mask_size}, sum={mask_sum}, mean={mask_mean:.6f}, min={mask_min:.6f}, max={mask_max:.6f}")
            print(f"  DEBUG: valid_esp_masks: n_valid={n_valid}, n_total={n_total}, fraction={n_valid/n_total:.6f}")
            # Also check first few mask values from first batch
            if len(valid_esp_masks_list) > 0:
                first_mask = valid_esp_masks_list[0]
                print(f"  DEBUG: first_mask shape={first_mask.shape}, first_mask[:10]={first_mask.ravel()[:10]}")
        
        # Block once for statistics computation (better GPU utilization)
        jax.block_until_ready(valid_esp_errors)
        
        # Aggregate loss components
        valid_loss_components_agg = {}
        for component_name in valid_loss_components[0].keys():
            values = jnp.array([comp[component_name] for comp in valid_loss_components])
            valid_loss_components_agg[component_name] = float(jnp.mean(values))
        
        # Block once for statistics computation (better GPU utilization)
        jax.block_until_ready(valid_esp_errors)
        
        # Aggregate loss components
        valid_loss_components_agg = {}
        for component_name in valid_loss_components[0].keys():
            values = jnp.array([comp[component_name] for comp in valid_loss_components])
            valid_loss_components_agg[component_name] = float(jnp.mean(values))
        
        # Compute validation statistics
        valid_mono_stats = compute_statistics(valid_mono_preds.sum(axis=-1), valid_mono_targets)
        
        # Compute R² for ESP (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        # SS_res = sum((y_pred - y_true)²)
        # SS_tot = sum((y_true - y_mean)²)
        
        # Unmasked R² (all points)
        valid_esp_ss_res_unmasked = jnp.sum((valid_esp_preds - valid_esp_targets)**2)
        valid_esp_target_mean_unmasked = jnp.mean(valid_esp_targets)
        valid_esp_ss_tot_unmasked = jnp.sum((valid_esp_targets - valid_esp_target_mean_unmasked)**2)
        # Clamp to avoid division by zero and extreme values
        valid_esp_r2_unmasked = 1.0 - (valid_esp_ss_res_unmasked / jnp.maximum(valid_esp_ss_tot_unmasked, 1e-10))
        # Clamp R² to reasonable range [-10, 1] to avoid extreme values
        valid_esp_r2_unmasked = jnp.clip(valid_esp_r2_unmasked, -10.0, 1.0)
        
        # Masked R² (only valid points)
        # Ensure masks and predictions have matching shapes for boolean indexing
        # Initialize diagnostic variables
        valid_esp_r2_masked_unclamped_val = float(valid_esp_r2_unmasked)
        valid_esp_ss_res_masked_val = 0.0
        valid_esp_ss_tot_masked_val = 0.0
        
        if valid_esp_masks.shape != valid_esp_preds.shape:
            # Reshape mask to match predictions if needed
            if valid_esp_masks.size == valid_esp_preds.size:
                valid_esp_masks = valid_esp_masks.reshape(valid_esp_preds.shape)
            else:
                # If sizes don't match, fall back to unmasked R²
                valid_esp_r2_masked = valid_esp_r2_unmasked
                valid_mask_fraction = 0.0
        else:
            mask_bool = valid_esp_masks > 0.5
            n_masked_points = float(jnp.sum(mask_bool))
            n_total_points = float(valid_esp_masks.size)
            valid_mask_fraction = n_masked_points / n_total_points if n_total_points > 0 else 0.0
            if n_masked_points > 10:  # Need at least 10 points for meaningful R²
                valid_esp_preds_masked = valid_esp_preds[mask_bool]
                valid_esp_targets_masked = valid_esp_targets[mask_bool]
                valid_esp_ss_res_masked = jnp.sum((valid_esp_preds_masked - valid_esp_targets_masked)**2)
                valid_esp_target_mean_masked = jnp.mean(valid_esp_targets_masked)
                valid_esp_ss_tot_masked = jnp.sum((valid_esp_targets_masked - valid_esp_target_mean_masked)**2)
                # Clamp to avoid division by zero and extreme values
                valid_esp_r2_masked_unclamped = 1.0 - (valid_esp_ss_res_masked / jnp.maximum(valid_esp_ss_tot_masked, 1e-10))
                # Clamp R² to reasonable range [-10, 1] to avoid extreme values
                valid_esp_r2_masked = jnp.clip(valid_esp_r2_masked_unclamped, -10.0, 1.0)
                
                # Store diagnostics for warnings
                valid_esp_r2_masked_unclamped_val = float(valid_esp_r2_masked_unclamped)
                valid_esp_ss_res_masked_val = float(valid_esp_ss_res_masked)
                valid_esp_ss_tot_masked_val = float(valid_esp_ss_tot_masked)
            else:
                # Too few masked points - use unmasked R²
                valid_esp_r2_masked = valid_esp_r2_unmasked
                valid_esp_r2_masked_unclamped_val = float(valid_esp_r2_unmasked)
                valid_esp_ss_res_masked_val = 0.0
                valid_esp_ss_tot_masked_val = 0.0
        
        # Check for systematic bias and scale mismatches
        valid_esp_pred_mean = float(jnp.mean(valid_esp_preds))
        valid_esp_target_mean = float(jnp.mean(valid_esp_targets))
        valid_esp_pred_std = float(jnp.std(valid_esp_preds))
        valid_esp_target_std = float(jnp.std(valid_esp_targets))
        valid_esp_error_mean = float(jnp.mean(valid_esp_errors))
        
        # Relative bias (as fraction of target std)
        valid_esp_bias_relative = valid_esp_error_mean / (valid_esp_target_std + 1e-10)
        # Scale mismatch (ratio of stds)
        valid_esp_std_ratio = valid_esp_pred_std / (valid_esp_target_std + 1e-10)
        
        valid_esp_stats = {
            'mae': float(jnp.mean(jnp.abs(valid_esp_errors))),
            'rmse': float(jnp.sqrt(jnp.mean(valid_esp_errors**2))),
            'r2_unmasked': float(valid_esp_r2_unmasked),
            'r2_masked': float(valid_esp_r2_masked),
            'r2_masked_unclamped': valid_esp_r2_masked_unclamped_val,
            'ss_res_masked': valid_esp_ss_res_masked_val,
            'ss_tot_masked': valid_esp_ss_tot_masked_val,
            'mask_fraction': float(valid_mask_fraction),  # Fraction of points that passed masking
            'pred_mean': valid_esp_pred_mean,
            'pred_std': valid_esp_pred_std,
            'target_mean': valid_esp_target_mean,
            'error_mean': valid_esp_error_mean,
            'error_std': float(jnp.std(valid_esp_errors)),
            'bias_relative': valid_esp_bias_relative,  # Bias relative to target std
            'std_ratio': valid_esp_std_ratio,  # Ratio of pred std to target std
        }
        
        # Verify monopoles are non-zero (after imputation)
        valid_mono_sum_abs = float(jnp.mean(jnp.abs(valid_mono_targets)))
        if valid_mono_sum_abs < 1e-6:
            print(f"⚠️  WARNING: Validation monopoles are all zeros (mean_abs={valid_mono_sum_abs:.2e})")
        else:
            print(f"✓  Validation monopoles OK (mean_abs={valid_mono_sum_abs:.6f} e)")
        
        valid_stats = {
            'loss': float(valid_loss),
            'esp_loss': valid_loss_components_agg['esp_loss'],
            'mono_loss': valid_loss_components_agg['mono_loss'],
            'charge_conservation_loss': valid_loss_components_agg['charge_conservation_loss'],
            'esp_loss_weighted': valid_loss_components_agg['esp_loss_weighted'],
            'mono_loss_weighted': valid_loss_components_agg['mono_loss_weighted'],
            'charge_conservation_loss_weighted': valid_loss_components_agg['charge_conservation_loss_weighted'],
            'mono_mae': valid_mono_stats['mae'],
            'mono_rmse': valid_mono_stats['rmse'],
            'mono_mean': valid_mono_stats['mean'],
            'mono_std': valid_mono_stats['std'],
            'mono_min': valid_mono_stats['min'],
            'mono_max': valid_mono_stats['max'],
            'esp_mae': valid_esp_stats['mae'],
            'esp_rmse': valid_esp_stats['rmse'],
            'esp_r2_unmasked': valid_esp_stats['r2_unmasked'],
            'esp_r2_masked': valid_esp_stats['r2_masked'],
            'esp_mask_fraction': valid_esp_stats['mask_fraction'],
            'esp_pred_mean': valid_esp_stats['pred_mean'],
            'esp_pred_std': valid_esp_stats['pred_std'],
            'esp_target_mean': valid_esp_stats['target_mean'],
            'esp_target_std': valid_esp_target_std,  # Add target std for warnings
            'esp_error_mean': valid_esp_stats['error_mean'],
            'esp_error_std': valid_esp_stats['error_std'],
            'bias_relative': valid_esp_stats['bias_relative'],
            'std_ratio': valid_esp_stats['std_ratio'],
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


def preprocess_monopoles(data, mono_imputation_fn, num_atoms=60, batch_size=1000, verbose=False, cache_dir=None):
    """
    Pre-impute monopoles for all samples in the dataset before training.
    
    This is more efficient than imputing during batch preparation, as it only
    needs to be done once before the training loop starts.
    
    Parameters
    ----------
    data : dict
        Data dictionary containing 'R', 'Z', etc.
    mono_imputation_fn : callable
        Function to impute monopoles. Takes a batch dict and returns monopoles.
    num_atoms : int
        Number of atoms per system
    batch_size : int
        Batch size to use for imputation
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    dict
        Updated data dictionary with imputed monopoles
        
    Raises
    ------
    RuntimeError
        If monopole imputation fails for any batch
    """
    # Check if monopoles need imputation
    if "mono" in data:
        mono_data = jnp.array(data["mono"])
        mono_sum = jnp.abs(mono_data).sum()
        if mono_sum > 1e-6:
            # Monopoles are not all zeros, no need to impute
            if verbose:
                print(f"  Monopoles already present (sum_abs={float(mono_sum):.2e}). Skipping imputation.")
            return data
    
    # Infer padded size from data shape (this is what the data format expects)
    # Check R or Z shape to determine padded size
    padded_size = num_atoms  # Default to num_atoms
    if "R" in data and len(data["R"]) > 0:
        first_r = np.array(data["R"][0])
        if first_r.ndim == 2:
            padded_size = first_r.shape[0]  # (n_atoms, 3)
        elif first_r.ndim == 1:
            # Flattened, try to infer from total size
            padded_size = num_atoms
    elif "Z" in data and len(data["Z"]) > 0:
        first_z = np.array(data["Z"][0])
        if isinstance(first_z, (list, tuple, np.ndarray)):
            padded_size = len(first_z)
    
    # Infer actual number of atoms from data (may be less than padded_size due to padding)
    # Check the actual number of atoms from Z or R data (non-zero elements)
    actual_num_atoms = padded_size  # Default to padded_size
    if "Z" in data and len(data["Z"]) > 0:
        first_z = np.array(data["Z"][0])
        if isinstance(first_z, (list, tuple, np.ndarray)):
            # Count non-zero elements (actual atoms, not padding)
            non_zero_mask = np.array(first_z) != 0
            if np.any(non_zero_mask):
                actual_num_atoms = int(np.sum(non_zero_mask))
            else:
                actual_num_atoms = len(first_z)  # Fallback if all zeros
    
    # Create cache path based on data hash
    n_samples = len(data["R"])
    cache_path = None
    if cache_dir is not None:
        # Create a hash from sample count and actual_num_atoms to identify the dataset
        cache_key = f"mono_imputed_n{n_samples}_atoms{actual_num_atoms}_padded{padded_size}.npz"
        cache_path = Path(cache_dir) / cache_key
        
        # Try to load from cache
        if cache_path.exists():
            try:
                if verbose:
                    print(f"  Loading imputed monopoles from cache: {cache_path}")
                cached_data = np.load(cache_path)
                cached_mono = jnp.array(cached_data["mono"])
                
                # Verify shape matches
                if cached_mono.shape == (n_samples, padded_size):
                    data_updated = data.copy()
                    data_updated["mono"] = cached_mono
                    if verbose:
                        cached_mean = float(jnp.mean(jnp.abs(cached_mono)))
                        print(f"  ✓ Loaded from cache. Mean_abs={cached_mean:.6f} e")
                    return data_updated
                else:
                    if verbose:
                        print(f"  Warning: Cache shape mismatch ({cached_mono.shape} vs expected ({n_samples}, {padded_size})). Recomputing...")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to load cache ({e}). Recomputing...")
    
    # Need to impute monopoles
    if verbose:
        print(f"  Imputing monopoles for {n_samples} samples...")
        print(f"  Using padded_size={padded_size} (from data shape), actual_atoms={actual_num_atoms} per molecule, batch_size={batch_size}")
    
    # Create batches for imputation
    imputed_monopoles = []
    
    # Use a dummy key for deterministic batching (order doesn't matter for imputation)
    dummy_key = jax.random.PRNGKey(0)
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_indices = list(range(i, end_idx))
        actual_batch_size = len(batch_indices)
        
        # Create a batch dict for this subset
        batch_dict = {}
        for k, v in data.items():
            if k == "R":
                batch_dict[k] = jnp.array([jnp.array(data["R"][idx]) for idx in batch_indices]).reshape(-1, 3)
            elif k == "Z":
                batch_dict[k] = jnp.array([jnp.array(data["Z"][idx]) for idx in batch_indices]).reshape(-1)
            elif k == "N":
                batch_dict[k] = jnp.array([jnp.array(data["N"][idx]) for idx in batch_indices])
            else:
                # For other fields, try to index if possible
                try:
                    batch_dict[k] = jnp.array([jnp.array(data[k][idx]) for idx in batch_indices])
                except:
                    pass
        
        # Create message passing indices for this batch
        # Use padded_size for message passing indices (imputation function expects padded format)
        batch_segments = jnp.repeat(jnp.arange(actual_batch_size), padded_size)
        offsets = jnp.arange(actual_batch_size) * padded_size
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(padded_size)
        dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
        src_idx = (src_idx + offsets[:, None]).reshape(-1)
        
        batch_dict["dst_idx"] = dst_idx
        batch_dict["src_idx"] = src_idx
        batch_dict["batch_segments"] = batch_segments
        
        # Impute monopoles for this batch
        try:
            imputed_batch = mono_imputation_fn(batch_dict)
            
            # The imputation function might return monopoles for different numbers of atoms.
            # It could return: actual_num_atoms, num_atoms (padded), or even 60 (hardcoded padding).
            # We need to infer the number of atoms per molecule from the output size.
            atoms_per_molecule = imputed_batch.size // actual_batch_size
            
            if atoms_per_molecule < actual_num_atoms:
                raise ValueError(
                    f"Imputation function returned too few atoms: got {atoms_per_molecule} atoms per molecule, "
                    f"but need at least {actual_num_atoms} (actual atoms)"
                )
            
            # Reshape to (actual_batch_size, atoms_per_molecule)
            imputed_reshaped = imputed_batch.reshape(actual_batch_size, atoms_per_molecule)
            
            # Extract only the actual atoms (first actual_num_atoms)
            imputed_actual = imputed_reshaped[:, :actual_num_atoms]
            
            # Pad with zeros to match padded_size (the padded size expected by the data format)
            if actual_num_atoms < padded_size:
                padding = jnp.zeros((actual_batch_size, padded_size - actual_num_atoms), dtype=imputed_actual.dtype)
                imputed_padded = jnp.concatenate([imputed_actual, padding], axis=1)
            else:
                imputed_padded = imputed_actual[:, :padded_size]  # Trim if somehow larger
            
            # Flatten back to (actual_batch_size * padded_size,) for concatenation
            imputed_monopoles.append(imputed_padded.reshape(-1))
        except Exception as e:
            # If imputation fails, raise an error rather than silently using zeros
            raise RuntimeError(
                f"Failed to impute monopoles for batch {i//batch_size + 1} "
                f"(samples {i} to {end_idx-1}): {e}"
            ) from e
    
    # Concatenate all imputed monopoles
    all_imputed = jnp.concatenate(imputed_monopoles)
    
    # Verify the shape is correct before reshaping
    expected_size = n_samples * padded_size
    if all_imputed.size != expected_size:
        raise ValueError(
            f"Imputed monopoles size mismatch: got {all_imputed.size} elements, "
            f"expected {expected_size} (n_samples={n_samples} * padded_size={padded_size})"
        )
    
    # Reshape to (n_samples, padded_size) format
    try:
        imputed_reshaped = all_imputed.reshape(n_samples, padded_size)
    except Exception as e:
        raise ValueError(
            f"Failed to reshape imputed monopoles: got shape {all_imputed.shape} "
            f"(size {all_imputed.size}), trying to reshape to ({n_samples}, {padded_size}) "
            f"(size {n_samples * padded_size}). Check that padded_size={padded_size} is correct."
        ) from e
    
    # Update data dictionary
    data_updated = data.copy()
    data_updated["mono"] = imputed_reshaped
    
    if verbose:
        imputed_mean = float(jnp.mean(jnp.abs(all_imputed)))
        imputed_std = float(jnp.std(all_imputed))
        print(f"  ✓ Monopole imputation complete. Mean_abs={imputed_mean:.6f} e, Std={imputed_std:.6f} e")
    
    # Save to cache for next time
    if cache_path is not None:
        try:
            if verbose:
                print(f"  Saving imputed monopoles to cache: {cache_path}")
            np.savez(cache_path, mono=imputed_reshaped)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to save cache ({e}). Continuing anyway...")
    
    return data_updated


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

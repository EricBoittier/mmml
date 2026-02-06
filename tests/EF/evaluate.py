#!/usr/bin/env python3
"""
Evaluate trained model on test/validation data.

Creates scatter plots and computes metrics (MAE, RMSE, R²) for energy predictions.

Note: Expects input data in eV/angstrom units. All errors are reported in kcal/mol.

Usage:
    python evaluate.py --params params.json --data data-full.npz --output-dir results/
"""

import os

# --- Environment (must be set before importing jax) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

import e3x
import matplotlib.pyplot as plt
import seaborn as sns
import functools

# Import model and utilities from training script
import sys
from pathlib import Path
# Add parent directory to path to import training module
sys.path.insert(0, str(Path(__file__).parent))

from training import MessagePassingModel, prepare_batches

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Unit conversion constants
# Data is expected in eV/angstrom units
EV_TO_KCAL_MOL = 23.06035  # 1 eV = 23.06035 kcal/mol

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--params", type=str, default="params.json",
                       help="Path to parameters JSON file (can be params-UUID.json or params.json)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON file (will be auto-detected from params UUID if not provided)")
    parser.add_argument("--data", type=str, default="data-full.npz",
                       help="Path to dataset NPZ file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for plots and metrics")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--num-test", type=int, default=None,
                       help="Number of test samples to use (None = use all)")
    parser.add_argument("--model-config", type=str, default=None,
                       help="Path to model config JSON (deprecated: use --config)")
    parser.add_argument("--features", type=int, default=None,
                       help="Model features (will be inferred from params/config if not provided)")
    parser.add_argument("--max-degree", type=int, default=None,
                       help="Max degree (default: 2)")
    parser.add_argument("--num-iterations", type=int, default=None,
                       help="Number of iterations (default: 2)")
    parser.add_argument("--num-basis-functions", type=int, default=None,
                       help="Number of basis functions (default: 64)")
    parser.add_argument("--cutoff", type=float, default=None,
                       help="Cutoff radius (default: 10.0)")
    parser.add_argument("--max-atomic-number", type=int, default=None,
                       help="Max atomic number (default: 55)")
    
    args = parser.parse_args()
    return args



def load_params(params_path):
    """Load parameters from JSON file."""
    with open(params_path, 'r') as f:
        params_dict = json.load(f)
    
    # Convert numpy arrays back from lists
    def convert_to_jax(obj):
        if isinstance(obj, dict):
            return {k: convert_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            arr = np.array(obj)
            if arr.dtype == np.float64:
                return jnp.array(arr, dtype=jnp.float32)
            elif arr.dtype == np.int64:
                return jnp.array(arr, dtype=jnp.int32)
            return jnp.array(arr)
        return obj
    
    params = convert_to_jax(params_dict)
    return params


def compute_metrics(predictions, targets, convert_to_kcal_mol=True):
    """Compute error metrics.
    
    Parameters
    ----------
    predictions : array-like
        Predicted values (in eV)
    targets : array-like
        Target values (in eV)
    convert_to_kcal_mol : bool, default=True
        If True, convert errors to kcal/mol for display
    
    Returns
    -------
    dict
        Dictionary containing metrics in eV (and kcal/mol if convert_to_kcal_mol=True)
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    errors = predictions - targets
    
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # R² score
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Mean error and std
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_abs_error = np.max(np.abs(errors))
    
    result = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_abs_error': max_abs_error,
    }
    
    # Add kcal/mol conversions
    if convert_to_kcal_mol:
        result['mae_kcal_mol'] = mae * EV_TO_KCAL_MOL
        result['rmse_kcal_mol'] = rmse * EV_TO_KCAL_MOL
        result['mean_error_kcal_mol'] = mean_error * EV_TO_KCAL_MOL
        result['std_error_kcal_mol'] = std_error * EV_TO_KCAL_MOL
        result['max_abs_error_kcal_mol'] = max_abs_error * EV_TO_KCAL_MOL
    
    return result


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size"))
def inference_step(model_apply, batch, batch_size, params):
    """Single inference step - returns energy, forces, and dipole."""
    # Compute energy and dipole
    energy, dipole = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx_flat=batch["dst_idx_flat"],
        src_idx_flat=batch["src_idx_flat"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    
    # Compute forces
    from training import energy_and_forces
    _, forces, _ = energy_and_forces(
        model_apply, params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx_flat=batch["dst_idx_flat"],
        src_idx_flat=batch["src_idx_flat"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    
    return energy, forces, dipole


def compute_force_metrics(predictions, targets):
    """Compute force error metrics per component and magnitude."""
    predictions = np.asarray(predictions)  # (N, 3)
    targets = np.asarray(targets)  # (N, 3)
    
    errors = predictions - targets  # (N, 3)
    
    # Per component errors
    mae_x = np.mean(np.abs(errors[:, 0]))
    mae_y = np.mean(np.abs(errors[:, 1]))
    mae_z = np.mean(np.abs(errors[:, 2]))
    mae_components = np.array([mae_x, mae_y, mae_z])
    
    # Magnitude errors
    pred_mags = np.linalg.norm(predictions, axis=1)
    target_mags = np.linalg.norm(targets, axis=1)
    mag_errors = pred_mags - target_mags
    mae_magnitude = np.mean(np.abs(mag_errors))
    rmse_magnitude = np.sqrt(np.mean(mag_errors**2))
    
    # Overall force MAE (average over all components)
    mae_overall = np.mean(np.abs(errors))
    rmse_overall = np.sqrt(np.mean(errors**2))
    
    return {
        'mae_overall': mae_overall,
        'rmse_overall': rmse_overall,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_z': mae_z,
        'mae_components': mae_components,
        'mae_magnitude': mae_magnitude,
        'rmse_magnitude': rmse_magnitude,
    }


def evaluate_dataset(model, params, data, batch_size=64, dataset_name="test"):
    """Evaluate model on a dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} dataset")
    print(f"{'='*60}")
    
    has_forces = 'forces' in data and data['forces'] is not None
    has_dipoles = ('dipoles' in data and data['dipoles'] is not None) or ('D' in data and data['D'] is not None)
    
    # Prepare batches
    key = jax.random.PRNGKey(42)
    batches = prepare_batches(key, data, batch_size)
    
    # Collect predictions and targets
    all_energy_predictions = []
    all_energy_targets = []
    all_force_predictions = []
    all_force_targets = []
    all_dipole_predictions = []
    all_dipole_targets = []
    
    print(f"Processing {len(batches)} batches...")
    for i, batch in enumerate(batches):
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(batches)}")
        
        energy_pred, forces_pred, dipole_pred = inference_step(
            model_apply=model.apply,
            batch=batch,
            batch_size=batch_size,
            params=params,
        )
        
        # Convert to numpy
        energy_pred = np.asarray(energy_pred)
        forces_pred = np.asarray(forces_pred)
        dipole_pred = np.asarray(dipole_pred)
        energy_targets = np.asarray(batch["energies"]).reshape(-1)
        
        all_energy_predictions.append(energy_pred)
        all_energy_targets.append(energy_targets)
        
        if has_forces:
            force_targets = np.asarray(batch["forces"])
            all_force_predictions.append(forces_pred)
            all_force_targets.append(force_targets)
        
        # Check for dipoles in batch (could be "dipoles" or "D" key)
        batch_has_dipoles = "dipoles" in batch or "D" in batch
        if has_dipoles and batch_has_dipoles:
            # Get dipoles from batch (prefer "dipoles" key, fallback to "D")
            dipole_targets = np.asarray(batch.get("dipoles", batch.get("D")))
            all_dipole_predictions.append(dipole_pred)
            all_dipole_targets.append(dipole_targets)
    
    # Concatenate all predictions and targets
    all_energy_predictions = np.concatenate(all_energy_predictions)
    all_energy_targets = np.concatenate(all_energy_targets)
    
    # Compute energy metrics
    energy_metrics = compute_metrics(all_energy_predictions, all_energy_targets)
    
    print(f"\nEnergy Metrics for {dataset_name}:")
    print(f"  MAE:  {energy_metrics['mae_kcal_mol']:.4f} kcal/mol ({energy_metrics['mae']:.6f} eV)")
    print(f"  RMSE: {energy_metrics['rmse_kcal_mol']:.4f} kcal/mol ({energy_metrics['rmse']:.6f} eV)")
    print(f"  R²:   {energy_metrics['r2']:.6f}")
    print(f"  Mean Error: {energy_metrics['mean_error_kcal_mol']:.4f} kcal/mol ({energy_metrics['mean_error']:.6f} eV)")
    print(f"  Std Error:  {energy_metrics['std_error_kcal_mol']:.4f} kcal/mol ({energy_metrics['std_error']:.6f} eV)")
    print(f"  Max |Error|: {energy_metrics['max_abs_error_kcal_mol']:.4f} kcal/mol ({energy_metrics['max_abs_error']:.6f} eV)")
    
    # Compute force metrics if available
    force_metrics = {}
    if has_forces:
        all_force_predictions = np.concatenate(all_force_predictions)
        all_force_targets = np.concatenate(all_force_targets)
        
        force_metrics = compute_force_metrics(all_force_predictions, all_force_targets)
        
        # Convert force metrics to kcal/mol/angstrom
        EV_ANG_TO_KCAL_MOL_ANG = EV_TO_KCAL_MOL  # Same conversion factor
        force_metrics['mae_overall_kcal_mol_ang'] = force_metrics['mae_overall'] * EV_ANG_TO_KCAL_MOL_ANG
        force_metrics['rmse_overall_kcal_mol_ang'] = force_metrics['rmse_overall'] * EV_ANG_TO_KCAL_MOL_ANG
        force_metrics['mae_x_kcal_mol_ang'] = force_metrics['mae_x'] * EV_ANG_TO_KCAL_MOL_ANG
        force_metrics['mae_y_kcal_mol_ang'] = force_metrics['mae_y'] * EV_ANG_TO_KCAL_MOL_ANG
        force_metrics['mae_z_kcal_mol_ang'] = force_metrics['mae_z'] * EV_ANG_TO_KCAL_MOL_ANG
        force_metrics['mae_magnitude_kcal_mol_ang'] = force_metrics['mae_magnitude'] * EV_ANG_TO_KCAL_MOL_ANG
        force_metrics['rmse_magnitude_kcal_mol_ang'] = force_metrics['rmse_magnitude'] * EV_ANG_TO_KCAL_MOL_ANG
        
        print(f"\nForce Metrics for {dataset_name}:")
        print(f"  MAE (overall): {force_metrics['mae_overall_kcal_mol_ang']:.4f} kcal/(mol·Å) ({force_metrics['mae_overall']:.6f} eV/Å)")
        print(f"  RMSE (overall): {force_metrics['rmse_overall_kcal_mol_ang']:.4f} kcal/(mol·Å) ({force_metrics['rmse_overall']:.6f} eV/Å)")
        print(f"  MAE (magnitude): {force_metrics['mae_magnitude_kcal_mol_ang']:.4f} kcal/(mol·Å) ({force_metrics['mae_magnitude']:.6f} eV/Å)")
        print(f"  RMSE (magnitude): {force_metrics['rmse_magnitude_kcal_mol_ang']:.4f} kcal/(mol·Å) ({force_metrics['rmse_magnitude']:.6f} eV/Å)")
        print(f"  MAE per component:")
        print(f"    X: {force_metrics['mae_x_kcal_mol_ang']:.4f} kcal/(mol·Å) ({force_metrics['mae_x']:.6f} eV/Å)")
        print(f"    Y: {force_metrics['mae_y_kcal_mol_ang']:.4f} kcal/(mol·Å) ({force_metrics['mae_y']:.6f} eV/Å)")
        print(f"    Z: {force_metrics['mae_z_kcal_mol_ang']:.4f} kcal/(mol·Å) ({force_metrics['mae_z']:.6f} eV/Å)")
    else:
        all_force_predictions = None
        all_force_targets = None
    
    # Compute dipole metrics if available
    dipole_metrics = {}
    if has_dipoles and len(all_dipole_predictions) > 0:
        all_dipole_predictions = np.concatenate(all_dipole_predictions)
        all_dipole_targets = np.concatenate(all_dipole_targets)
        
        # Compute dipole metrics (similar to force metrics)
        dipole_errors = all_dipole_predictions - all_dipole_targets  # (N, 3)
        dipole_metrics['mae_overall'] = np.mean(np.abs(dipole_errors))
        dipole_metrics['rmse_overall'] = np.sqrt(np.mean(dipole_errors**2))
        dipole_metrics['mae_x'] = np.mean(np.abs(dipole_errors[:, 0]))
        dipole_metrics['mae_y'] = np.mean(np.abs(dipole_errors[:, 1]))
        dipole_metrics['mae_z'] = np.mean(np.abs(dipole_errors[:, 2]))
        
        # Magnitude errors
        dipole_pred_mag = np.linalg.norm(all_dipole_predictions, axis=1)
        dipole_target_mag = np.linalg.norm(all_dipole_targets, axis=1)
        dipole_mag_errors = dipole_pred_mag - dipole_target_mag
        dipole_metrics['mae_magnitude'] = np.mean(np.abs(dipole_mag_errors))
        dipole_metrics['rmse_magnitude'] = np.sqrt(np.mean(dipole_mag_errors**2))
        
        # R² score (flattened components)
        dipole_pred_flat = all_dipole_predictions.flatten()
        dipole_target_flat = all_dipole_targets.flatten()
        ss_res = np.sum((dipole_pred_flat - dipole_target_flat)**2)
        ss_tot = np.sum((dipole_target_flat - np.mean(dipole_target_flat))**2)
        dipole_metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Correlation coefficient R
        dipole_metrics['r'] = np.corrcoef(dipole_pred_flat, dipole_target_flat)[0, 1]
        
        # Convert to Debye (assuming input is in Debye, or atomic units)
        # Note: dipole is typically in Debye units, so no conversion needed
        # But we'll add Debye suffix for clarity
        dipole_metrics['mae_overall_debye'] = dipole_metrics['mae_overall']
        dipole_metrics['rmse_overall_debye'] = dipole_metrics['rmse_overall']
        dipole_metrics['mae_x_debye'] = dipole_metrics['mae_x']
        dipole_metrics['mae_y_debye'] = dipole_metrics['mae_y']
        dipole_metrics['mae_z_debye'] = dipole_metrics['mae_z']
        dipole_metrics['mae_magnitude_debye'] = dipole_metrics['mae_magnitude']
        dipole_metrics['rmse_magnitude_debye'] = dipole_metrics['rmse_magnitude']
        
        print(f"\nDipole Metrics for {dataset_name}:")
        print(f"  MAE (overall): {dipole_metrics['mae_overall_debye']:.4f} Debye")
        print(f"  RMSE (overall): {dipole_metrics['rmse_overall_debye']:.4f} Debye")
        print(f"  R²:   {dipole_metrics['r2']:.6f}")
        print(f"  R:    {dipole_metrics['r']:.6f}")
        print(f"  MAE (magnitude): {dipole_metrics['mae_magnitude_debye']:.4f} Debye")
        print(f"  RMSE (magnitude): {dipole_metrics['rmse_magnitude_debye']:.4f} Debye")
        print(f"  MAE per component:")
        print(f"    X: {dipole_metrics['mae_x_debye']:.4f} Debye")
        print(f"    Y: {dipole_metrics['mae_y_debye']:.4f} Debye")
        print(f"    Z: {dipole_metrics['mae_z_debye']:.4f} Debye")
    else:
        all_dipole_predictions = None
        all_dipole_targets = None
    
    # Combine metrics
    all_metrics = {**energy_metrics, **force_metrics, **dipole_metrics}
    
    return all_energy_predictions, all_energy_targets, all_force_predictions, all_force_targets, all_dipole_predictions, all_dipole_targets, all_metrics


def plot_scatter(predictions, targets, metrics, title="Energy Predictions", 
                 save_path=None, ax=None):
    """Create scatter plot of predictions vs targets."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert to kcal/mol for plotting
    targets_kcal = targets * EV_TO_KCAL_MOL
    predictions_kcal = predictions * EV_TO_KCAL_MOL
    
    # Scatter plot
    ax.scatter(targets_kcal, predictions_kcal, alpha=0.5, s=20, edgecolors='none')
    
    # Perfect prediction line
    lims = [
        min(targets_kcal.min(), predictions_kcal.min()),
        max(targets_kcal.max(), predictions_kcal.max())
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect prediction')
    
    # Labels and title
    ax.set_xlabel('True Energy (kcal/mol)', fontsize=12)
    ax.set_ylabel('Predicted Energy (kcal/mol)', fontsize=12)
    mae_kcal = metrics.get('mae_kcal_mol', metrics['mae'] * EV_TO_KCAL_MOL)
    rmse_kcal = metrics.get('rmse_kcal_mol', metrics['rmse'] * EV_TO_KCAL_MOL)
    ax.set_title(f'{title}\nMAE: {mae_kcal:.4f} kcal/mol | RMSE: {rmse_kcal:.4f} kcal/mol | R²: {metrics["r2"]:.4f}', 
                fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_error_distribution(predictions, targets, metrics, title="Error Distribution",
                           save_path=None, ax=None):
    """Plot error distribution histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    errors = predictions - targets
    errors_kcal = errors * EV_TO_KCAL_MOL
    
    ax.hist(errors_kcal, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    mean_error_kcal = metrics.get('mean_error_kcal_mol', metrics['mean_error'] * EV_TO_KCAL_MOL)
    ax.axvline(mean_error_kcal, color='g', linestyle='--', linewidth=2, 
               label=f"Mean: {mean_error_kcal:.4f} kcal/mol")
    
    ax.set_xlabel('Error (Predicted - True) [kcal/mol]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    std_error_kcal = metrics.get('std_error_kcal_mol', metrics['std_error'] * EV_TO_KCAL_MOL)
    ax.set_title(f'{title}\nStd: {std_error_kcal:.4f} kcal/mol', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_residuals(predictions, targets, metrics, title="Residual Plot",
                  save_path=None, ax=None):
    """Plot residuals vs predicted values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    residuals = predictions - targets
    residuals_kcal = residuals * EV_TO_KCAL_MOL
    predictions_kcal = predictions * EV_TO_KCAL_MOL
    
    ax.scatter(predictions_kcal, residuals_kcal, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Zero residual')
    mean_error_kcal = metrics.get('mean_error_kcal_mol', metrics['mean_error'] * EV_TO_KCAL_MOL)
    ax.axhline(mean_error_kcal, color='g', linestyle='--', linewidth=2,
               label=f"Mean: {mean_error_kcal:.4f} kcal/mol")
    
    ax.set_xlabel('Predicted Energy (kcal/mol)', fontsize=12)
    ax.set_ylabel('Residual (Predicted - True) [kcal/mol]', fontsize=12)
    ax.set_title(f'{title}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_force_scatter(predictions, targets, metrics, save_path=None, ax=None):
    """Create scatter plot of force components (flattened)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten all force components (X, Y, Z for all atoms)
    pred_flat = predictions.flatten() * EV_TO_KCAL_MOL
    target_flat = targets.flatten() * EV_TO_KCAL_MOL
    
    ax.scatter(target_flat, pred_flat, alpha=0.5, s=20, edgecolors='none')
    
    lims = [min(target_flat.min(), pred_flat.min()), max(target_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect prediction')
    
    # Compute R² and R (correlation coefficient)
    errors = pred_flat - target_flat
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((target_flat - np.mean(target_flat))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Pearson correlation coefficient R
    correlation_matrix = np.corrcoef(target_flat, pred_flat)
    r = correlation_matrix[0, 1] if correlation_matrix.size > 1 else 0.0
    
    ax.set_xlabel('True Force Component (kcal/(mol·Å))', fontsize=12)
    ax.set_ylabel('Predicted Force Component (kcal/(mol·Å))', fontsize=12)
    mae_overall = metrics.get('mae_overall_kcal_mol_ang', metrics.get('mae_overall', 0.0) * EV_TO_KCAL_MOL)
    rmse_overall = metrics.get('rmse_overall_kcal_mol_ang', metrics.get('rmse_overall', 0.0) * EV_TO_KCAL_MOL)
    ax.set_title(f'Force Component Predictions\nMAE: {mae_overall:.4f} | RMSE: {rmse_overall:.4f} kcal/(mol·Å) | R²: {r2:.6f} | R: {r:.6f}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_force_component_errors(predictions, targets, metrics, save_path=None, ax=None):
    """Plot force errors per component."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    errors = (predictions - targets) * EV_TO_KCAL_MOL  # Convert to kcal/(mol·Å)
    
    components = ['X', 'Y', 'Z']
    mae_values = [
        metrics.get('mae_x_kcal_mol_ang', metrics['mae_x'] * EV_TO_KCAL_MOL),
        metrics.get('mae_y_kcal_mol_ang', metrics['mae_y'] * EV_TO_KCAL_MOL),
        metrics.get('mae_z_kcal_mol_ang', metrics['mae_z'] * EV_TO_KCAL_MOL),
    ]
    
    bars = ax.bar(components, mae_values, alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_ylabel('MAE (kcal/(mol·Å))', fontsize=12)
    ax.set_title('Force Error per Component', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_force_magnitude_scatter(predictions, targets, metrics, save_path=None, ax=None):
    """Plot force magnitude errors vs magnitude."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    pred_mags = np.linalg.norm(predictions, axis=1) * EV_TO_KCAL_MOL
    target_mags = np.linalg.norm(targets, axis=1) * EV_TO_KCAL_MOL
    mag_errors = (pred_mags - target_mags)
    
    ax.scatter(target_mags, mag_errors, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    
    ax.set_xlabel('True Force Magnitude (kcal/(mol·Å))', fontsize=12)
    ax.set_ylabel('Magnitude Error (kcal/(mol·Å))', fontsize=12)
    ax.set_title('Force Magnitude Error vs Magnitude', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_force_error_distribution(predictions, targets, metrics, save_path=None, ax=None):
    """Plot distribution of force errors."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    errors = (predictions - targets) * EV_TO_KCAL_MOL  # Convert to kcal/(mol·Å)
    errors_flat = errors.flatten()
    
    ax.hist(errors_flat, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    mean_error = np.mean(errors_flat)
    ax.axvline(mean_error, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    
    ax.set_xlabel('Force Error (kcal/(mol·Å))', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    std_error = np.std(errors_flat)
    ax.set_title(f'Force Error Distribution\nStd: {std_error:.4f} kcal/(mol·Å)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_dipole_scatter(predictions, targets, metrics, save_path=None, ax=None):
    """Create scatter plot of dipole components (flattened)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten all dipole components (X, Y, Z for all molecules)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    ax.scatter(target_flat, pred_flat, alpha=0.5, s=20, edgecolors='none')
    
    lims = [min(target_flat.min(), pred_flat.min()), max(target_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect prediction')
    
    # Compute R² and R (correlation coefficient)
    errors = pred_flat - target_flat
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((target_flat - np.mean(target_flat))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Pearson correlation coefficient R
    correlation_matrix = np.corrcoef(target_flat, pred_flat)
    r = correlation_matrix[0, 1] if correlation_matrix.size > 1 else 0.0
    
    ax.set_xlabel('True Dipole Component (Debye)', fontsize=12)
    ax.set_ylabel('Predicted Dipole Component (Debye)', fontsize=12)
    mae_overall = metrics.get('mae_overall_debye', metrics.get('mae_overall', 0.0))
    rmse_overall = metrics.get('rmse_overall_debye', metrics.get('rmse_overall', 0.0))
    ax.set_title(f'Dipole Component Predictions\nMAE: {mae_overall:.4f} | RMSE: {rmse_overall:.4f} Debye | R²: {r2:.6f} | R: {r:.6f}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_dipole_component_errors(predictions, targets, metrics, save_path=None, ax=None):
    """Plot dipole errors per component."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    errors = predictions - targets  # Already in Debye
    
    components = ['X', 'Y', 'Z']
    mae_values = [
        metrics.get('mae_x_debye', metrics.get('mae_x', 0.0)),
        metrics.get('mae_y_debye', metrics.get('mae_y', 0.0)),
        metrics.get('mae_z_debye', metrics.get('mae_z', 0.0)),
    ]
    
    bars = ax.bar(components, mae_values, alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_ylabel('MAE (Debye)', fontsize=12)
    ax.set_title('Dipole Error per Component', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_dipole_magnitude_scatter(predictions, targets, metrics, save_path=None, ax=None):
    """Plot dipole magnitude errors vs magnitude."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    pred_mags = np.linalg.norm(predictions, axis=1)
    target_mags = np.linalg.norm(targets, axis=1)
    mag_errors = pred_mags - target_mags
    
    ax.scatter(target_mags, mag_errors, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    
    ax.set_xlabel('True Dipole Magnitude (Debye)', fontsize=12)
    ax.set_ylabel('Error (Predicted - True) [Debye]', fontsize=12)
    mae_mag = metrics.get('mae_magnitude_debye', metrics.get('mae_magnitude', 0.0))
    ax.set_title(f'Dipole Magnitude Error vs Magnitude\nMAE: {mae_mag:.4f} Debye', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_dipole_error_distribution(predictions, targets, metrics, save_path=None, ax=None):
    """Plot distribution of dipole errors."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    errors = predictions - targets  # (N, 3)
    error_mags = np.linalg.norm(errors, axis=1)  # (N,)
    
    ax.hist(error_mags, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    
    mae_overall = metrics.get('mae_overall_debye', metrics.get('mae_overall', 0.0))
    rmse_overall = metrics.get('rmse_overall_debye', metrics.get('rmse_overall', 0.0))
    ax.set_xlabel('Dipole Error Magnitude (Debye)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Dipole Error Distribution\nMAE: {mae_overall:.4f} | RMSE: {rmse_overall:.4f} Debye', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_dipole_component_comparison(predictions, targets, metrics, save_path=None, ax=None):
    """Plot dipole component predictions vs targets."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    components = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (comp, color) in enumerate(zip(components, colors)):
        pred_comp = predictions[:, i]
        target_comp = targets[:, i]
        
        ax.scatter(target_comp, pred_comp, alpha=0.5, s=20, label=f'{comp} component', 
                  edgecolors='none', c=color)
    
    # Perfect prediction line
    all_targets = targets.flatten()
    all_preds = predictions.flatten()
    lims = [min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())]
    ax.plot(lims, lims, 'k--', alpha=0.75, linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('True Dipole Component (Debye)', fontsize=12)
    ax.set_ylabel('Predicted Dipole Component (Debye)', fontsize=12)
    mae_overall = metrics.get('mae_overall_debye', metrics.get('mae_overall', 0.0))
    ax.set_title(f'Dipole Component Predictions\nMAE: {mae_overall:.4f} Debye', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def plot_force_component_comparison(predictions, targets, metrics, save_path=None, ax=None):
    """Plot comparison of force components."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    errors = (predictions - targets) * EV_TO_KCAL_MOL
    
    # Flatten all components for comparison
    errors_x = errors[:, 0].flatten()
    errors_y = errors[:, 1].flatten()
    errors_z = errors[:, 2].flatten()
    
    ax.scatter(errors_x, errors_y, alpha=0.3, s=10, label='X vs Y', c='blue')
    ax.scatter(errors_x, errors_z, alpha=0.3, s=10, label='X vs Z', c='red')
    ax.scatter(errors_y, errors_z, alpha=0.3, s=10, label='Y vs Z', c='green')
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Force Error (kcal/(mol·Å))', fontsize=12)
    ax.set_ylabel('Force Error (kcal/(mol·Å))', fontsize=12)
    ax.set_title('Force Component Error Comparison', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return ax


def main():
    args = get_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    print(f"Parameters: {args.params}")
    print(f"Data: {args.data}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    dataset = np.load(args.data, allow_pickle=True)
    print(f"Dataset keys: {dataset.files}")
    print(f"Dataset shapes:")
    for key in dataset.files:
        print(f"  {key}: {dataset[key].shape}")
    
    # Load parameters
    print(f"\nLoading parameters from {args.params}...")
    params = load_params(args.params)
    print("✓ Parameters loaded")
    
    # Try to find config file from UUID in params filename
    config_path = args.config
    if config_path is None:
        # Try to extract UUID from params filename
        params_path = Path(args.params)
        if params_path.stem.startswith('params-'):
            # Extract UUID from filename like params-UUID.json
            uuid_part = params_path.stem.replace('params-', '')
            config_candidate = params_path.parent / f'config-{uuid_part}.json'
            if config_candidate.exists():
                config_path = str(config_candidate)
                print(f"✓ Found matching config file: {config_path}")
        elif args.model_config:
            # Fallback to deprecated argument
            config_path = args.model_config
        else:
            # Try config.json symlink
            config_candidate = params_path.parent / 'config.json'
            if config_candidate.exists():
                config_path = str(config_candidate)
                print(f"✓ Found config.json symlink: {config_path}")
    
    # Determine model config (try to load from config file, then infer from params, then use defaults)
    model_config = {
        'max_degree': 2,
        'num_iterations': 2,
        'num_basis_functions': 64,
        'cutoff': 10.0,
        'max_atomic_number': 55,
        'include_pseudotensors': True,
    }
    
    # Try to load from config file first
    config_loaded = False
    if config_path and Path(config_path).exists():
        print(f"\nLoading model config from {config_path}...")
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            if 'model' in saved_config:
                model_config.update(saved_config['model'])
                print(f"✓ Loaded model config from {config_path}")
                if 'uuid' in saved_config:
                    print(f"  Config UUID: {saved_config['uuid']}")
                config_loaded = True
            else:
                # Config file might be in old format
                model_config.update(saved_config)
                print(f"✓ Loaded model config from {config_path}")
                config_loaded = True
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    elif config_path:
        print(f"Warning: Config file {config_path} not found, will infer from params")
    
    # Try to infer features from params if not loaded from config or features missing
    if not config_loaded or 'features' not in model_config or model_config.get('features') is None:
        try:
            def get_shape(obj):
                """Get shape from JAX array, numpy array, or list."""
                if hasattr(obj, 'shape'):
                    return obj.shape
                elif isinstance(obj, (list, tuple)):
                    if len(obj) > 0 and isinstance(obj[0], (list, tuple)):
                        # Nested list, try to get shape
                        return tuple(len(obj) if isinstance(obj, (list, tuple)) else 1 for _ in range(4))
                    return None
            return None

        except Exception as e:
            import traceback
            print(f"Warning: Error inferring features: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            model_config['features'] = 64
    
    # Allow user to override with model config file
    if args.model_config:
        with open(args.model_config, 'r') as f:
            user_config = json.load(f)
        model_config.update(user_config)
        print(f"✓ Loaded model config from {args.model_config}")
    
    # Allow command-line overrides
    if args.features is not None:
        model_config['features'] = args.features
    if args.max_degree is not None:
        model_config['max_degree'] = args.max_degree
    if args.num_iterations is not None:
        model_config['num_iterations'] = args.num_iterations
    if args.num_basis_functions is not None:
        model_config['num_basis_functions'] = args.num_basis_functions
    if args.cutoff is not None:
        model_config['cutoff'] = args.cutoff
    if args.max_atomic_number is not None:
        model_config['max_atomic_number'] = args.max_atomic_number
    
    print(f"\nModel configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = MessagePassingModel(**model_config)
    
    # Prepare test data
    print(f"\nPreparing test data...")
    key = jax.random.PRNGKey(0)
    
    # Use all data as test if num_test is None, otherwise use first num_test
    if args.num_test is None:
        indices = None
        test_data = {
            'atomic_numbers': jnp.asarray(dataset["Z"], dtype=jnp.int32),
            'positions': jnp.asarray(dataset["R"], dtype=jnp.float32),
            'electric_field': jnp.asarray(dataset["Ef"], dtype=jnp.float32),
            'energies': jnp.asarray(dataset["E"], dtype=jnp.float32),
        }
        # Handle case where R might have extra dimension
        if test_data['positions'].ndim == 4 and test_data['positions'].shape[1] == 1:
            test_data['positions'] = test_data['positions'].squeeze(axis=1)
    else:
        indices = np.arange(min(args.num_test, len(dataset["R"])))
        test_data = {
            'atomic_numbers': jnp.asarray(dataset["Z"][indices], dtype=jnp.int32),
            'positions': jnp.asarray(dataset["R"][indices], dtype=jnp.float32),
            'electric_field': jnp.asarray(dataset["Ef"][indices], dtype=jnp.float32),
            'energies': jnp.asarray(dataset["E"][indices], dtype=jnp.float32),
        }
        if test_data['positions'].ndim == 4 and test_data['positions'].shape[1] == 1:
            test_data['positions'] = test_data['positions'].squeeze(axis=1)
    
    # Handle forces: F has shape (num_data, 1, N, 3) - squeeze out the extra dimension
    if 'F' in dataset.files:
        forces_raw = jnp.asarray(dataset["F"], dtype=jnp.float32)
        if forces_raw.ndim == 4 and forces_raw.shape[1] == 1:
            forces_raw = forces_raw.squeeze(axis=1)  # (num_data, N, 3)
        
        if indices is None:
            test_data['forces'] = forces_raw
        else:
            test_data['forces'] = forces_raw[indices]
    else:
        print("Warning: No forces (F) found in dataset. Force evaluation will be skipped.")
        test_data['forces'] = None
    
    # Handle dipoles: key "D" in dataset
    if 'D' in dataset.files:
        dipoles_raw = jnp.asarray(dataset['D'], dtype=jnp.float32)
        if dipoles_raw.ndim == 3 and dipoles_raw.shape[1] == 1:
            dipoles_raw = dipoles_raw.squeeze(axis=1)  # (num_data, 3)
        
        if indices is None:
            test_data['dipoles'] = dipoles_raw
            test_data['D'] = dipoles_raw  # Also add as 'D' for prepare_batches compatibility
        else:
            test_data['dipoles'] = dipoles_raw[indices]
            test_data['D'] = dipoles_raw[indices]  # Also add as 'D' for prepare_batches compatibility
    else:
        test_data['dipoles'] = None
        test_data['D'] = None
    
    print(f"Test data size: {len(test_data['energies'])}")
    if test_data['forces'] is not None:
        print(f"Test forces shape: {test_data['forces'].shape}")
    if test_data.get('dipoles') is not None:
        print(f"Test dipoles shape: {test_data['dipoles'].shape}")
    
    # Evaluate
    if test_data['forces'] is not None:
        energy_pred, energy_targets, force_pred, force_targets, dipole_pred, dipole_targets, metrics = evaluate_dataset(
            model, params, test_data, batch_size=args.batch_size, dataset_name="test"
        )
    else:
        # Fallback if no forces
        print("Warning: Evaluating without forces (forces not available in dataset)")
        energy_pred, energy_targets, _, _, dipole_pred, dipole_targets, metrics = evaluate_dataset(
            model, params, test_data, batch_size=args.batch_size, dataset_name="test"
        )
        force_pred = None
        force_targets = None
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    
    # Convert metrics to JSON-serializable format
    def to_json_serializable(obj):
        """Convert numpy arrays and other non-serializable types to JSON-compatible formats."""
        if isinstance(obj, (np.ndarray, np.generic)):
            if obj.size == 1:
                return float(obj.item()) if np.issubdtype(obj.dtype, np.floating) else int(obj.item())
            else:
                # Convert array to list
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_json_serializable(item) for item in obj]
        else:
            return obj
    
    with open(metrics_path, 'w') as f:
        json.dump(to_json_serializable(metrics), f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Create plots
    print(f"\nCreating plots...")
    
    # Energy scatter plot
    scatter_path = output_dir / "scatter_energy.png"
    plot_scatter(energy_pred, energy_targets, metrics, 
                title="Energy Predictions", save_path=scatter_path)
    
    # Energy error distribution
    error_path = output_dir / "error_distribution.png"
    plot_error_distribution(energy_pred, energy_targets, metrics,
                          title="Energy Error Distribution", save_path=error_path)
    
    # Energy residual plot
    residual_path = output_dir / "residuals.png"
    plot_residuals(energy_pred, energy_targets, metrics,
                  title="Energy Residual Plot", save_path=residual_path)
    
    # Force scatter plot (if forces available) - flattened components
    if force_pred is not None and force_targets is not None:
        force_scatter_path = output_dir / "scatter_forces.png"
        plot_force_scatter(force_pred, force_targets, metrics, save_path=force_scatter_path)
    
    # Dipole scatter plot (if dipoles available) - flattened components
    if dipole_pred is not None and dipole_targets is not None:
        dipole_scatter_path = output_dir / "scatter_dipoles.png"
        plot_dipole_scatter(dipole_pred, dipole_targets, metrics, save_path=dipole_scatter_path)
    
    # Combined figure - adjust layout based on available data
    has_forces = force_pred is not None and force_targets is not None
    has_dipoles = dipole_pred is not None and dipole_targets is not None
    
    if has_forces and has_dipoles:
        # 4x3 grid: Energy (row 0), Forces (row 1), Dipoles (row 2), Mixed (row 3)
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Energy plots (top row)
        plot_scatter(energy_pred, energy_targets, metrics, 
                    title="Energy Predictions", ax=fig.add_subplot(gs[0, 0]))
        plot_error_distribution(energy_pred, energy_targets, metrics,
                              title="Energy Error Distribution", ax=fig.add_subplot(gs[0, 1]))
        plot_residuals(energy_pred, energy_targets, metrics,
                      title="Energy Residual Plot", ax=fig.add_subplot(gs[0, 2]))
        
        # Force plot (second row) - flattened components
        plot_force_scatter(force_pred, force_targets, metrics, ax=fig.add_subplot(gs[1, 0]))
        # Leave other two subplots empty or use for other plots
        fig.add_subplot(gs[1, 1]).axis('off')
        fig.add_subplot(gs[1, 2]).axis('off')
        
        # Dipole plot (third row) - flattened components
        plot_dipole_scatter(dipole_pred, dipole_targets, metrics, ax=fig.add_subplot(gs[2, 0]))
        # Leave other two subplots empty
        fig.add_subplot(gs[2, 1]).axis('off')
        fig.add_subplot(gs[2, 2]).axis('off')
        
        # Mixed plots (bottom row)
        try:
            from scipy import stats
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            stats.probplot(errors_kcal, dist="norm", plot=fig.add_subplot(gs[3, 0]))
            fig.axes[-1].set_title("Q-Q Plot of Energy Errors", fontsize=11)
            fig.axes[-1].set_xlabel('Theoretical Quantiles', fontsize=12)
            fig.axes[-1].set_ylabel('Sample Quantiles (kcal/mol)', fontsize=12)
            fig.axes[-1].grid(True, alpha=0.3)
        except ImportError:
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            fig.add_subplot(gs[3, 0]).hist(errors_kcal, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            fig.axes[-1].axvline(0, color='r', linestyle='--', linewidth=2)
            fig.axes[-1].set_title("Energy Error Histogram", fontsize=11)
            fig.axes[-1].set_xlabel('Error (kcal/mol)', fontsize=12)
            fig.axes[-1].set_ylabel('Frequency', fontsize=12)
            fig.axes[-1].grid(True, alpha=0.3)
        
        # Leave subplots empty or use for other plots
        fig.add_subplot(gs[3, 1]).axis('off')
        # Leave subplot empty
        fig.add_subplot(gs[3, 2]).axis('off')
        
    elif has_forces:
        # 3x3 grid: Energy, Forces, Mixed
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Energy plots (top row)
        plot_scatter(energy_pred, energy_targets, metrics, 
                    title="Energy Predictions", ax=fig.add_subplot(gs[0, 0]))
        plot_error_distribution(energy_pred, energy_targets, metrics,
                              title="Energy Error Distribution", ax=fig.add_subplot(gs[0, 1]))
        plot_residuals(energy_pred, energy_targets, metrics,
                      title="Energy Residual Plot", ax=fig.add_subplot(gs[0, 2]))
        
        # Force plot (middle row) - flattened components
        plot_force_scatter(force_pred, force_targets, metrics, ax=fig.add_subplot(gs[1, 0]))
        # Leave other two subplots empty
        fig.add_subplot(gs[1, 1]).axis('off')
        fig.add_subplot(gs[1, 2]).axis('off')
        
        # Q-Q plot (bottom row)
        try:
            from scipy import stats
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            stats.probplot(errors_kcal, dist="norm", plot=fig.add_subplot(gs[2, 0]))
            fig.axes[-1].set_title("Q-Q Plot of Energy Errors", fontsize=11)
            fig.axes[-1].set_xlabel('Theoretical Quantiles', fontsize=12)
            fig.axes[-1].set_ylabel('Sample Quantiles (kcal/mol)', fontsize=12)
            fig.axes[-1].grid(True, alpha=0.3)
        except ImportError:
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            fig.add_subplot(gs[2, 0]).hist(errors_kcal, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            fig.axes[-1].axvline(0, color='r', linestyle='--', linewidth=2)
            fig.axes[-1].set_title("Energy Error Histogram", fontsize=11)
            fig.axes[-1].set_xlabel('Error (kcal/mol)', fontsize=12)
            fig.axes[-1].set_ylabel('Frequency', fontsize=12)
            fig.axes[-1].grid(True, alpha=0.3)
        
        # Leave other subplots empty
        fig.add_subplot(gs[2, 1]).axis('off')
        fig.add_subplot(gs[2, 2]).axis('off')
    
    elif has_dipoles:
        # 3x3 grid: Energy, Dipoles, Mixed
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Energy plots (top row)
        plot_scatter(energy_pred, energy_targets, metrics, 
                    title="Energy Predictions", ax=fig.add_subplot(gs[0, 0]))
        plot_error_distribution(energy_pred, energy_targets, metrics,
                              title="Energy Error Distribution", ax=fig.add_subplot(gs[0, 1]))
        plot_residuals(energy_pred, energy_targets, metrics,
                      title="Energy Residual Plot", ax=fig.add_subplot(gs[0, 2]))
        
        # Dipole plot (middle row) - flattened components
        plot_dipole_scatter(dipole_pred, dipole_targets, metrics, ax=fig.add_subplot(gs[1, 0]))
        # Leave other two subplots empty
        fig.add_subplot(gs[1, 1]).axis('off')
        fig.add_subplot(gs[1, 2]).axis('off')
        
        # Q-Q plot and dipole error distribution (bottom row)
        try:
            from scipy import stats
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            stats.probplot(errors_kcal, dist="norm", plot=fig.add_subplot(gs[2, 0]))
            fig.axes[-1].set_title("Q-Q Plot of Energy Errors", fontsize=11)
            fig.axes[-1].set_xlabel('Theoretical Quantiles', fontsize=12)
            fig.axes[-1].set_ylabel('Sample Quantiles (kcal/mol)', fontsize=12)
            fig.axes[-1].grid(True, alpha=0.3)
        except ImportError:
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            fig.add_subplot(gs[2, 0]).hist(errors_kcal, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            fig.axes[-1].axvline(0, color='r', linestyle='--', linewidth=2)
            fig.axes[-1].set_title("Energy Error Histogram", fontsize=11)
            fig.axes[-1].set_xlabel('Error (kcal/mol)', fontsize=12)
            fig.axes[-1].set_ylabel('Frequency', fontsize=12)
            fig.axes[-1].grid(True, alpha=0.3)
        
        # Leave subplots empty
        fig.add_subplot(gs[2, 1]).axis('off')
        fig.add_subplot(gs[2, 2]).axis('off')
    else:
        # Energy-only figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        plot_scatter(energy_pred, energy_targets, metrics, 
                    title="Energy Predictions", ax=axes[0, 0])
        plot_error_distribution(energy_pred, energy_targets, metrics,
                              title="Energy Error Distribution", ax=axes[0, 1])
        plot_residuals(energy_pred, energy_targets, metrics,
                      title="Energy Residual Plot", ax=axes[1, 0])
        
        # Q-Q plot for errors
        try:
            from scipy import stats
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            stats.probplot(errors_kcal, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title("Q-Q Plot of Energy Errors", fontsize=11)
            axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
            axes[1, 1].set_ylabel('Sample Quantiles (kcal/mol)', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
        except ImportError:
            errors = energy_pred - energy_targets
            errors_kcal = errors * EV_TO_KCAL_MOL
            axes[1, 1].hist(errors_kcal, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
            axes[1, 1].set_title("Energy Error Histogram", fontsize=11)
            axes[1, 1].set_xlabel('Error (kcal/mol)', fontsize=12)
            axes[1, 1].set_ylabel('Frequency', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
    
    combined_path = output_dir / "evaluation_summary.png"
    plt.savefig(combined_path, bbox_inches='tight')
    print(f"✓ Saved combined plot to {combined_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Energy scatter plot: {scatter_path}")
    print(f"  - Energy error distribution: {error_path}")
    print(f"  - Energy residual plot: {residual_path}")
    if force_pred is not None and force_targets is not None:
        print(f"  - Force scatter plot (flattened components): {force_scatter_path}")
    if dipole_pred is not None and dipole_targets is not None:
        print(f"  - Dipole scatter plot (flattened components): {dipole_scatter_path}")
    print(f"  - Summary: {combined_path}")


if __name__ == "__main__":
    main()

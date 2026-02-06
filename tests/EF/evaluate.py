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
    """Single inference step."""
    energy = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        Ef=batch["electric_field"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        dst_idx_flat=batch["dst_idx_flat"],
        src_idx_flat=batch["src_idx_flat"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    return energy


def evaluate_dataset(model, params, data, batch_size=64, dataset_name="test"):
    """Evaluate model on a dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Prepare batches
    key = jax.random.PRNGKey(42)
    batches = prepare_batches(key, data, batch_size)
    
    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    
    print(f"Processing {len(batches)} batches...")
    for i, batch in enumerate(batches):
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(batches)}")
        
        predictions = inference_step(
            model_apply=model.apply,
            batch=batch,
            batch_size=batch_size,
            params=params,
        )
        
        # Convert to numpy
        predictions = np.asarray(predictions)
        targets = np.asarray(batch["energies"]).reshape(-1)
        
        all_predictions.append(predictions)
        all_targets.append(targets)
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    print(f"\nMetrics for {dataset_name}:")
    print(f"  MAE:  {metrics['mae_kcal_mol']:.4f} kcal/mol ({metrics['mae']:.6f} eV)")
    print(f"  RMSE: {metrics['rmse_kcal_mol']:.4f} kcal/mol ({metrics['rmse']:.6f} eV)")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  Mean Error: {metrics['mean_error_kcal_mol']:.4f} kcal/mol ({metrics['mean_error']:.6f} eV)")
    print(f"  Std Error:  {metrics['std_error_kcal_mol']:.4f} kcal/mol ({metrics['std_error']:.6f} eV)")
    print(f"  Max |Error|: {metrics['max_abs_error_kcal_mol']:.4f} kcal/mol ({metrics['max_abs_error']:.6f} eV)")
    
    return all_predictions, all_targets, metrics


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


def main():
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
    
    print(f"Test data size: {len(test_data['energies'])}")
    
    # Evaluate
    predictions, targets, metrics = evaluate_dataset(
        model, params, test_data, batch_size=args.batch_size, dataset_name="test"
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Create plots
    print(f"\nCreating plots...")
    
    # Scatter plot
    scatter_path = output_dir / "scatter_energy.png"
    plot_scatter(predictions, targets, metrics, 
                title="Energy Predictions", save_path=scatter_path)
    
    # Error distribution
    error_path = output_dir / "error_distribution.png"
    plot_error_distribution(predictions, targets, metrics,
                          title="Error Distribution", save_path=error_path)
    
    # Residual plot
    residual_path = output_dir / "residuals.png"
    plot_residuals(predictions, targets, metrics,
                  title="Residual Plot", save_path=residual_path)
    
    # Combined figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    plot_scatter(predictions, targets, metrics, 
                title="Energy Predictions", ax=axes[0, 0])
    plot_error_distribution(predictions, targets, metrics,
                          title="Error Distribution", ax=axes[0, 1])
    plot_residuals(predictions, targets, metrics,
                  title="Residual Plot", ax=axes[1, 0])
    
    # Q-Q plot for errors
    try:
        from scipy import stats
        errors = predictions - targets
        errors_kcal = errors * EV_TO_KCAL_MOL
        stats.probplot(errors_kcal, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot of Errors", fontsize=11)
        axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
        axes[1, 1].set_ylabel('Sample Quantiles (kcal/mol)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
    except ImportError:
        # If scipy not available, show error histogram instead
        errors = predictions - targets
        errors_kcal = errors * EV_TO_KCAL_MOL
        axes[1, 1].hist(errors_kcal, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_title("Error Histogram", fontsize=11)
        axes[1, 1].set_xlabel('Error (kcal/mol)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
    
    combined_path = output_dir / "evaluation_summary.png"
    plt.tight_layout()
    plt.savefig(combined_path, bbox_inches='tight')
    print(f"✓ Saved combined plot to {combined_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Scatter plot: {scatter_path}")
    print(f"  - Error distribution: {error_path}")
    print(f"  - Residual plot: {residual_path}")
    print(f"  - Summary: {combined_path}")


if __name__ == "__main__":
    main()

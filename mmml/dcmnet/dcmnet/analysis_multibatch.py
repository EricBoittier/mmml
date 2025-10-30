"""
Enhanced analysis tools for multi-batch trained DCMNet models.

Provides comprehensive analysis including:
- Batch-wise performance metrics
- Per-atom and per-molecule statistics
- Error distribution analysis
- Checkpoint comparison
- Training history visualization
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import prepare_batches
from .loss import esp_mono_loss
from .training_config import ExperimentConfig


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
        
    Returns
    -------
    Dict
        Checkpoint dictionary containing params, config, metrics, etc.
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def analyze_checkpoint(
    checkpoint_path: Path,
    model,
    test_data: Dict,
    batch_size: int = 1,
    num_atoms: int = 60
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a trained checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint
    model
        Model instance
    test_data : Dict
        Test dataset
    batch_size : int
        Batch size for evaluation
    num_atoms : int
        Number of atoms per system
        
    Returns
    -------
    Dict
        Analysis results
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint.get('ema_params', checkpoint['params'])
    config_dict = checkpoint['config']
    
    # Extract model config
    if isinstance(config_dict, dict) and 'model' in config_dict:
        model_config = config_dict['model']
        esp_w = config_dict['training'].get('esp_w', 1.0)
        chg_w = config_dict['training'].get('chg_w', 0.01)
        ndcm = model_config['n_dcm']
    else:
        # Fallback for older checkpoints
        esp_w = 1.0
        chg_w = 0.01
        ndcm = 2
    
    # Prepare batches
    key = jax.random.PRNGKey(42)
    test_batches = prepare_batches(key, test_data, batch_size, num_atoms=num_atoms)
    
    # Collect predictions and metrics
    results = {
        'batch_losses': [],
        'batch_mono_mae': [],
        'batch_mono_rmse': [],
        'predictions': [],
        'targets': [],
        'system_errors': []
    }
    
    print(f"Analyzing {len(test_batches)} batches...")
    
    for batch in tqdm(test_batches):
        # Forward pass
        mono_pred, dipo_pred = model.apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
        )
        
        # Compute loss
        loss = esp_mono_loss(
            dipo_prediction=dipo_pred,
            mono_prediction=mono_pred,
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
        
        # Compute monopole statistics
        mono_sum = mono_pred.sum(axis=-1)  # Sum over DCM
        mono_target = batch["mono"]
        error = mono_sum - mono_target
        
        results['batch_losses'].append(float(loss))
        results['batch_mono_mae'].append(float(jnp.mean(jnp.abs(error))))
        results['batch_mono_rmse'].append(float(jnp.sqrt(jnp.mean(error**2))))
        results['predictions'].append(mono_sum)
        results['targets'].append(mono_target)
        results['system_errors'].append(error)
    
    # Aggregate results
    all_preds = jnp.concatenate(results['predictions'], axis=0)
    all_targets = jnp.concatenate(results['targets'], axis=0)
    all_errors = jnp.concatenate(results['system_errors'], axis=0)
    
    summary = {
        'total_loss': float(jnp.mean(jnp.array(results['batch_losses']))),
        'loss_std': float(jnp.std(jnp.array(results['batch_losses']))),
        'mae': float(jnp.mean(jnp.abs(all_errors))),
        'rmse': float(jnp.sqrt(jnp.mean(all_errors**2))),
        'max_error': float(jnp.max(jnp.abs(all_errors))),
        'pred_mean': float(jnp.mean(all_preds)),
        'pred_std': float(jnp.std(all_preds)),
        'target_mean': float(jnp.mean(all_targets)),
        'target_std': float(jnp.std(all_targets)),
        'correlation': float(jnp.corrcoef(all_preds.flatten(), all_targets.flatten())[0, 1]),
    }
    
    # Add percentile information
    error_percentiles = jnp.percentile(jnp.abs(all_errors), [50, 75, 90, 95, 99])
    summary['error_median'] = float(error_percentiles[0])
    summary['error_75th'] = float(error_percentiles[1])
    summary['error_90th'] = float(error_percentiles[2])
    summary['error_95th'] = float(error_percentiles[3])
    summary['error_99th'] = float(error_percentiles[4])
    
    results['summary'] = summary
    results['all_predictions'] = all_preds
    results['all_targets'] = all_targets
    results['all_errors'] = all_errors
    
    return results


def print_analysis_summary(analysis: Dict[str, Any]):
    """
    Print formatted analysis summary.
    
    Parameters
    ----------
    analysis : Dict
        Analysis results from analyze_checkpoint
    """
    summary = analysis['summary']
    
    print(f"\n{'='*80}")
    print("Model Analysis Summary")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Value':>20} {'Unit':>15}")
    print(f"{'-'*80}")
    
    metrics = [
        ('Total Loss', summary['total_loss'], ''),
        ('Loss Std Dev', summary['loss_std'], ''),
        ('MAE', summary['mae'], 'a.u.'),
        ('RMSE', summary['rmse'], 'a.u.'),
        ('Max Error', summary['max_error'], 'a.u.'),
        ('Correlation', summary['correlation'], ''),
        ('Pred Mean', summary['pred_mean'], 'a.u.'),
        ('Pred Std', summary['pred_std'], 'a.u.'),
        ('Target Mean', summary['target_mean'], 'a.u.'),
        ('Target Std', summary['target_std'], 'a.u.'),
    ]
    
    for name, value, unit in metrics:
        print(f"{name:<30} {value:>20.6e} {unit:>15}")
    
    print(f"{'-'*80}")
    print("Error Distribution (Absolute):")
    print(f"  Median (50th): {summary['error_median']:.6e}")
    print(f"  75th percentile: {summary['error_75th']:.6e}")
    print(f"  90th percentile: {summary['error_90th']:.6e}")
    print(f"  95th percentile: {summary['error_95th']:.6e}")
    print(f"  99th percentile: {summary['error_99th']:.6e}")
    print(f"{'='*80}\n")


def compare_checkpoints(
    checkpoint_paths: List[Path],
    model,
    test_data: Dict,
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple checkpoints on test data.
    
    Parameters
    ----------
    checkpoint_paths : List[Path]
        List of checkpoint paths to compare
    model
        Model instance
    test_data : Dict
        Test dataset
    labels : Optional[List[str]]
        Labels for each checkpoint
        
    Returns
    -------
    pd.DataFrame
        Comparison dataframe
    """
    if labels is None:
        labels = [f"Checkpoint {i+1}" for i in range(len(checkpoint_paths))]
    
    results = []
    
    for path, label in zip(checkpoint_paths, labels):
        print(f"\nAnalyzing {label}...")
        analysis = analyze_checkpoint(path, model, test_data)
        summary = analysis['summary']
        summary['label'] = label
        summary['checkpoint'] = str(path)
        results.append(summary)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['label', 'mae', 'rmse', 'max_error', 'correlation', 
            'total_loss', 'pred_mean', 'pred_std']
    df = df[[c for c in cols if c in df.columns]]
    
    return df


def analyze_training_history(exp_dir: Path) -> Dict[str, Any]:
    """
    Analyze training history from checkpoints.
    
    Parameters
    ----------
    exp_dir : Path
        Experiment directory
        
    Returns
    -------
    Dict
        Training history analysis
    """
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all epoch checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pkl"))
    
    history = {
        'epochs': [],
        'train_loss': [],
        'valid_loss': [],
        'train_mae': [],
        'valid_mae': [],
    }
    
    for ckpt_path in checkpoints:
        checkpoint = load_checkpoint(ckpt_path)
        epoch = checkpoint['epoch']
        metrics = checkpoint.get('metrics', {})
        
        history['epochs'].append(epoch)
        history['train_loss'].append(metrics.get('train', {}).get('loss', np.nan))
        history['valid_loss'].append(metrics.get('valid', {}).get('loss', np.nan))
        history['train_mae'].append(metrics.get('train', {}).get('mono_mae', np.nan))
        history['valid_mae'].append(metrics.get('valid', {}).get('mono_mae', np.nan))
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Find best epoch
    if len(df) > 0:
        best_epoch_idx = df['valid_loss'].idxmin()
        best_epoch = df.loc[best_epoch_idx]
        
        summary = {
            'total_epochs': len(df),
            'best_epoch': int(best_epoch['epochs']),
            'best_valid_loss': float(best_epoch['valid_loss']),
            'best_valid_mae': float(best_epoch['valid_mae']),
            'final_train_loss': float(df['train_loss'].iloc[-1]),
            'final_valid_loss': float(df['valid_loss'].iloc[-1]),
            'history_df': df
        }
    else:
        summary = {'history_df': df}
    
    return summary


def export_analysis_report(
    analysis: Dict[str, Any],
    output_path: Path,
    format: str = 'json'
):
    """
    Export analysis results to file.
    
    Parameters
    ----------
    analysis : Dict
        Analysis results
    output_path : Path
        Output file path
    format : str
        Format: 'json', 'csv', or 'pkl'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'summary': analysis['summary'],
            'batch_losses': [float(x) for x in analysis['batch_losses']],
            'batch_mono_mae': [float(x) for x in analysis['batch_mono_mae']],
            'batch_mono_rmse': [float(x) for x in analysis['batch_mono_rmse']],
        }
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    elif format == 'csv':
        # Export predictions to CSV
        df = pd.DataFrame({
            'prediction': analysis['all_predictions'].flatten(),
            'target': analysis['all_targets'].flatten(),
            'error': analysis['all_errors'].flatten()
        })
        df.to_csv(output_path, index=False)
    
    elif format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(analysis, f)
    
    print(f"Analysis exported to {output_path}")


def batch_analysis_summary(
    exp_dir: Path,
    model,
    test_data: Dict,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate complete analysis summary for an experiment.
    
    Parameters
    ----------
    exp_dir : Path
        Experiment directory
    model
        Model instance
    test_data : Dict
        Test dataset
    output_dir : Optional[Path]
        Directory to save analysis outputs
        
    Returns
    -------
    Dict
        Complete analysis results
    """
    if output_dir is None:
        output_dir = exp_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Batch Analysis Summary: {exp_dir.name}")
    print(f"{'='*80}\n")
    
    # Analyze best checkpoint
    best_ckpt = exp_dir / "checkpoints" / "checkpoint_best.pkl"
    if best_ckpt.exists():
        print("Analyzing best checkpoint...")
        best_analysis = analyze_checkpoint(best_ckpt, model, test_data)
        print_analysis_summary(best_analysis)
        
        # Export results
        export_analysis_report(best_analysis, output_dir / "best_analysis.json", 'json')
        export_analysis_report(best_analysis, output_dir / "predictions.csv", 'csv')
    else:
        print("No best checkpoint found!")
        best_analysis = None
    
    # Analyze training history
    try:
        print("\nAnalyzing training history...")
        history = analyze_training_history(exp_dir)
        
        if 'best_epoch' in history:
            print(f"\nTraining History Summary:")
            print(f"  Total Epochs: {history['total_epochs']}")
            print(f"  Best Epoch: {history['best_epoch']}")
            print(f"  Best Valid Loss: {history['best_valid_loss']:.6e}")
            print(f"  Best Valid MAE: {history['best_valid_mae']:.6e}")
            print(f"  Final Train Loss: {history['final_train_loss']:.6e}")
            print(f"  Final Valid Loss: {history['final_valid_loss']:.6e}")
        
        # Export history
        history['history_df'].to_csv(output_dir / "training_history.csv", index=False)
    except FileNotFoundError as e:
        print(f"Could not analyze history: {e}")
        history = None
    
    results = {
        'best_checkpoint_analysis': best_analysis,
        'training_history': history,
        'experiment_dir': str(exp_dir),
        'output_dir': str(output_dir)
    }
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"{'='*80}\n")
    
    return results


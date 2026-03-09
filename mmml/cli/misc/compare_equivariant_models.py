#!/usr/bin/env python3
"""
Compare Equivariant (DCMNet) vs Non-Equivariant Models

This tool analyzes comparison_results.json files from multiple training runs
and creates comprehensive plots comparing:
- Accuracy vs number of distributed charges (n_dcm)
- Accuracy vs number of parameters
- Training efficiency
- Model performance across different configurations

Usage:
    python -m mmml.cli.compare_equivariant_models \
        --comparison-dirs examples/co2/dcmnet_physnet_train/comparisons/*/ \
        --output-dir analysis/equivariant_comparison/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Matplotlib not available. Install with: pip install matplotlib")


def load_comparison_data(comparison_dir: Path) -> Dict:
    """Load comparison_results.json from a directory."""
    json_file = comparison_dir / "comparison_results.json"
    
    if not json_file.exists():
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def extract_metrics(data: Dict, model_type: str) -> Dict:
    """Extract key metrics from comparison data."""
    if model_type not in data:
        return None
    
    model_data = data[model_type]
    args = data.get('args', {})
    
    metrics = {
        'n_dcm': args.get('n_dcm', None),
        'n_params': model_data.get('num_parameters', model_data.get('n_params', None)),
        'model_type': 'DCMNet' if model_type == 'dcmnet' else 'NonEquivariant',
    }
    
    # Extract validation metrics (try multiple naming conventions)
    metric_mappings = {
        'energy_mae': ['val_energy_mae', 'test_energy_mae', 'valid_energy_mae', 'energy_mae'],
        'forces_mae': ['val_forces_mae', 'test_forces_mae', 'valid_forces_mae', 'forces_mae'],
        'dipole_mae': ['val_dipole_mae', 'test_dipole_mae', 'valid_dipole_mae', 'dipole_mae'],
        'esp_rmse': ['val_esp_mae', 'test_esp_mae', 'valid_esp_mae', 'esp_rmse', 'esp_mae'],
    }
    
    for metric_name, possible_keys in metric_mappings.items():
        for key in possible_keys:
            if key in model_data:
                metrics[metric_name] = model_data[key]
                break
    
    # Extract training metrics
    for metric_name in ['train_loss', 'val_loss', 'best_epoch', 'training_time', 
                       'inference_time', 'rotation_error_dipole', 'rotation_error_esp',
                       'translation_error_dipole', 'translation_error_esp']:
        if metric_name in model_data:
            metrics[metric_name] = model_data[metric_name]
    
    # Extract configuration
    for key in ['physnet_features', 'dcmnet_features', 'noneq_features', 
                'physnet_iterations', 'dcmnet_iterations']:
        if key in args:
            metrics[key] = args[key]
    
    return metrics


def aggregate_comparisons(comparison_dirs: List[Path], verbose: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """
    Aggregate data from all comparison directories.
    
    Returns
    -------
    dcmnet_results : list
        List of metrics dictionaries for DCMNet models
    noneq_results : list
        List of metrics dictionaries for NonEquivariant models
    """
    dcmnet_results = []
    noneq_results = []
    
    for comp_dir in comparison_dirs:
        if not comp_dir.is_dir():
            continue
        
        data = load_comparison_data(comp_dir)
        if data is None:
            if verbose:
                print(f"  ⚠️  Skipping {comp_dir.name}: no comparison_results.json")
            continue
        
        # Extract DCMNet metrics
        dcm_metrics = extract_metrics(data, 'dcmnet')
        if dcm_metrics:
            dcm_metrics['run_name'] = comp_dir.name
            dcmnet_results.append(dcm_metrics)
        
        # Extract NonEq metrics
        noneq_metrics = extract_metrics(data, 'noneq')
        if noneq_metrics:
            noneq_metrics['run_name'] = comp_dir.name
            noneq_results.append(noneq_metrics)
        
        if verbose:
            print(f"  ✅ Loaded {comp_dir.name}")
    
    return dcmnet_results, noneq_results


def plot_accuracy_vs_ndcm(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Plot accuracy metrics vs n_dcm for both model types."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('energy_mae', 'Energy MAE (eV)', 0, 0),
        ('forces_mae', 'Forces MAE (eV/Å)', 0, 1),
        ('dipole_mae', 'Dipole MAE (e·Å)', 1, 0),
        ('esp_rmse', 'ESP RMSE (Ha/e)', 1, 1),
    ]
    
    for metric_key, ylabel, row, col in metrics_to_plot:
        ax = axes[row, col]
        
        # Extract data for DCMNet
        dcm_ndcm = []
        dcm_values = []
        for result in dcmnet_results:
            if metric_key in result and result.get('n_dcm') is not None:
                dcm_ndcm.append(result['n_dcm'])
                dcm_values.append(result[metric_key])
        
        # Extract data for NonEq
        noneq_ndcm = []
        noneq_values = []
        for result in noneq_results:
            if metric_key in result and result.get('n_dcm') is not None:
                noneq_ndcm.append(result['n_dcm'])
                noneq_values.append(result[metric_key])
        
        # Plot DCMNet
        if dcm_ndcm:
            ax.scatter(dcm_ndcm, dcm_values, alpha=0.7, s=100, 
                      label='DCMNet (Equivariant)', marker='o', color='#0072B2')
        
        # Plot NonEq
        if noneq_ndcm:
            ax.scatter(noneq_ndcm, noneq_values, alpha=0.7, s=100,
                      label='NonEquivariant', marker='s', color='#D55E00')
        
        ax.set_xlabel('Number of Distributed Charges (n_dcm)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split('(')[0].strip(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add trend lines if enough data
        if len(dcm_ndcm) > 2:
            z = np.polyfit(dcm_ndcm, dcm_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(dcm_ndcm), max(dcm_ndcm), 100)
            ax.plot(x_line, p(x_line), '--', alpha=0.5, color='#0072B2', linewidth=2)
        
        if len(noneq_ndcm) > 2:
            z = np.polyfit(noneq_ndcm, noneq_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(noneq_ndcm), max(noneq_ndcm), 100)
            ax.plot(x_line, p(x_line), '--', alpha=0.5, color='#D55E00', linewidth=2)
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_vs_ndcm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_accuracy_vs_params(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Plot accuracy metrics vs number of parameters."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('energy_mae', 'Energy MAE (eV)', 0, 0),
        ('forces_mae', 'Forces MAE (eV/Å)', 0, 1),
        ('dipole_mae', 'Dipole MAE (e·Å)', 1, 0),
        ('esp_rmse', 'ESP RMSE (Ha/e)', 1, 1),
    ]
    
    for metric_key, ylabel, row, col in metrics_to_plot:
        ax = axes[row, col]
        
        # Extract data for DCMNet
        dcm_params = []
        dcm_values = []
        dcm_labels = []
        for result in dcmnet_results:
            if metric_key in result and result.get('n_params') is not None:
                dcm_params.append(result['n_params'])
                dcm_values.append(result[metric_key])
                dcm_labels.append(f"n_dcm={result.get('n_dcm', '?')}")
        
        # Extract data for NonEq
        noneq_params = []
        noneq_values = []
        noneq_labels = []
        for result in noneq_results:
            if metric_key in result and result.get('n_params') is not None:
                noneq_params.append(result['n_params'])
                noneq_values.append(result[metric_key])
                noneq_labels.append(f"n_dcm={result.get('n_dcm', '?')}")
        
        # Plot DCMNet
        if dcm_params:
            scatter = ax.scatter(dcm_params, dcm_values, alpha=0.7, s=100,
                                label='DCMNet (Equivariant)', marker='o', color='#0072B2')
            # Annotate some points
            for i in range(0, len(dcm_params), max(1, len(dcm_params)//5)):
                ax.annotate(f"n={dcm_labels[i].split('=')[1]}", 
                           (dcm_params[i], dcm_values[i]),
                           fontsize=7, alpha=0.7, xytext=(5, 5), 
                           textcoords='offset points')
        
        # Plot NonEq
        if noneq_params:
            scatter = ax.scatter(noneq_params, noneq_values, alpha=0.7, s=100,
                                label='NonEquivariant', marker='s', color='#D55E00')
            # Annotate some points
            for i in range(0, len(noneq_params), max(1, len(noneq_params)//5)):
                ax.annotate(f"n={noneq_labels[i].split('=')[1]}", 
                           (noneq_params[i], noneq_values[i]),
                           fontsize=7, alpha=0.7, xytext=(5, -10), 
                           textcoords='offset points')
        
        ax.set_xlabel('Number of Parameters', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{ylabel.split("(")[0].strip()} vs Model Size', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis if range is large
        if dcm_params or noneq_params:
            all_params = dcm_params + noneq_params
            if max(all_params) / min(all_params) > 10:
                ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_vs_params.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_model_comparison(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Create comprehensive comparison plots."""
    if not HAS_MATPLOTLIB:
        return
    
    # Create paired comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = [
        ('energy_mae', 'Energy MAE (eV)'),
        ('forces_mae', 'Forces MAE (eV/Å)'),
        ('dipole_mae', 'Dipole MAE (e·Å)'),
        ('esp_rmse', 'ESP RMSE (Ha/e)'),
        ('val_loss', 'Validation Loss'),
        ('n_params', 'Number of Parameters'),
    ]
    
    for idx, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Get matching pairs (same run name)
        run_names = set([r['run_name'] for r in dcmnet_results])
        
        dcm_values = []
        noneq_values = []
        labels = []
        
        for run_name in sorted(run_names):
            dcm_data = [r for r in dcmnet_results if r['run_name'] == run_name]
            noneq_data = [r for r in noneq_results if r['run_name'] == run_name]
            
            if dcm_data and noneq_data:
                dcm_val = dcm_data[0].get(metric_key)
                noneq_val = noneq_data[0].get(metric_key)
                if dcm_val is not None and noneq_val is not None:
                    dcm_values.append(dcm_val)
                    noneq_values.append(noneq_val)
                    labels.append(run_name[:15])  # Truncate long names
        
        if dcm_values and noneq_values:
            x = np.arange(len(labels))
            width = 0.35
            
            ax.bar(x - width/2, dcm_values, width, label='DCMNet', 
                  alpha=0.8, color='#0072B2')
            ax.bar(x + width/2, noneq_values, width, label='NonEq',
                  alpha=0.8, color='#D55E00')
            
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(ylabel.split('(')[0].strip(), fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'model_comparison_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_pareto_front(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Plot Pareto front: accuracy vs computational cost."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: ESP RMSE vs Parameters
    ax = axes[0]
    
    dcm_params = [r['n_params'] for r in dcmnet_results if 'esp_rmse' in r and 'n_params' in r]
    dcm_esp = [r['esp_rmse'] for r in dcmnet_results if 'esp_rmse' in r and 'n_params' in r]
    dcm_ndcm = [r['n_dcm'] for r in dcmnet_results if 'esp_rmse' in r and 'n_params' in r]
    
    noneq_params = [r['n_params'] for r in noneq_results if 'esp_rmse' in r and 'n_params' in r]
    noneq_esp = [r['esp_rmse'] for r in noneq_results if 'esp_rmse' in r and 'n_params' in r]
    noneq_ndcm = [r['n_dcm'] for r in noneq_results if 'esp_rmse' in r and 'n_params' in r]
    
    if dcm_params:
        scatter = ax.scatter(dcm_params, dcm_esp, c=dcm_ndcm, cmap='Blues',
                            alpha=0.7, s=150, marker='o', edgecolors='black',
                            linewidth=1.5, label='DCMNet')
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('n_dcm', fontsize=10)
    
    if noneq_params:
        scatter = ax.scatter(noneq_params, noneq_esp, c=noneq_ndcm, cmap='Oranges',
                            alpha=0.7, s=150, marker='s', edgecolors='black',
                            linewidth=1.5, label='NonEq')
        # Add colorbar for noneq
        cbar2 = plt.colorbar(scatter, ax=ax)
        cbar2.set_label('n_dcm', fontsize=10)
    
    ax.set_xlabel('Number of Parameters', fontsize=11)
    ax.set_ylabel('ESP RMSE (Ha/e)', fontsize=11)
    ax.set_title('ESP Accuracy vs Model Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Energy MAE vs Parameters
    ax = axes[1]
    
    dcm_params_e = [r['n_params'] for r in dcmnet_results if 'energy_mae' in r and 'n_params' in r]
    dcm_energy = [r['energy_mae'] for r in dcmnet_results if 'energy_mae' in r and 'n_params' in r]
    dcm_ndcm_e = [r['n_dcm'] for r in dcmnet_results if 'energy_mae' in r and 'n_params' in r]
    
    noneq_params_e = [r['n_params'] for r in noneq_results if 'energy_mae' in r and 'n_params' in r]
    noneq_energy = [r['energy_mae'] for r in noneq_results if 'energy_mae' in r and 'n_params' in r]
    noneq_ndcm_e = [r['n_dcm'] for r in noneq_results if 'energy_mae' in r and 'n_params' in r]
    
    if dcm_params_e:
        scatter = ax.scatter(dcm_params_e, dcm_energy, c=dcm_ndcm_e, cmap='Blues',
                            alpha=0.7, s=150, marker='o', edgecolors='black',
                            linewidth=1.5, label='DCMNet')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('n_dcm', fontsize=10)
    
    if noneq_params_e:
        scatter = ax.scatter(noneq_params_e, noneq_energy, c=noneq_ndcm_e, cmap='Oranges',
                            alpha=0.7, s=150, marker='s', edgecolors='black',
                            linewidth=1.5, label='NonEq')
        cbar2 = plt.colorbar(scatter, ax=ax)
        cbar2.set_label('n_dcm', fontsize=10)
    
    ax.set_xlabel('Number of Parameters', fontsize=11)
    ax.set_ylabel('Energy MAE (eV)', fontsize=11)
    ax.set_title('Energy Accuracy vs Model Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Pareto Front: Accuracy vs Computational Cost', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'pareto_front.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def create_summary_table(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Create summary statistics table."""
    
    summary_lines = []
    summary_lines.append("="*100)
    summary_lines.append("EQUIVARIANT vs NON-EQUIVARIANT MODEL COMPARISON")
    summary_lines.append("="*100)
    summary_lines.append("")
    
    # Overall statistics
    summary_lines.append(f"Total DCMNet runs: {len(dcmnet_results)}")
    summary_lines.append(f"Total NonEq runs: {len(noneq_results)}")
    summary_lines.append("")
    
    # Metrics comparison
    metrics = ['energy_mae', 'forces_mae', 'dipole_mae', 'esp_rmse', 'n_params',
               'training_time', 'inference_time', 'rotation_error_dipole', 'rotation_error_esp']
    metric_names = {
        'energy_mae': 'Energy MAE (eV)',
        'forces_mae': 'Forces MAE (eV/Å)',
        'dipole_mae': 'Dipole MAE (e·Å)',
        'esp_rmse': 'ESP RMSE (Ha/e)',
        'n_params': 'Parameters (count)',
        'training_time': 'Training Time (seconds)',
        'inference_time': 'Inference Time (seconds)',
        'rotation_error_dipole': 'Rotation Error Dipole (e·Å)',
        'rotation_error_esp': 'Rotation Error ESP (Ha/e)',
    }
    
    summary_lines.append("AVERAGE METRICS")
    summary_lines.append("-"*100)
    summary_lines.append(f"{'Metric':<30} {'DCMNet (mean±std)':<30} {'NonEq (mean±std)':<30} {'Winner':<10}")
    summary_lines.append("-"*100)
    
    for metric in metrics:
        dcm_vals = [r[metric] for r in dcmnet_results if metric in r]
        noneq_vals = [r[metric] for r in noneq_results if metric in r]
        
        if dcm_vals and noneq_vals:
            dcm_mean = np.mean(dcm_vals)
            dcm_std = np.std(dcm_vals)
            noneq_mean = np.mean(noneq_vals)
            noneq_std = np.std(noneq_vals)
            
            # Lower is better for all these metrics
            winner = 'DCMNet' if dcm_mean < noneq_mean else 'NonEq'
            
            # Special annotation for rotation errors (equivariance test)
            if 'rotation_error' in metric:
                winner = winner + ' ⭐' if winner == 'DCMNet' else winner
            
            summary_lines.append(
                f"{metric_names[metric]:<30} {dcm_mean:.6f}±{dcm_std:.6f}     "
                f"{noneq_mean:.6f}±{noneq_std:.6f}     {winner:<10}"
            )
    
    summary_lines.append("-"*100)
    summary_lines.append("")
    
    # Parameter efficiency
    summary_lines.append("PARAMETER EFFICIENCY (ESP RMSE per 1000 parameters)")
    summary_lines.append("-"*100)
    
    dcm_efficiency = []
    for r in dcmnet_results:
        if 'esp_rmse' in r and 'n_params' in r:
            dcm_efficiency.append(r['esp_rmse'] / (r['n_params'] / 1000))
    
    noneq_efficiency = []
    for r in noneq_results:
        if 'esp_rmse' in r and 'n_params' in r:
            noneq_efficiency.append(r['esp_rmse'] / (r['n_params'] / 1000))
    
    if dcm_efficiency:
        summary_lines.append(f"DCMNet:     {np.mean(dcm_efficiency):.8f} ± {np.std(dcm_efficiency):.8f}")
    if noneq_efficiency:
        summary_lines.append(f"NonEq:      {np.mean(noneq_efficiency):.8f} ± {np.std(noneq_efficiency):.8f}")
    
    if dcm_efficiency and noneq_efficiency:
        improvement = (np.mean(noneq_efficiency) - np.mean(dcm_efficiency)) / np.mean(noneq_efficiency) * 100
        winner = 'DCMNet' if improvement > 0 else 'NonEq'
        summary_lines.append(f"\nBetter: {winner} ({abs(improvement):.1f}% more parameter-efficient)")
    
    summary_lines.append("")
    summary_lines.append("="*100)
    
    # Save summary
    summary_text = '\n'.join(summary_lines)
    output_path = output_dir / 'comparison_summary.txt'
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    print(f"  ✅ Saved: {output_path}")
    print("")
    print(summary_text)


def plot_training_curves_comparison(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Plot training efficiency comparison."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Best epoch comparison
    ax = axes[0]
    
    dcm_epochs = [r['best_epoch'] for r in dcmnet_results if 'best_epoch' in r]
    noneq_epochs = [r['best_epoch'] for r in noneq_results if 'best_epoch' in r]
    
    if dcm_epochs and noneq_epochs:
        data_to_plot = [dcm_epochs, noneq_epochs]
        labels = ['DCMNet\n(Equivariant)', 'NonEquivariant']
        colors = ['#0072B2', '#D55E00']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       widths=0.6, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Best Epoch', fontsize=11)
        ax.set_title('Training Convergence Speed', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        ax.text(0.02, 0.98, 
               f"DCMNet: {np.mean(dcm_epochs):.1f}±{np.std(dcm_epochs):.1f}\n"
               f"NonEq: {np.mean(noneq_epochs):.1f}±{np.std(noneq_epochs):.1f}",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Final validation loss
    ax = axes[1]
    
    dcm_val_loss = [r['val_loss'] for r in dcmnet_results if 'val_loss' in r]
    noneq_val_loss = [r['val_loss'] for r in noneq_results if 'val_loss' in r]
    
    if dcm_val_loss and noneq_val_loss:
        data_to_plot = [dcm_val_loss, noneq_val_loss]
        labels = ['DCMNet\n(Equivariant)', 'NonEquivariant']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       widths=0.6, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Validation Loss', fontsize=11)
        ax.set_title('Final Model Performance', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        ax.text(0.02, 0.98,
               f"DCMNet: {np.mean(dcm_val_loss):.6f}±{np.std(dcm_val_loss):.6f}\n"
               f"NonEq: {np.mean(noneq_val_loss):.6f}±{np.std(noneq_val_loss):.6f}",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'training_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_equivariance_test(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Plot equivariance test results - rotation errors."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Rotation error for dipole
    ax = axes[0]
    
    dcm_rot_dipole = [r['rotation_error_dipole'] for r in dcmnet_results if 'rotation_error_dipole' in r]
    noneq_rot_dipole = [r['rotation_error_dipole'] for r in noneq_results if 'rotation_error_dipole' in r]
    
    if dcm_rot_dipole and noneq_rot_dipole:
        data = [dcm_rot_dipole, noneq_rot_dipole]
        labels = ['DCMNet\n(Equivariant)', 'NonEquivariant\n(NOT Equivariant)']
        colors = ['#0072B2', '#D55E00']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Rotation Error - Dipole (e·Å)', fontsize=11)
        ax.set_title('Equivariance Test: Dipole Rotation Error', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        dcm_median = np.median(dcm_rot_dipole)
        noneq_median = np.median(noneq_rot_dipole)
        ratio = noneq_median / dcm_median if dcm_median > 0 else float('inf')
        
        ax.text(0.02, 0.98,
               f"DCMNet: {dcm_median:.2e}\n"
               f"NonEq: {noneq_median:.2e}\n"
               f"Ratio: {ratio:.1f}x worse",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Rotation error for ESP
    ax = axes[1]
    
    dcm_rot_esp = [r['rotation_error_esp'] for r in dcmnet_results if 'rotation_error_esp' in r]
    noneq_rot_esp = [r['rotation_error_esp'] for r in noneq_results if 'rotation_error_esp' in r]
    
    if dcm_rot_esp and noneq_rot_esp:
        data = [dcm_rot_esp, noneq_rot_esp]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Rotation Error - ESP (Ha/e)', fontsize=11)
        ax.set_title('Equivariance Test: ESP Rotation Error', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        dcm_median = np.median(dcm_rot_esp)
        noneq_median = np.median(noneq_rot_esp)
        ratio = noneq_median / dcm_median if dcm_median > 0 else float('inf')
        
        ax.text(0.02, 0.98,
               f"DCMNet: {dcm_median:.2e}\n"
               f"NonEq: {noneq_median:.2e}\n"
               f"Ratio: {ratio:.1f}x worse",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Equivariance Testing: Prediction Error After Rotation', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'equivariance_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_computational_efficiency(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Plot computational efficiency metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training time
    ax = axes[0]
    dcm_train_time = [r['training_time'] for r in dcmnet_results if 'training_time' in r]
    noneq_train_time = [r['training_time'] for r in noneq_results if 'training_time' in r]
    
    if dcm_train_time and noneq_train_time:
        data = [dcm_train_time, noneq_train_time]
        labels = ['DCMNet', 'NonEq']
        colors = ['#0072B2', '#D55E00']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Training Time (seconds)', fontsize=11)
        ax.set_title('Training Speed', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        speedup = np.mean(dcm_train_time) / np.mean(noneq_train_time)
        winner = 'NonEq' if speedup > 1 else 'DCMNet'
        ax.text(0.02, 0.98,
               f"DCMNet: {np.mean(dcm_train_time):.1f}s\n"
               f"NonEq: {np.mean(noneq_train_time):.1f}s\n"
               f"{winner}: {abs(speedup):.2f}x faster",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Inference time
    ax = axes[1]
    dcm_inf_time = [r['inference_time'] for r in dcmnet_results if 'inference_time' in r]
    noneq_inf_time = [r['inference_time'] for r in noneq_results if 'inference_time' in r]
    
    if dcm_inf_time and noneq_inf_time:
        data = [dcm_inf_time, noneq_inf_time]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Inference Time (seconds)', fontsize=11)
        ax.set_title('Inference Speed', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        speedup = np.mean(dcm_inf_time) / np.mean(noneq_inf_time)
        winner = 'NonEq' if speedup > 1 else 'DCMNet'
        ax.text(0.02, 0.98,
               f"DCMNet: {np.mean(dcm_inf_time):.3f}s\n"
               f"NonEq: {np.mean(noneq_inf_time):.3f}s\n"
               f"{winner}: {abs(speedup):.2f}x faster",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Parameters vs ESP accuracy (efficiency scatter)
    ax = axes[2]
    
    dcm_params = [r['n_params'] for r in dcmnet_results if 'esp_rmse' in r and 'n_params' in r]
    dcm_esp = [r['esp_rmse'] for r in dcmnet_results if 'esp_rmse' in r and 'n_params' in r]
    
    noneq_params = [r['n_params'] for r in noneq_results if 'esp_rmse' in r and 'n_params' in r]
    noneq_esp = [r['esp_rmse'] for r in noneq_results if 'esp_rmse' in r and 'n_params' in r]
    
    if dcm_params and noneq_params:
        ax.scatter(dcm_params, dcm_esp, s=120, alpha=0.7, color='#0072B2', 
                  marker='o', edgecolors='black', linewidth=1.5, label='DCMNet')
        ax.scatter(noneq_params, noneq_esp, s=120, alpha=0.7, color='#D55E00',
                  marker='s', edgecolors='black', linewidth=1.5, label='NonEq')
        
        # Add efficiency lines (lower-left is better)
        # Draw lines from origin to each point
        for p, e in zip(dcm_params[:3], dcm_esp[:3]):
            ax.plot([0, p], [0, e], ':', alpha=0.3, color='#0072B2', linewidth=1)
        for p, e in zip(noneq_params[:3], noneq_esp[:3]):
            ax.plot([0, p], [0, e], ':', alpha=0.3, color='#D55E00', linewidth=1)
        
        ax.set_xlabel('Number of Parameters', fontsize=11)
        ax.set_ylabel('ESP RMSE (Ha/e)', fontsize=11)
        ax.set_title('Parameter Efficiency', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotate Pareto optimal points
        ax.text(0.98, 0.98, 'Lower-left = Better\n(fewer params, lower error)',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'computational_efficiency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_ndcm_scaling(
    dcmnet_results: List[Dict],
    noneq_results: List[Dict],
    output_dir: Path,
):
    """Plot how metrics scale with n_dcm."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group by n_dcm and compute statistics
    ndcm_values = sorted(set([r['n_dcm'] for r in dcmnet_results + noneq_results if 'n_dcm' in r]))
    
    dcm_esp_by_ndcm = {n: [] for n in ndcm_values}
    noneq_esp_by_ndcm = {n: [] for n in ndcm_values}
    
    for r in dcmnet_results:
        if 'esp_rmse' in r and 'n_dcm' in r:
            dcm_esp_by_ndcm[r['n_dcm']].append(r['esp_rmse'])
    
    for r in noneq_results:
        if 'esp_rmse' in r and 'n_dcm' in r:
            noneq_esp_by_ndcm[r['n_dcm']].append(r['esp_rmse'])
    
    # Plot DCMNet
    dcm_x = []
    dcm_y_mean = []
    dcm_y_std = []
    for n in ndcm_values:
        if dcm_esp_by_ndcm[n]:
            dcm_x.append(n)
            dcm_y_mean.append(np.mean(dcm_esp_by_ndcm[n]))
            dcm_y_std.append(np.std(dcm_esp_by_ndcm[n]))
    
    if dcm_x:
        ax.errorbar(dcm_x, dcm_y_mean, yerr=dcm_y_std, fmt='o-', 
                   label='DCMNet (Equivariant)', linewidth=2, markersize=8,
                   capsize=5, capthick=2, color='#0072B2', alpha=0.8)
    
    # Plot NonEq
    noneq_x = []
    noneq_y_mean = []
    noneq_y_std = []
    for n in ndcm_values:
        if noneq_esp_by_ndcm[n]:
            noneq_x.append(n)
            noneq_y_mean.append(np.mean(noneq_esp_by_ndcm[n]))
            noneq_y_std.append(np.std(noneq_esp_by_ndcm[n]))
    
    if noneq_x:
        ax.errorbar(noneq_x, noneq_y_mean, yerr=noneq_y_std, fmt='s-',
                   label='NonEquivariant', linewidth=2, markersize=8,
                   capsize=5, capthick=2, color='#D55E00', alpha=0.8)
    
    ax.set_xlabel('Number of Distributed Charges (n_dcm)', fontsize=12)
    ax.set_ylabel('ESP RMSE (Ha/e)', fontsize=12)
    ax.set_title('ESP Accuracy Scaling with n_dcm', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'ndcm_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare equivariant vs non-equivariant model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all runs in comparisons directory
  python -m mmml.cli.compare_equivariant_models \\
      --comparison-dirs examples/co2/dcmnet_physnet_train/comparisons/*/ \\
      --output-dir analysis/model_comparison/
  
  # Specific runs only
  python -m mmml.cli.compare_equivariant_models \\
      --comparison-dirs comparisons/run1/ comparisons/run2/ \\
      --output-dir analysis/
        """
    )
    
    parser.add_argument('--comparison-dirs', nargs='+', type=Path, required=True,
                       help='Directories containing comparison_results.json files')
    parser.add_argument('-o', '--output-dir', type=Path, required=True,
                       help='Output directory for plots and summary')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "="*100)
        print("EQUIVARIANT vs NON-EQUIVARIANT MODEL COMPARISON")
        print("="*100)
        print(f"\nComparison directories: {len(args.comparison_dirs)}")
        print(f"Output directory: {args.output_dir}")
        print("")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all comparison data
    if verbose:
        print("📂 Loading comparison data...")
    
    dcmnet_results, noneq_results = aggregate_comparisons(args.comparison_dirs, verbose=verbose)
    
    if not dcmnet_results and not noneq_results:
        print("\n❌ No comparison data found!")
        return 1
    
    if verbose:
        print(f"\n✅ Loaded {len(dcmnet_results)} DCMNet results")
        print(f"✅ Loaded {len(noneq_results)} NonEq results")
        print("")
    
    # Create plots
    if HAS_MATPLOTLIB:
        if verbose:
            print("📊 Creating comparison plots...")
        
        plot_accuracy_vs_ndcm(dcmnet_results, noneq_results, args.output_dir)
        plot_accuracy_vs_params(dcmnet_results, noneq_results, args.output_dir)
        plot_pareto_front(dcmnet_results, noneq_results, args.output_dir)
        plot_model_comparison(dcmnet_results, noneq_results, args.output_dir)
        plot_training_curves_comparison(dcmnet_results, noneq_results, args.output_dir)
        plot_ndcm_scaling(dcmnet_results, noneq_results, args.output_dir)
        plot_equivariance_test(dcmnet_results, noneq_results, args.output_dir)
        plot_computational_efficiency(dcmnet_results, noneq_results, args.output_dir)
        
        if verbose:
            print("")
    else:
        print("⚠️  Matplotlib not available, skipping plots")
    
    # Create summary table
    if verbose:
        print("📋 Creating summary statistics...")
    
    create_summary_table(dcmnet_results, noneq_results, args.output_dir)
    
    if verbose:
        print("")
        print("="*100)
        print("✅ COMPARISON COMPLETE!")
        print("="*100)
        print(f"\nResults saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - accuracy_vs_ndcm.png")
        print("  - accuracy_vs_params.png")
        print("  - pareto_front.png")
        print("  - model_comparison_bars.png")
        print("  - training_comparison.png")
        print("  - ndcm_scaling.png")
        print("  - equivariance_test.png")
        print("  - computational_efficiency.png")
        print("  - comparison_summary.txt")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


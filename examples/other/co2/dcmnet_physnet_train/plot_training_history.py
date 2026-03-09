#!/usr/bin/env python3
"""
CLI tool for plotting training history and analyzing model parameters

Visualizes training curves, convergence, and parameter statistics from saved checkpoints.

Usage:
    # Plot single training run
    python plot_training_history.py checkpoints/my_model/history.json

    # Compare two training runs
    python plot_training_history.py \
        comparisons/test1/dcmnet_equivariant/history.json \
        comparisons/test1/noneq_model/history.json \
        --compare

    # Include parameter analysis
    python plot_training_history.py history.json \
        --params best_params.pkl \
        --analyze-params

    # Customization
    python plot_training_history.py history.json \
        --output-dir plots --dpi 300 --format pdf \
        --smoothing 0.9
"""

import argparse
import json
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    print("❌ Error: matplotlib not found")
    print("Install with: pip install matplotlib")
    sys.exit(1)

try:
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_flatten, tree_map, tree_structure
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def load_history(json_path: Path) -> Dict[str, List]:
    """Load training history from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {json_path}: {e}")
        sys.exit(1)


def smooth_curve(data: List[float], alpha: float = 0.9) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    if alpha <= 0 or alpha >= 1:
        return np.array(data)
    
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]
    return smoothed


def plot_single_training(
    history: Dict[str, List],
    output_dir: Path,
    name: str = "model",
    dpi: int = 150,
    format: str = 'png',
    smoothing: float = 0.0,
    color: str = '#2E86AB',
):
    """Plot training curves for a single model."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Hatching pattern
    hatch = '///'
    
    # 1. Loss curves
    ax = fig.add_subplot(gs[0, :])
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    
    if smoothing > 0:
        train_loss_smooth = smooth_curve(train_loss, smoothing)
        val_loss_smooth = smooth_curve(val_loss, smoothing)
        ax.plot(epochs, train_loss, alpha=0.2, color=color, linewidth=1)
        ax.plot(epochs, val_loss, alpha=0.2, color='#A23B72', linewidth=1)
        ax.plot(epochs, train_loss_smooth, label='Train Loss (smoothed)', 
               color=color, linewidth=2.5)
        ax.plot(epochs, val_loss_smooth, label='Val Loss (smoothed)', 
               color='#A23B72', linewidth=2.5)
    else:
        ax.plot(epochs, train_loss, label='Train Loss', 
               color=color, linewidth=2.5, alpha=0.8)
        ax.plot(epochs, val_loss, label='Val Loss', 
               color='#A23B72', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'{name} - Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_yscale('log')
    
    # Add best epoch marker
    best_epoch = np.argmin(val_loss) + 1
    ax.axvline(x=best_epoch, color='gold', linestyle='--', linewidth=2.5, 
              alpha=0.7, label=f'Best Epoch: {best_epoch}')
    ax.legend(fontsize=12, framealpha=0.9)
    
    # 2-5. Individual MAE metrics
    metrics = [
        ('val_energy_mae', 'Energy MAE (eV)', '#06A77D'),
        ('val_forces_mae', 'Forces MAE (eV/Å)', '#F45B69'),
        ('val_dipole_mae', 'Dipole MAE (e·Å)', '#4ECDC4'),
        ('val_esp_mae', 'ESP MAE (Ha/e)', '#FF6B6B'),
    ]
    
    for idx, (key, label, col) in enumerate(metrics):
        row = 1 + idx // 2
        col_idx = idx % 2
        ax = fig.add_subplot(gs[row, col_idx])
        
        if key in history:
            data = np.array(history[key])
            if smoothing > 0:
                data_smooth = smooth_curve(data, smoothing)
                ax.plot(epochs, data, alpha=0.2, color=col, linewidth=1)
                ax.plot(epochs, data_smooth, color=col, linewidth=2.5)
            else:
                ax.plot(epochs, data, color=col, linewidth=2.5, alpha=0.8)
            
            # Mark best
            best_idx = np.argmin(data)
            best_val = data[best_idx]
            ax.scatter(best_idx + 1, best_val, s=150, color='gold', 
                      edgecolor='black', linewidth=2.5, zorder=5, marker='*')
            
            # Annotate best
            ax.text(0.98, 0.98, f'Best: {best_val:.6f}\n@ Epoch {best_idx + 1}',
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, linewidth=2))
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_yscale('log')
    
    plt.suptitle(f'{name} - Training History', fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / f'training_history_{name.lower().replace(" ", "_")}.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def plot_comparison_training(
    history1: Dict[str, List],
    history2: Dict[str, List],
    name1: str,
    name2: str,
    output_dir: Path,
    dpi: int = 150,
    format: str = 'png',
    smoothing: float = 0.0,
):
    """Compare training curves from two models."""
    
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)
    
    colors = {
        name1: '#2E86AB',
        name2: '#A23B72',
    }
    
    hatches = {
        name1: '///',
        name2: '\\\\\\',
    }
    
    epochs1 = np.arange(1, len(history1['train_loss']) + 1)
    epochs2 = np.arange(1, len(history2['train_loss']) + 1)
    
    # 1. Loss comparison
    ax = fig.add_subplot(gs[0, :])
    
    for epochs, history, name in [(epochs1, history1, name1), (epochs2, history2, name2)]:
        val_loss = np.array(history['val_loss'])
        if smoothing > 0:
            val_loss_smooth = smooth_curve(val_loss, smoothing)
            ax.plot(epochs, val_loss, alpha=0.15, color=colors[name], linewidth=1)
            ax.plot(epochs, val_loss_smooth, label=f'{name} (smoothed)', 
                   color=colors[name], linewidth=3)
        else:
            ax.plot(epochs, val_loss, label=name, 
                   color=colors[name], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Comparison - Validation Loss', fontsize=14, fontweight='bold')
    legend = ax.legend(fontsize=12, framealpha=0.9, loc='best')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_yscale('log')
    
    # 2-5. Individual metrics comparison
    metrics = [
        ('val_energy_mae', 'Energy MAE (eV)'),
        ('val_forces_mae', 'Forces MAE (eV/Å)'),
        ('val_dipole_mae', 'Dipole MAE (e·Å)'),
        ('val_esp_mae', 'ESP MAE (Ha/e)'),
    ]
    
    for idx, (key, label) in enumerate(metrics):
        row = 1 + idx // 2
        col_idx = idx % 2
        ax = fig.add_subplot(gs[row, col_idx])
        
        for epochs, history, name in [(epochs1, history1, name1), (epochs2, history2, name2)]:
            if key in history:
                data = np.array(history[key])
                if smoothing > 0:
                    data_smooth = smooth_curve(data, smoothing)
                    ax.plot(epochs, data, alpha=0.15, color=colors[name], linewidth=1)
                    ax.plot(epochs, data_smooth, label=name, 
                           color=colors[name], linewidth=2.5)
                else:
                    ax.plot(epochs, data, label=name, 
                           color=colors[name], linewidth=2.5, alpha=0.8)
                
                # Mark best
                best_idx = np.argmin(data)
                best_val = data[best_idx]
                ax.scatter(best_idx + 1, best_val, s=100, color=colors[name], 
                          edgecolor='black', linewidth=2, zorder=5)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        legend = ax.legend(fontsize=10, framealpha=0.9, loc='best')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_yscale('log')
    
    # 6. Convergence rate comparison (last row)
    ax = fig.add_subplot(gs[3, :])
    
    for epochs, history, name in [(epochs1, history1, name1), (epochs2, history2, name2)]:
        if 'epoch_times' in history:
            times = np.array(history['epoch_times'])
            # Remove outliers (like first epoch or negative times)
            times_clean = times[(times > 0) & (times < np.percentile(times[times > 0], 95))]
            
            if smoothing > 0:
                times_smooth = smooth_curve(times_clean[:len(times_clean)], 0.7)
                ax.plot(np.arange(1, len(times_clean) + 1), times_clean, 
                       alpha=0.15, color=colors[name], linewidth=1)
                ax.plot(np.arange(1, len(times_clean) + 1), times_smooth, 
                       label=f'{name} (avg: {np.mean(times_clean):.2f}s)', 
                       color=colors[name], linewidth=2.5)
            else:
                ax.plot(np.arange(1, len(times_clean) + 1), times_clean, 
                       label=f'{name} (avg: {np.mean(times_clean):.2f}s)', 
                       color=colors[name], linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time per Epoch (s)', fontsize=12, fontweight='bold')
    ax.set_title('Training Speed', fontsize=13, fontweight='bold')
    legend = ax.legend(fontsize=11, framealpha=0.9, loc='best')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=10)
    
    plt.suptitle(f'Training Comparison: {name1} vs {name2}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / f'training_comparison.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def analyze_parameters(params_path: Path) -> Dict[str, Any]:
    """Analyze parameter structure from pickle file."""
    try:
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Parameters file not found: {params_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return None
    
    if not HAS_JAX:
        print("⚠️  Warning: JAX not available, parameter analysis limited")
        return {'total_params': 0, 'structure': 'Unknown'}
    
    # Handle nested 'params' dict
    if isinstance(params, dict) and 'params' in params:
        params = params['params']
    
    # Recursively count parameters
    layer_stats = []
    module_counts = defaultdict(int)
    
    def count_recursive(obj, path=""):
        """Recursively count parameters in nested structure."""
        total = 0
        
        if hasattr(obj, 'shape'):
            # It's an array
            size = int(np.prod(obj.shape))
            layer_stats.append({
                'name': path,
                'shape': tuple(obj.shape),
                'size': size
            })
            # Extract module name (top-level key)
            module = path.split('/')[0] if '/' in path else path
            module_counts[module] += size
            return size
            
        elif isinstance(obj, dict):
            # Recursively process dictionary
            for key, value in obj.items():
                new_path = f"{path}/{key}" if path else key
                total += count_recursive(value, new_path)
            return total
            
        elif isinstance(obj, (list, tuple)):
            # Process lists/tuples
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                total += count_recursive(item, new_path)
            return total
            
        else:
            # Unknown type, skip
            return 0
    
    total_params = count_recursive(params)
    
    return {
        'total_params': total_params,
        'layer_stats': layer_stats,
        'module_counts': dict(module_counts),
        'num_layers': len(layer_stats),
    }


def plot_parameter_analysis(
    params_info: Dict[str, Any],
    output_dir: Path,
    name: str = "model",
    dpi: int = 150,
    format: str = 'png',
):
    """Visualize parameter structure and statistics."""
    
    if params_info is None or params_info['total_params'] == 'Unknown' or params_info['total_params'] == 0:
        print(f"⚠️  Skipping parameter analysis for {name} (data not available)")
        return
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Module parameter counts (pie chart)
    ax = fig.add_subplot(gs[0, 0])
    
    modules = list(params_info['module_counts'].keys())
    counts = list(params_info['module_counts'].values())
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(modules)))
    
    wedges, texts, autotexts = ax.pie(counts, labels=modules, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Parameters by Module', fontsize=13, fontweight='bold')
    
    # 2. Module parameter counts (bar chart)
    ax = fig.add_subplot(gs[0, 1])
    
    y_pos = np.arange(len(modules))
    bars = ax.barh(y_pos, counts, color=colors_pie, 
                   alpha=0.85, edgecolor='black', linewidth=2)
    
    # Add hatching
    for i, bar in enumerate(bars):
        bar.set_hatch(['///', '\\\\\\', 'xxx', '...', '|||'][i % 5])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modules, fontsize=11, fontweight='bold')
    ax.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Count by Module', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='x', labelsize=10)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{count/1000:.1f}K' if count < 1e6 else f'{count/1e6:.2f}M',
               ha='left', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 3. Layer size distribution
    ax = fig.add_subplot(gs[1, 0])
    
    layer_sizes = [stat['size'] for stat in params_info['layer_stats']]
    
    bins = np.logspace(np.log10(min(layer_sizes)), np.log10(max(layer_sizes)), 20)
    n, bins, patches = ax.hist(layer_sizes, bins=bins, color='#2E86AB', 
                               alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add hatching
    for patch in patches:
        patch.set_hatch('///')
    
    ax.set_xlabel('Layer Size (# parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Layer Size Distribution', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=10)
    
    # 4. Parameter tree visualization
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    
    # Create summary text
    total = params_info['total_params']
    summary_text = f"""
    PARAMETER SUMMARY
    {'='*40}
    
    Total Parameters: {total:,}
                     ({total/1e6:.2f} Million)
    
    Number of Layers: {params_info['num_layers']}
    
    Module Breakdown:
    """
    
    for module, count in sorted(params_info['module_counts'].items(), 
                               key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100
        summary_text += f"\n    {module:20s}: {count/1000:7.1f}K ({pct:5.1f}%)"
    
    summary_text += f"\n\n    Largest Layers:"
    sorted_layers = sorted(params_info['layer_stats'], key=lambda x: x['size'], reverse=True)[:5]
    for stat in sorted_layers:
        summary_text += f"\n    {stat['name'][:30]:30s}: {stat['size']/1000:7.1f}K"
    
    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, linewidth=2))
    
    plt.suptitle(f'{name} - Parameter Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / f'parameter_analysis_{name.lower().replace(" ", "_")}.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def plot_convergence_analysis(
    history: Dict[str, List],
    output_dir: Path,
    name: str = "model",
    dpi: int = 150,
    format: str = 'png',
    window: int = 50,
):
    """Analyze and plot convergence characteristics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    val_loss = np.array(history['val_loss'])
    epochs = np.arange(1, len(val_loss) + 1)
    
    # 1. Loss improvement rate
    ax = axes[0, 0]
    loss_improvement = -np.diff(val_loss)  # Negative diff = improvement
    ax.plot(epochs[1:], loss_improvement, color='#2E86AB', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax.fill_between(epochs[1:], 0, loss_improvement, 
                    where=(loss_improvement > 0), alpha=0.3, color='green',
                    label='Improving')
    ax.fill_between(epochs[1:], 0, loss_improvement, 
                    where=(loss_improvement < 0), alpha=0.3, color='red',
                    label='Degrading')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Improvement', fontsize=12, fontweight='bold')
    ax.set_title('Per-Epoch Improvement', fontsize=13, fontweight='bold')
    legend = ax.legend(fontsize=10, framealpha=0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=10)
    
    # 2. Rolling average improvement
    ax = axes[0, 1]
    if len(loss_improvement) >= window:
        rolling_avg = np.convolve(loss_improvement, np.ones(window)/window, mode='valid')
        ax.plot(epochs[window:], rolling_avg, color='#A23B72', linewidth=2.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6)
        
        # Find when converged (rolling avg < threshold)
        threshold = np.max(np.abs(rolling_avg)) * 0.01
        converged = np.where(np.abs(rolling_avg) < threshold)[0]
        if len(converged) > 0:
            conv_epoch = converged[0] + window
            ax.axvline(x=conv_epoch, color='gold', linestyle='--', linewidth=2.5,
                      label=f'Converged @ Epoch {conv_epoch}')
            ax.legend(fontsize=10, framealpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{window}-Epoch Avg Improvement', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Trend', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=10)
    
    # 3. Relative progress
    ax = axes[1, 0]
    initial_loss = val_loss[0]
    best_loss = np.min(val_loss)
    progress = ((initial_loss - val_loss) / (initial_loss - best_loss)) * 100
    progress = np.clip(progress, 0, 100)
    
    ax.plot(epochs, progress, color='#06A77D', linewidth=2.5)
    ax.fill_between(epochs, 0, progress, alpha=0.3, color='#06A77D')
    ax.axhline(y=90, color='gold', linestyle='--', linewidth=2, alpha=0.6,
              label='90% of improvement')
    
    # Find 90% point
    idx_90 = np.where(progress >= 90)[0]
    if len(idx_90) > 0:
        epoch_90 = idx_90[0] + 1
        ax.axvline(x=epoch_90, color='gold', linestyle='--', linewidth=2.5,
                  label=f'90% @ Epoch {epoch_90}')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Progress to Best (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training Progress', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 105])
    legend = ax.legend(fontsize=10, framealpha=0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=10)
    
    # 4. Final metrics summary (bar chart)
    ax = axes[1, 1]
    
    final_metrics = {
        'Energy': history['val_energy_mae'][-1] if 'val_energy_mae' in history else 0,
        'Forces': history['val_forces_mae'][-1] if 'val_forces_mae' in history else 0,
        'Dipole': history['val_dipole_mae'][-1] if 'val_dipole_mae' in history else 0,
        'ESP': history['val_esp_mae'][-1] * 1000 if 'val_esp_mae' in history else 0,
    }
    
    metric_names = list(final_metrics.keys())
    values = list(final_metrics.values())
    colors_bar = ['#06A77D', '#F45B69', '#4ECDC4', '#FF6B6B']
    hatches_bar = ['///', '\\\\\\', 'xxx', '...']
    
    y_pos = np.arange(len(metric_names))
    bars = ax.barh(y_pos, values, color=colors_bar, 
                   alpha=0.85, edgecolor='black', linewidth=2)
    
    for bar, hatch in zip(bars, hatches_bar):
        bar.set_hatch(hatch)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.set_xlabel('Final MAE', fontsize=12, fontweight='bold')
    ax.set_title('Final Validation Metrics', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xscale('log')
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{val:.6f}',
               ha='left', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle(f'{name} - Convergence Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / f'convergence_analysis_{name.lower().replace(" ", "_")}.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def plot_parameter_tree_comparison(
    params_info1: Dict[str, Any],
    params_info2: Dict[str, Any],
    name1: str,
    name2: str,
    output_dir: Path,
    dpi: int = 150,
    format: str = 'png',
):
    """Compare parameter structures between two models."""
    
    if params_info1 is None or params_info2 is None:
        print("⚠️  Skipping parameter tree comparison (data not available)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {name1: '#2E86AB', name2: '#A23B72'}
    hatches = {name1: '///', name2: '\\\\\\'}
    
    # 1. Total parameters comparison
    ax = axes[0]
    
    totals = [params_info1['total_params'], params_info2['total_params']]
    names = [name1, name2]
    
    bars = ax.bar([0, 1], totals, 
                  color=[colors[name1], colors[name2]],
                  alpha=0.85, edgecolor='black', linewidth=2.5,
                  hatch=[hatches[name1], hatches[name2]])
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names, fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Parameters', fontsize=13, fontweight='bold')
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='y', labelsize=11)
    
    # Add value labels
    for bar, val, name in zip(bars, totals, names):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val/1e6:.2f}M\n({val:,})',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add difference
    diff = abs(totals[0] - totals[1])
    pct = (diff / max(totals)) * 100
    winner = names[0] if totals[0] < totals[1] else names[1]
    ax.text(0.5, 0.95, f'{winner} is {pct:.1f}% smaller',
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, linewidth=2))
    
    # 2. Module-by-module comparison
    ax = axes[1]
    
    # Get common modules
    modules1 = set(params_info1['module_counts'].keys())
    modules2 = set(params_info2['module_counts'].keys())
    all_modules = sorted(modules1 | modules2)
    
    counts1 = [params_info1['module_counts'].get(m, 0) for m in all_modules]
    counts2 = [params_info2['module_counts'].get(m, 0) for m in all_modules]
    
    x = np.arange(len(all_modules))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts1, width, label=name1,
                   color=colors[name1], alpha=0.85, edgecolor='black', 
                   linewidth=2, hatch=hatches[name1])
    bars2 = ax.bar(x + width/2, counts2, width, label=name2,
                   color=colors[name2], alpha=0.85, edgecolor='black', 
                   linewidth=2, hatch=hatches[name2])
    
    ax.set_xlabel('Module', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameters', fontsize=12, fontweight='bold')
    ax.set_title('Module-Level Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_modules, rotation=45, ha='right', fontsize=10, fontweight='bold')
    legend = ax.legend(fontsize=11, framealpha=0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='y', labelsize=10)
    
    plt.suptitle(f'Parameter Structure Comparison: {name1} vs {name2}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'parameter_comparison.{format}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def print_training_summary(history: Dict[str, List], name: str = "Model"):
    """Print text summary of training."""
    print(f"\n{'='*70}")
    print(f"{name} - TRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    total_epochs = len(history['train_loss'])
    print(f"Total Epochs: {total_epochs}")
    
    # Loss
    final_train = history['train_loss'][-1]
    final_val = history['val_loss'][-1]
    best_val = np.min(history['val_loss'])
    best_epoch = np.argmin(history['val_loss']) + 1
    
    print(f"\nLoss:")
    print(f"  Final Train Loss: {final_train:.4f}")
    print(f"  Final Val Loss:   {final_val:.4f}")
    print(f"  Best Val Loss:    {best_val:.4f} @ Epoch {best_epoch}")
    
    # MAE metrics
    if 'val_energy_mae' in history:
        print(f"\nFinal Validation MAE:")
        print(f"  Energy: {history['val_energy_mae'][-1]:.6f} eV")
        print(f"  Forces: {history['val_forces_mae'][-1]:.6f} eV/Å")
        print(f"  Dipole: {history['val_dipole_mae'][-1]:.6f} e·Å")
        print(f"  ESP:    {history['val_esp_mae'][-1]:.6f} Ha/e")
        
        print(f"\nBest Validation MAE:")
        print(f"  Energy: {np.min(history['val_energy_mae']):.6f} eV @ Epoch {np.argmin(history['val_energy_mae']) + 1}")
        print(f"  Forces: {np.min(history['val_forces_mae']):.6f} eV/Å @ Epoch {np.argmin(history['val_forces_mae']) + 1}")
        print(f"  Dipole: {np.min(history['val_dipole_mae']):.6f} e·Å @ Epoch {np.argmin(history['val_dipole_mae']) + 1}")
        print(f"  ESP:    {np.min(history['val_esp_mae']):.6f} Ha/e @ Epoch {np.argmin(history['val_esp_mae']) + 1}")
    
    # Timing
    if 'epoch_times' in history:
        times = np.array(history['epoch_times'])
        times_clean = times[(times > 0) & (times < np.percentile(times[times > 0], 95))]
        
        print(f"\nTraining Speed:")
        print(f"  Avg time per epoch: {np.mean(times_clean):.2f}s")
        print(f"  Min time per epoch: {np.min(times_clean):.2f}s")
        print(f"  Max time per epoch: {np.max(times_clean):.2f}s")
        print(f"  Total training time: {np.sum(times_clean)/3600:.2f}h")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training history and analyze model parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single training
  python plot_training_history.py checkpoints/my_model/history.json
  
  # Compare two runs
  python plot_training_history.py hist1.json hist2.json --compare
  
  # With parameter analysis
  python plot_training_history.py history.json \
      --params best_params.pkl --analyze-params
  
  # Full comparison with parameters
  python plot_training_history.py \
      dcmnet/history.json noneq/history.json \
      --compare \
      --params dcmnet/best_params.pkl noneq/best_params.pkl \
      --analyze-params
  
  # High-resolution
  python plot_training_history.py history.json --dpi 300 --format pdf
        """
    )
    
    parser.add_argument('history_files', type=Path, nargs='+',
                       help='Training history JSON file(s)')
    
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple training runs (requires 2 history files)')
    
    parser.add_argument('--params', type=Path, nargs='+', default=None,
                       help='Parameter pickle file(s) for analysis')
    
    parser.add_argument('--analyze-params', action='store_true',
                       help='Analyze and plot parameter structure')
    
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: same as history file)')
    
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output images')
    
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg', 'jpg'],
                       help='Output format')
    
    parser.add_argument('--smoothing', type=float, default=0.0,
                       help='Exponential smoothing factor (0-1, 0=none, 0.9=heavy)')
    
    parser.add_argument('--names', type=str, nargs='+', default=None,
                       help='Custom names for models')
    
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print text summary, no plots')
    
    parser.add_argument('--convergence', action='store_true',
                       help='Plot convergence analysis')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.compare and len(args.history_files) != 2:
        print("❌ Error: --compare requires exactly 2 history files")
        sys.exit(1)
    
    if args.analyze_params and not HAS_JAX:
        print("⚠️  Warning: JAX not available, parameter analysis will be limited")
    
    # Determine names
    if args.names:
        names = args.names
    elif args.compare:
        names = ['Model 1', 'Model 2']
    else:
        names = ['Model']
    
    # Load histories
    print(f"\n📊 Loading training history...")
    histories = [load_history(f) for f in args.history_files]
    
    # Print summaries
    for hist, name in zip(histories, names):
        print_training_summary(hist, name)
    
    if args.summary_only:
        return
    
    # Setup output directory
    output_dir = args.output_dir or args.history_files[0].parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🎨 Creating plots...")
    print(f"  Output directory: {output_dir}")
    print(f"  Format: {args.format}, DPI: {args.dpi}")
    if args.smoothing > 0:
        print(f"  Smoothing: {args.smoothing}")
    print()
    
    # Plot training curves
    if args.compare:
        plot_comparison_training(
            histories[0], histories[1],
            names[0], names[1],
            output_dir, args.dpi, args.format, args.smoothing
        )
    else:
        plot_single_training(
            histories[0], output_dir, names[0],
            args.dpi, args.format, args.smoothing
        )
    
    # Convergence analysis
    if args.convergence:
        for hist, name in zip(histories, names):
            plot_convergence_analysis(hist, output_dir, name, args.dpi, args.format)
    
    # Parameter analysis
    if args.analyze_params and args.params:
        print(f"\n🔍 Analyzing parameters...")
        
        params_infos = []
        for param_file in args.params:
            print(f"  Loading: {param_file}")
            params_infos.append(analyze_parameters(param_file))
        
        # Plot parameter analysis
        if len(params_infos) == 1:
            plot_parameter_analysis(params_infos[0], output_dir, names[0], 
                                   args.dpi, args.format)
        elif len(params_infos) == 2:
            plot_parameter_analysis(params_infos[0], output_dir, names[0], 
                                   args.dpi, args.format)
            plot_parameter_analysis(params_infos[1], output_dir, names[1], 
                                   args.dpi, args.format)
            plot_parameter_tree_comparison(params_infos[0], params_infos[1],
                                          names[0], names[1], output_dir,
                                          args.dpi, args.format)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print(f"\nFiles created:")
    for f in sorted(output_dir.glob(f'*.{args.format}')):
        if f.name.startswith('training_') or f.name.startswith('convergence_') or f.name.startswith('parameter_'):
            print(f"  - {f.name}")
    print()


if __name__ == '__main__':
    main()


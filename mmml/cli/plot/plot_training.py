#!/usr/bin/env python3
"""
CLI tool for plotting training history and analyzing model parameters

Visualizes training curves, convergence, and parameter statistics from saved checkpoints.

Usage:
    # Plot single training run
    python -m mmml.cli.plot_training checkpoints/my_model/history.json

    # Compare two training runs
    python -m mmml.cli.plot_training \
        model1/history.json \
        model2/history.json \
        --compare

    # Include parameter analysis
    python -m mmml.cli.plot_training history.json \
        --params best_params.pkl \
        --analyze-params

    # Customization
    python -m mmml.cli.plot_training history.json \
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
    
    # 2-5. Individual MAE metrics (if available)
    metrics = []
    if 'val_energy_mae' in history:
        metrics.append(('val_energy_mae', 'Energy MAE (eV)', '#06A77D'))
    if 'val_forces_mae' in history:
        metrics.append(('val_forces_mae', 'Forces MAE (eV/Å)', '#F45B69'))
    if 'val_dipole_mae' in history:
        metrics.append(('val_dipole_mae', 'Dipole MAE (e·Å)', '#4ECDC4'))
    if 'val_esp_mae' in history:
        metrics.append(('val_esp_mae', 'ESP MAE (Ha/e)', '#FF6B6B'))
    
    for idx, (key, label, col) in enumerate(metrics):
        row = 1 + idx // 2
        col_idx = idx % 2
        ax = fig.add_subplot(gs[row, col_idx])
        
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
    
    # 2-5. Individual metrics comparison (if available)
    available_metrics = []
    if 'val_energy_mae' in history1 and 'val_energy_mae' in history2:
        available_metrics.append(('val_energy_mae', 'Energy MAE (eV)'))
    if 'val_forces_mae' in history1 and 'val_forces_mae' in history2:
        available_metrics.append(('val_forces_mae', 'Forces MAE (eV/Å)'))
    if 'val_dipole_mae' in history1 and 'val_dipole_mae' in history2:
        available_metrics.append(('val_dipole_mae', 'Dipole MAE (e·Å)'))
    if 'val_esp_mae' in history1 and 'val_esp_mae' in history2:
        available_metrics.append(('val_esp_mae', 'ESP MAE (Ha/e)'))
    
    for idx, (key, label) in enumerate(available_metrics):
        row = 1 + idx // 2
        col_idx = idx % 2
        ax = fig.add_subplot(gs[row, col_idx])
        
        for epochs, history, name in [(epochs1, history1, name1), (epochs2, history2, name2)]:
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
            times_clean = times[(times > 0) & (times < np.percentile(times[times > 0], 95))] if len(times) > 0 else times
            
            if len(times_clean) > 0:
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
    
    # MAE metrics (if available)
    if 'val_energy_mae' in history:
        print(f"\nFinal Validation MAE:")
        if 'val_energy_mae' in history:
            print(f"  Energy: {history['val_energy_mae'][-1]:.6f} eV")
        if 'val_forces_mae' in history:
            print(f"  Forces: {history['val_forces_mae'][-1]:.6f} eV/Å")
        if 'val_dipole_mae' in history:
            print(f"  Dipole: {history['val_dipole_mae'][-1]:.6f} e·Å")
        if 'val_esp_mae' in history:
            print(f"  ESP:    {history['val_esp_mae'][-1]:.6f} Ha/e")
        
        print(f"\nBest Validation MAE:")
        if 'val_energy_mae' in history:
            print(f"  Energy: {np.min(history['val_energy_mae']):.6f} eV @ Epoch {np.argmin(history['val_energy_mae']) + 1}")
        if 'val_forces_mae' in history:
            print(f"  Forces: {np.min(history['val_forces_mae']):.6f} eV/Å @ Epoch {np.argmin(history['val_forces_mae']) + 1}")
        if 'val_dipole_mae' in history:
            print(f"  Dipole: {np.min(history['val_dipole_mae']):.6f} e·Å @ Epoch {np.argmin(history['val_dipole_mae']) + 1}")
        if 'val_esp_mae' in history:
            print(f"  ESP:    {np.min(history['val_esp_mae']):.6f} Ha/e @ Epoch {np.argmin(history['val_esp_mae']) + 1}")
    
    # Timing
    if 'epoch_times' in history:
        times = np.array(history['epoch_times'])
        times_clean = times[(times > 0) & (times < np.percentile(times[times > 0], 95))] if len(times) > 0 else times
        
        if len(times_clean) > 0:
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
  python -m mmml.cli.plot_training checkpoints/my_model/history.json
  
  # Compare two runs
  python -m mmml.cli.plot_training hist1.json hist2.json --compare
  
  # With parameter analysis
  python -m mmml.cli.plot_training history.json \\
      --params best_params.pkl --analyze-params
  
  # High-resolution
  python -m mmml.cli.plot_training history.json --dpi 300 --format pdf
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
    
    # Parameter analysis
    if args.analyze_params and args.params:
        print(f"\n🔍 Analyzing parameters...")
        
        for param_file, name in zip(args.params, names):
            print(f"  Loading: {param_file}")
            params_info = analyze_parameters(param_file)
            if params_info and params_info['total_params'] > 0:
                total = params_info['total_params']
                print(f"    Total parameters: {total:,} ({total/1e6:.2f}M)")
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print(f"\nFiles created:")
    for f in sorted(output_dir.glob(f'*.{args.format}')):
        if f.name.startswith('training_'):
            print(f"  - {f.name}")
    print()


if __name__ == '__main__':
    main()


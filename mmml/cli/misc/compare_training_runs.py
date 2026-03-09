#!/usr/bin/env python3
"""
Compare Multiple Training Runs

Plot metrics from multiple training runs on the same axes for direct comparison.

Usage:
    python -m mmml.cli.compare_training_runs \
        --runs run1/ run2/ run3/ \
        --labels "Run 1" "Run 2" "Run 3" \
        --output comparison.png \
        --log-loss
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import re

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("❌ Matplotlib required")
    sys.exit(1)

try:
    from orbax.checkpoint import PyTreeCheckpointer
    HAS_ORBAX = True
except ImportError:
    HAS_ORBAX = False


def extract_metrics_from_orbax(epoch_dir: Path) -> Dict:
    """Extract metrics from Orbax checkpoint."""
    try:
        if HAS_ORBAX:
            checkpointer = PyTreeCheckpointer()
            restored = checkpointer.restore(str(epoch_dir.resolve()))
            
            metrics = {
                'epoch': restored.get('epoch', None),
                'best_loss': restored.get('best_loss', None),
                'objectives': restored.get('objectives', {}),
                'lr_eff': restored.get('lr_eff', None),
            }
            
            return metrics
    except Exception as e:
        return None


def collect_all_metrics(ckpt_dir: Path, verbose: bool = False) -> Dict[str, np.ndarray]:
    """Collect metrics from all epoch checkpoints."""
    
    epoch_dirs = sorted([d for d in ckpt_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('epoch-')],
                       key=lambda x: int(re.search(r'(\d+)', x.name).group(1)))
    
    if not epoch_dirs:
        return None
    
    all_metrics = {
        'epochs': [],
        'train_loss': [],
        'valid_loss': [],
        'train_energy_mae': [],
        'valid_energy_mae': [],
        'train_forces_mae': [],
        'valid_forces_mae': [],
        'train_dipole_mae': [],
        'valid_dipole_mae': [],
        'lr_eff': [],
        'best_loss': [],
    }
    
    for epoch_dir in epoch_dirs:
        metrics = extract_metrics_from_orbax(epoch_dir)
        
        if metrics is None:
            continue
        
        epoch_num = metrics.get('epoch')
        if epoch_num is None:
            match = re.search(r'epoch-(\d+)', epoch_dir.name)
            if match:
                epoch_num = int(match.group(1))
        
        if epoch_num is not None:
            all_metrics['epochs'].append(epoch_num)
        
        obj = metrics.get('objectives', {})
        all_metrics['train_loss'].append(obj.get('train_loss', np.nan))
        all_metrics['valid_loss'].append(obj.get('valid_loss', np.nan))
        all_metrics['train_energy_mae'].append(obj.get('train_energy_mae', np.nan))
        all_metrics['valid_energy_mae'].append(obj.get('valid_energy_mae', np.nan))
        all_metrics['train_forces_mae'].append(obj.get('train_forces_mae', np.nan))
        all_metrics['valid_forces_mae'].append(obj.get('valid_forces_mae', np.nan))
        all_metrics['train_dipole_mae'].append(obj.get('train_dipole_mae', np.nan))
        all_metrics['valid_dipole_mae'].append(obj.get('valid_dipole_mae', np.nan))
        all_metrics['lr_eff'].append(metrics.get('lr_eff', np.nan))
        all_metrics['best_loss'].append(metrics.get('best_loss', np.nan))
    
    for key in all_metrics:
        all_metrics[key] = np.array(all_metrics[key])
    
    if verbose:
        valid_count = np.sum(~np.isnan(all_metrics['valid_loss']))
        print(f"  ✅ Loaded {ckpt_dir.name}: {valid_count} checkpoints")
    
    return all_metrics


def plot_comparison(
    all_runs_metrics: List[Dict],
    labels: List[str],
    output_path: Path,
    log_loss: bool = True,
):
    """Plot comparison of multiple training runs."""
    
    # Okabe-Ito colorblind-friendly palette
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    eV_to_kcal = 23.0605  # Unit conversion
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Validation Loss Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    for i, (metrics, label) in enumerate(zip(all_runs_metrics, labels)):
        valid_idx = ~np.isnan(metrics['valid_loss'])
        epochs = metrics['epochs'][valid_idx]
        valid_loss = metrics['valid_loss'][valid_idx]
        
        ax1.plot(epochs, valid_loss, marker=markers[i % len(markers)], 
                label=label, color=colors[i % len(colors)],
                linewidth=2.5, markersize=5, alpha=0.8, markevery=max(1, len(epochs)//20))
        
        # Mark best
        best_idx = np.argmin(valid_loss)
        ax1.scatter(epochs[best_idx], valid_loss[best_idx],
                   s=200, color=colors[i % len(colors)], 
                   edgecolor='black', linewidth=2, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    if log_loss:
        ax1.set_yscale('log')
        ax1.set_title('Validation Loss Comparison (Log Scale)', fontsize=14, fontweight='bold')
    else:
        ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    for i, (metrics, label) in enumerate(zip(all_runs_metrics, labels)):
        valid_idx = ~np.isnan(metrics['train_loss'])
        epochs = metrics['epochs'][valid_idx]
        train_loss = metrics['train_loss'][valid_idx]
        
        ax2.plot(epochs, train_loss, marker=markers[i % len(markers)],
                label=label, color=colors[i % len(colors)],
                linewidth=2, markersize=4, alpha=0.7, markevery=max(1, len(epochs)//20))
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Training Loss', fontsize=11)
    ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    if log_loss:
        ax2.set_yscale('log')
    
    # Plot 3: Energy MAE Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, (metrics, label) in enumerate(zip(all_runs_metrics, labels)):
        if not np.all(np.isnan(metrics['valid_energy_mae'])):
            valid_idx = ~np.isnan(metrics['valid_energy_mae'])
            epochs = metrics['epochs'][valid_idx]
            energy_mae = metrics['valid_energy_mae'][valid_idx] * eV_to_kcal
            
            ax3.plot(epochs, energy_mae, marker=markers[i % len(markers)],
                    label=label, color=colors[i % len(colors)],
                    linewidth=2.5, markersize=5, alpha=0.8, markevery=max(1, len(epochs)//20))
            
            # Mark best
            best_idx = np.argmin(energy_mae)
            ax3.scatter(epochs[best_idx], energy_mae[best_idx],
                       s=150, color=colors[i % len(colors)],
                       edgecolor='black', linewidth=2, zorder=5, marker='*')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Energy MAE (kcal/mol)', fontsize=11)
    ax3.set_title('Energy Prediction', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    if log_loss:
        ax3.set_yscale('log')
    
    # Plot 4: Forces MAE Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, (metrics, label) in enumerate(zip(all_runs_metrics, labels)):
        if not np.all(np.isnan(metrics['valid_forces_mae'])):
            valid_idx = ~np.isnan(metrics['valid_forces_mae'])
            epochs = metrics['epochs'][valid_idx]
            forces_mae = metrics['valid_forces_mae'][valid_idx] * eV_to_kcal
            
            ax4.plot(epochs, forces_mae, marker=markers[i % len(markers)],
                    label=label, color=colors[i % len(colors)],
                    linewidth=2.5, markersize=5, alpha=0.8, markevery=max(1, len(epochs)//20))
            
            # Mark best
            best_idx = np.argmin(forces_mae)
            ax4.scatter(epochs[best_idx], forces_mae[best_idx],
                       s=150, color=colors[i % len(colors)],
                       edgecolor='black', linewidth=2, zorder=5, marker='*')
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Forces MAE (kcal/mol/Å)', fontsize=11)
    ax4.set_title('Forces Prediction', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    if log_loss:
        ax4.set_yscale('log')
    
    # Plot 5: Best Loss Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    
    for i, (metrics, label) in enumerate(zip(all_runs_metrics, labels)):
        valid_idx = ~np.isnan(metrics['best_loss'])
        epochs = metrics['epochs'][valid_idx]
        best_loss = metrics['best_loss'][valid_idx]
        
        ax5.plot(epochs, best_loss, marker=markers[i % len(markers)],
                label=label, color=colors[i % len(colors)],
                linewidth=2.5, markersize=5, alpha=0.8, markevery=max(1, len(epochs)//20))
    
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Best Loss So Far', fontsize=11)
    ax5.set_title('Best Model Progress', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    if log_loss:
        ax5.set_yscale('log')
    
    plt.suptitle(f'Training Comparison: {len(all_runs_metrics)} Runs',
                 fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple training runs side by side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two glycol runs
  python -m mmml.cli.compare_training_runs \\
      --runs checkpoints/run1/ checkpoints/run2/ \\
      --labels "Run 1" "Run 2" \\
      --output comparison.png --log-loss
        """
    )
    
    parser.add_argument('--runs', nargs='+', type=Path, required=True,
                       help='Checkpoint directories to compare (space-separated)')
    parser.add_argument('--labels', nargs='+', type=str, default=None,
                       help='Labels for each run (default: Run 1, Run 2, ...)')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output plot file (PNG)')
    parser.add_argument('--log-loss', action='store_true',
                       help='Use log scale for loss axes (recommended)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Resolve wildcards in paths
    resolved_runs = []
    for run_path in args.runs:
        if '*' in str(run_path):
            from glob import glob
            matches = glob(str(run_path))
            if matches:
                resolved_runs.append(Path(matches[0]))
            else:
                print(f"⚠️  No match for: {run_path}")
        else:
            resolved_runs.append(run_path)
    
    args.runs = resolved_runs
    
    # Generate labels if not provided
    if args.labels is None:
        args.labels = [f"Run {i+1}" for i in range(len(args.runs))]
    elif len(args.labels) != len(args.runs):
        print(f"❌ Number of labels ({len(args.labels)}) must match number of runs ({len(args.runs)})")
        return 1
    
    # Validate directories
    for run_dir in args.runs:
        if not run_dir.exists():
            print(f"❌ Directory not found: {run_dir}")
            return 1
    
    if not HAS_ORBAX:
        print("❌ Orbax checkpoint library required")
        print("   Install with: pip install orbax-checkpoint")
        return 1
    
    if verbose:
        print("\n" + "="*80)
        print("COMPARING TRAINING RUNS")
        print("="*80)
        print(f"\nRuns to compare: {len(args.runs)}")
        for label, run_dir in zip(args.labels, args.runs):
            print(f"  - {label}: {run_dir.name}")
        print(f"\nOutput: {args.output}")
        print("")
    
    # Load metrics from all runs
    if verbose:
        print("📂 Loading metrics from all runs...")
    
    all_runs_metrics = []
    for run_dir in args.runs:
        metrics = collect_all_metrics(run_dir, verbose=verbose)
        if metrics is None:
            print(f"⚠️  No metrics found in {run_dir.name}, skipping")
            continue
        all_runs_metrics.append(metrics)
    
    if not all_runs_metrics:
        print("❌ No valid training runs found")
        return 1
    
    if verbose:
        print("")
    
    # Create comparison plots
    if verbose:
        print("📊 Creating comparison plots...")
    
    plot_comparison(all_runs_metrics, args.labels, args.output, log_loss=args.log_loss)
    
    # Print summary statistics
    if verbose:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print("")
        
        for i, (metrics, label) in enumerate(zip(all_runs_metrics, args.labels)):
            valid_idx = ~np.isnan(metrics['valid_loss'])
            
            print(f"{label}:")
            print(f"  Epochs: {len(metrics['epochs'][valid_idx])}")
            print(f"  Best valid loss: {np.nanmin(metrics['valid_loss']):.6f}")
            print(f"  Final valid loss: {metrics['valid_loss'][valid_idx][-1]:.6f}")
            
            if not np.all(np.isnan(metrics['valid_energy_mae'])):
                energy_kcal = np.nanmin(metrics['valid_energy_mae']) * 23.0605
                print(f"  Best energy MAE: {energy_kcal:.4f} kcal/mol")
            
            if not np.all(np.isnan(metrics['valid_forces_mae'])):
                forces_kcal = np.nanmin(metrics['valid_forces_mae']) * 23.0605
                print(f"  Best forces MAE: {forces_kcal:.4f} kcal/mol/Å")
            
            print("")
        
        print("="*80)
    
    print(f"\n✅ Comparison complete! View: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


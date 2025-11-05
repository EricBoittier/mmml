#!/usr/bin/env python3
"""
Extract and Plot Training Metrics from Orbax Checkpoints

This tool extracts loss values and metrics from Orbax checkpoint files
and creates comprehensive training plots with log-scale loss.

Usage:
    python -m mmml.cli.extract_checkpoint_metrics \
        checkpoints/run/run-uuid/ \
        --output training_plots.png \
        --log-loss
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import re
import pickle

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
    print("⚠️  Orbax not available, will try pickle fallback")


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


def collect_all_metrics(ckpt_dir: Path, verbose: bool = True) -> Dict[str, List]:
    """Collect metrics from all epoch checkpoints."""
    
    # Find epoch directories
    epoch_dirs = sorted([d for d in ckpt_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('epoch-')],
                       key=lambda x: int(re.search(r'(\d+)', x.name).group(1)))
    
    if not epoch_dirs:
        raise ValueError(f"No epoch checkpoints found in {ckpt_dir}")
    
    if verbose:
        print(f"Found {len(epoch_dirs)} epoch checkpoints")
    
    # Collect metrics
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
        
        # Extract from objectives dict
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
    
    # Convert to numpy arrays and remove NaNs
    for key in all_metrics:
        all_metrics[key] = np.array(all_metrics[key])
    
    if verbose:
        valid_count = np.sum(~np.isnan(all_metrics['valid_loss']))
        print(f"Extracted metrics from {valid_count}/{len(epoch_dirs)} checkpoints")
    
    return all_metrics


def plot_training_metrics(
    metrics: Dict[str, np.ndarray],
    output_path: Path,
    ckpt_name: str = "Training",
    log_loss: bool = True,
    verbose: bool = True,
):
    """Create comprehensive training plots."""
    
    epochs = metrics['epochs']
    
    # Remove NaN values
    valid_idx = ~np.isnan(metrics['valid_loss'])
    epochs_valid = epochs[valid_idx]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Training and Validation Loss (Large, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    train_loss_valid = metrics['train_loss'][valid_idx]
    valid_loss_valid = metrics['valid_loss'][valid_idx]
    
    ax1.plot(epochs_valid, train_loss_valid, 'o-', label='Training Loss',
            color='#0072B2', linewidth=2.5, markersize=6, alpha=0.8)
    ax1.plot(epochs_valid, valid_loss_valid, 's-', label='Validation Loss',
            color='#D55E00', linewidth=2.5, markersize=6, alpha=0.8)
    
    # Mark best
    best_idx = np.argmin(valid_loss_valid)
    ax1.scatter(epochs_valid[best_idx], valid_loss_valid[best_idx], 
               s=300, color='gold', edgecolor='black', linewidth=3, 
               zorder=5, marker='*', label='Best')
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    if log_loss:
        ax1.set_yscale('log')
        ax1.set_title('Training Progress (Log Scale)', fontsize=14, fontweight='bold')
    else:
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Annotations removed per user request
    
    # Plot 2: Best Loss Progression
    ax2 = fig.add_subplot(gs[0, 2])
    best_loss_valid = metrics['best_loss'][valid_idx]
    ax2.plot(epochs_valid, best_loss_valid, 'o-', color='#009E73',
            linewidth=2.5, markersize=6, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Best Loss So Far', fontsize=11)
    ax2.set_title('Best Model Progress', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if log_loss:
        ax2.set_yscale('log')
    
    # Unit conversion constant
    eV_to_kcal = 23.0605  # 1 eV = 23.0605 kcal/mol
    
    # Plot 3: Energy MAE
    ax3 = fig.add_subplot(gs[1, 0])
    if not np.all(np.isnan(metrics['valid_energy_mae'])):
        valid_energy_valid = metrics['valid_energy_mae'][valid_idx] * eV_to_kcal
        ax3.plot(epochs_valid, valid_energy_valid, 's-', color='#E69F00',
                linewidth=2.5, markersize=6, alpha=0.8, label='Validation')
        
        if not np.all(np.isnan(metrics['train_energy_mae'])):
            train_energy_valid = metrics['train_energy_mae'][valid_idx] * eV_to_kcal
            ax3.plot(epochs_valid, train_energy_valid, 'o-', color='#E69F00',
                    linewidth=1.5, markersize=4, alpha=0.4, label='Training')
        
        # Mark best
        best_idx = np.argmin(valid_energy_valid)
        ax3.scatter(epochs_valid[best_idx], valid_energy_valid[best_idx],
                   s=200, color='gold', edgecolor='black', linewidth=2, zorder=5, marker='*')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Energy MAE (kcal/mol)', fontsize=11)
    ax3.set_title('Energy Prediction', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    if log_loss:
        ax3.set_yscale('log')
    
    # Plot 4: Forces MAE
    ax4 = fig.add_subplot(gs[1, 1])
    if not np.all(np.isnan(metrics['valid_forces_mae'])):
        valid_forces_valid = metrics['valid_forces_mae'][valid_idx] * eV_to_kcal
        ax4.plot(epochs_valid, valid_forces_valid, 's-', color='#CC79A7',
                linewidth=2.5, markersize=6, alpha=0.8, label='Validation')
        
        if not np.all(np.isnan(metrics['train_forces_mae'])):
            train_forces_valid = metrics['train_forces_mae'][valid_idx] * eV_to_kcal
            ax4.plot(epochs_valid, train_forces_valid, 'o-', color='#CC79A7',
                    linewidth=1.5, markersize=4, alpha=0.4, label='Training')
        
        # Mark best
        best_idx = np.argmin(valid_forces_valid)
        ax4.scatter(epochs_valid[best_idx], valid_forces_valid[best_idx],
                   s=200, color='gold', edgecolor='black', linewidth=2, zorder=5, marker='*')
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Forces MAE (kcal/mol/Å)', fontsize=11)
    ax4.set_title('Forces Prediction', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    if log_loss:
        ax4.set_yscale('log')
    
    # Plot 5: Dipole MAE
    ax5 = fig.add_subplot(gs[1, 2])
    if not np.all(np.isnan(metrics['valid_dipole_mae'])):
        valid_dipole_valid = metrics['valid_dipole_mae'][valid_idx]
        ax5.plot(epochs_valid, valid_dipole_valid, 's-', color='#56B4E9',
                linewidth=2.5, markersize=6, alpha=0.8, label='Validation')
        
        if not np.all(np.isnan(metrics['train_dipole_mae'])):
            train_dipole_valid = metrics['train_dipole_mae'][valid_idx]
            ax5.plot(epochs_valid, train_dipole_valid, 'o-', color='#56B4E9',
                    linewidth=1.5, markersize=4, alpha=0.4, label='Training')
        
        # Mark best
        best_idx = np.argmin(valid_dipole_valid)
        ax5.scatter(epochs_valid[best_idx], valid_dipole_valid[best_idx],
                   s=200, color='gold', edgecolor='black', linewidth=2, zorder=5, marker='*')
    
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Dipole MAE (e·Å)', fontsize=11)
    ax5.set_title('Dipole Prediction', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    if log_loss:
        ax5.set_yscale('log')
    
    # Plot 6: Learning Rate
    ax6 = fig.add_subplot(gs[2, 0])
    if not np.all(np.isnan(metrics['lr_eff'])):
        lr_valid = metrics['lr_eff'][valid_idx]
        ax6.plot(epochs_valid, lr_valid, 'o-', color='#F0E442',
                linewidth=2.5, markersize=6, alpha=0.8)
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Learning Rate', fontsize=11)
        ax6.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
    
    # Plot 7: Loss Components Breakdown
    ax7 = fig.add_subplot(gs[2, 1:])
    
    # Calculate improvement from start
    if len(valid_loss_valid) > 1:
        improvement = (valid_loss_valid[0] - valid_loss_valid) / valid_loss_valid[0] * 100
        ax7.plot(epochs_valid, improvement, 'o-', color='#009E73',
                linewidth=2.5, markersize=6, alpha=0.8)
        ax7.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax7.set_xlabel('Epoch', fontsize=11)
        ax7.set_ylabel('Improvement from Start (%)', fontsize=11)
        ax7.set_title('Validation Loss Improvement', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.fill_between(epochs_valid, 0, improvement, alpha=0.2, color='#009E73')
        
        # Annotations removed per user request
    
    plt.suptitle(f'Training Analysis: {ckpt_name}\n{len(epochs_valid)} checkpoints analyzed',
                 fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✅ Saved: {output_path}")


def print_metrics_summary(metrics: Dict[str, np.ndarray], ckpt_dir: Path):
    """Print summary statistics."""
    
    valid_idx = ~np.isnan(metrics['valid_loss'])
    epochs_valid = metrics['epochs'][valid_idx]
    
    print("\n" + "="*80)
    print("TRAINING METRICS SUMMARY")
    print("="*80)
    print(f"\nCheckpoint: {ckpt_dir.name}")
    print(f"Epochs analyzed: {len(epochs_valid)}")
    print(f"Epoch range: {int(np.min(epochs_valid))} to {int(np.max(epochs_valid))}")
    print("")
    
    # Loss statistics
    train_loss = metrics['train_loss'][valid_idx]
    valid_loss = metrics['valid_loss'][valid_idx]
    
    print("LOSS STATISTICS")
    print("-"*80)
    print(f"Training Loss:")
    print(f"  Initial: {train_loss[0]:.6f}")
    print(f"  Final:   {train_loss[-1]:.6f}")
    print(f"  Best:    {np.nanmin(train_loss):.6f} (epoch {epochs_valid[np.nanargmin(train_loss)]})")
    print(f"  Improvement: {(train_loss[0] - train_loss[-1])/train_loss[0]*100:.1f}%")
    print("")
    print(f"Validation Loss:")
    print(f"  Initial: {valid_loss[0]:.6f}")
    print(f"  Final:   {valid_loss[-1]:.6f}")
    print(f"  Best:    {np.nanmin(valid_loss):.6f} (epoch {epochs_valid[np.nanargmin(valid_loss)]})")
    print(f"  Improvement: {(valid_loss[0] - valid_loss[-1])/valid_loss[0]*100:.1f}%")
    print("")
    
    # MAE statistics
    print("VALIDATION MAE METRICS")
    print("-"*80)
    
    # Unit conversion
    eV_to_kcal = 23.0605  # 1 eV = 23.0605 kcal/mol
    
    mae_metrics = [
        ('valid_energy_mae', 'Energy', 'kcal/mol', eV_to_kcal),
        ('valid_forces_mae', 'Forces', 'kcal/mol/Å', eV_to_kcal),
        ('valid_dipole_mae', 'Dipole', 'e·Å', 1.0),
    ]
    
    for key, name, unit, conversion in mae_metrics:
        if not np.all(np.isnan(metrics[key])):
            data = metrics[key][valid_idx] * conversion
            print(f"{name:8s} MAE: {data[-1]:.6f} {unit:12s} (best: {np.nanmin(data):.6f} @ epoch {epochs_valid[np.nanargmin(data)]})")
    
    print("")
    
    # Learning rate
    if not np.all(np.isnan(metrics['lr_eff'])):
        lr = metrics['lr_eff'][valid_idx]
        print("LEARNING RATE")
        print("-"*80)
        print(f"Initial: {lr[0]:.6e}")
        print(f"Final:   {lr[-1]:.6e}")
        if lr[-1] < lr[0]:
            print(f"Decay:   {(1 - lr[-1]/lr[0])*100:.1f}%")
        print("")
    
    # Convergence analysis
    print("CONVERGENCE ANALYSIS")
    print("-"*80)
    
    # Check if converged (last 10 epochs stable)
    if len(valid_loss) >= 10:
        last_10_std = np.std(valid_loss[-10:])
        last_10_mean = np.mean(valid_loss[-10:])
        relative_std = last_10_std / last_10_mean * 100
        
        print(f"Last 10 epochs:")
        print(f"  Mean loss: {last_10_mean:.6f}")
        print(f"  Std dev:   {last_10_std:.6f}")
        print(f"  Relative std: {relative_std:.2f}%")
        
        if relative_std < 1.0:
            print(f"  ✅ Converged (std < 1%)")
        elif relative_std < 5.0:
            print(f"  ⚠️  Nearly converged (std < 5%)")
        else:
            print(f"  ❌ Not converged (std > 5%) - consider more epochs")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and plot training metrics from Orbax checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot glycol training with log scale
  python -m mmml.cli.extract_checkpoint_metrics \\
      examples/glycol/checkpoints/glycol_production/glycol_production-*/ \\
      --output glycol_training.png \\
      --log-loss
  
  # Without log scale
  python -m mmml.cli.extract_checkpoint_metrics \\
      checkpoints/run/run-uuid/ \\
      --output training.png
        """
    )
    
    parser.add_argument('checkpoint_dir', type=Path,
                       help='Checkpoint directory containing epoch-* subdirectories')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output plot file (PNG)')
    parser.add_argument('--log-loss', action='store_true',
                       help='Use log scale for loss axes (recommended)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Resolve checkpoint directory (handle wildcards)
    if '*' in str(args.checkpoint_dir):
        from glob import glob
        matches = glob(str(args.checkpoint_dir))
        if not matches:
            print(f"❌ No directories match: {args.checkpoint_dir}")
            return 1
        args.checkpoint_dir = Path(matches[0])
        if verbose:
            print(f"📂 Resolved to: {args.checkpoint_dir.name}")
    
    if not args.checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
        return 1
    
    if not HAS_ORBAX:
        print("❌ Orbax checkpoint library required")
        print("   Install with: pip install orbax-checkpoint")
        return 1
    
    if verbose:
        print("\n" + "="*80)
        print("EXTRACTING TRAINING METRICS FROM CHECKPOINTS")
        print("="*80)
        print(f"\nCheckpoint directory: {args.checkpoint_dir.name}")
        print(f"Output: {args.output}")
        print("")
    
    # Extract metrics
    if verbose:
        print("📊 Extracting metrics from checkpoints...")
    
    metrics = collect_all_metrics(args.checkpoint_dir, verbose=verbose)
    
    # Create plots
    if verbose:
        print("\n📈 Creating training plots...")
    
    plot_training_metrics(metrics, args.output, ckpt_name=args.checkpoint_dir.name,
                         log_loss=args.log_loss, verbose=verbose)
    
    # Print summary
    print_metrics_summary(metrics, args.checkpoint_dir)
    
    if verbose:
        print("\n✅ ANALYSIS COMPLETE!")
        print(f"\nTo evaluate the best checkpoint:")
        best_epoch = int(metrics['epochs'][np.nanargmin(metrics['valid_loss'])])
        print(f"  python -m mmml.cli.evaluate_model \\")
        print(f"      {args.checkpoint_dir}/epoch-{best_epoch} \\")
        print(f"      --test-data splits/data_test.npz")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


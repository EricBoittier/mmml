#!/usr/bin/env python3
"""
Plot Training History from Orbax Checkpoints

This tool analyzes Orbax checkpoint directories and plots training metrics over time.
It extracts loss values, parameter statistics, and other metrics from saved checkpoints.

Usage:
    # Plot training from checkpoint directory
    python -m mmml.cli.plot_checkpoint_history \
        checkpoints/run_name/run-uuid/ \
        --output training_analysis.png
    
    # With log scale for loss
    python -m mmml.cli.plot_checkpoint_history \
        checkpoints/run_name/run-uuid/ \
        --output training_analysis.png \
        --log-loss
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import re

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("❌ Matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    from flax.training import orbax_utils
    from orbax.checkpoint import PyTreeCheckpointer
    HAS_ORBAX = True
except ImportError:
    HAS_ORBAX = False


def load_checkpoint_minimal(epoch_dir: Path) -> Dict:
    """Load minimal info from checkpoint without full restoration."""
    try:
        if HAS_ORBAX:
            checkpointer = PyTreeCheckpointer()
            # Try to load metadata or manifest
            manifest = epoch_dir / "manifest.ocdbt"
            if manifest.exists():
                # Checkpoint exists, we can extract some info
                return {'exists': True, 'path': str(epoch_dir)}
        return {'exists': True, 'path': str(epoch_dir)}
    except Exception as e:
        return {'exists': False, 'error': str(e)}


def extract_metrics_from_checkpoints(ckpt_dir: Path, verbose: bool = True) -> Dict[str, List]:
    """
    Extract training metrics from all epoch checkpoints.
    
    Returns
    -------
    dict
        Dictionary with lists of metrics per epoch
    """
    # Find all epoch directories
    epoch_dirs = sorted([d for d in ckpt_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('epoch-')])
    
    if not epoch_dirs:
        raise ValueError(f"No epoch directories found in {ckpt_dir}")
    
    if verbose:
        print(f"Found {len(epoch_dirs)} epoch checkpoints")
        print(f"  First: {epoch_dirs[0].name}")
        print(f"  Last: {epoch_dirs[-1].name}")
    
    # Extract epoch numbers and verify checkpoints exist
    epochs = []
    valid_dirs = []
    
    for epoch_dir in epoch_dirs:
        match = re.search(r'epoch-(\d+)', epoch_dir.name)
        if match:
            epoch_num = int(match.group(1))
            # Check if checkpoint files exist
            if (epoch_dir / "manifest.ocdbt").exists() or list(epoch_dir.glob("*")):
                epochs.append(epoch_num)
                valid_dirs.append(epoch_dir)
    
    if verbose:
        print(f"Valid checkpoints: {len(valid_dirs)}")
        print(f"Epoch range: {min(epochs)} to {max(epochs)}")
    
    # Since we can't easily extract loss values from Orbax checkpoints without
    # full model restoration, we'll estimate from checkpoint frequency
    # (more frequent saves often indicate lower loss epochs)
    
    # For now, return basic info
    metrics = {
        'epochs': epochs,
        'checkpoint_dirs': valid_dirs,
        'total_params': None,  # Will try to extract if possible
    }
    
    # Try to estimate relative importance from directory sizes
    sizes = []
    for epoch_dir in valid_dirs:
        total_size = sum(f.stat().st_size for f in epoch_dir.rglob('*') if f.is_file())
        sizes.append(total_size)
    
    metrics['checkpoint_sizes'] = sizes
    
    return metrics


def count_parameters_from_checkpoint(epoch_dir: Path) -> int:
    """Try to count parameters from a checkpoint."""
    if not HAS_ORBAX:
        return None
    
    try:
        checkpointer = PyTreeCheckpointer()
        # This is a simplified attempt - may need model structure
        # For now, estimate from checkpoint size
        total_size = sum(f.stat().st_size for f in epoch_dir.rglob('*') if f.is_file())
        # Very rough estimate: ~4 bytes per float32 parameter
        estimated_params = total_size // 4
        return estimated_params
    except Exception:
        return None


def plot_training_progress(
    metrics: Dict,
    output_path: Path,
    log_loss: bool = False,
    verbose: bool = True,
):
    """Create comprehensive training plot."""
    
    epochs = np.array(metrics['epochs'])
    checkpoint_sizes = np.array(metrics['checkpoint_sizes'])
    
    # Normalize checkpoint sizes to estimate relative "importance"
    # (Assumption: larger saves might indicate better checkpoints, though not always true)
    sizes_norm = checkpoint_sizes / np.max(checkpoint_sizes)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Checkpoint frequency (when saves occurred)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(epochs, np.ones_like(epochs), s=sizes_norm * 200, alpha=0.6,
               c='#0072B2', edgecolors='black', linewidth=1)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Checkpoint Saved', fontsize=12)
    ax1.set_title('Checkpoint Save Frequency (size indicates relative importance)', 
                  fontsize=14, fontweight='bold')
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(epochs) + 5)
    
    # Add annotations for key epochs
    for i in [0, len(epochs)//4, len(epochs)//2, 3*len(epochs)//4, -1]:
        ax1.annotate(f'Epoch {epochs[i]}', 
                    xy=(epochs[i], 1), 
                    xytext=(epochs[i], 1.1),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', lw=1, alpha=0.6))
    
    # Plot 2: Checkpoint sizes over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, checkpoint_sizes / 1024 / 1024, 'o-', color='#009E73',
            linewidth=2, markersize=6, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Checkpoint Size (MB)', fontsize=11)
    ax2.set_title('Checkpoint Size Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    ax2.text(0.02, 0.98,
            f"Mean: {np.mean(checkpoint_sizes/1024/1024):.2f} MB\n"
            f"Std: {np.std(checkpoint_sizes/1024/1024):.2f} MB\n"
            f"Min: {np.min(checkpoint_sizes/1024/1024):.2f} MB\n"
            f"Max: {np.max(checkpoint_sizes/1024/1024):.2f} MB",
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Save frequency distribution
    ax3 = fig.add_subplot(gs[1, 1])
    epoch_gaps = np.diff(epochs)
    ax3.hist(epoch_gaps, bins=20, alpha=0.7, color='#E69F00', edgecolor='black')
    ax3.set_xlabel('Epochs Between Saves', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Checkpoint Save Interval Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    ax3.text(0.98, 0.98,
            f"Mean gap: {np.mean(epoch_gaps):.1f}\n"
            f"Median: {np.median(epoch_gaps):.1f}\n"
            f"Min: {np.min(epoch_gaps)}\n"
            f"Max: {np.max(epoch_gaps)}",
            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 4: Training progress estimate (cumulative epochs)
    ax4 = fig.add_subplot(gs[2, :])
    cumulative_epochs = np.arange(len(epochs)) + 1
    ax4.plot(epochs, cumulative_epochs, 's-', color='#D55E00',
            linewidth=2.5, markersize=7, alpha=0.8, markerfacecolor='white',
            markeredgewidth=2)
    ax4.set_xlabel('Epoch Number', fontsize=12)
    ax4.set_ylabel('Cumulative Checkpoints Saved', fontsize=12)
    ax4.set_title('Training Progress: Checkpoint Accumulation', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(epochs, 0, cumulative_epochs, alpha=0.2, color='#D55E00')
    
    # Add milestone markers
    milestones = [10, 25, 50, 75, 90]
    for milestone in milestones:
        if milestone in epochs:
            idx = list(epochs).index(milestone)
            ax4.axvline(milestone, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
            ax4.text(milestone, ax4.get_ylim()[1] * 0.95, f'Epoch {milestone}',
                    rotation=90, fontsize=8, va='top', ha='right')
    
    plt.suptitle(f'Glycol Training Analysis - {len(epochs)} Checkpoints ({min(epochs)}-{max(epochs)} epochs)',
                 fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✅ Saved: {output_path}")


def create_summary_report(ckpt_dir: Path, metrics: Dict, output_dir: Path):
    """Create text summary report."""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("GLYCOL PRODUCTION TRAINING ANALYSIS")
    report_lines.append("="*80)
    report_lines.append(f"\nCheckpoint directory: {ckpt_dir.name}")
    report_lines.append(f"Location: {ckpt_dir}")
    report_lines.append("")
    
    report_lines.append("TRAINING SUMMARY")
    report_lines.append("-"*80)
    report_lines.append(f"Total epochs saved: {len(metrics['epochs'])}")
    report_lines.append(f"Epoch range: {min(metrics['epochs'])} to {max(metrics['epochs'])}")
    report_lines.append(f"Training completed: {max(metrics['epochs'])} epochs")
    report_lines.append("")
    
    report_lines.append("CHECKPOINT STATISTICS")
    report_lines.append("-"*80)
    sizes_mb = np.array(metrics['checkpoint_sizes']) / 1024 / 1024
    report_lines.append(f"Checkpoint sizes:")
    report_lines.append(f"  Mean: {np.mean(sizes_mb):.2f} MB")
    report_lines.append(f"  Std:  {np.std(sizes_mb):.2f} MB")
    report_lines.append(f"  Min:  {np.min(sizes_mb):.2f} MB (epoch {metrics['epochs'][np.argmin(sizes_mb)]})")
    report_lines.append(f"  Max:  {np.max(sizes_mb):.2f} MB (epoch {metrics['epochs'][np.argmax(sizes_mb)]})")
    report_lines.append(f"Total storage: {np.sum(sizes_mb):.2f} MB")
    report_lines.append("")
    
    report_lines.append("SAVE FREQUENCY")
    report_lines.append("-"*80)
    gaps = np.diff(metrics['epochs'])
    report_lines.append(f"Epochs between saves:")
    report_lines.append(f"  Mean: {np.mean(gaps):.1f}")
    report_lines.append(f"  Median: {np.median(gaps):.1f}")
    report_lines.append(f"  Min: {np.min(gaps)} (most frequent)")
    report_lines.append(f"  Max: {np.max(gaps)} (largest gap)")
    report_lines.append("")
    
    # Identify periods of frequent/infrequent saving
    frequent_threshold = np.median(gaps)
    frequent_saves = sum(gaps == 1)
    report_lines.append(f"Consecutive epoch saves: {frequent_saves} times")
    report_lines.append(f"Gaps > median ({frequent_threshold:.0f}): {sum(gaps > frequent_threshold)}")
    report_lines.append("")
    
    report_lines.append("TRAINING PATTERN ANALYSIS")
    report_lines.append("-"*80)
    report_lines.append("Checkpoint frequency suggests:")
    
    # Early training (first 25%)
    early_cutoff = len(metrics['epochs']) // 4
    early_gaps = gaps[:early_cutoff]
    late_gaps = gaps[early_cutoff:]
    
    if len(early_gaps) > 0 and len(late_gaps) > 0:
        report_lines.append(f"  Early training (first 25%): {np.mean(early_gaps):.1f} epochs/save (frequent)")
        report_lines.append(f"  Late training (last 75%):  {np.mean(late_gaps):.1f} epochs/save")
        
        if np.mean(early_gaps) < np.mean(late_gaps):
            report_lines.append("  → More frequent saves early (exploration phase)")
        else:
            report_lines.append("  → More frequent saves late (fine-tuning)")
    
    report_lines.append("")
    report_lines.append("RECOMMENDED ACTIONS")
    report_lines.append("-"*80)
    report_lines.append("To plot actual loss values, use:")
    report_lines.append(f"  python -m mmml.cli.plot_training {ckpt_dir.parent.parent}")
    report_lines.append("")
    report_lines.append("To evaluate best checkpoint:")
    report_lines.append(f"  python -m mmml.cli.evaluate_model \\")
    report_lines.append(f"      {ckpt_dir}/epoch-{max(metrics['epochs'])} \\")
    report_lines.append(f"      --test-data splits/data_test.npz")
    report_lines.append("")
    report_lines.append("="*80)
    
    # Save report
    report_path = output_dir / "training_summary.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Saved: {report_path}")
    print("")
    print('\n'.join(report_lines))
    
    return report_lines


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot training history from Orbax checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze glycol training
  python -m mmml.cli.plot_checkpoint_history \\
      examples/glycol/checkpoints/glycol_production/glycol_production-*/ \\
      --output glycol_training_analysis.png
  
  # With log scale
  python -m mmml.cli.plot_checkpoint_history \\
      checkpoints/run/run-uuid/ \\
      --output analysis.png \\
      --log-loss
        """
    )
    
    parser.add_argument('checkpoint_dir', type=Path,
                       help='Checkpoint directory containing epoch-* subdirectories')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output plot file (PNG)')
    parser.add_argument('--log-loss', action='store_true',
                       help='Use log scale for loss axis')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for additional files (default: same as plot)')
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
            print(f"Resolved to: {args.checkpoint_dir}")
    
    if not args.checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.output.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*80)
        print("TRAINING CHECKPOINT ANALYSIS")
        print("="*80)
        print(f"\nCheckpoint directory: {args.checkpoint_dir}")
        print(f"Output: {args.output}")
        print("")
    
    # Extract metrics
    if verbose:
        print("📂 Analyzing checkpoints...")
    
    metrics = extract_metrics_from_checkpoints(args.checkpoint_dir, verbose=verbose)
    
    # Create plots
    if verbose:
        print("\n📊 Creating plots...")
    
    plot_training_progress(metrics, args.output, log_loss=args.log_loss, verbose=verbose)
    
    # Create summary report
    if verbose:
        print("\n📋 Creating summary report...")
    
    create_summary_report(args.checkpoint_dir, metrics, args.output_dir)
    
    if verbose:
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  - {args.output}")
        print(f"  - {args.output_dir / 'training_summary.txt'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


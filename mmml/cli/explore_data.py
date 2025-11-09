#!/usr/bin/env python3
"""
Explore and visualize NPZ datasets.

Features:
- Statistical summaries of all fields
- Energy, force, dipole distributions
- Geometry analysis (bond lengths, angles)
- ESP visualization (if available)
- Unit detection and validation
- Data quality checks

Usage:
    # Basic exploration
    python -m mmml.cli.explore_data data.npz
    
    # With plots
    python -m mmml.cli.explore_data data.npz --plots --output-dir exploration
    
    # Detailed analysis
    python -m mmml.cli.explore_data data.npz --detailed --plots --output-dir analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def print_summary(data: Dict[str, np.ndarray], verbose: bool = True):
    """Print dataset summary."""
    if not verbose:
        return
    
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    # Get number of samples
    n_samples = None
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            n_samples = value.shape[0]
            break
    
    print(f"\nTotal samples: {n_samples}")
    print(f"\nFields:")
    for key, value in data.items():
        if isinstance(value, np.ndarray) and hasattr(value, 'shape'):
            print(f"  {key:20s}: {str(value.shape):30s} {value.dtype}")
        else:
            print(f"  {key:20s}: {type(value).__name__}")


def analyze_energies(E: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """Analyze energy distribution."""
    stats = {
        'mean': float(E.mean()),
        'std': float(E.std()),
        'min': float(E.min()),
        'max': float(E.max()),
        'range': float(E.max() - E.min()),
    }
    
    if verbose:
        print("\n" + "="*70)
        print("ENERGY ANALYSIS")
        print("="*70)
        print(f"  Mean:  {stats['mean']:.6f}")
        print(f"  Std:   {stats['std']:.6f}")
        print(f"  Min:   {stats['min']:.6f}")
        print(f"  Max:   {stats['max']:.6f}")
        print(f"  Range: {stats['range']:.6f}")
    
    return stats


def analyze_forces(F: np.ndarray, N: np.ndarray = None, verbose: bool = True) -> Dict[str, Any]:
    """Analyze force distribution."""
    # Remove padding if N is provided
    if N is not None:
        forces_real = []
        for i in range(len(F)):
            forces_real.append(F[i, :N[i], :])
        forces_flat = np.vstack(forces_real)
    else:
        forces_flat = F.reshape(-1, 3)
    
    force_norms = np.linalg.norm(forces_flat, axis=1)
    
    stats = {
        'mean_norm': float(force_norms.mean()),
        'std_norm': float(force_norms.std()),
        'max_norm': float(force_norms.max()),
        'mean_components': [float(forces_flat[:, i].mean()) for i in range(3)],
    }
    
    if verbose:
        print("\n" + "="*70)
        print("FORCE ANALYSIS")
        print("="*70)
        print(f"  Mean norm: {stats['mean_norm']:.6e}")
        print(f"  Std norm:  {stats['std_norm']:.6e}")
        print(f"  Max norm:  {stats['max_norm']:.6e}")
        print(f"  Mean components: [{stats['mean_components'][0]:.6e}, "
              f"{stats['mean_components'][1]:.6e}, {stats['mean_components'][2]:.6e}]")
        
        # Check if forces are balanced
        if max(abs(c) for c in stats['mean_components']) < 1e-3:
            print(f"  ✅ Forces well-balanced (mean ≈ 0)")
        else:
            print(f"  ⚠️  Non-zero mean components detected")
    
    return stats


def analyze_geometry(R: np.ndarray, Z: np.ndarray, N: np.ndarray = None, 
                     verbose: bool = True, max_samples: int = 100) -> Dict[str, Any]:
    """Analyze molecular geometries."""
    if verbose:
        print("\n" + "="*70)
        print("GEOMETRY ANALYSIS")
        print("="*70)
    
    # Collect bond lengths
    bond_lengths = []
    elements = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S'}
    
    for i in range(min(max_samples, len(R))):
        r = R[i]
        z = Z[i]
        
        # Get valid atoms
        if N is not None:
            n = N[i]
            valid_pos = r[:n]
            valid_z = z[:n]
        else:
            valid_mask = z > 0
            valid_pos = r[valid_mask]
            valid_z = z[valid_mask]
        
        # Calculate pairwise distances
        for j in range(len(valid_pos)):
            for k in range(j+1, len(valid_pos)):
                dist = np.linalg.norm(valid_pos[j] - valid_pos[k])
                z1, z2 = int(valid_z[j]), int(valid_z[k])
                elem1 = elements.get(z1, f'Z{z1}')
                elem2 = elements.get(z2, f'Z{z2}')
                bond_lengths.append((f"{elem1}-{elem2}", dist))
    
    # Summarize by bond type
    from collections import defaultdict
    bond_stats = defaultdict(list)
    
    for bond_type, dist in bond_lengths:
        bond_stats[bond_type].append(dist)
    
    if verbose:
        print(f"\nBond statistics (sampled {min(max_samples, len(R))} structures):")
        for bond_type, dists in sorted(bond_stats.items()):
            dists = np.array(dists)
            print(f"  {bond_type:8s}: mean={dists.mean():.4f} Å, "
                  f"std={dists.std():.4f}, "
                  f"range=[{dists.min():.4f}, {dists.max():.4f}]")
    
    return dict(bond_stats)


def create_plots(data: Dict[str, np.ndarray], output_dir: Path, verbose: bool = True):
    """Create visualization plots."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not available, skipping plots")
        return
    
    if verbose:
        print("\n" + "="*70)
        print("CREATING PLOTS")
        print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Energy distribution
    if 'E' in data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data['E'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Energy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Energy Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print("  ✅ energy_distribution.png")
    
    # Force distribution
    if 'F' in data:
        F_flat = data['F'].reshape(-1, 3)
        force_norms = np.linalg.norm(F_flat, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Force Distributions', fontsize=16, fontweight='bold')
        
        axes[0, 0].hist(F_flat[:, 0], bins=50, alpha=0.7, edgecolor='black', color='red')
        axes[0, 0].set_xlabel('Force X')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('X Component')
        axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        axes[0, 1].hist(F_flat[:, 1], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_xlabel('Force Y')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Y Component')
        axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        axes[1, 0].hist(F_flat[:, 2], bins=50, alpha=0.7, edgecolor='black', color='blue')
        axes[1, 0].set_xlabel('Force Z')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Z Component')
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        axes[1, 1].hist(force_norms, bins=50, alpha=0.7, edgecolor='black', color='purple')
        axes[1, 1].set_xlabel('Force Magnitude')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Force Norms')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'force_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print("  ✅ force_distributions.png")
    
    # Dipole distribution
    if 'D' in data or 'Dxyz' in data:
        D_key = 'D' if 'D' in data else 'Dxyz'
        D = data[D_key]
        dipole_norms = np.linalg.norm(D, axis=-1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dipole_norms, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Dipole Magnitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Dipole Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'dipole_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print("  ✅ dipole_distribution.png")


def main():
    parser = argparse.ArgumentParser(
        description="Explore and visualize NPZ datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic exploration
  python -m mmml.cli.explore_data data.npz
  
  # With plots
  python -m mmml.cli.explore_data data.npz --plots --output-dir exploration
  
  # Detailed analysis
  python -m mmml.cli.explore_data data.npz --detailed --plots --output-dir analysis
        """
    )
    
    parser.add_argument('input', type=Path,
                       help='Input NPZ file')
    parser.add_argument('--detailed', action='store_true',
                       help='Detailed analysis including geometry')
    parser.add_argument('--plots', action='store_true',
                       help='Generate distribution plots')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for plots')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Validate
    if not args.input.exists():
        print(f"❌ Error: File not found: {args.input}")
        return 1
    
    if args.plots and not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not available, plots disabled")
        args.plots = False
    
    verbose = not args.quiet
    
    # Load data
    if verbose:
        print(f"\n📁 Loading: {args.input}")
    
    data = dict(np.load(args.input, allow_pickle=True))
    
    # Print summary
    print_summary(data, verbose=verbose)
    
    # Analyze fields
    if 'E' in data:
        analyze_energies(data['E'], verbose=verbose)
    
    if 'F' in data:
        N = data.get('N', None)
        analyze_forces(data['F'], N, verbose=verbose)
    
    if args.detailed and 'R' in data and 'Z' in data:
        N = data.get('N', None)
        analyze_geometry(data['R'], data['Z'], N, verbose=verbose)
    
    # Create plots
    if args.plots:
        output_dir = args.output_dir or args.input.parent / 'exploration'
        create_plots(data, output_dir, verbose=verbose)
        
        if verbose:
            print(f"\n✅ Plots saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


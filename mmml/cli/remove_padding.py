#!/usr/bin/env python3
"""
Remove padding from NPZ datasets.

Datasets often contain padding (extra zeros) to handle variable molecule sizes.
This tool removes the padding to create compact datasets.

Benefits:
- Smaller file sizes
- Faster training (fewer atoms to process)
- More efficient memory usage

Usage:
    # Auto-detect padding from N field
    python -m mmml.cli.remove_padding input.npz -o unpadded.npz
    
    # Specify maximum atoms
    python -m mmml.cli.remove_padding input.npz -o unpadded.npz --max-atoms 10
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import numpy as np


def remove_padding(
    data: Dict[str, np.ndarray],
    max_atoms: int = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Remove padding from dataset.
    
    Parameters
    ----------
    data : dict
        Dataset with padded arrays
    max_atoms : int, optional
        Maximum number of atoms (auto-detected from N if not provided)
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Dataset with padding removed
    """
    # Auto-detect max_atoms from N field
    if max_atoms is None:
        if 'N' in data:
            max_atoms = int(np.max(data['N']))
            if verbose:
                print(f"✅ Auto-detected max_atoms = {max_atoms} from N field")
        else:
            raise ValueError("Cannot auto-detect max_atoms. Please specify --max-atoms")
    
    # Get current padding size
    if 'R' in data:
        current_atoms = data['R'].shape[1]
        padding = current_atoms - max_atoms
        
        if verbose:
            print(f"\nPadding analysis:")
            print(f"   Current atoms per structure: {current_atoms}")
            print(f"   Actual atoms (max): {max_atoms}")
            print(f"   Padding atoms: {padding}")
        
        if padding <= 0:
            if verbose:
                print(f"\n✅ No padding detected - dataset already compact")
            return data
    
    # Remove padding from relevant fields
    unpadded = {}
    
    for key, value in data.items():
        if not isinstance(value, np.ndarray):
            unpadded[key] = value
            continue
        
        # Fields that need unpadding (per-atom arrays)
        if key == 'R' and value.ndim == 3:
            # Positions: (n_samples, n_atoms_padded, 3) → (n_samples, max_atoms, 3)
            unpadded[key] = value[:, :max_atoms, :]
            if verbose:
                print(f"   ✅ {key}: {value.shape} → {unpadded[key].shape}")
        
        elif key == 'Z' and value.ndim == 2:
            # Atomic numbers: (n_samples, n_atoms_padded) → (n_samples, max_atoms)
            unpadded[key] = value[:, :max_atoms]
            if verbose:
                print(f"   ✅ {key}: {value.shape} → {unpadded[key].shape}")
        
        elif key == 'F' and value.ndim == 3:
            # Forces: (n_samples, n_atoms_padded, 3) → (n_samples, max_atoms, 3)
            unpadded[key] = value[:, :max_atoms, :]
            if verbose:
                print(f"   ✅ {key}: {value.shape} → {unpadded[key].shape}")
        
        else:
            # Keep as-is (E, N, D, Dxyz, etc.)
            unpadded[key] = value
            if verbose and key in ['E', 'N', 'D', 'Dxyz', 'dipoles']:
                print(f"   ✅ {key}: {value.shape} (no padding)")
    
    return unpadded


def main():
    parser = argparse.ArgumentParser(
        description="Remove padding from NPZ datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect from N field
  python -m mmml.cli.remove_padding data.npz -o unpadded.npz
  
  # Specify max atoms
  python -m mmml.cli.remove_padding data.npz -o unpadded.npz --max-atoms 10
  
  # Process all splits
  for f in splits/*.npz; do
    python -m mmml.cli.remove_padding "$f" -o "unpadded/$(basename $f)"
  done
        """
    )
    
    parser.add_argument('input', type=Path,
                       help='Input NPZ file (with padding)')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output NPZ file (without padding)')
    parser.add_argument('--max-atoms', type=int, default=None,
                       help='Maximum number of atoms (auto-detected from N if not specified)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Validate
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "="*70)
        print("REMOVING PADDING FROM DATASET")
        print("="*70)
        print(f"\nInput: {args.input}")
        print(f"Output: {args.output}")
    
    # Load data
    data = dict(np.load(args.input, allow_pickle=True))
    
    # Remove padding
    unpadded = remove_padding(data, max_atoms=args.max_atoms, verbose=verbose)
    
    # Save
    if verbose:
        print(f"\n💾 Saving unpadded dataset...")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **unpadded)
    
    if verbose:
        old_size = args.input.stat().st_size / 1024 / 1024
        new_size = args.output.stat().st_size / 1024 / 1024
        reduction = (1 - new_size / old_size) * 100
        
        print(f"✅ Saved: {args.output}")
        print(f"   Original size: {old_size:.1f} MB")
        print(f"   New size: {new_size:.1f} MB")
        print(f"   Reduction: {reduction:.1f}%")
        
        print(f"\n{'='*70}")
        print("✅ PADDING REMOVED!")
        print(f"{'='*70}")
        print(f"\nNow you can train with --num_atoms {unpadded['R'].shape[1] if 'R' in unpadded else max_atoms}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


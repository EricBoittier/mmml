#!/usr/bin/env python3
"""
Split datasets into train/valid/test sets with optional unit conversion.

Supports:
- Random splits with configurable ratios
- Reproducible splits (seed-based)
- Optional unit conversion (Hartree→eV, Hartree/Bohr→eV/Å)
- Validation of split consistency
- Multiple file handling (EFD + ESP grids)

Usage:
    # Basic split (no conversion)
    python -m mmml.cli.split_dataset data.npz -o output_dir --train 0.8 --valid 0.1 --test 0.1
    
    # With unit conversion (Hartree → eV)
    python -m mmml.cli.split_dataset data.npz -o output_dir --convert-units
    
    # Split EFD + ESP grid files together
    python -m mmml.cli.split_dataset \
        --efd energies_forces_dipoles.npz \
        --grid grids_esp.npz \
        -o training_data \
        --convert-units --train 0.8 --valid 0.1 --test 0.1
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np


def create_splits(
    n_samples: int,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create train/valid/test split indices.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    train_frac : float
        Training fraction
    valid_frac : float
        Validation fraction
    test_frac : float
        Test fraction
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary with 'train', 'valid', 'test' indices
    """
    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {train_frac + valid_frac + test_frac}")
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    n_train = int(n_samples * train_frac)
    n_valid = int(n_samples * valid_frac)
    
    return {
        'train': indices[:n_train],
        'valid': indices[n_train:n_train + n_valid],
        'test': indices[n_train + n_valid:]
    }


def convert_units(
    data: Dict[str, np.ndarray],
    convert_energy: bool = True,
    convert_forces: bool = True,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Convert units to ASE standard.
    
    Conversions:
    - Energy: Hartree → eV (×27.211386)
    - Forces: Hartree/Bohr → eV/Angstrom (×51.42208)
    
    Parameters
    ----------
    data : dict
        Dataset dictionary
    convert_energy : bool
        Convert energy units
    convert_forces : bool
        Convert force units
    verbose : bool
        Print conversion info
    
    Returns
    -------
    dict
        Data with converted units
    """
    HARTREE_TO_EV = 27.211386
    HARTREE_BOHR_TO_EV_ANG = 51.42208
    
    converted = data.copy()
    
    if convert_energy and 'E' in data:
        if verbose:
            print(f"\n🔄 Converting energies: Hartree → eV")
            print(f"   Before: mean={data['E'].mean():.6f} Ha")
        converted['E'] = data['E'] * HARTREE_TO_EV
        if verbose:
            print(f"   After:  mean={converted['E'].mean():.6f} eV")
    
    if convert_forces and 'F' in data:
        if verbose:
            print(f"\n🔄 Converting forces: Hartree/Bohr → eV/Å")
            f_before = np.linalg.norm(data['F'].reshape(-1, 3), axis=1).mean()
            print(f"   Before: mean norm={f_before:.6e} Ha/Bohr")
        converted['F'] = data['F'] * HARTREE_BOHR_TO_EV_ANG
        if verbose:
            f_after = np.linalg.norm(converted['F'].reshape(-1, 3), axis=1).mean()
            print(f"   After:  mean norm={f_after:.6e} eV/Å")
    
    return converted


def split_and_save(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    splits: Dict[str, np.ndarray],
    prefix: str = "data",
    verbose: bool = True
):
    """
    Split dataset and save to files.
    
    Only keeps essential training fields to prevent indexing errors.
    
    Parameters
    ----------
    data : dict
        Dataset to split
    output_dir : Path
        Output directory
    splits : dict
        Split indices (train, valid, test)
    prefix : str
        File prefix (e.g., "energies_forces_dipoles")
    verbose : bool
        Print progress
    """
    # Only keep essential training fields
    essential_fields = {'E', 'F', 'R', 'Z', 'N', 'D', 'Dxyz', 'esp', 'vdw_grid', 'vdw_surface'}
    
    n_samples = len(next(iter(data.values())))
    
    for split_name, split_indices in splits.items():
        if verbose:
            print(f"\n💾 Saving {split_name} split ({len(split_indices)} samples)...")
        
        # Create split
        split_data = {}
        for key, value in data.items():
            # Skip non-essential fields
            if key not in essential_fields:
                if verbose:
                    print(f"   Skipping non-essential field: {key}")
                continue
            
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                if value.shape[0] == n_samples:
                    split_data[key] = value[split_indices]
                    if verbose:
                        print(f"   ✅ {key}: {split_data[key].shape}")
                else:
                    # Skip fields with wrong dimensions
                    if verbose:
                        print(f"   ⚠️  Skipping {key}: shape {value.shape} doesn't match n_samples={n_samples}")
            else:
                # Skip scalars
                if verbose:
                    print(f"   Skipping scalar: {key}")
        
        # Save
        output_file = output_dir / f"{prefix}_{split_name}.npz"
        np.savez_compressed(output_file, **split_data)
        
        if verbose:
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"   💾 {output_file.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Split datasets into train/valid/test with optional unit conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic split
  python -m mmml.cli.split_dataset data.npz -o splits/
  
  # Custom split ratios
  python -m mmml.cli.split_dataset data.npz -o splits/ \\
      --train 0.7 --valid 0.15 --test 0.15
  
  # With unit conversion (Hartree → eV)
  python -m mmml.cli.split_dataset data.npz -o splits/ --convert-units
  
  # Split multiple related files (EFD + ESP grids)
  python -m mmml.cli.split_dataset \\
      --efd energies_forces_dipoles.npz \\
      --grid grids_esp.npz \\
      -o training_data --convert-units
  
  # Custom seed for different splits
  python -m mmml.cli.split_dataset data.npz -o splits/ --seed 123
        """
    )
    
    # Input files
    parser.add_argument('input', type=Path, nargs='?',
                       help='Input NPZ file (single file mode)')
    parser.add_argument('--efd', type=Path,
                       help='Energy/force/dipole NPZ file (multi-file mode)')
    parser.add_argument('--grid', type=Path,
                       help='ESP grid NPZ file (multi-file mode)')
    
    # Output
    parser.add_argument('-o', '--output-dir', type=Path, required=True,
                       help='Output directory for split files')
    
    # Split ratios
    parser.add_argument('--train', type=float, default=0.8,
                       help='Training fraction (default: 0.8)')
    parser.add_argument('--valid', type=float, default=0.1,
                       help='Validation fraction (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                       help='Test fraction (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Unit conversion
    parser.add_argument('--convert-units', action='store_true',
                       help='Convert Hartree→eV and Hartree/Bohr→eV/Å')
    parser.add_argument('--no-convert-energy', action='store_true',
                       help='Skip energy conversion')
    parser.add_argument('--no-convert-forces', action='store_true',
                       help='Skip force conversion')
    
    # Options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input and not (args.efd or args.grid):
        print("❌ Error: Must specify either 'input' or --efd/--grid")
        return 1
    
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "="*70)
        print("DATASET SPLITTING")
        print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single file mode
    if args.input:
        if not args.input.exists():
            print(f"❌ Error: Input file not found: {args.input}")
            return 1
        
        if verbose:
            print(f"\n📁 Loading: {args.input}")
        
        data = dict(np.load(args.input, allow_pickle=True))
        n_samples = len(data.get('E', data.get('R', next(iter(data.values())))))
        
        if verbose:
            print(f"   Samples: {n_samples}")
            print(f"   Fields: {list(data.keys())}")
        
        # Convert units if requested
        if args.convert_units:
            data = convert_units(
                data,
                convert_energy=not args.no_convert_energy,
                convert_forces=not args.no_convert_forces,
                verbose=verbose
            )
        
        # Create splits
        if verbose:
            print(f"\n🔀 Creating splits (train: {args.train}, valid: {args.valid}, test: {args.test})...")
        
        splits = create_splits(n_samples, args.train, args.valid, args.test, args.seed)
        
        if verbose:
            print(f"   Train: {len(splits['train'])} samples")
            print(f"   Valid: {len(splits['valid'])} samples")
            print(f"   Test:  {len(splits['test'])} samples")
        
        # Save splits
        split_and_save(data, args.output_dir, splits, prefix="data", verbose=verbose)
        
        # Save split indices
        indices_file = args.output_dir / "split_indices.npz"
        np.savez(indices_file, **splits)
        if verbose:
            print(f"\n✅ Split indices saved: {indices_file.name}")
    
    # Multi-file mode (EFD + Grid)
    elif args.efd or args.grid:
        if args.efd and not args.efd.exists():
            print(f"❌ Error: EFD file not found: {args.efd}")
            return 1
        if args.grid and not args.grid.exists():
            print(f"❌ Error: Grid file not found: {args.grid}")
            return 1
        
        # Load EFD
        if args.efd:
            if verbose:
                print(f"\n📁 Loading EFD: {args.efd}")
            efd_data = dict(np.load(args.efd, allow_pickle=True))
            n_samples = len(efd_data.get('E', efd_data.get('R')))
            
            if args.convert_units:
                efd_data = convert_units(
                    efd_data,
                    convert_energy=not args.no_convert_energy,
                    convert_forces=not args.no_convert_forces,
                    verbose=verbose
                )
        
        # Load Grid
        if args.grid:
            if verbose:
                print(f"\n📁 Loading Grid: {args.grid}")
            grid_data = dict(np.load(args.grid, allow_pickle=True))
            if not args.efd:
                n_samples = len(grid_data.get('esp', grid_data.get('R')))
        
        # Create splits
        if verbose:
            print(f"\n🔀 Creating splits (train: {args.train}, valid: {args.valid}, test: {args.test})...")
        
        splits = create_splits(n_samples, args.train, args.valid, args.test, args.seed)
        
        if verbose:
            print(f"   Train: {len(splits['train'])} samples")
            print(f"   Valid: {len(splits['valid'])} samples")
            print(f"   Test:  {len(splits['test'])} samples")
        
        # Save EFD splits
        if args.efd:
            split_and_save(efd_data, args.output_dir, splits, 
                          prefix="energies_forces_dipoles", verbose=verbose)
        
        # Save Grid splits
        if args.grid:
            split_and_save(grid_data, args.output_dir, splits, 
                          prefix="grids_esp", verbose=verbose)
        
        # Save split indices
        indices_file = args.output_dir / "split_indices.npz"
        np.savez(indices_file, **splits)
        if verbose:
            print(f"\n✅ Split indices saved: {indices_file.name}")
    
    if verbose:
        print(f"\n{'='*70}")
        print("✅ SPLITTING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nOutput directory: {args.output_dir}")
        print(f"Files created:")
        for f in sorted(args.output_dir.glob("*.npz")):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


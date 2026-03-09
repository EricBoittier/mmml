#!/usr/bin/env python3
"""
CLI tool for cleaning and validating NPZ datasets.

Removes structures with:
- NaN or Inf values in energy, forces, or positions
- Abnormally large forces (SCF failures)
- Very short interatomic distances (overlapping atoms, optional)
- Other quality issues

Keeps only essential training fields:
- E, F, R, Z, N: Required for energy/force training
- D, Dxyz: Optional dipole data
- Removes: cube_*, orbital_*, metadata, and other QM-specific fields

Usage:
    # Basic cleaning (remove SCF failures only, recommended)
    python -m mmml.cli.clean_data input.npz -o cleaned.npz --no-check-distances
    
    # With geometric filtering (stricter, removes more data)
    python -m mmml.cli.clean_data input.npz -o cleaned.npz --max-force 10.0 --min-distance 0.5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


def validate_structure(
    idx: int,
    E: float,
    F: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    max_force: float = 10.0,
    min_distance: float = 0.4,
    check_distances: bool = True,
    min_energy: float = -1e6,
    max_energy: float = -1e-3,
) -> Tuple[bool, str]:
    """
    Validate a single structure.
    
    Returns
    -------
    valid : bool
        True if structure passes all checks
    reason : str
        Reason for failure if not valid
    """
    # Check for NaN/Inf in energy
    if np.isnan(E) or np.isinf(E):
        return False, f"Invalid energy: {E}"
    
    # Check for zero or suspiciously high energies (failed calculations)
    if np.abs(E) < 1e-3:
        return False, f"Zero energy: {E:.6f} (failed calculation)"
    
    if E > max_energy:
        return False, f"Energy too high: {E:.4f} eV > {max_energy} (failed calculation)"
    
    if E < min_energy:
        return False, f"Energy too low: {E:.4f} eV < {min_energy} (check units)"
    
    # Check for NaN/Inf in forces
    if np.any(np.isnan(F)) or np.any(np.isinf(F)):
        return False, "NaN or Inf in forces"
    
    # Check for NaN/Inf in positions
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        return False, "NaN or Inf in positions"
    
    # Check force magnitudes
    F_mag = np.linalg.norm(F, axis=-1)
    if np.any(F_mag > max_force):
        return False, f"Large force: max={F_mag.max():.2f} eV/Å > {max_force}"
    
    # Check interatomic distances (optional, can be slow)
    if check_distances:
        n_atoms = len(R)
        for i in range(n_atoms):
            if Z[i] == 0:  # Skip padding
                continue
            for j in range(i+1, n_atoms):
                if Z[j] == 0:  # Skip padding
                    continue
                dist = np.linalg.norm(R[i] - R[j])
                if dist < min_distance:
                    return False, f"Short distance: {dist:.3f} Å < {min_distance}"
    
    return True, "OK"


def clean_dataset(
    input_file: Path,
    output_file: Path,
    max_force: float = 10.0,
    min_distance: float = 0.4,
    check_distances: bool = True,
    min_energy: float = -1e6,
    max_energy: float = -1e-3,
    verbose: bool = True,
) -> Dict:
    """
    Clean NPZ dataset by removing invalid structures.
    
    Parameters
    ----------
    input_file : Path
        Input NPZ file
    output_file : Path
        Output NPZ file (cleaned)
    max_force : float
        Maximum allowed force magnitude (eV/Å)
    min_distance : float
        Minimum allowed interatomic distance (Å)
    check_distances : bool
        Whether to check interatomic distances (slower)
    min_energy : float
        Minimum allowed energy (eV, default: -1e6)
    max_energy : float
        Maximum allowed energy (eV, default: -1e-3, catches zeros and positive)
    verbose : bool
        Print progress
    
    Returns
    -------
    stats : dict
        Statistics about cleaning
    """
    if verbose:
        print(f"\n🧹 Cleaning dataset: {input_file}")
        print(f"   Max force threshold: {max_force} eV/Å")
        print(f"   Energy range: [{min_energy}, {max_energy}] eV")
        print(f"   Min distance threshold: {min_distance} Å")
        print(f"   Check distances: {check_distances}")
        print()
    
    # Load data
    data = np.load(input_file, allow_pickle=True)
    
    # Required keys
    required_keys = ['E', 'F', 'R', 'Z']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    E = data['E']
    F = data['F']
    R = data['R']
    Z = data['Z']
    
    n_total = len(E)
    
    # Validate each structure
    valid_indices = []
    failure_reasons = {}
    
    if verbose:
        print(f"Validating {n_total} structures...")
    
    for i in range(n_total):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{n_total} ({(i+1)/n_total*100:.1f}%)")
        
        is_valid, reason = validate_structure(
            i, E[i], F[i], R[i], Z[i],
            max_force=max_force,
            min_distance=min_distance,
            check_distances=check_distances,
            min_energy=min_energy,
            max_energy=max_energy,
        )
        
        if is_valid:
            valid_indices.append(i)
        else:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    valid_indices = np.array(valid_indices)
    n_valid = len(valid_indices)
    n_removed = n_total - n_valid
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"   Total structures: {n_total}")
        print(f"   Valid structures: {n_valid}")
        print(f"   Removed: {n_removed} ({n_removed/n_total*100:.1f}%)")
        
        if failure_reasons:
            print(f"\n   Failure breakdown:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                print(f"     • {reason}: {count}")
    
    if n_valid == 0:
        print("\n❌ Error: All structures were removed! Try relaxing thresholds.")
        sys.exit(1)
    
    # Create cleaned dataset
    if verbose:
        print(f"\n💾 Saving cleaned dataset to: {output_file}")
    
    # Only keep essential training fields (per-structure data)
    # Note: D and Dxyz are dipole fields - keep both if they exist
    essential_fields = {'E', 'F', 'R', 'Z', 'N', 'D', 'Dxyz', 'dipoles', 'dipole'}  # Core training data
    
    cleaned_data = {}
    for key in data.keys():
        # Skip non-essential fields
        if key not in essential_fields:
            if verbose:
                print(f"   Skipping non-essential field: {key}")
            continue
            
        if hasattr(data[key], 'shape') and len(data[key].shape) > 0:
            if data[key].shape[0] == n_total:
                # Filter this array
                cleaned_data[key] = data[key][valid_indices]
                if verbose:
                    print(f"   ✅ Keeping {key}: {cleaned_data[key].shape}")
            else:
                # Warn about size mismatch
                if verbose:
                    print(f"   ⚠️  Skipping {key}: shape {data[key].shape} doesn't match n_total={n_total}")
        else:
            # Skip scalars
            if verbose:
                print(f"   Skipping scalar field: {key}")
    
    if not cleaned_data:
        print("\n❌ Error: No valid fields found in dataset!")
        sys.exit(1)
    
    np.savez_compressed(output_file, **cleaned_data)
    
    if verbose:
        print(f"✅ Cleaned dataset saved!")
        print(f"   Output size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return {
        'n_total': n_total,
        'n_valid': n_valid,
        'n_removed': n_removed,
        'failure_reasons': failure_reasons,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean and validate NPZ datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cleaning (remove SCF failures and bad geometries)
  python -m mmml.cli.clean_data input.npz -o cleaned.npz
  
  # Custom thresholds
  python -m mmml.cli.clean_data input.npz -o cleaned.npz --max-force 5.0 --min-distance 0.3
  
  # Skip distance checks (faster)
  python -m mmml.cli.clean_data input.npz -o cleaned.npz --no-check-distances
  
  # Quiet mode
  python -m mmml.cli.clean_data input.npz -o cleaned.npz --quiet
        """
    )
    
    parser.add_argument('input', type=Path,
                       help='Input NPZ file')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output NPZ file (cleaned)')
    
    parser.add_argument('--max-force', type=float, default=10.0,
                       help='Maximum allowed force magnitude (eV/Å), default: 10.0')
    parser.add_argument('--max-energy', type=float, default=-1e-3,
                       help='Maximum allowed energy (eV), default: -0.001 (catches zeros)')
    parser.add_argument('--min-energy', type=float, default=-1e6,
                       help='Minimum allowed energy (eV), default: -1e6')
    parser.add_argument('--min-distance', type=float, default=0.4,
                       help='Minimum allowed interatomic distance (Å), default: 0.4')
    
    parser.add_argument('--no-check-distances', action='store_true',
                       help='Skip distance checks (faster)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.output.exists():
        print(f"⚠️  Warning: Output file already exists: {args.output}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean dataset
    stats = clean_dataset(
        args.input,
        args.output,
        max_force=args.max_force,
        min_distance=args.min_distance,
        check_distances=not args.no_check_distances,
        min_energy=args.min_energy,
        max_energy=args.max_energy,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print(f"\n✨ Done! Removed {stats['n_removed']} / {stats['n_total']} structures")


if __name__ == '__main__':
    main()


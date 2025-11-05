#!/usr/bin/env python3
"""
Convert NPZ datasets to ASE trajectory format for visualization.

Useful for:
- Visualizing training data
- Checking structures with external tools
- Creating animations
- Converting between formats

Usage:
    # Convert entire dataset
    python -m mmml.cli.convert_npz_traj data.npz -o trajectory.traj
    
    # Convert subset
    python -m mmml.cli.convert_npz_traj data.npz -o traj.traj --max-structures 100
    
    # With stride
    python -m mmml.cli.convert_npz_traj data.npz -o traj.traj --stride 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    from ase import Atoms
    from ase.io import write
    from ase.io.trajectory import Trajectory
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("❌ Error: ASE not installed")
    print("Install with: pip install ase")
    sys.exit(1)


def npz_to_trajectory(
    npz_file: Path,
    output_file: Path,
    max_structures: int = None,
    stride: int = 1,
    verbose: bool = True,
):
    """
    Convert NPZ dataset to ASE trajectory.
    
    Parameters
    ----------
    npz_file : Path
        Input NPZ file
    output_file : Path
        Output trajectory file (.traj, .xyz, etc.)
    max_structures : int, optional
        Maximum number of structures to convert
    stride : int
        Use every Nth structure
    verbose : bool
        Print progress
    """
    if verbose:
        print(f"\n🔄 Converting NPZ to trajectory...")
        print(f"   Input: {npz_file}")
        print(f"   Output: {output_file}")
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    
    # Get required fields
    if 'R' not in data or 'Z' not in data:
        raise ValueError("NPZ file must contain 'R' (positions) and 'Z' (atomic numbers)")
    
    R = data['R']
    Z = data['Z']
    
    # Optional fields
    E = data.get('E', None)
    F = data.get('F', None)
    
    n_structures = len(R)
    
    if verbose:
        print(f"   Structures in file: {n_structures}")
        print(f"   Stride: {stride}")
    
    # Apply stride and max_structures
    indices = range(0, n_structures, stride)
    if max_structures:
        indices = list(indices)[:max_structures]
    
    if verbose:
        print(f"   Converting: {len(list(indices))} structures")
        print()
    
    # Convert
    atoms_list = []
    
    for i, idx in enumerate(indices):
        if verbose and (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(list(indices))}")
        
        # Get positions and atomic numbers
        positions = R[idx]
        atomic_numbers = Z[idx]
        
        # Remove padding (Z==0)
        mask = atomic_numbers > 0
        positions = positions[mask]
        atomic_numbers = atomic_numbers[mask]
        
        # Create Atoms object
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        
        # Add energy and forces if available
        if E is not None:
            atoms.info['energy'] = float(E[idx])
        if F is not None:
            forces = F[idx][mask]
            atoms.arrays['forces'] = forces
        
        atoms_list.append(atoms)
    
    # Write trajectory
    if output_file.suffix == '.traj':
        # ASE trajectory format
        traj = Trajectory(str(output_file), 'w')
        for atoms in atoms_list:
            traj.write(atoms)
        traj.close()
    else:
        # Other formats (XYZ, PDB, etc.)
        write(str(output_file), atoms_list)
    
    if verbose:
        print(f"\n✅ Conversion complete!")
        print(f"   Output: {output_file}")
        print(f"   Structures: {len(atoms_list)}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ datasets to ASE trajectory format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire dataset
  python -m mmml.cli.convert_npz_traj data.npz -o trajectory.traj
  
  # Convert first 100 structures
  python -m mmml.cli.convert_npz_traj data.npz -o traj.traj --max-structures 100
  
  # Every 10th structure
  python -m mmml.cli.convert_npz_traj data.npz -o traj.traj --stride 10
  
  # To XYZ format
  python -m mmml.cli.convert_npz_traj data.npz -o structures.xyz
  
  # Quiet mode
  python -m mmml.cli.convert_npz_traj data.npz -o traj.traj --quiet
        """
    )
    
    parser.add_argument('input', type=Path,
                       help='Input NPZ file')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output trajectory file (.traj, .xyz, .pdb, etc.)')
    
    parser.add_argument('--max-structures', type=int, default=None,
                       help='Maximum number of structures to convert')
    parser.add_argument('--stride', type=int, default=1,
                       help='Use every Nth structure (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Validate
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Convert
    try:
        npz_to_trajectory(
            args.input,
            args.output,
            max_structures=args.max_structures,
            stride=args.stride,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


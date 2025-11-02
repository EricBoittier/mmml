#!/usr/bin/env python3
"""
Convert JAX MD NPZ trajectory to ASE trajectory format.
"""

import sys
from pathlib import Path
import numpy as np
import argparse

from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator


def convert_npz_to_traj(npz_file, output_traj, formula='CO2'):
    """Convert NPZ trajectory to ASE format."""
    
    print(f"Loading {npz_file}...")
    data = np.load(npz_file)
    
    print(f"\nData keys: {list(data.keys())}")
    
    trajectory = data['trajectory']
    energies = data['energies']
    atomic_numbers = data['atomic_numbers']
    
    n_frames = len(trajectory)
    print(f"Frames: {n_frames}")
    print(f"Atoms: {len(atomic_numbers)}")
    
    # Create trajectory
    print(f"\nWriting to {output_traj}...")
    traj = Trajectory(str(output_traj), 'w')
    
    for i in range(n_frames):
        atoms = Atoms(numbers=atomic_numbers, positions=trajectory[i])
        
        # Add energy
        calc = SinglePointCalculator(atoms, energy=float(energies[i]))
        atoms.calc = calc
        
        traj.write(atoms)
        
        if (i + 1) % 100 == 0:
            print(f"  Wrote {i+1}/{n_frames} frames...")
    
    traj.close()
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_traj}")
    print(f"   Frames: {n_frames}")
    print(f"   File size: {output_traj.stat().st_size / 1024**2:.1f} MB")
    print(f"\nView with: ase gui {output_traj}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_file', type=Path, help='Input NPZ file')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output trajectory file (default: same name with .traj)')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.npz_file.with_suffix('.traj')
    
    convert_npz_to_traj(args.npz_file, args.output)


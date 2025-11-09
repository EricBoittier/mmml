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


def convert_npz_to_traj(npz_file, output_traj, metadata_candidates=None, formula='CO2'):
    """Convert NPZ trajectory to ASE format."""
    if metadata_candidates is None:
        metadata_candidates = []
    print(f"Loading {npz_file}...")
    data = np.load(npz_file)
    metadata = None
    print(f"\nData keys: {list(data.keys())}")
    
    if 'trajectory' in data:
        trajectory = data['trajectory']
    elif 'positions' in data:
        trajectory = data['positions']
    else:
        raise KeyError("Expected 'trajectory' or 'positions' in NPZ file.")
    
    if 'energies' in data:
        energies = data['energies']
    else:
        energies = None
    
    atomic_numbers = None
    if 'atomic_numbers' in data:
        atomic_numbers = data['atomic_numbers']
    else:
        for candidate in metadata_candidates:
            candidate = Path(candidate)
            if candidate.exists():
                with np.load(candidate) as meta:
                    if 'atomic_numbers' in meta:
                        atomic_numbers = meta['atomic_numbers']
                        metadata = dict(meta)
                        print(f"  Loaded atomic numbers from {candidate}")
                        break
    if atomic_numbers is None:
        raise KeyError("Expected 'atomic_numbers' in NPZ file or metadata.")
    
    if trajectory.ndim == 4:  # (steps, replicas, atoms, 3)
        steps, replicas, atoms = trajectory.shape[0], trajectory.shape[1], trajectory.shape[2]
        print(f"Detected multi-replica trajectory: steps={steps}, replicas={replicas}, atoms={atoms}")
        trajectory = trajectory.reshape(-1, atoms, 3)
        if energies is not None:
            energies = np.broadcast_to(energies[:, None], (steps, replicas)).reshape(-1)
    else:
        print(f"Detected single trajectory: frames={trajectory.shape[0]}")
    
    n_frames = trajectory.shape[0]
    print(f"Frames: {n_frames}")
    print(f"Atoms: {len(atomic_numbers)}")
    
    # Create trajectory
    print(f"\nWriting to {output_traj}...")
    traj = Trajectory(str(output_traj), 'w')
    
    for i in range(n_frames):
        atoms = Atoms(numbers=atomic_numbers, positions=trajectory[i])
        
        if energies is not None:
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
    parser.add_argument('--metadata', type=Path, default=None,
                       help='Optional metadata NPZ containing atomic_numbers/masses')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output trajectory file (default: same name with .traj)')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.npz_file.with_suffix('.traj')
    
    metadata_candidates = []
    if args.metadata is not None:
        metadata_candidates.append(args.metadata)
    metadata_candidates.append(args.npz_file.with_name('multi_copy_metadata.npz'))
    metadata_candidates.append(args.npz_file.parent / 'multi_copy_metadata.npz')
    
    convert_npz_to_traj(args.npz_file, args.output, metadata_candidates)


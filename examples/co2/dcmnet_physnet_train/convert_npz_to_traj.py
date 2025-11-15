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


def write_ase_traj(frames: np.ndarray,
                   atomic_numbers: np.ndarray,
                   output_path: Path,
                   energies: np.ndarray | None = None):
    """Write a set of frames (n_frames, n_atoms, 3) to an ASE .traj file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(str(output_path), 'w')

    for i, positions in enumerate(frames):
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        if energies is not None:
            calc = SinglePointCalculator(atoms, energy=float(energies[i]))
            atoms.calc = calc
        traj.write(atoms)

        if (i + 1) % 100 == 0:
            print(f"    wrote {i + 1}/{len(frames)} frames to {output_path.name}")

    traj.close()
    print(f"  ✅ Saved {len(frames)} frames -> {output_path}")


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
    
    output_traj = Path(output_traj)

    if trajectory.ndim == 4:  # (steps, replicas, atoms, 3)
        steps, replicas, atoms = trajectory.shape[0], trajectory.shape[1], trajectory.shape[2]
        print(f"Detected multi-replica trajectory: steps={steps}, replicas={replicas}, atoms={atoms}")
        if energies is not None:
            energies = np.broadcast_to(energies[:, None], (steps, replicas))

        for rep in range(replicas):
            frames_rep = trajectory[:, rep, :, :]
            energies_rep = energies[:, rep] if energies is not None else None
            rep_output = output_traj.with_name(f"{output_traj.stem}_rep{rep:02d}{output_traj.suffix}")
            print(f"\nWriting replica {rep} to {rep_output} ...")
            write_ase_traj(frames_rep, atomic_numbers, rep_output, energies_rep)
    else:
        print(f"Detected single trajectory: frames={trajectory.shape[0]}")
        print(f"\nWriting to {output_traj}...")
        write_ase_traj(trajectory, atomic_numbers, output_traj, energies)
    
    print("\n✅ Conversion complete!")
    print("   View with: ase gui <output.traj>")


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


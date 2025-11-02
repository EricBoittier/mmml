#!/usr/bin/env python3
"""
Test force evaluation to check if forces are reasonable.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from trainer import JointPhysNetDCMNet
from ase.io import read
import ase.data


def test_forces(checkpoint_dir, molecule_xyz):
    """Test force evaluation at a geometry."""
    
    # Load model
    with open(checkpoint_dir / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    with open(checkpoint_dir / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    # Load molecule
    atoms = read(molecule_xyz)
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    
    print(f"Molecule: {atoms.get_chemical_formula()}")
    print(f"Positions:\n{positions}")
    print(f"Atomic numbers: {atomic_numbers}")
    
    # Build edge list
    cutoff = 10.0
    n_atoms = len(atoms)
    dst_list, src_list = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    dst_list.append(i)
                    src_list.append(j)
                    print(f"Edge: {i} → {j}, dist = {dist:.4f} Å")
    
    dst_idx = jnp.array(dst_list, dtype=jnp.int32)
    src_idx = jnp.array(src_list, dtype=jnp.int32)
    
    # Evaluate model
    batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
    batch_mask = jnp.ones(len(dst_idx), dtype=jnp.float32)
    atom_mask = jnp.ones(n_atoms, dtype=jnp.float32)
    
    output = model.apply(
        params,
        atomic_numbers=jnp.array(atomic_numbers),
        positions=jnp.array(positions),
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments=batch_segments,
        batch_size=1,
        batch_mask=batch_mask,
        atom_mask=atom_mask,
    )
    
    energy = float(output['energy'][0])
    forces = np.array(output['forces'][:n_atoms])
    
    print(f"\nEnergy: {energy:.6f} eV")
    print(f"Forces:\n{forces}")
    print(f"Force RMS: {np.sqrt(np.mean(forces**2)):.6f} eV/Å")
    print(f"Force max: {np.abs(forces).max():.6f} eV/Å")
    
    # Test numerical forces (finite difference)
    print(f"\n{'='*70}")
    print("NUMERICAL FORCES (finite difference)")
    print(f"{'='*70}")
    
    delta = 0.001  # Å
    forces_numerical = np.zeros_like(forces)
    
    for atom_idx in range(n_atoms):
        for coord_idx in range(3):
            # Forward
            pos_plus = positions.copy()
            pos_plus[atom_idx, coord_idx] += delta
            
            # Rebuild edge list
            dst_p, src_p = [], []
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j and np.linalg.norm(pos_plus[i] - pos_plus[j]) < cutoff:
                        dst_p.append(i)
                        src_p.append(j)
            
            out_plus = model.apply(
                params,
                atomic_numbers=jnp.array(atomic_numbers),
                positions=jnp.array(pos_plus),
                dst_idx=jnp.array(dst_p, dtype=jnp.int32),
                src_idx=jnp.array(src_p, dtype=jnp.int32),
                batch_segments=batch_segments,
                batch_size=1,
                batch_mask=jnp.ones(len(dst_p), dtype=jnp.float32),
                atom_mask=atom_mask,
            )
            E_plus = float(out_plus['energy'][0])
            
            # Backward
            pos_minus = positions.copy()
            pos_minus[atom_idx, coord_idx] -= delta
            
            dst_m, src_m = [], []
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j and np.linalg.norm(pos_minus[i] - pos_minus[j]) < cutoff:
                        dst_m.append(i)
                        src_m.append(j)
            
            out_minus = model.apply(
                params,
                atomic_numbers=jnp.array(atomic_numbers),
                positions=jnp.array(pos_minus),
                dst_idx=jnp.array(dst_m, dtype=jnp.int32),
                src_idx=jnp.array(src_m, dtype=jnp.int32),
                batch_segments=batch_segments,
                batch_size=1,
                batch_mask=jnp.ones(len(dst_m), dtype=jnp.float32),
                atom_mask=atom_mask,
            )
            E_minus = float(out_minus['energy'][0])
            
            # F = -dE/dr
            forces_numerical[atom_idx, coord_idx] = -(E_plus - E_minus) / (2 * delta)
    
    print(f"Numerical forces:\n{forces_numerical}")
    print(f"Analytical forces:\n{forces}")
    print(f"\nDifference:\n{forces - forces_numerical}")
    print(f"Max difference: {np.abs(forces - forces_numerical).max():.6f} eV/Å")
    
    if np.abs(forces - forces_numerical).max() > 0.01:
        print(f"\n⚠️  WARNING: Analytical and numerical forces don't match!")
        print(f"   This suggests the model's force calculation is wrong.")
    else:
        print(f"\n✅ Forces match numerical derivatives")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    args = parser.parse_args()
    
    test_forces(args.checkpoint, args.geometry)


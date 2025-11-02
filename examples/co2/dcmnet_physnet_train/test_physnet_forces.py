#!/usr/bin/env python3
"""
Test PhysNet forces directly (without DCMNet) to isolate the issue.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from mmml.physnetjax.physnetjax.models.model import EF
from ase.io import read


def test_physnet_forces(checkpoint_dir, molecule_xyz):
    """Test PhysNet force calculation."""
    
    # Load config
    with open(checkpoint_dir / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    with open(checkpoint_dir / 'best_params.pkl', 'rb') as f:
        all_params = pickle.load(f)
    
    # Extract just PhysNet params
    physnet_params = all_params['params']['physnet']
    
    # Create PhysNet model
    physnet = EF(**config['physnet_config'])
    
    # Load molecule
    atoms = read(molecule_xyz)
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atoms)
    
    print(f"Testing PURE PhysNet (no DCMNet)")
    print(f"Molecule: {atoms.get_chemical_formula()}")
    print(f"Positions:\n{positions}")
    
    # Build edge list
    cutoff = 10.0
    dst_list, src_list = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    dst_list.append(i)
                    src_list.append(j)
    
    dst_idx = jnp.array(dst_list, dtype=jnp.int32)
    src_idx = jnp.array(src_list, dtype=jnp.int32)
    batch_mask = jnp.ones(len(dst_idx), dtype=jnp.float32)
    
    # Call PhysNet directly
    output = physnet.apply(
        {'params': physnet_params},
        atomic_numbers=jnp.array(atomic_numbers),
        positions=jnp.array(positions),
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_mask=batch_mask,
    )
    
    energy = float(output['energy'])
    forces = np.array(output['forces'])
    
    print(f"\nPhysNet output:")
    print(f"Energy: {energy:.6f} eV")
    print(f"Forces shape: {forces.shape}")
    print(f"Forces:\n{forces}")
    
    # Numerical forces from PhysNet energy
    print(f"\n{'='*70}")
    print("NUMERICAL FORCES from PhysNet energy")
    print(f"{'='*70}")
    
    delta = 0.001
    forces_numerical = np.zeros((n_atoms, 3))
    
    for atom_idx in range(n_atoms):
        for coord_idx in range(3):
            # Plus
            pos_plus = positions.copy()
            pos_plus[atom_idx, coord_idx] += delta
            
            dst_p, src_p = [], []
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j and np.linalg.norm(pos_plus[i] - pos_plus[j]) < cutoff:
                        dst_p.append(i)
                        src_p.append(j)
            
            out_p = physnet.apply(
                {'params': physnet_params},
                atomic_numbers=jnp.array(atomic_numbers),
                positions=jnp.array(pos_plus),
                dst_idx=jnp.array(dst_p, dtype=jnp.int32),
                src_idx=jnp.array(src_p, dtype=jnp.int32),
                batch_mask=jnp.ones(len(dst_p), dtype=jnp.float32),
            )
            E_plus = float(out_p['energy'])
            
            # Minus
            pos_minus = positions.copy()
            pos_minus[atom_idx, coord_idx] -= delta
            
            dst_m, src_m = [], []
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j and np.linalg.norm(pos_minus[i] - pos_minus[j]) < cutoff:
                        dst_m.append(i)
                        src_m.append(j)
            
            out_m = physnet.apply(
                {'params': physnet_params},
                atomic_numbers=jnp.array(atomic_numbers),
                positions=jnp.array(pos_minus),
                dst_idx=jnp.array(dst_m, dtype=jnp.int32),
                src_idx=jnp.array(src_m, dtype=jnp.int32),
                batch_mask=jnp.ones(len(dst_m), dtype=jnp.float32),
            )
            E_minus = float(out_m['energy'])
            
            # F = -dE/dr
            forces_numerical[atom_idx, coord_idx] = -(E_plus - E_minus) / (2 * delta)
    
    print(f"Numerical forces:\n{forces_numerical}")
    print(f"Analytical forces:\n{forces}")
    print(f"\nDifference:\n{forces - forces_numerical}")
    print(f"Max difference: {np.abs(forces - forces_numerical).max():.6f} eV/Å")
    
    if np.abs(forces - forces_numerical).max() > 0.01:
        print(f"\n❌ PhysNet analytical forces don't match numerical!")
        print(f"   This is a bug in PhysNet's autodiff force calculation.")
    else:
        print(f"\n✅ PhysNet forces are correct")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    args = parser.parse_args()
    
    test_physnet_forces(args.checkpoint, args.geometry)


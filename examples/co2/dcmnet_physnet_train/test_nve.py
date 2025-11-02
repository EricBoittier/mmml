#!/usr/bin/env python3
"""
Simple NVE test to verify units and stability.

This runs pure NVE (no thermostat) to isolate integration issues.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from jax import random

from trainer import JointPhysNetDCMNet
from ase.io import read
import ase.data


def simple_nve_test(checkpoint_dir, molecule_xyz, nsteps=1000, timestep=0.05):
    """
    Simple NVE test with detailed diagnostics.
    """
    print("="*70)
    print("SIMPLE NVE TEST")
    print("="*70)
    
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
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    
    print(f"\nMolecule: {atoms.get_chemical_formula()}")
    print(f"Atoms: {len(atoms)}")
    print(f"Masses: {masses}")
    print(f"Timestep: {timestep} fs")
    print(f"Steps: {nsteps}")
    
    # Energy and force function (recomputes edge list each time!)
    cutoff = 10.0
    n_atoms = len(atoms)
    
    def compute_energy_forces(R):
        """Compute energy and forces with DYNAMIC edge list."""
        # Rebuild edge list for current positions
        dst_list, src_list = [], []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(R[i] - R[j])
                    if dist < cutoff:
                        dst_list.append(i)
                        src_list.append(j)
        
        dst_idx = jnp.array(dst_list, dtype=jnp.int32)
        src_idx = jnp.array(src_list, dtype=jnp.int32)
        
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
        batch_mask = jnp.ones(len(dst_idx), dtype=jnp.float32)
        atom_mask = jnp.ones(n_atoms, dtype=jnp.float32)
        
        output = model.apply(
            params,
            atomic_numbers=jnp.array(atomic_numbers),
            positions=jnp.array(R),
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )
        
        energy = output['energy'][0]
        forces = output['forces'][:n_atoms]
        return float(energy), np.array(forces)
    
    # Test initial edge list
    dst_test, src_test = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    dst_test.append(i)
                    src_test.append(j)
    print(f"Initial edges: {len(dst_test)}")
    
    # Initialize velocities (very small for test)
    key = random.PRNGKey(42)
    T_init = 100.0  # K
    kB = 8.617333262e-5  # eV/K
    
    # v_rms² = kB*T / (m * 0.01036427)  for each component
    sigma = np.sqrt(kB * T_init / (masses[:, None] * 0.01036427))
    velocities = sigma * np.array(random.normal(key, shape=(n_atoms, 3)))
    
    # Remove COM motion
    total_p = np.sum(masses[:, None] * velocities, axis=0)
    velocities -= total_p / masses.sum()
    
    print(f"\nInitial velocities:")
    print(f"  RMS: {np.sqrt(np.mean(velocities**2)):.6f} Å/fs")
    print(f"  Max: {np.abs(velocities).max():.6f} Å/fs")
    
    # Initial energy and forces
    E0, F0 = compute_energy_forces(positions)
    print(f"\nInitial state:")
    print(f"  Energy: {E0:.6f} eV")
    print(f"  Force RMS: {np.sqrt(np.mean(F0**2)):.6f} eV/Å")
    print(f"  Force max: {np.abs(F0).max():.6f} eV/Å")
    
    KE0 = 0.5 * np.sum(masses[:, None] * velocities**2) * 0.01036427
    T0 = 2 * KE0 / (3 * n_atoms * kB)
    print(f"  Kinetic energy: {KE0:.6f} eV")
    print(f"  Temperature: {T0:.1f} K")
    print(f"  Total energy: {E0 + KE0:.6f} eV")
    
    # Simple Velocity Verlet
    print(f"\n{'='*70}")
    print("RUNNING NVE")
    print(f"{'='*70}")
    
    R = positions.copy()
    V = velocities.copy()
    
    E_total_history = []
    T_history = []
    
    for step in range(nsteps):
        # Get forces at current position (edge list rebuilt inside!)
        E, F = compute_energy_forces(R)
        
        # VERBOSE: Print every step initially
        if step < 5:
            print(f"\n--- Step {step} ---")
            print(f"  R: {R[0]}")  # Just first atom
            print(f"  V: {V[0]}")
            print(f"  F: {F[0]}")
            print(f"  E: {E:.6f} eV")
            print(f"  F_max: {np.abs(F).max():.6f} eV/Å")
            
            # Check edge list
            n_edges = 0
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j and np.linalg.norm(R[i] - R[j]) < cutoff:
                        n_edges += 1
            print(f"  Edges: {n_edges}")
        
        # Acceleration
        # a [Å/fs²] = F [eV/Å] / (m [amu] * 0.01036427)
        a = F / (masses[:, None] * 0.01036427)
        
        if step < 5:
            print(f"  a_max: {np.abs(a).max():.6f} Å/fs²")
        
        # Velocity Verlet - Step 1: velocity half-step
        V_half = V + 0.5 * a * timestep
        
        if step < 5:
            print(f"  V_half: {V_half[0]}")
        
        # Step 2: position update
        R_new = R + V_half * timestep
        
        if step < 5:
            print(f"  R_new: {R_new[0]}")
            print(f"  ΔR: {R_new[0] - R[0]}")
        
        # Step 3: force at new position (edge list rebuilt inside!)
        E_new, F_new = compute_energy_forces(R_new)
        
        if step < 5:
            print(f"  F_new: {F_new[0]}")
            print(f"  F_new_max: {np.abs(F_new).max():.6f} eV/Å")
        
        a_new = F_new / (masses[:, None] * 0.01036427)
        
        # Step 4: velocity full-step
        V_new = V_half + 0.5 * a_new * timestep
        
        if step < 5:
            print(f"  V_new: {V_new[0]}")
        
        # Update state
        R = R_new
        V = V_new
        
        # Compute conserved quantities
        KE = 0.5 * np.sum(masses[:, None] * V**2) * 0.01036427
        E_total = E_new + KE
        T = 2 * KE / (3 * n_atoms * kB)
        
        E_total_history.append(E_total)
        T_history.append(T)
        
        if step % 100 == 0 or step < 5:
            print(f"\nStep {step:4d} | E_tot = {E_total:10.6f} eV | T = {T:6.1f} K | "
                  f"max|R| = {np.abs(R).max():6.2f} Å | max|V| = {np.abs(V).max():.4f} Å/fs | "
                  f"max|F| = {np.abs(F_new).max():.4f} eV/Å")
        
        # Safety check
        if np.abs(F_new).max() > 5.0 or np.abs(V).max() > 1.0:
            print(f"\n❌ EXPLOSION at step {step}")
            print(f"   max|F| = {np.abs(F_new).max():.4f} eV/Å")
            print(f"   max|V| = {np.abs(V).max():.4f} Å/fs")
            print(f"   max|a| = {np.abs(a_new).max():.4f} Å/fs²")
            break
    
    # Check energy conservation
    E_total_arr = np.array(E_total_history)
    E_drift = E_total_arr - E_total_arr[0]
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Energy drift: {E_drift[-1]:.6f} eV ({100*E_drift[-1]/E_total_arr[0]:.2f}%)")
    print(f"Energy std dev: {np.std(E_total_arr):.6f} eV")
    print(f"Avg temperature: {np.mean(T_history):.1f} K")
    print(f"Temp std dev: {np.std(T_history):.1f} K")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True,
                       help='Optimized geometry XYZ file')
    parser.add_argument('--nsteps', type=int, default=1000)
    parser.add_argument('--timestep', type=float, default=0.05)
    args = parser.parse_args()
    
    simple_nve_test(args.checkpoint, args.geometry, args.nsteps, args.timestep)


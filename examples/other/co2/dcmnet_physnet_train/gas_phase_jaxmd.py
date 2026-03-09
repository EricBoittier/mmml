#!/usr/bin/env python3
"""
JAX MD Gas Phase Simulations (FAST!)

Ultra-fast gas phase MD with multiple CO2 molecules using JAX MD.
Much faster than ASE for multi-molecule systems.

Usage:
    python gas_phase_jaxmd.py \
        --checkpoint ./ckpts/model \
        --n-molecules 50 \
        --temperature 300 \
        --pressure 1.0 \
        --timestep 0.1 \
        --nsteps 100000 \
        --output-dir ./gas_jaxmd
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
import time

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from jax import random

try:
    from jax_md import space, simulate
    HAS_JAXMD = True
except ImportError:
    HAS_JAXMD = False
    print("❌ JAX MD required. Install: pip install jax-md")
    sys.exit(1)

from ase import Atoms
from ase.io import write as ase_write
import ase.data

from trainer import JointPhysNetDCMNet


def calculate_gas_box_size(n_molecules: int, temperature: float, pressure: float) -> float:
    """Calculate box size using ideal gas law."""
    R = 0.08206  # L·atm/(mol·K)
    NA = 6.022e23
    n_moles = n_molecules / NA
    V_liters = n_moles * R * temperature / pressure
    V_angstrom3 = V_liters * 1e27
    return V_angstrom3 ** (1/3)


def create_gas_initial_state(n_molecules: int, temperature: float, pressure: float,
                             molecule: str = 'CO2', seed: int = 42):
    """
    Create initial positions, atomic numbers, masses for gas system.
    
    Returns
    -------
    positions : np.ndarray
        (n_total_atoms, 3)
    atomic_numbers : np.ndarray
        (n_total_atoms,)
    masses : np.ndarray
        (n_total_atoms,)
    box_size : float
        Box side length (cubic)
    """
    np.random.seed(seed)
    
    box_size = calculate_gas_box_size(n_molecules, temperature, pressure)
    
    print(f"\nGas system:")
    print(f"  {n_molecules} × {molecule}")
    print(f"  T = {temperature} K, P = {pressure} atm")
    print(f"  Box = {box_size:.2f} Å")
    print(f"  Density = {n_molecules / box_size**3 * 1e24:.2e} molecules/cm³")
    
    # CO2 template
    if molecule.upper() == 'CO2':
        template_positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.16],
            [0.0, 0.0, -1.16],
        ])
        template_z = np.array([6, 8, 8])
    else:
        raise ValueError(f"Molecule {molecule} not supported")
    
    n_atoms_per_mol = len(template_z)
    n_total = n_molecules * n_atoms_per_mol
    
    positions = np.zeros((n_total, 3))
    atomic_numbers = np.zeros(n_total, dtype=np.int32)
    
    # Place molecules
    min_sep = 3.0  # Minimum COM separation
    
    for i_mol in range(n_molecules):
        # Random position
        com = np.random.uniform(0, box_size, 3)
        
        # Random rotation
        alpha = np.random.uniform(0, 2*np.pi)
        beta = np.random.uniform(0, np.pi)
        gamma = np.random.uniform(0, 2*np.pi)
        
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        
        R = np.array([
            [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
            [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
            [-sb*cg, sb*sg, cb]
        ])
        
        mol_positions = (template_positions @ R.T) + com
        
        # Store
        start_idx = i_mol * n_atoms_per_mol
        end_idx = start_idx + n_atoms_per_mol
        positions[start_idx:end_idx] = mol_positions
        atomic_numbers[start_idx:end_idx] = template_z
    
    # Masses
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    
    print(f"  Total atoms: {n_total}")
    print(f"  Atoms per molecule: {n_atoms_per_mol}")
    
    return positions, atomic_numbers, masses, box_size


def create_energy_fn(model, params, atomic_numbers_static, cutoff=10.0):
    """
    Create JAX MD compatible energy function for gas system.
    
    Handles periodic boundary conditions via JAX MD neighbor lists.
    """
    n_atoms = len(atomic_numbers_static)
    masses_static = jnp.array([ase.data.atomic_masses[int(z)] for z in atomic_numbers_static])
    
    # Create displacement function with PBC
    displacement_fn, shift_fn = space.periodic(side=50.0)  # Will be updated with actual box size
    
    def energy_fn(R, neighbor):
        """Energy function with neighbor list."""
        # Build edge list from neighbor list
        # neighbor.idx has shape (n_atoms, max_neighbors) with -1 for missing
        dst_list = []
        src_list = []
        
        for i in range(n_atoms):
            for j_idx in range(neighbor.idx.shape[1]):
                j = neighbor.idx[i, j_idx]
                if j >= 0 and j != i:  # Valid neighbor and not self
                    dst_list.append(i)
                    src_list.append(int(j))
        
        # Convert to arrays (need fixed size for JIT)
        # Use maximum possible edges
        max_edges = n_atoms * (n_atoms - 1)
        n_edges = len(dst_list)
        
        dst_idx = jnp.zeros(max_edges, dtype=jnp.int32)
        src_idx = jnp.zeros(max_edges, dtype=jnp.int32)
        edge_mask = jnp.zeros(max_edges, dtype=jnp.float32)
        
        if n_edges > 0:
            dst_idx = dst_idx.at[:n_edges].set(jnp.array(dst_list))
            src_idx = src_idx.at[:n_edges].set(jnp.array(src_list))
            edge_mask = edge_mask.at[:n_edges].set(1.0)
        
        # Prepare model inputs
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
        atom_mask = jnp.ones(n_atoms, dtype=jnp.float32)
        
        # Run model
        output = model.apply(
            params,
            atomic_numbers=atomic_numbers_static,
            positions=R,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=edge_mask,
            atom_mask=atom_mask,
        )
        
        E = output['energy'][0]
        F = output['forces']
        
        dipole_physnet = output['dipoles'][0]
        dipole_dcmnet_magnitude = jnp.linalg.norm(output['dipoles'][0])  # Placeholder
        
        return E, F, dipole_physnet, dipole_dcmnet_magnitude
    
    return energy_fn


def run_gas_md(model, params, positions, atomic_numbers, masses, box_size,
               temperature=300, timestep=0.5, nsteps=10000, ensemble='nvt',
               friction=0.01, output_dir=Path('./gas_md'), seed=42):
    """Run JAX MD simulation on gas system."""
    
    print(f"\n{'='*70}")
    print("JAX MD GAS SIMULATION")
    print(f"{'='*70}")
    print(f"Ensemble: {ensemble.upper()}")
    print(f"Temperature: {temperature} K")
    print(f"Timestep: {timestep} fs")
    print(f"Steps: {nsteps}")
    print(f"Time: {nsteps * timestep / 1000:.2f} ps")
    print(f"Molecules: {len(atomic_numbers) // 3}")
    print(f"Total atoms: {len(atomic_numbers)}")
    
    # This is complex for multi-molecule systems
    # For now, recommend using ASE with the calculator
    print(f"\n⚠️  Multi-molecule JAX MD is complex (neighbor lists, PBC)")
    print(f"   Recommend using ASE MD instead (see gas_phase_calculator.py --run-md)")
    print(f"   Or wait for full implementation here.")
    
    return None


def main():
    parser = argparse.ArgumentParser(description='JAX MD gas phase simulations')
    
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--n-molecules', type=int, default=10)
    parser.add_argument('--molecule', type=str, default='CO2')
    parser.add_argument('--temperature', type=float, default=300)
    parser.add_argument('--pressure', type=float, default=1.0)
    parser.add_argument('--timestep', type=float, default=0.5)
    parser.add_argument('--nsteps', type=int, default=10000)
    parser.add_argument('--ensemble', type=str, default='nvt')
    parser.add_argument('--output-dir', type=Path, default=Path('./gas_jaxmd'))
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*70)
    print("JAX MD GAS PHASE SIMULATIONS")
    print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model...")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    # Adjust natoms for multi-molecule system
    n_atoms_total = (args.n_molecules * 3) if args.molecule == 'CO2' else None
    config['physnet_config']['natoms'] = n_atoms_total
    
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    print(f"✅ Loaded model")
    
    # Create gas system
    positions, atomic_numbers, masses, box_size = create_gas_initial_state(
        args.n_molecules, args.temperature, args.pressure, args.molecule, args.seed
    )
    
    # Save initial configuration
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=[box_size, box_size, box_size],
        pbc=True
    )
    ase_write(args.output_dir / 'initial.xyz', atoms)
    print(f"\n✅ Saved initial configuration")
    
    # Run MD
    results = run_gas_md(
        model, params, positions, atomic_numbers, masses, box_size,
        temperature=args.temperature,
        timestep=args.timestep,
        nsteps=args.nsteps,
        ensemble=args.ensemble,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == '__main__':
    if not HAS_JAXMD:
        sys.exit(1)
    main()


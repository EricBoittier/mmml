#!/usr/bin/env python3
"""
Gas Phase Multi-Molecule Calculator

Creates a system with multiple CO2 molecules at gas density.
Handles periodic boundary conditions and intermolecular interactions.

Usage:
    # Create gas at 1 atm, 300 K
    python gas_phase_calculator.py \
        --checkpoint ./ckpts/model \
        --n-molecules 10 \
        --temperature 300 \
        --pressure 1.0 \
        --output-dir ./gas_phase
    
    # Run MD on gas
    python gas_phase_calculator.py \
        --checkpoint ./ckpts/model \
        --n-molecules 20 \
        --temperature 300 \
        --pressure 1.0 \
        --run-md \
        --md-steps 100000 \
        --output-dir ./gas_md
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
from typing import Tuple

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from ase import Atoms
from ase.io import write as ase_write
from ase.visualize import view
import ase.data

from trainer import JointPhysNetDCMNet
from dynamics_calculator import JointPhysNetDCMNetCalculator


def calculate_gas_box_size(n_molecules: int, temperature: float, pressure: float) -> float:
    """
    Calculate box size for gas at given conditions using ideal gas law.
    
    PV = nRT
    V = nRT/P
    box_size = V^(1/3)
    
    Parameters
    ----------
    n_molecules : int
        Number of molecules
    temperature : float
        Temperature in Kelvin
    pressure : float
        Pressure in atmospheres
        
    Returns
    -------
    float
        Box side length in Angstroms
    """
    # Constants
    R = 0.08206  # L·atm/(mol·K)
    NA = 6.022e23  # Avogadro's number
    
    # Number of moles
    n_moles = n_molecules / NA
    
    # Volume in liters
    V_liters = n_moles * R * temperature / pressure
    
    # Convert to Angstroms^3 (1 L = 10^27 Å^3)
    V_angstrom3 = V_liters * 1e27
    
    # Cubic box side length
    box_size = V_angstrom3 ** (1/3)
    
    return box_size


def create_gas_system(n_molecules: int, temperature: float, pressure: float,
                     molecule: str = 'CO2', min_separation: float = 3.0,
                     seed: int = 42) -> Atoms:
    """
    Create a gas-phase system with multiple molecules.
    
    Parameters
    ----------
    n_molecules : int
        Number of molecules
    temperature : float
        Temperature (K)
    pressure : float
        Pressure (atm)
    molecule : str
        Molecule type (currently only CO2)
    min_separation : float
        Minimum intermolecular separation (Å)
    seed : int
        Random seed for placement
        
    Returns
    -------
    Atoms
        ASE Atoms object with all molecules
    """
    np.random.seed(seed)
    
    # Calculate box size
    box_size = calculate_gas_box_size(n_molecules, temperature, pressure)
    
    print(f"\n{'='*70}")
    print("GAS PHASE SYSTEM SETUP")
    print(f"{'='*70}")
    print(f"Molecules: {n_molecules} × {molecule}")
    print(f"Temperature: {temperature} K")
    print(f"Pressure: {pressure:.2f} atm")
    print(f"Box size: {box_size:.2f} Å")
    print(f"Density: {n_molecules / box_size**3 * 1e24:.2e} molecules/cm³")
    
    # Create single molecule template
    if molecule.upper() == 'CO2':
        # Equilibrium CO2 geometry
        template_positions = np.array([
            [0.0, 0.0, 0.0],      # C
            [0.0, 0.0, 1.16],     # O
            [0.0, 0.0, -1.16],    # O
        ])
        template_symbols = ['C', 'O', 'O']
    else:
        raise ValueError(f"Molecule {molecule} not supported yet")
    
    n_atoms_per_mol = len(template_symbols)
    
    # Place molecules randomly
    all_positions = []
    all_symbols = []
    
    for i_mol in range(n_molecules):
        # Random position in box
        com = np.random.uniform(0, box_size, 3)
        
        # Random rotation
        # Generate random rotation matrix using Euler angles
        alpha = np.random.uniform(0, 2*np.pi)
        beta = np.random.uniform(0, np.pi)
        gamma = np.random.uniform(0, 2*np.pi)
        
        # Rotation matrix
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        
        R = np.array([
            [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
            [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
            [-sb*cg, sb*sg, cb]
        ])
        
        # Rotate and translate molecule
        mol_positions = (template_positions @ R.T) + com
        
        # Check minimum separation (simple check: COM distance)
        if i_mol > 0:
            existing_coms = np.array([np.mean(all_positions[j*n_atoms_per_mol:(j+1)*n_atoms_per_mol], axis=0) 
                                     for j in range(i_mol)])
            
            # Apply periodic boundary conditions for distance check
            delta = com - existing_coms
            delta = delta - np.round(delta / box_size) * box_size  # Minimum image convention
            distances = np.linalg.norm(delta, axis=1)
            
            if np.any(distances < min_separation):
                # Try again with new random position (simple rejection sampling)
                # In practice, at gas density this rarely happens
                max_attempts = 100
                for attempt in range(max_attempts):
                    com = np.random.uniform(0, box_size, 3)
                    mol_positions = (template_positions @ R.T) + com
                    
                    delta = com - existing_coms
                    delta = delta - np.round(delta / box_size) * box_size
                    distances = np.linalg.norm(delta, axis=1)
                    
                    if np.all(distances >= min_separation):
                        break
        
        all_positions.extend(mol_positions)
        all_symbols.extend(template_symbols)
    
    # Create ASE Atoms object
    atoms = Atoms(
        symbols=all_symbols,
        positions=all_positions,
        cell=[box_size, box_size, box_size],
        pbc=True  # Periodic boundary conditions
    )
    
    print(f"\n✅ Created gas system:")
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Box: [{box_size:.2f}, {box_size:.2f}, {box_size:.2f}] Å")
    print(f"  PBC: {atoms.pbc}")
    
    return atoms


def main():
    parser = argparse.ArgumentParser(description='Gas phase multi-molecule calculator')
    
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Model checkpoint directory')
    parser.add_argument('--n-molecules', type=int, default=10,
                       help='Number of molecules')
    parser.add_argument('--molecule', type=str, default='CO2',
                       help='Molecule type (currently only CO2)')
    parser.add_argument('--temperature', type=float, default=300,
                       help='Temperature (K)')
    parser.add_argument('--pressure', type=float, default=1.0,
                       help='Pressure (atm)')
    parser.add_argument('--min-separation', type=float, default=3.0,
                       help='Minimum intermolecular separation (Å)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for molecule placement')
    
    # Actions
    parser.add_argument('--save-xyz', action='store_true',
                       help='Save initial configuration to XYZ')
    parser.add_argument('--run-md', action='store_true',
                       help='Run molecular dynamics')
    parser.add_argument('--run-opt', action='store_true',
                       help='Optimize geometry')
    
    # MD parameters
    parser.add_argument('--md-ensemble', type=str, default='nvt',
                       choices=['nve', 'nvt', 'npt'],
                       help='MD ensemble')
    parser.add_argument('--md-timestep', type=float, default=0.5,
                       help='MD timestep (fs)')
    parser.add_argument('--md-steps', type=int, default=10000,
                       help='Number of MD steps')
    
    parser.add_argument('--output-dir', type=Path, default=Path('./gas_phase'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    print(f"✅ Loaded {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Create gas system
    atoms = create_gas_system(
        n_molecules=args.n_molecules,
        temperature=args.temperature,
        pressure=args.pressure,
        molecule=args.molecule,
        min_separation=args.min_separation,
        seed=args.seed,
    )
    
    # Create calculator
    print(f"\n{'='*70}")
    print("SETTING UP CALCULATOR")
    print(f"{'='*70}")
    
    calculator = JointPhysNetDCMNetCalculator(
        model=model,
        params=params,
        cutoff=10.0,
    )
    
    atoms.calc = calculator
    
    # Save initial configuration
    if args.save_xyz:
        xyz_file = args.output_dir / f'gas_{args.n_molecules}mol_initial.xyz'
        ase_write(xyz_file, atoms)
        print(f"✅ Saved initial configuration: {xyz_file}")
    
    # Single point evaluation
    print(f"\n{'='*70}")
    print("SINGLE POINT EVALUATION")
    print(f"{'='*70}")
    
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    print(f"Energy: {energy:.6f} eV")
    print(f"Energy per molecule: {energy/args.n_molecules:.6f} eV")
    print(f"Force RMS: {np.sqrt(np.mean(forces**2)):.6f} eV/Å")
    print(f"Force max: {np.max(np.abs(forces)):.6f} eV/Å")
    
    # Optimize geometry
    if args.run_opt:
        print(f"\n{'='*70}")
        print("GEOMETRY OPTIMIZATION")
        print(f"{'='*70}")
        
        from ase.optimize import BFGS
        
        opt = BFGS(atoms, logfile=str(args.output_dir / 'optimization.log'))
        opt.run(fmax=0.05, steps=200)
        
        print(f"✅ Optimization converged in {opt.nsteps} steps")
        print(f"  Final energy: {atoms.get_potential_energy():.6f} eV")
        print(f"  Final max force: {np.max(np.abs(atoms.get_forces())):.6f} eV/Å")
        
        # Save optimized
        opt_file = args.output_dir / f'gas_{args.n_molecules}mol_optimized.xyz'
        ase_write(opt_file, atoms)
        print(f"✅ Saved optimized configuration: {opt_file}")
    
    # Run MD
    if args.run_md:
        print(f"\n{'='*70}")
        print("MOLECULAR DYNAMICS")
        print(f"{'='*70}")
        print(f"Ensemble: {args.md_ensemble.upper()}")
        print(f"Timestep: {args.md_timestep} fs")
        print(f"Steps: {args.md_steps}")
        print(f"Total time: {args.md_steps * args.md_timestep / 1000:.2f} ps")
        
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase.md.langevin import Langevin
        from ase.md import MDLogger
        from ase import units
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
        
        # Set up MD
        if args.md_ensemble == 'nve':
            dyn = VelocityVerlet(atoms, args.md_timestep * units.fs)
        elif args.md_ensemble == 'nvt':
            dyn = Langevin(
                atoms,
                args.md_timestep * units.fs,
                temperature_K=args.temperature,
                friction=0.01  # 1/fs
            )
        else:
            raise ValueError(f"Ensemble {args.md_ensemble} not supported yet")
        
        # Attach trajectory writer
        traj_file = args.output_dir / f'gas_{args.md_ensemble}.traj'
        from ase.io.trajectory import Trajectory
        traj = Trajectory(traj_file, 'w', atoms)
        dyn.attach(traj.write, interval=10)
        
        # Attach logger
        log_file = args.output_dir / f'gas_{args.md_ensemble}.log'
        dyn.attach(MDLogger(dyn, atoms, str(log_file), header=True, mode='w'), interval=10)
        
        # Run
        print(f"\n🚀 Running MD...")
        dyn.run(args.md_steps)
        
        traj.close()
        
        print(f"\n✅ MD complete!")
        print(f"  Trajectory: {traj_file}")
        print(f"  Log: {log_file}")
        print(f"\nView with: ase gui {traj_file}")
    
    print(f"\n{'='*70}")
    print("✅ DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()


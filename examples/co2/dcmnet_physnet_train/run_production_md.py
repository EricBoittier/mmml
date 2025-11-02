#!/usr/bin/env python3
"""
Production MD script with conservative settings for stable dynamics.

This uses the timestep that was verified to work (0.01 fs).
Automatically converts output to ASE trajectory format.

Usage:
    # 50 ps NVE simulation
    python run_production_md.py \
        --checkpoint /path/to/ckpt \
        --molecule CO2 \
        --nsteps 500000 \
        --output-dir ./production_md
    
    # 100 ps with IR analysis
    python run_production_md.py \
        --checkpoint /path/to/ckpt \
        --molecule CO2 \
        --nsteps 1000000 \
        --analyze-ir \
        --output-dir ./long_md
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

from jaxmd_dynamics import run_jaxmd_simulation, initialize_system
from trainer import JointPhysNetDCMNet
from convert_npz_to_traj import convert_npz_to_traj
from dynamics_calculator import compute_ir_from_md


def main():
    parser = argparse.ArgumentParser(description='Production MD with conservative settings')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--molecule', type=str, default='CO2',
                       help='Molecule name')
    parser.add_argument('--geometry', type=Path, default=None,
                       help='Optimized geometry XYZ (optional)')
    parser.add_argument('--output-dir', type=Path, default=Path('./production_md'),
                       help='Output directory')
    
    # MD parameters
    parser.add_argument('--ensemble', type=str, default='nve',
                       choices=['nve', 'nvt'],
                       help='MD ensemble (default: nve for testing)')
    parser.add_argument('--integrator', type=str, default='velocity-verlet',
                       choices=['velocity-verlet', 'langevin', 'berendsen'],
                       help='Integrator (default: velocity-verlet for NVE)')
    parser.add_argument('--temperature', type=float, default=300,
                       help='Temperature (K)')
    parser.add_argument('--timestep', type=float, default=0.01,
                       help='Timestep (fs) - default 0.01 for stability')
    parser.add_argument('--nsteps', type=int, default=500000,
                       help='Number of steps (default: 500k = 50 ps at 0.01 fs)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save every N steps')
    parser.add_argument('--equilibration-steps', type=int, default=0,
                       help='NVE equilibration before NVT (default: 0 for NVE)')
    
    # Analysis
    parser.add_argument('--analyze-ir', action='store_true',
                       help='Compute IR spectrum from MD')
    parser.add_argument('--convert-to-ase', action='store_true', default=True,
                       help='Convert to ASE trajectory (default: True)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PRODUCTION MD SIMULATION")
    print("="*70)
    print(f"Molecule: {args.molecule}")
    print(f"Ensemble: {args.ensemble.upper()}")
    print(f"Timestep: {args.timestep} fs (conservative for stability)")
    print(f"Total steps: {args.nsteps:,}")
    print(f"Total time: {args.nsteps * args.timestep / 1000:.1f} ps")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n1. Loading checkpoint...")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"✅ Loaded {sum(x.size for x in __import__('jax').tree_util.tree_leaves(params)):,} parameters")
    
    # Create model
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    # Initialize and optimize system
    print(f"\n2. Initializing and optimizing system...")
    
    from ase import Atoms
    from ase.io import read, write
    from ase.optimize import BFGS
    from dynamics_calculator import JointPhysNetDCMNetCalculator
    import ase.data
    
    # Create calculator for optimization
    calc = JointPhysNetDCMNetCalculator(
        model=model,
        params=params,
        cutoff=10.0,
    )
    
    # Load or create initial geometry
    if args.geometry:
        atoms = read(args.geometry)
        print(f"Loaded geometry from {args.geometry}")
    else:
        if args.molecule.upper() == 'CO2':
            atoms = Atoms('CO2', positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.16],
                [0.0, 0.0, -1.16],
            ])
        elif args.molecule.upper() == 'H2O':
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.757, 0.586],
                [0.0, -0.757, 0.586],
            ])
        else:
            raise ValueError(f"Unknown molecule: {args.molecule}")
        print(f"Using default {args.molecule} geometry")
    
    # Optimize geometry
    atoms.calc = calc
    E_init = atoms.get_potential_energy()
    F_init = atoms.get_forces()
    F_max_init = np.abs(F_init).max()
    
    print(f"\nBefore optimization:")
    print(f"  Energy: {E_init:.6f} eV")
    print(f"  Max force: {F_max_init:.6f} eV/Å")
    
    if F_max_init > 0.1:
        print(f"\n⚠️  Forces too large, optimizing geometry...")
        opt = BFGS(atoms, logfile=str(args.output_dir / 'optimization.log'))
        opt.run(fmax=0.01, steps=100)
        print(f"✅ Optimization converged in {opt.nsteps} steps")
        print(f"  Final energy: {atoms.get_potential_energy():.6f} eV")
        print(f"  Final max force: {np.abs(atoms.get_forces()).max():.6f} eV/Å")
        
        # Save optimized geometry
        write(args.output_dir / f'{args.molecule}_optimized.xyz', atoms)
    else:
        print(f"✅ Geometry already well optimized (F_max < 0.1 eV/Å)")
    
    # Extract for JAX MD
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    
    # Center in box
    box_size = 50.0
    positions = positions - positions.mean(axis=0) + box_size / 2
    box = np.array([[box_size, 0.0, 0.0],
                   [0.0, box_size, 0.0],
                   [0.0, 0.0, box_size]])
    
    # Run JAX MD
    print(f"\n3. Running JAX MD simulation...")
    results = run_jaxmd_simulation(
        model=model,
        params=params,
        positions=positions,
        atomic_numbers=atomic_numbers,
        masses=masses,
        box=box,
        ensemble=args.ensemble,
        integrator=args.integrator,
        temperature=args.temperature,
        timestep=args.timestep,
        nsteps=args.nsteps,
        save_interval=args.save_interval,
        equilibration_steps=args.equilibration_steps,
        output_dir=args.output_dir,
    )
    
    # Convert to ASE trajectory
    if args.convert_to_ase:
        print(f"\n4. Converting to ASE trajectory...")
        convert_npz_to_traj(
            args.output_dir / 'trajectory.npz',
            args.output_dir / 'trajectory.traj'
        )
    
    # Analyze IR spectrum
    if args.analyze_ir:
        print(f"\n5. Analyzing IR spectrum from MD...")
        ir_results = compute_ir_from_md(results, output_dir=args.output_dir)
    
    print(f"\n{'='*70}")
    print("✅ PRODUCTION RUN COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs in: {args.output_dir}")
    print(f"  - trajectory.npz (JAX MD native)")
    if args.convert_to_ase:
        print(f"  - trajectory.traj (ASE format)")
    print(f"  - jaxmd_{args.ensemble}_results.png")
    if args.analyze_ir:
        print(f"  - ir_spectrum_md.png")
    
    print(f"\n🎉 Trajectory ready for analysis!")


if __name__ == '__main__':
    main()


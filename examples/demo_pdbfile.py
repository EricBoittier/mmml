#!/usr/bin/env python3
"""PDB file processing and molecular dynamics simulation demo.

This demo loads PDB files and runs molecular dynamics simulations using
the hybrid MM/ML calculator with PyCHARMM integration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from demo_base import (
    load_model_parameters,
    parse_base_args,
    resolve_checkpoint_paths,
    setup_ase_imports,
    setup_mmml_imports,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDB file processing and MD simulation demo"
    )
    
    # Add base arguments
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to the acetone dataset (.npz). Defaults to $MMML_DATA or "
            "mmml/data/fixed-acetone-only_MP2_21000.npz."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory used for the ML model. Defaults to $MMML_CKPT "
            "or mmml/physnetjax/ckpts."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the configuration to evaluate (default: 0).",
    )
    parser.add_argument(
        "--n-monomers",
        type=int,
        default=2,
        help="Number of monomers in the system (default: 2).",
    )
    parser.add_argument(
        "--atoms-per-monomer",
        type=int,
        default=None,
        help=(
            "Number of atoms per monomer. Defaults to total_atoms/n_monomers "
            "derived from the dataset."
        ),
    )
    parser.add_argument(
        "--ml-cutoff",
        type=float,
        default=2.0,
        help="ML cutoff distance passed to the calculator factory (default: 2.0 Å).",
    )
    parser.add_argument(
        "--mm-switch-on",
        type=float,
        default=5.0,
        help="MM switch-on distance for the hybrid calculator (default: 5.0 Å).",
    )
    parser.add_argument(
        "--mm-cutoff",
        type=float,
        default=1.0,
        help="MM cutoff width for the hybrid calculator (default: 1.0 Å).",
    )
    parser.add_argument(
        "--include-mm",
        action="store_true",
        help="Keep MM contributions enabled when evaluating the hybrid calculator.",
    )
    parser.add_argument(
        "--skip-ml-dimers",
        action="store_true",
        help="If set, skip the ML dimer correction in the hybrid calculator.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output inside the calculator factory.",
    )
    parser.add_argument(
        "--units",
        choices=("eV", "kcal/mol"),
        default="eV",
        help=(
            "Output units for energies/forces. Use 'kcal/mol' to apply the "
            "ASE conversion factor."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a JSON report containing the comparison results.",
    )
    
    # Add specific arguments for this demo
    parser.add_argument(
        "--pdbfile",
        type=Path,
        required=True,
        help="Path to the PDB file to load for pycharmm [requires correct atom names and types].",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=210.0,
        help="Temperature for MD simulation in Kelvin (default: 300.0).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.1,
        help="Timestep for MD simulation in fs (default: 0.5).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100_000,
        help="Number of MD steps to run (default: 100000).",
    )
    parser.add_argument(
        "--minimize-first",
        action="store_true",
        default=True,
        help="Minimize structure before starting MD (default: True).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="md_simulation",
        help="Prefix for output files (default: md_simulation).",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main function for PDB file demo."""
    args = parse_args()

    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("--------------------------------")


    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)

    # Setup imports
    Atoms = setup_ase_imports()
    CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()
    
    # Additional imports for this demo
    try:
        import pycharmm
        import ase
        import ase.calculators.calculator as ase_calc
        import ase.io as ase_io
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet
        import ase.optimize as ase_opt
        import matplotlib.pyplot as plt
        import py3Dmol
        from mmml.pycharmmInterface.import_pycharmm import coor
        from mmml.pycharmmInterface.setupBox import setup_box_generic
        import pandas as pd
        from mmml.pycharmmInterface.import_pycharmm import minimize
    except ModuleNotFoundError as exc:
        sys.exit(f"Required modules not available: {exc}")

    pdbfilename = str(args.pdbfile)
    
    # Setup box and load PDB
    # setup_box_generic(pdbfilename, side_length=1000)
    from mmml.pycharmmInterface.setupBox import initialize_psf
    initialize_psf("ACO", args.n_monomers, 30, None, pdbfilename)
    pdb_ase_atoms = ase_io.read(pdbfilename)

    print(f"Loaded PDB file: {pdb_ase_atoms}")
    print(f"PyCHARMM coordinates: {coor.get_positions()}")
    print(f"PyCHARMM coordinate info: {coor.show()}")
    
    # Load model parameters
    natoms = len(pdb_ase_atoms)
    params, model = load_model_parameters(epoch_dir, natoms)
    model.natoms = natoms
    print(f"Model loaded: {model}")
    
    # Get atomic numbers and positions
    Z, R = pdb_ase_atoms.get_atomic_numbers(), pdb_ase_atoms.get_positions()
    
    print("--------------------------------")
    print(f"N atoms: {natoms}")
    print(f"N monomers: {args.n_monomers}")
    print(f"Atoms per monomer: {args.atoms_per_monomer}")
    print(f"ML cutoff: {args.ml_cutoff}")
    print(f"MM switch on: {args.mm_switch_on}")
    print(f"MM cutoff: {args.mm_cutoff}")
    print(f"Include MM: {args.include_mm}")
    print(f"Cutoff parameters: {CutoffParameters(ml_cutoff=args.ml_cutoff, mm_switch_on=args.mm_switch_on, mm_cutoff=args.mm_cutoff)}")
    print(f"Do ML: {True}")
    print(f"Do MM: {args.include_mm}")
    print(f"Do ML dimer: {not args.skip_ml_dimers}")
    print(f"Debug: {args.debug}")
    print(f"Model restart path: {base_ckpt_dir}")
    print(f"MAX_ATOMS_PER_SYSTEM: {natoms}")
    print("--------------------------------")

    # Setup calculator factory
    calculator_factory = setup_calculator(
        ATOMS_PER_MONOMER=args.atoms_per_monomer,
        N_MONOMERS=args.n_monomers,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        debug=args.debug,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=natoms,
        ml_energy_conversion_factor=1,
        ml_force_conversion_factor=1,
    )
    
    print("--------------------------------")
    print(f"Calculator factory: {calculator_factory}")


    # Create hybrid calculator
    hybrid_calc, _ = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=args.n_monomers,
        cutoff_params=CutoffParameters(
            ml_cutoff=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
        ),
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        backprop=True,
        debug=args.debug,
        energy_conversion_factor=1,
        force_conversion_factor=1,
    )
    
    print(f"Hybrid calculator created: {hybrid_calc}")
    atoms = pdb_ase_atoms
    print(f"ASE atoms: {atoms}")
    atoms.calc = hybrid_calc
    
    # Get initial energy and forces
    hybrid_energy = float(atoms.get_potential_energy())
    hybrid_forces = np.asarray(atoms.get_forces())
    print(f"Initial energy: {hybrid_energy:.6f} eV")
    
    # Minimize structure if requested
    if args.minimize_first:
        print("Minimizing structure with hybrid calculator")
        _ = ase_opt.BFGS(atoms).run(fmax=0.05, steps=100)
        
        # Sync with PyCHARMM
        xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(xyz)
        print(f"PyCHARMM coordinates after ASE minimization: {coor.show()}")
        
        # Additional PyCHARMM minimization
        minimize.run_abnr(nstep=1000, tolenr=1e-4, tolgrd=1e-4)
        pycharmm.lingo.charmm_script("ENER")
        print(f"PyCHARMM coordinates after PyCHARMM minimization: {coor.show()}")
        
        # Final ASE minimization
        atoms.set_positions(coor.get_positions())
        _ = ase_opt.BFGS(atoms).run(fmax=0.0001, steps=100)
        print("Minimization complete")

    # Setup MD simulation
    temperature = args.temperature
    timestep_fs = args.timestep
    num_steps = args.num_steps
    ase_atoms = atoms
    
    # Draw initial momenta
    MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
    Stationary(ase_atoms)  # Remove center of mass translation
    ZeroRotation(ase_atoms)  # Remove rotations

    # Initialize Velocity Verlet integrator
    integrator = VelocityVerlet(ase_atoms, timestep=timestep_fs*ase.units.fs)

    # Open trajectory file
    traj_filename = f'{args.output_prefix}_trajectory_{temperature}K_{num_steps}steps.xyz'
    traj = ase_io.Trajectory(traj_filename, 'w')

    # Run molecular dynamics
    frames = np.zeros((num_steps, len(ase_atoms), 3))
    potential_energy = np.zeros((num_steps,))
    kinetic_energy = np.zeros((num_steps,))
    total_energy = np.zeros((num_steps,))

    breakcount = 0
    for i in range(num_steps):
        # Run 1 time step
        integrator.run(1)
        
        # Save current frame and keep track of energies
        frames[i] = ase_atoms.get_positions()
        potential_energy[i] = ase_atoms.get_potential_energy()
        kinetic_energy[i] = ase_atoms.get_kinetic_energy()
        total_energy[i] = ase_atoms.get_total_energy()
        traj.write(ase_atoms)
        
        # Check for energy spikes and re-minimize if needed
        if kinetic_energy[i] > 200 or potential_energy[i] > 0:
            print(f"Energy spike detected at step {i}, re-minimizing...")
            pycharmm.lingo.charmm_script("ENER")
            xyz = pd.DataFrame(ase_atoms.get_positions(), columns=["x", "y", "z"])
            coor.set_positions(xyz)
            minimize.run_abnr(nstep=1000, tolenr=1e-5, tolgrd=1e-5)
            pycharmm.lingo.charmm_script("ENER")
            ase_atoms.set_positions(coor.get_positions())
            _ = ase_opt.BFGS(atoms).run(fmax=0.001, steps=100)
            MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
            Stationary(ase_atoms)
            ZeroRotation(ase_atoms)
            breakcount += 1
            if breakcount > 100:
                print("Maximum number of breaks reached")
                break
        
        # Occasionally print progress and adjust temperature
        if i % 10_000 == 0:
            temperature += 1
            Stationary(ase_atoms)
            ZeroRotation(ase_atoms)
            MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
            print(f"Temperature adjusted to: {temperature} K")

        if i % 100 == 0:
            print(f"step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}")

    # Plot the time series of the energy
    plt.figure(figsize=(10, 6))
    plt.plot(total_energy)
    plt.xlabel('time [fs]')
    plt.ylabel('energy [eV]')
    plt.title('Total energy')
    plt.grid(True)
    
    plot_filename = f'{args.output_prefix}_total_energy_dt_{timestep_fs}fs_{temperature}K_{num_steps}steps.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Energy plot saved to: {plot_filename}")
    
    # Close trajectory file
    traj.close()
    print(f"Trajectory saved to: {traj_filename}")
    print("MD simulation complete!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

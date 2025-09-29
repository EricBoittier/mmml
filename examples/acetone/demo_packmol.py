#!/usr/bin/env python3
"""Packmol box creation demo for MMML.

This demo creates boxes of molecules using Packmol and then processes them
with the hybrid MM/ML calculator. It demonstrates how to:
1. Create molecular boxes using Packmol
2. Convert Packmol output to ASE atoms
3. Setup and run calculations on the boxed systems
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

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
        description="Packmol box creation and processing demo"
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
        "--molecule-file",
        type=Path,
        required=True,
        help="Path to the molecule file (PDB, XYZ, etc.) to pack into boxes.",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=20.0,
        help="Size of the cubic box in Angstroms (default: 20.0).",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=10,
        help="Number of molecules to pack into the box (default: 10).",
    )
    parser.add_argument(
        "--packmol-executable",
        type=str,
        default="packmol",
        help="Path to packmol executable (default: packmol).",
    )
    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol tolerance in Angstroms (default: 2.0).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="packmol_box",
        help="Prefix for output files (default: packmol_box).",
    )
    parser.add_argument(
        "--minimize-box",
        action="store_true",
        default=True,
        help="Minimize the packed box structure (default: True).",
    )
    parser.add_argument(
        "--run-md",
        action="store_true",
        help="Run molecular dynamics on the packed box.",
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=10000,
        help="Number of MD steps if --run-md is used (default: 10000).",
    )
    
    return parser.parse_args()


def check_packmol_available(packmol_exec: str) -> bool:
    """Check if packmol executable is available."""
    try:
        result = subprocess.run([packmol_exec, "--help"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def create_packmol_input(molecule_file: Path, box_size: float, n_molecules: int, 
                        tolerance: float, output_file: Path) -> str:
    """Create Packmol input file content."""
    input_content = f"""# Packmol input file for creating molecular box
# Box size: {box_size} x {box_size} x {box_size} Angstroms
# Number of molecules: {n_molecules}
# Tolerance: {tolerance} Angstroms

tolerance {tolerance}
filetype pdb
output {output_file}

# Define the box
structure {molecule_file}
  number {n_molecules}
  inside box 0.0 0.0 0.0 {box_size} {box_size} {box_size}
end structure
"""
    return input_content


def run_packmol(packmol_exec: str, input_file: Path, timeout: int = 300) -> bool:
    """Run Packmol with the given input file."""
    try:
        result = subprocess.run(
            [packmol_exec, str(input_file)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print("Packmol completed successfully!")
            return True
        else:
            print(f"Packmol failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Packmol timed out after {timeout} seconds")
        return False
    except FileNotFoundError:
        print(f"Packmol executable not found: {packmol_exec}")
        return False


def analyze_packed_box(atoms, box_size: float) -> Dict[str, Any]:
    """Analyze the packed box structure."""
    positions = atoms.get_positions()
    
    # Calculate density
    volume = box_size ** 3
    n_atoms = len(atoms)
    density = n_atoms / volume
    
    # Calculate center of mass
    masses = atoms.get_masses()
    com = np.average(positions, axis=0, weights=masses)
    
    # Calculate distances from box center
    box_center = np.array([box_size/2, box_size/2, box_size/2])
    distances_from_center = np.linalg.norm(positions - box_center, axis=1)
    
    analysis = {
        "n_atoms": n_atoms,
        "box_size": box_size,
        "volume": volume,
        "density": density,
        "center_of_mass": com.tolist(),
        "max_distance_from_center": float(np.max(distances_from_center)),
        "min_distance_from_center": float(np.min(distances_from_center)),
        "mean_distance_from_center": float(np.mean(distances_from_center)),
    }
    
    return analysis


def main() -> int:
    """Main function for packmol demo."""
    args = parse_args()
    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)

    # Check if packmol is available
    if not check_packmol_available(args.packmol_executable):
        sys.exit(f"Packmol executable not found or not working: {args.packmol_executable}")

    # Setup imports
    Atoms = setup_ase_imports()
    CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()
    
    # Additional imports
    try:
        import ase.io as ase_io
        import ase.optimize as ase_opt
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet
        import ase
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        sys.exit(f"Required modules not available: {exc}")

    # Check if molecule file exists
    if not args.molecule_file.exists():
        sys.exit(f"Molecule file not found: {args.molecule_file}")

    print(f"Creating molecular box with:")
    print(f"  Molecule file: {args.molecule_file}")
    print(f"  Box size: {args.box_size} Å")
    print(f"  Number of molecules: {args.n_molecules}")
    print(f"  Packmol tolerance: {args.packmol_tolerance} Å")

    # Create temporary directory for packmol files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create packmol input file
        packmol_input = temp_path / "packmol_input.inp"
        packmol_output = temp_path / f"{args.output_prefix}.pdb"
        
        input_content = create_packmol_input(
            args.molecule_file, 
            args.box_size, 
            args.n_molecules, 
            args.packmol_tolerance,
            packmol_output
        )
        
        with open(packmol_input, 'w') as f:
            f.write(input_content)
        
        print(f"Packmol input file created: {packmol_input}")
        
        # Run packmol
        print("Running Packmol...")
        success = run_packmol(args.packmol_executable, packmol_input)
        
        if not success:
            sys.exit("Packmol failed to create the molecular box")
        
        # Copy output to permanent location
        final_output = Path(f"{args.output_prefix}.pdb")
        packmol_output.rename(final_output)
        print(f"Packed box saved to: {final_output}")
        
        # Load the packed box with ASE
        packed_atoms = ase_io.read(str(final_output))
        print(f"Loaded packed box: {packed_atoms}")
        
        # Analyze the packed box
        analysis = analyze_packed_box(packed_atoms, args.box_size)
        print("\nBox analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Set up periodic boundary conditions
        packed_atoms.set_cell([args.box_size, args.box_size, args.box_size])
        packed_atoms.set_pbc(True)
        
        # Load model parameters
        natoms = len(packed_atoms)
        params, model = load_model_parameters(epoch_dir, natoms)
        model.natoms = natoms
        
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
        
        # Create hybrid calculator
        Z, R = packed_atoms.get_atomic_numbers(), packed_atoms.get_positions()
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
        
        packed_atoms.calc = hybrid_calc
        
        # Get initial energy
        initial_energy = float(packed_atoms.get_potential_energy())
        print(f"\nInitial energy of packed box: {initial_energy:.6f} eV")
        
        # Minimize if requested
        if args.minimize_box:
            print("Minimizing packed box structure...")
            _ = ase_opt.BFGS(packed_atoms).run(fmax=0.01, steps=100)
            minimized_energy = float(packed_atoms.get_potential_energy())
            print(f"Minimized energy: {minimized_energy:.6f} eV")
            print(f"Energy change: {minimized_energy - initial_energy:.6f} eV")
            
            # Save minimized structure
            minimized_output = Path(f"{args.output_prefix}_minimized.pdb")
            ase_io.write(str(minimized_output), packed_atoms)
            print(f"Minimized structure saved to: {minimized_output}")
        
        # Run MD if requested
        if args.run_md:
            print(f"\nRunning molecular dynamics for {args.md_steps} steps...")
            
            # Setup MD
            temperature = 300.0
            timestep_fs = 0.5
            
            # Draw initial momenta
            MaxwellBoltzmannDistribution(packed_atoms, temperature_K=temperature)
            Stationary(packed_atoms)
            ZeroRotation(packed_atoms)
            
            # Initialize integrator
            integrator = VelocityVerlet(packed_atoms, timestep=timestep_fs*ase.units.fs)
            
            # Open trajectory file
            traj_filename = f'{args.output_prefix}_md_trajectory.xyz'
            traj = ase_io.Trajectory(traj_filename, 'w')
            
            # Run MD
            energies = []
            for i in range(args.md_steps):
                integrator.run(1)
                traj.write(packed_atoms)
                
                if i % 100 == 0:
                    epot = packed_atoms.get_potential_energy()
                    ekin = packed_atoms.get_kinetic_energy()
                    etot = packed_atoms.get_total_energy()
                    energies.append(etot)
                    print(f"step {i:5d} epot {epot: 5.3f} ekin {ekin: 5.3f} etot {etot: 5.3f}")
            
            traj.close()
            print(f"MD trajectory saved to: {traj_filename}")
            
            # Plot energy evolution
            if energies:
                plt.figure(figsize=(10, 6))
                plt.plot(energies)
                plt.xlabel('MD Step')
                plt.ylabel('Total Energy [eV]')
                plt.title('Energy Evolution During MD')
                plt.grid(True)
                
                plot_filename = f'{args.output_prefix}_md_energy.png'
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"Energy plot saved to: {plot_filename}")
        
        print("\nPackmol box creation and processing complete!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

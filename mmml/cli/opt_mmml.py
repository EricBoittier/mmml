"""
Finds cut-offs and MM non-bonded parameters to better fit the QM data.
"""


#!/usr/bin/env python3
from __future__ import annotations
"""...
"""
import jax
# jax.config.update("jax_enable_x64", True)


import argparse
import sys
from pathlib import Path

import numpy as np

from mmml.cli.base import (
    load_model_parameters,
    resolve_checkpoint_paths,
    setup_ase_imports,
    setup_mmml_imports,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDB file processing and MD simulation demo"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset file to load for pycharmm",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the force field (default: False).",
    )

    # ========================================================================
    # MD SIMULATION ARGUMENTS
    # ========================================================================
    parser.add_argument(
        "--energy-catch",
        type=float,
        default=0.5,
        help="Energy catch factor for the simulation (default: 0.05).",
    )

    # parser.add_argument(
    #     "--cell",
    #     default=None,
    #     help="Use cell for the simulation (default: False) as a float for a cubic cell length (Å).",
    # )

    parser.add_argument(
        "--n-monomers",
        type=int,
        required=True,
        help="Number of monomers in the system (default: 2).",
    )
    parser.add_argument(
        "--n-atoms-monomer",
        type=int,
        required=True,
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


    return parser.parse_args()


def main() -> int:
    """Main function for PDB file demo."""
    args = parse_args()
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
        import jax_md
        # JAX-MD imports
        from jax_md import space, smap, energy, quantity, simulate, partition, units
        from ase.units import _amu

        import jax.numpy as jnp

        import jax, e3x
        from jax import jit, grad, lax, ops, random
        import jax.numpy as jnp
        from ase.io import Trajectory
    except ModuleNotFoundError as exc:
        sys.exit(f"Required modules not available: {exc}")

    pdbfilename = str(args.pdbfile)
    
    # Setup box and load PDB
    setup_box_generic(pdbfilename, side_length=1000)
    pdb_ase_atoms = ase_io.read(pdbfilename)

    print(f"Loaded PDB file: {pdb_ase_atoms}")
    
    # ========================================================================
    # MASS SETUP FOR JAX-MD SIMULATION
    # ========================================================================
    # JAX-MD requires proper mass arrays for temperature calculation and dynamics
    
    # Get atomic masses from ASE (in atomic mass units)
    raw_masses = pdb_ase_atoms.get_masses()
    print(f"Raw masses from ASE: {raw_masses}")
    Si_mass = jnp.array(raw_masses)  # Use ASE masses directly (in amu)
    Si_mass_sum = Si_mass.sum()
    print(f"Si_mass (ASE masses in amu): {Si_mass}")
    print(f"Si_mass sum: {Si_mass_sum}")
    
    # Expand mass array to match momentum dimensions for JAX-MD broadcasting
    # Momentum has shape (n_atoms, 3), so mass must also have shape (n_atoms, 3)
    Si_mass_expanded = jnp.repeat(Si_mass[:, None], 3, axis=1)  # Shape: (20, 3)
    print(f"Si_mass_expanded shape: {Si_mass_expanded.shape}")
    print(f"Si_mass_expanded sample: {Si_mass_expanded[0]}")


    print(f"PyCHARMM coordinates: {coor.get_positions()}")
    print(f"Ase coordinates: {pdb_ase_atoms.get_positions()}")
    print(f"{coor.get_positions() == pdb_ase_atoms.get_positions()}")

    print(coor.get_positions())
    
    # Load model parameters
    natoms = len(pdb_ase_atoms)
    n_monomers = args.n_monomers
    n_atoms_monomer = args.n_atoms_monomer
    assert n_atoms_monomer * n_monomers == natoms, "n_atoms_monomer * n_monomers != natoms"
    params, model = load_model_parameters(epoch_dir, natoms)
    model.natoms = natoms
    print(f"Model loaded: {model}")
    
    # Get atomic numbers and positions
    Z, R = pdb_ase_atoms.get_atomic_numbers(), pdb_ase_atoms.get_positions()
    
    # Setup calculator factory
    calculator_factory = setup_calculator(
        ATOMS_PER_MONOMER=args.n_atoms_monomer,
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
        cell=args.cell,
    )
    

    CUTOFF_PARAMS = CutoffParameters(
            ml_cutoff=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
        )

    print(f"Cutoff parameters: {CUTOFF_PARAMS}")

    # Create hybrid calculator
    hybrid_calc, _ = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=args.n_monomers,
        cutoff_params=CUTOFF_PARAMS,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        backprop=True,
        debug=args.debug,
        energy_conversion_factor=1,
        force_conversion_factor=1,
        # do_pbc_map=args.cell is not None,
        # pbc_map=calculator_factory.pbc_map if hasattr(calculator_factory, 'pbc_map') else None,
    )
 
    print(f"Hybrid calculator created: {hybrid_calc}")
    atoms = pdb_ase_atoms

    
    if args.cell is not None:
        print("Setting cell")
        from ase.cell import Cell
        print("Creating cell")
        cell = Cell.fromcellpar([float(args.cell), float(args.cell), float(args.cell), 90., 90., 90.])
        atoms.set_cell(cell)
        # Enable periodic boundary conditions
        atoms.set_pbc(True)
        print(f"Cell: {cell}")
        print(f"PBC enabled: {atoms.pbc}")
        print(f"Cell shape: {cell.shape}")
        print(f"Cell type: {type(cell)}")
        print(f"Cell dtype: {cell.dtype}")
        print(f"Cell size: {cell.size}")
        print(f"Cell dtype: {cell.dtype}")
        print(f"Cell ndim: {cell.ndim}")
        print(f"Cell dtype: {cell.dtype}")
    else:
        cell = None
        print("No cell provided")

    print(f"ASE atoms: {atoms}")
    atoms.calc = hybrid_calc
    # Get initial energy and forces
    hybrid_energy = float(atoms.get_potential_energy())
    hybrid_forces = np.asarray(atoms.get_forces())
    print(f"Initial energy: {hybrid_energy:.6f} eV")
    print(f"Initial forces: {hybrid_forces}")


    # load dataset
    dataset = np.load(args.dataset)
    print(f"Dataset: {dataset}")
    print(f"Dataset keys: {dataset.keys()}")
    print(f"Dataset shape: {dataset['R'].shape}")
    print(f"Dataset dtype: {dataset['R'].dtype}")
    print(f"Dataset ndim: {dataset['R'].ndim}")
    print(f"Dataset size: {dataset['R'].size}")
    
    


if __name__ == "__main__":
    raise SystemExit(main())

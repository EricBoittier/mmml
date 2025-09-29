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
import itertools
import json
import time

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
        "--pdbfile",
        type=Path,
        required=True,
        help="Path to the PDB file to load for pycharmm [requires correct atom names and types].",
    )


    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    # Optimization controls
    parser.add_argument(
        "--ml-cutoff-grid",
        type=str,
        default="1.5,2.0,2.5,3.0",
        help="Comma-separated ML cutoff grid in Å (e.g., '1.5,2.0,2.5').",
    )
    parser.add_argument(
        "--mm-switch-on-grid",
        type=str,
        default="4.0,5.0,6.0,7.0",
        help="Comma-separated MM switch-on grid in Å (e.g., '4.0,5.0,6.0').",
    )
    parser.add_argument(
        "--mm-cutoff-grid",
        type=str,
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated MM cutoff width grid in Å (e.g., '0.5,1.0,1.5').",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=1.0,
        help="Weight for energy MSE term in the objective.",
    )
    parser.add_argument(
        "--force-weight",
        type=float,
        default=1.0,
        help="Weight for force MSE term in the objective.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum number of frames from dataset to evaluate (for speed). Use -1 for all.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save best parameters and scores as JSON.",
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

    parser.add_argument(
        "--cell",
        default=None,
        help="Use cell for the simulation (default: False) as a float for a cubic cell length (Å).",
    )

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
    print("Note: for testing the dimer calculator, the pdb file should contain a dimer, and" \
     "the atom types should be consistent with the dimer calculator.")
    
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
    R_all = dataset["R"]  # (n_frames, natoms, 3)
    Z_ds = dataset.get("Z", Z)
    if Z_ds.ndim > 1:
        Z_ds = np.array(Z_ds[0]).astype(int)
    print(f"R shape: {R_all.shape}")
    E_all = dataset.get("E", None)
    F_all = dataset.get("F", None)
    has_E = E_all is not None and np.size(E_all) > 0
    has_F = F_all is not None and np.size(F_all) > 0
    n_frames = R_all.shape[0]
    if args.max_frames is not None and args.max_frames > 0:
        n_eval = min(n_frames, args.max_frames)
    elif args.max_frames == -1:
        n_eval = n_frames
    else:
        n_eval = n_frames
    # arrange frames by center of mass distances between the two monomers
    com_distances = []
    count_non_dimer = 0
    for i in range(len(R_all)):
        # Calculate COM for each monomer
        com1 = R_all[i][:args.n_atoms_monomer].mean(axis=0)  # First monomer
        com2 = R_all[i][args.n_atoms_monomer:].mean(axis=0)  # Second monomer
        # Distance between monomer COMs
        if dataset["N"][i] != args.n_atoms_monomer*2:
            count_non_dimer += 1
        com_distances.append(np.linalg.norm(com1 - com2))
    com_distances = np.array(com_distances)
    frame_indices = np.argsort(com_distances)[:-count_non_dimer][::(len(com_distances)-count_non_dimer)//n_eval]
    print(f"Evaluating {n_eval} frames (out of {n_frames}). E available: {has_E}, F available: {has_F}")

    # Utility to parse grids
    def _parse_grid(s: str) -> list[float]:
        return [float(x) for x in s.split(",") if x.strip() != ""]

    ml_grid = _parse_grid(args.ml_cutoff_grid)
    mm_on_grid = _parse_grid(args.mm_switch_on_grid)
    mm_cut_grid = _parse_grid(args.mm_cutoff_grid)
    print(f"Grid sizes -> ml:{len(ml_grid)} mm_on:{len(mm_on_grid)} mm_cut:{len(mm_cut_grid)}")

    # Objective evaluation for a given cutoff triple
    def evaluate_objective(ml_cutoff: float, mm_switch_on: float, mm_cutoff: float) -> dict:
        nonlocal atoms
        local_params = CutoffParameters(
            ml_cutoff=ml_cutoff,
            mm_switch_on=mm_switch_on,
            mm_cutoff=mm_cutoff,
        )
        # Rebuild calculator with new cutoffs
        calc, _ = calculator_factory(
            atomic_numbers=Z_ds,
            atomic_positions=R_all[0],
            n_monomers=args.n_monomers,
            cutoff_params=local_params,
            doML=True,
            doMM=args.include_mm,
            doML_dimer=not args.skip_ml_dimers,
            backprop=True,
            debug=args.debug,
            energy_conversion_factor=1,
            force_conversion_factor=1,
        )
        atoms.calc = calc
        se_e = 0.0
        se_f = 0.0
        n_e = 0
        n_f = 0
        for i in frame_indices:
            atoms.positions = R_all[i]
            pred_E = float(atoms.get_potential_energy())
            pred_F = np.asarray(atoms.get_forces())
            if has_E:
                ref_E = float(E_all[i])
                se_e += (pred_E - ref_E) ** 2
                n_e += 1
            if has_F:
                ref_F = np.asarray(F_all[i])
                se_f += float(np.mean((pred_F - ref_F) ** 2))
                n_f += 1
        mse_e = (se_e / max(n_e, 1)) if has_E else 0.0
        mse_f = (se_f / max(n_f, 1)) if has_F else 0.0
        obj = args.energy_weight * mse_e + args.force_weight * mse_f
        return {
            "ml_cutoff": ml_cutoff,
            "mm_switch_on": mm_switch_on,
            "mm_cutoff": mm_cutoff,
            "mse_energy": mse_e,
            "mse_forces": mse_f,
            "objective": obj,
        }

    # Grid search
    start = time.time()
    best = None
    results = []
    for ml_c, mm_on, mm_c in itertools.product(ml_grid, mm_on_grid, mm_cut_grid):
        res = evaluate_objective(ml_c, mm_on, mm_c)
        results.append(res)
        if best is None or res["objective"] < best["objective"]:
            best = res
        print(
            f"ml={ml_c:.3f} mm_on={mm_on:.3f} mm_cut={mm_c:.3f} -> obj={res['objective']:.6e} (E={res['mse_energy']:.6e}, F={res['mse_forces']:.6e})"
        )
    elapsed = time.time() - start
    print(f"Grid search completed in {elapsed:.1f}s over {len(results)} combos.")
    print(f"Best: {best}")

    if args.out is not None:
        payload = {
            "best": best,
            "results": results,
            "n_eval_frames": int(n_eval),
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.out}")
    
    


if __name__ == "__main__":
    raise SystemExit(main())



# python -m mmml.cli.opt_mmml \
#   --dataset /path/to/data.npz \
#   --checkpoint /path/to/checkpoint \
#   --n-monomers 2 \
#   --n-atoms-monomer 10 \
#   --ml-cutoff-grid 1.5,2.0,2.5 \
#   --mm-switch-on-grid 4.0,5.0,6.0 \
#   --mm-cutoff-grid 0.5,1.0,1.5 \
#   --energy-weight 1.0 \
#   --force-weight 1.0 \
#   --max-frames 200 \
#   --out /tmp/cutoff_opt.json
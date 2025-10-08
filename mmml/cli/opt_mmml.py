"""
Finds cut-offs and MM non-bonded parameters to better fit the QM data.
"""


#!/usr/bin/env python3
from __future__ import annotations
"""...
"""
import jax
# jax.config.update("jax_enable_x64", True)

from functools import partial
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
        "--out-npz",
        type=Path,
        default=None,
        help="Optional path to save detailed results as NPZ file.",
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
        # default=2.0,
        help="ML cutoff distance passed to the calculator factory (default: 2.0 Å).",
    )
    parser.add_argument(
        "--mm-switch-on",
        type=float,
        # default=5.0,
        help="MM switch-on distance for the hybrid calculator (default: 5.0 Å).",
    )
    parser.add_argument(
        "--mm-cutoff",
        type=float,
        # default=1.0,
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

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    
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
        verbose=True,
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
    CUTOFF_PARAMS.plot_cutoff_parameters(Path.cwd())

    # Create hybrid calculator
    hybrid_calc, _ = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=args.n_monomers,
        # cutoff_params=CUTOFF_PARAMS,
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
    # --------------------------------------------------------------------
    # Visualize cutoff meanings (schematic)
    # --------------------------------------------------------------------
    try:
        _save_dir = args.out.parent if (args.out is not None) else (args.out_npz.parent if (args.out_npz is not None) else Path.cwd())
        CUTOFF_PARAMS.plot_cutoff_parameters(_save_dir)
        
    except Exception as _plot_exc:
        print(f"Warning: could not render cutoff schematic: {_plot_exc}")
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
        # else:
        #     print(f"Frame {i} is a dimer", f"com1: {com1}", f"com2: {com2}", dataset["Z"][i])
        com_distances.append(np.linalg.norm(com1 - com2))

        # center all atoms to the origin
        R_all[i] = R_all[i] - R_all[i].mean(axis=0)
        # # rotate all atoms to have the first two principal axes in the xy plane
        # R_all[i] = rotate_to_principal_axes(R_all[i])
        # if we rotate we need to rotate the forces... seems daft


    com_distances = np.array(com_distances)
    # sort by com_distances and then remove the non-dimer frames
    com_distances = np.array(com_distances)
    # sort by com_distances and then remove the non-dimer frames
    valid_mask = (dataset["N"] == args.n_atoms_monomer * 2)
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        raise RuntimeError("No valid dimer frames found in dataset.")
    sorted_valid = valid_idx[np.argsort(com_distances[valid_idx])]
    n_valid = len(sorted_valid)
    # ensure stride >= 1 and do not overshoot n_valid
    stride = max(1, n_valid // max(1, n_eval))
    frame_indices = sorted_valid[::stride][:n_eval]
    print(f"Evaluating {len(frame_indices)} frames (out of {n_frames}). E available: {has_E}, F available: {has_F}")



    # sometimes the atoms are in different orders (CHARMM will be based on the topology, the pdb has to match this)
    # the data usually comes from quantum chemistry calculations, so we need to sort the atoms to match the pdb
    if not np.all(Z == Z_ds):
        print("Z does not match Z_ds")
        print(f"Z: {Z}")
        print(f"Z_ds: {Z_ds}")
        # sort R, Z, and F to match the pdb... the mapping has to be known aprioi
        # for the test case:
        mapping = np.array([3, 1, 2, 0, 4, 5, 6, 7, 8, 9, 13, 11, 12, 10, 14, 15, 16, 17, 18, 19])
        print(f"Mapping: {mapping}")
        Z_ds = Z_ds[mapping]
        print("R_all before mapping: ", R_all.shape)
        R_all = np.array([R_all[i][mapping] for i, z in enumerate(R_all)])
        if has_F:
            F_all = np.array([F_all[i][mapping] for i, z in enumerate(F_all)])
        print(f"Z_ds: {Z_ds}")
        print(f"R_all: {R_all.shape}")
        print(f"F_all: {F_all.shape}")



    # --------------------------------------------------------------------
    # Diagnostics: energy scatter (pred vs ref) and energy vs COM distance
    # for the current calculator settings, before grid search.
    # --------------------------------------------------------------------
    try:
        if has_E:
            r_sel = com_distances[frame_indices]
            pred_E_list = []
            ref_E_list = []
            for i in frame_indices:
                # atoms.positions = R_all[i]
                atoms.set_positions(R_all[i])
                _pot = atoms.get_potential_energy()
                pred_E_list.append(float(_pot))
                import jax
                jax.debug.print("{x}", x=atoms.calc.results["out"])
                ref_E_list.append(float(E_all[i]))
            pred_E_arr = np.array(pred_E_list)
            ref_E_arr = np.array(ref_E_list)

            # Determine save dir
            _save_dir = args.out.parent if (args.out is not None) else (args.out_npz.parent if (args.out_npz is not None) else Path.cwd())
            _save_dir.mkdir(parents=True, exist_ok=True)

            # 1) Build per-frame diagnostics DataFrame (energies, force errors)
            rows = []
            for i, pred_e, ref_e in zip(frame_indices, pred_E_arr, ref_E_arr):
                if has_F:
                    atoms.set_positions(R_all[i])
                    pred_F_i = np.asarray(atoms.get_forces())
                    ref_F_i = np.asarray(F_all[i])
                    dF = pred_F_i - ref_F_i
                    mse_f_i = float(np.mean(dF**2))
                    max_f_err_i = float(np.abs(dF).max())
                else:
                    mse_f_i = float('nan')
                    max_f_err_i = float('nan')
                rows.append({
                    "frame_index": int(i),
                    "com_dist": float(com_distances[i]),
                    "pred_e": float(pred_e),
                    "ref_e": float(ref_e),
                    "mse_f": mse_f_i,
                    "max_f_err": max_f_err_i,
                })
            df_diag = pd.DataFrame(rows).sort_values(by=["com_dist", "frame_index"]).reset_index(drop=True)
            out_csv = _save_dir / f"diagnostics_{args.ml_cutoff:.2f}_{args.mm_switch_on:.2f}_{args.mm_cutoff:.2f}.csv"
            df_diag.to_csv(out_csv, index=False)
            print(f"Saved diagnostics CSV to {out_csv}")

            # 2) Scatter: predicted vs reference energy
            plt.figure(figsize=(5,5))
            plt.scatter(ref_E_arr, pred_E_arr, s=12, alpha=0.8)
            lim_min = float(min(ref_E_arr.min(), pred_E_arr.min()))
            lim_max = float(max(ref_E_arr.max(), pred_E_arr.max()))
            plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1)
            plt.xlabel("Reference energy (E)")
            plt.ylabel("Predicted energy (E)")
            plt.title(
                f"Energy: predicted vs reference | ml={args.ml_cutoff:.2f}, mm_on={args.mm_switch_on:.2f}, mm_cut={args.mm_cutoff:.2f}"
            )
            plt.tight_layout()
            out_scatter = _save_dir / f"energy_scatter_{args.ml_cutoff:.2f}_{args.mm_switch_on:.2f}_{args.mm_cutoff:.2f}.png"
            plt.savefig(out_scatter, dpi=150)
            try:
                plt.show()
            except Exception:
                pass
            print(f"Saved energy scatter to {out_scatter}")

            # 3) Energy vs COM distance
            order = np.argsort(r_sel)
            r_sorted = r_sel[order]
            pred_sorted = pred_E_arr[order]
            ref_sorted = ref_E_arr[order]
            plt.figure(figsize=(6,4))
            plt.plot(r_sorted, ref_sorted, label="Reference", lw=1.5)
            plt.plot(r_sorted, pred_sorted, label="Predicted", lw=1.5)
            plt.xlabel("COM distance (Å)")
            plt.ylabel("Energy (E)")
            plt.title(
                f"Energy vs COM distance | ml={args.ml_cutoff:.2f}, mm_on={args.mm_switch_on:.2f}, mm_cut={args.mm_cutoff:.2f}"
            )
            plt.legend()
            plt.tight_layout()
            out_curve = _save_dir / f"energy_vs_r_{args.ml_cutoff:.2f}_{args.mm_switch_on:.2f}_{args.mm_cutoff:.2f}.png"
            plt.savefig(out_curve, dpi=150)
            try:
                plt.show()
            except Exception:
                pass
            print(f"Saved energy vs r to {out_curve}")
    except Exception as _diag_exc:
        print(f"Warning: diagnostics plotting failed: {_diag_exc}")

    # --------------------------------------------------------------------
    # Write reference trajectory with ground truth energies and forces
    # --------------------------------------------------------------------
    try:
        if has_E:
            from ase.io import Trajectory
            from ase.calculators.singlepoint import SinglePointCalculator
            ref_traj_path = _save_dir / f"reference_trajectory_{args.ml_cutoff:.2f}_{args.mm_switch_on:.2f}_{args.mm_cutoff:.2f}.traj"
            
           
            with Trajectory(ref_traj_path, 'w') as traj:
                for idx, i in enumerate(frame_indices):
                    # Create atoms object with reference positions
                    ref_atoms = Atoms(
                        numbers=Z_ds,
                        positions=R_all[i],
                        cell=atoms.cell if atoms.cell is not None else None,
                        pbc=atoms.pbc if hasattr(atoms, 'pbc') else False
                    )
                    # Attach reference energy and forces via calculator
                    ref_E = float(E_all[i])
                    ref_F = F_all[i] if has_F else None
                    ref_atoms.calc = SinglePointCalculator(ref_atoms, energy=ref_E, forces=ref_F)                   
                    # Write to trajectory
                    traj.write(ref_atoms)
            
            print(f"Saved reference trajectory ({len(frame_indices)} frames) to {ref_traj_path}")
            print(f"  - Contains ground truth energies and forces from dataset")
            print(f"  - Can be visualized with ASE GUI: ase gui {ref_traj_path}")
    except Exception as _traj_exc:
        print(f"Warning: reference trajectory writing failed: {_traj_exc}")



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
        traj = Trajectory(f"traj_{ml_cutoff:.2f}_{mm_switch_on:.2f}_{mm_cutoff:.2f}.traj",  'w', atoms=atoms)
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
            atoms.set_positions(R_all[i])
            pred_E = float(atoms.get_potential_energy())
            pred_F = np.asarray(atoms.get_forces())
            traj.write(atoms)
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
        _out_dict = {
            "ml_cutoff": ml_cutoff,
            "mm_switch_on": mm_switch_on,
            "mm_cutoff": mm_cutoff,
            "mse_energy": mse_e,
            "mse_forces": mse_f,
            "objective": obj,
        }
        print(f"Objective: {_out_dict}")
        return _out_dict

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
    
    if args.out_npz is not None:
        # Save detailed results as NPZ
        npz_data = {
            "ml_cutoffs": np.array([r["ml_cutoff"] for r in results]),
            "mm_switch_ons": np.array([r["mm_switch_on"] for r in results]),
            "mm_cutoffs": np.array([r["mm_cutoff"] for r in results]),
            "mse_energies": np.array([r["mse_energy"] for r in results]),
            "mse_forces": np.array([r["mse_forces"] for r in results]),
            "objectives": np.array([r["objective"] for r in results]),
            "best_ml_cutoff": best["ml_cutoff"],
            "best_mm_switch_on": best["mm_switch_on"],
            "best_mm_cutoff": best["mm_cutoff"],
            "best_mse_energy": best["mse_energy"],
            "best_mse_forces": best["mse_forces"],
            "best_objective": best["objective"],
            "n_eval_frames": n_eval,
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
        }
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_npz, **npz_data)
        print(f"Saved detailed results to {args.out_npz}")
    
    


if __name__ == "__main__":
    raise SystemExit(main())



# python -m mmml.cli.opt_mmml \
#   --dataset /path/to/data.npz \
#   --checkpoint /path/to/checkpoint \
#   --n-monomers 2 \
#   --pdbfile /path/to/pdb \
#   --include-mm \
#   --n-atoms-monomer 10 \
#   --ml-cutoff-grid 1.5,2.0,2.5 \
#   --mm-switch-on-grid 4.0,5.0,6.0 \
#   --mm-cutoff-grid 0.5,1.0,1.5 \
#   --energy-weight 1.0 \
#   --force-weight 1.0 \
#   --max-frames 200 \
#   --out /tmp/cutoff_opt.json
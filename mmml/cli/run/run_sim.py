"""
Runs an MD simulation.

Requires the box size and system (as a pdb file) to be specified.

Performance tips:
    - Set JAX_COMPILATION_CACHE_DIR for persistent JAX JIT cache (subsequent runs reuse
      compiled code; first run may take minutes to compile).
    - Use --skip-setup-energy-show to avoid slow CHARMM energy.show() during setup
      (less validation of the initial structure).
Temperature, pressure, and density can be specified.
If no solvent is specified, the system will be vacuum.
If no temperature is specified, the system will be at room temperature.
If no pressure is specified, the system will be at atmospheric pressure.
If no density is specified, the system will be at the density of the box.
If no side length is specified, the system will be at the side length of the box.
If no residue is specified, the system will be at the residue of the box.

Args:
    --pdbfile
    simulation conditions

Run from a notebook:
    from pathlib import Path
    import argparse
    from mmml.cli.run_sim import run

    args = argparse.Namespace(
        pdbfile=Path("pdb/init-packmol.pdb"),
        checkpoint=Path("ACO-..."),
        n_monomers=50,
        n_atoms_monomer=10,
        cell=40.0,
        # optional: temperature=200.0, timestep=0.3, ensemble="nve",
        # ml_batch_size=512,  # chunk ML batches to reduce GPU memory
    )
    run(args)
"""


#!/usr/bin/env python3
from __future__ import annotations
"""PDB file processing and molecular dynamics simulation demo.

This demo loads PDB files and runs molecular dynamics simulations using
the hybrid MM/ML calculator with PyCHARMM integration.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from mmml.cli.base import (
    load_model_parameters,
    resolve_checkpoint_paths,
    setup_ase_imports,
    setup_mmml_imports,
)
from mmml.cli.run.shared import run_sim_loop, save_trajectory
from mmml.cli.run.summaries import (
    print_charges_summary,
    print_forces_summary,
    print_masses_summary,
    print_positions_summary,
    print_system_summary,
)
from mmml.cli.run.utils import get_steps_per_frame, normalize_n_atoms_monomer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDB file processing and MD simulation demo"
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
        type=float,
        default=None,
        help="Cubic cell side length in Å for periodic boundary conditions (default: None = no PBC).",
    )
    parser.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="Å",
        help="Radius of flat bottom potential to constrain system COM to center (default: None = disabled). "
        "When set, V=0 for |d|<=R and V=k*(|d|-R)^2 for |d|>R. With --cell, center=box center; else origin.",
    )
    parser.add_argument(
        "--flat-bottom-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Force constant for flat bottom potential when outside radius (default: 1.0).",
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
        "--no-complementary-handoff",
        action="store_true",
        help="Use legacy MM switching (MM only in [mm_switch_on, mm_switch_on+mm_cutoff]). "
        "When set, mm_r_min defaults to mm_switch_on to exclude close monomers.",
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
        "--mm-r-min",
        type=float,
        default=None,
        metavar="Å",
        help="MM inner cutoff: exclude pairs with dimer COM < this. Defaults: legacy mode "
        "-> mm_switch_on*0.9; complementary -> (mm_switch_on-ml_cutoff)*0.9.",
    )
    parser.add_argument(
        "--ml-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Max systems per ML forward pass. When set, chunk large batches to reduce memory. "
        "Default: None (no chunking). Suggested: 256–512 for 8–16 GB GPU, 512–1024 for 24 GB+.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output inside the calculator factory.",
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
        "--nsteps_jaxmd",
        type=int,
        default=100_000,
        help="Number of MD steps to run in JAX-MD (default: 100000).",
    )
    parser.add_argument(
        "--steps-per-recording",
        type=int,
        default=None,
        help="Steps between recording blocks (default: 25 for NPT, 1000 for NVT/NVE). "
        "NPT requires frequent neighbor list updates; use smaller values if unstable.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="md_simulation",
        help="Prefix for output files (default: md_simulation).",
    )
    parser.add_argument(
        "--nsteps_ase",
        type=int,
        default=10000,
        help="Number of steps to run in ASE (default: 10000).",
    )
    parser.add_argument(
        "--optimize-monomers",
        action="store_true",
        help="If set, run monomer-wise optimization with simple_physnet before hybrid BFGS. "
        "Default: skip (use hybrid BFGS from CHARMM structure). Monomer optimization ignores "
        "inter-monomer interactions and can produce overlapping geometries.",
    )

    parser.add_argument(
        "--ensemble",
        type=str,
        default="nvt",
        help="Ensemble to run the simulation in (default: nvt).",
    )

    # Nose-Hoover chain (NHC) thermostat parameters for NVT ensemble
    parser.add_argument(
        "--nhc-chain-length",
        type=int,
        default=3,
        help="Number of chains in the Nose-Hoover chain thermostat (default: 3).",
    )
    parser.add_argument(
        "--nhc-chain-steps",
        type=int,
        default=2,
        help="Number of steps per chain in the Nose-Hoover chain thermostat (default: 2).",
    )
    parser.add_argument(
        "--nhc-sy-steps",
        type=int,
        default=3,
        help="Number of Suzuki-Yoshida steps in the Nose-Hoover chain thermostat (default: 3).",
    )
    parser.add_argument(
        "--nhc-tau",
        type=float,
        default=100.0,
        help="Thermostat coupling time multiplier (tau = nhc_tau * dt) (default: 100).",
    )

    # NPT barostat parameters
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Target pressure in atm for NPT ensemble (default: 1.0). "
        "Use 0 to preserve initial density (P = N*kT/V for N molecules).",
    )
    parser.add_argument(
        "--nhc-barostat-tau",
        type=float,
        default=10000.0,
        help="Barostat coupling time multiplier for NPT (tau = nhc_barostat_tau * dt) (default: 10000).",
    )
    parser.add_argument(
        "--npt-diagnose",
        action="store_true",
        help="Run NPT diagnostic tests before simulation (energy, stress, shift, step).",
    )
    parser.add_argument(
        "--nbr-monitor",
        action="store_true",
        help="Monitor neighbor list: log n_valid pairs, capacity, fill ratio to progress and HDF5 (NPT only).",
    )

    parser.add_argument(
        "--heating_interval",
        type=int,
        default=500,
        help="Interval to heat the system in ASE (default: 500).",
    )
    parser.add_argument(
        "--write_interval",
        type=int,
        default=100,
        help="Interval to write the trajectory in ASE (default: 100).",
    )

    parser.add_argument(
        "--charmm_heat",
        action="store_true",
        help="Run CHARMM heat (default: False).",
    )
    parser.add_argument(
        "--charmm_equilibration",
        action="store_true",
        help="Run CHARMM equilibration (default: False).",
    )
    parser.add_argument(
        "--charmm_production",
        action="store_true",
        help="Run CHARMM production (default: False).",
    )
    parser.add_argument(
        "--pycharmm-minimize/--no-pycharmm-minimize",
        dest="pycharmm_minimize",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run PyCHARMM nbonds/minimize before ASE/JAX-MD (default: True). "
        "Use --no-pycharmm-minimize to skip when going straight to JAX-MD (e.g. with --nsteps_ase 0) to avoid slow single-threaded CHARMM phase.",
    )
    parser.add_argument(
        "--pycharmm-minimize-steps",
        type=int,
        default=1000,
        metavar="N",
        help="Number of ABNR minimization steps when PyCHARMM minimize is enabled (default: 1000). "
        "Use fewer (e.g. 100) for faster startup when structure is already reasonable.",
    )
    parser.add_argument(
        "--skip-setup-energy-show",
        action="store_true",
        help="Skip energy.show() in setup_box to avoid slow CHARMM energy evaluation (Drude setup). "
        "Use for faster startup; less validation of the initial structure.",
    )
    parser.add_argument(
        "--jaxmd-minimize-steps",
        type=int,
        default=1000,
        metavar="N",
        help="Number of FIRE minimization steps before JAX-MD (default: 1000). Use 0 to skip.",
    )
    parser.add_argument(
        "--jaxmd-pbc-minimize-steps",
        type=int,
        default=1000,
        metavar="N",
        help="Number of PBC FIRE minimization steps when --cell is set (default: 1000). Use 0 to skip.",
    )

    parser.add_argument(
        "--use_physnet_calculator_for_full_system",
        action="store_true",
        help="Use the physnet calculator for the full system (default: False).",
    )

    parser.add_argument(
        "--trajectory-format",
        type=str,
        choices=["traj", "dcd"],
        default="traj",
        help="Output trajectory format: traj (ASE) or dcd (CHARMM-readable, pure Python) (default: traj).",
    )
    parser.add_argument(
        "--precompile",
        action="store_true",
        help="Compile JAX energy/force once and exit without running simulation. "
        "Use to separate slow first-run compilation from production MD.",
    )

    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    """Run MD simulation with the given arguments (CLI or notebook)."""
    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)

    # Setup imports
    Atoms = setup_ase_imports()
    CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()
    
    # Additional imports for this demo
    try:
        import pycharmm
        import pycharmm.psf as psf
        import ase
        import ase.calculators.calculator as ase_calc
        import ase.io as ase_io
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet
        import ase.optimize as ase_opt
        from mmml.interfaces.pycharmmInterface.setupBox import setup_box_generic
        import pandas as pd
        from mmml.interfaces.pycharmmInterface.import_pycharmm import minimize
        from mmml.interfaces.pycharmmInterface.cell_list import _wrap_groups_np
        import jax_md
        from jax_md import space, quantity, simulate, partition, units, units
        from ase.units import _amu

        import jax.numpy as jnp
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import wrap_groups

        import jax, e3x
        from jax import jit, grad, lax, ops, random
        import jax.numpy as jnp
        from mmml.utils.hdf5_reporter import make_jaxmd_reporter
    except ModuleNotFoundError as exc:
        sys.exit(f"Required modules not available: {exc}")

    pdbfilename = str(args.pdbfile)
    
    # Setup box and load PDB
    skip_energy_show = getattr(args, "skip_setup_energy_show", False)
    setup_box_generic(pdbfilename, side_length=1000, skip_energy_show=skip_energy_show)
    pdb_ase_atoms = ase_io.read(pdbfilename)

    atoms_per_monomer_list, n_monomers, total_atoms, n_atoms_first, monomer_offsets = (
        normalize_n_atoms_monomer(args.n_atoms_monomer, args.n_monomers)
    )
    print_system_summary(
        pdb_ase_atoms,
        n_monomers=n_monomers,
        atoms_per_monomer_list=atoms_per_monomer_list,
        cell=None,
        calculator_info=None,
    )
    
    # ========================================================================
    # MASS SETUP FOR JAX-MD SIMULATION
    # ========================================================================
    raw_masses = pdb_ase_atoms.get_masses()
    psf_masses = psf.get_amass()
    pdb_ase_atoms.set_masses(psf_masses)
    print_masses_summary(np.array(psf_masses[:total_atoms]))
    psf_masses_arr = np.array(psf_masses)[:, np.newaxis]
    correct_atomic_numbers_from_mass = np.argmin(
        np.abs(ase.data.atomic_masses_common[np.newaxis, :] - psf_masses_arr), axis=1)
    pdb_ase_atoms.set_atomic_numbers(correct_atomic_numbers_from_mass)
    # print(f"PDB ASE atoms: {pdb_ase_atoms}")

    # Actual masses for COM (Si_mass was misnamed - it held atomic numbers before)
    masses_jax = jnp.array(psf_masses[:total_atoms], dtype=jnp.float32)
    Si_mass = masses_jax  # keep name for compatibility with JAX-MD closure

    use_ase_calculator = args.nsteps_ase > 0 or getattr(
        args, "use_physnet_calculator_for_full_system", False
    )
    if use_ase_calculator:
        ase_monomer = pdb_ase_atoms[0:n_atoms_first]
        params_monomer, model_monomer = load_model_parameters(epoch_dir, n_atoms_first)
        simple_physnet_calculator = get_ase_calc(params_monomer, model_monomer, ase_monomer)
        ase_monomer.calc = simple_physnet_calculator
    else:
        simple_physnet_calculator = None

    natoms = len(pdb_ase_atoms)
    assert total_atoms == natoms, (
        f"total_atoms ({total_atoms}) != natoms from PDB ({natoms}). "
        f"atoms_per_monomer_list={atoms_per_monomer_list}"
    )
    # Calculator loads its own model from model_restart_path with natoms=max(monomer,dimer) size
    # Get atomic numbers and positions
    Z, R = pdb_ase_atoms.get_atomic_numbers(), pdb_ase_atoms.get_positions()
    atoms = pdb_ase_atoms
    print_positions_summary(R, atoms=atoms, title="Initial positions")
    if args.cell is not None:
        from ase.cell import Cell
        cell = Cell.fromcellpar([float(args.cell), float(args.cell), float(args.cell), 90., 90., 90.])
        atoms.set_cell(cell)
        atoms.set_pbc(True)
    else:
        cell = None


    # Unified calculator supports both uniform and heterogeneous monomer sizes
    calculator_factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_monomer_list,
        N_MONOMERS=n_monomers,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        complementary_handoff=not getattr(args, "no_complementary_handoff", False),
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        debug=args.debug,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=total_atoms,
        ml_energy_conversion_factor=1,
        ml_force_conversion_factor=1,
        cell=args.cell,
        flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
        flat_bottom_force_const=getattr(args, "flat_bottom_k", 1.0),
        ensemble=getattr(args, "ensemble", "nve"),
        ml_batch_size=getattr(args, "ml_batch_size", None),
        mm_r_min=getattr(args, "mm_r_min", None),
    )
    

    CUTOFF_PARAMS = CutoffParameters(
            ml_cutoff=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
            complementary_handoff=not getattr(args, "no_complementary_handoff", False),
        )

    # Print charges from PSF if available
    try:
        psf_charges = np.array(psf.get_charges())[:total_atoms]
        print_charges_summary(psf_charges)
    except Exception:
        pass

    # Create hybrid calculator (MIC-only: factory uses pbc_cell for PBC, no pbc_map/transform)
    # Skip ASE calculator when nsteps_ase == 0; use minimal pbc_map-only object for JAX-MD
    calc_result = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=n_monomers,
        cutoff_params=CUTOFF_PARAMS,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        backprop=True,
        debug=args.debug,
        # Final energies/forces are already in eV; MM terms are converted internally.
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        do_pbc_map=getattr(calculator_factory, "do_pbc_map", args.cell is not None),
        pbc_map=getattr(calculator_factory, "pbc_map", None),
        create_ase_calculator=use_ase_calculator,
    )
    if len(calc_result) == 3:
        hybrid_calc, spherical_cutoff_calculator, get_update_fn = calc_result
    else:
        hybrid_calc, spherical_cutoff_calculator = calc_result
        get_update_fn = None

    calc_type = "Hybrid calculator" if use_ase_calculator else "PbcMap-only"
    print_system_summary(
        atoms,
        n_monomers=n_monomers,
        atoms_per_monomer_list=atoms_per_monomer_list,
        cell=atoms.cell if hasattr(atoms, "cell") else cell,
        cutoff_params=CUTOFF_PARAMS,
        calculator_info=f"{calc_type} (do_pbc_map={getattr(hybrid_calc, 'do_pbc_map', 'N/A')})",
    )
    atoms = pdb_ase_atoms
    atoms.calc = hybrid_calc



    # Test invariance of energy under translation of monomer 0 by a lattice vector (PBC)
    # Skip when nsteps_ase == 0 to avoid slow first-use JIT; JAX-MD will trigger it anyway
    if args.cell is not None and args.nsteps_ase > 0:
        E0 = atoms.get_potential_energy()
        a = np.array([float(args.cell), 0.0, 0.0])  # first lattice vector for cubic cell
        g0 = np.arange(int(monomer_offsets[1]))  # first monomer's atoms
        R_shift = R.copy()
        R_shift[g0] += a
        atoms.set_positions(R_shift)
        E1 = atoms.get_potential_energy()
        from rich.console import Console
        from rich.table import Table
        t = Table(title="Energy invariance (PBC)")
        t.add_column("E0 (eV)", justify="right")
        t.add_column("E1 (eV)", justify="right")
        t.add_column("ΔE (eV)", justify="right")
        t.add_column("OK", justify="center")
        t.add_row(f"{E0:.6f}", f"{E1:.6f}", f"{E1-E0:.2e}", "✓" if np.isclose(E0, E1, rtol=1e-2, atol=0.01) else "✗")
        Console().print(t)
        assert np.isclose(E0, E1, rtol=1e-2, atol=0.01), (
            f"Energy invariance test failed: |E1-E0|={abs(E1-E0):.2e} eV"
        )
        atoms.set_positions(R)
    
    ##### add an option to just use the physnet calculator for the full system
    if getattr(args, "use_physnet_calculator_for_full_system", False):
        atoms.calc = simple_physnet_calculator
        print("Using physnet calculator for the full system")

    # Get initial energy and forces (skip when nsteps_ase == 0 to avoid slow first-use JIT;
    # JAX-MD will trigger compilation on its first step anyway)
    if args.nsteps_ase > 0:
        hybrid_energy = float(atoms.get_potential_energy())
        hybrid_forces = np.asarray(atoms.get_forces())
        print_forces_summary(hybrid_forces, energy_eV=hybrid_energy)
    

    from mmml.cli.run.pycharmm_runner import (
        run_equilibration,
        run_heat,
        run_production,
        run_pycharmm_setup_and_minimize,
    )
    atoms = run_pycharmm_setup_and_minimize(atoms, args)

    if getattr(args, "charmm_heat", False):
        atoms = run_heat(atoms, args)
    if getattr(args, "charmm_equilibration", False):
        atoms = run_equilibration(atoms, args)
    if getattr(args, "charmm_production", False):
        atoms = run_production(atoms, args)
    
    # Minimize structure if requested
    from mmml.cli.run.ase_runner import run_ase_md
    from mmml.cli.run.jaxmd_runner import set_up_nhc_sim_routine

    sim_key, data_key = jax.random.split(jax.random.PRNGKey(42), 2)
    temperature = args.temperature
    if args.nsteps_ase > 0:
        for i in range(1):
            run_ase_md(
                atoms,
                args=args,
                hybrid_calc=hybrid_calc,
                monomer_offsets=monomer_offsets,
                n_monomers=n_monomers,
                atoms_per_monomer_list=atoms_per_monomer_list,
                simple_physnet_calculator=simple_physnet_calculator,
                run_index=i,
                temperature=temperature,
            )


    if args.nsteps_jaxmd > 0 or getattr(args, "precompile", False):
        for j in range(1):
            sim_key, data_key = jax.random.split(data_key, 2)
            s = set_up_nhc_sim_routine(
                atoms, args, spherical_cutoff_calculator, get_update_fn,
                CUTOFF_PARAMS, n_monomers, monomer_offsets, Si_mass
            )
            if getattr(args, "precompile", False):
                print("Precompile done. Exiting (no simulation).")
                return atoms
            # Skip JAX-MD minimization when ASE already ran (positions are ASE-minimized)
            skip_jaxmd_min = args.nsteps_ase > 0
            out_positions, out_boxes, _ = run_sim_loop(
                s, sim_key, atoms, skip_minimization=skip_jaxmd_min
            )

            for idx, pos_block in enumerate(out_positions):
                print_positions_summary(pos_block, atoms=atoms, title=f"Trajectory positions (block {idx})")

            steps_per_frame = get_steps_per_frame(args)
            for i in range(len(out_positions)):
                traj_filename = f"{args.output_prefix}_md_{j}_{i}"
                Path(traj_filename).parent.mkdir(parents=True, exist_ok=True)
                traj_format = getattr(args, "trajectory_format", "traj")
                ext = "dcd" if traj_format == "dcd" else "traj"
                print(f"Saving trajectory to: {traj_filename}.{ext}")
                save_trajectory(
                    out_positions[i],
                    atoms,
                    filename=traj_filename,
                    format=traj_format,
                    boxes=out_boxes[i] if out_boxes and out_boxes[i] is not None else None,
                    dt_ps=args.timestep * 0.001,
                    steps_per_frame=steps_per_frame,
                    save_energy_forces=use_ase_calculator,
                    monomer_offsets=monomer_offsets,
                    cell=args.cell,
                    masses=np.asarray(atoms.get_masses()) if hasattr(atoms, "get_masses") else None,
                )

            # atoms = minimize_structure(atoms)

    print("Trajectories saved!")


    print("JAX-MD simulation complete!")
    return atoms

def main() -> int:
    """CLI entry point: parse args and run simulation."""
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

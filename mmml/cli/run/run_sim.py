"""
Runs an MD simulation.

Requires the box size and system (as a pdb file) to be specified.
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
        # optional: temperature=200.0, timestep=0.3, ensemble="nve", ...
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
from mmml.cli.run.shared import save_trajectory


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
        from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
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
    setup_box_generic(pdbfilename, side_length=1000)
    pdb_ase_atoms = ase_io.read(pdbfilename)

    print(f"Loaded PDB file: {pdb_ase_atoms}")

    # --- Normalise n_atoms_monomer: int or list ---------------------------------
    raw_n_atoms_monomer = args.n_atoms_monomer
    if isinstance(raw_n_atoms_monomer, (list, tuple, np.ndarray)):
        atoms_per_monomer_list = [int(x) for x in raw_n_atoms_monomer]
        n_monomers = len(atoms_per_monomer_list)
        total_atoms = sum(atoms_per_monomer_list)
        n_atoms_first = atoms_per_monomer_list[0]  # for single-monomer test calc
    else:
        n_atoms_first = int(raw_n_atoms_monomer)
        n_monomers = args.n_monomers
        atoms_per_monomer_list = [n_atoms_first] * n_monomers
        total_atoms = n_atoms_first * n_monomers
    # Build monomer offsets for slicing
    monomer_offsets = np.zeros(n_monomers + 1, dtype=int)
    for _mi, _na in enumerate(atoms_per_monomer_list):
        monomer_offsets[_mi + 1] = monomer_offsets[_mi] + _na
    print(f"[run_sim] atoms_per_monomer_list={atoms_per_monomer_list}, "
          f"n_monomers={n_monomers}, total_atoms={total_atoms}")
    
    # ========================================================================
    # MASS SETUP FOR JAX-MD SIMULATION
    # ========================================================================
    raw_masses = pdb_ase_atoms.get_masses()
    # print(f"Raw masses from ASE: {raw_masses}")
    psf_masses = psf.get_amass()
    # print(f"PSF masses: {psf_masses}")
    print(f"PSF masses sum: {sum(psf_masses)}")
    print("Setting the elements and masses from the psf")
    pdb_ase_atoms.set_masses(psf_masses)
    psf_masses_arr = np.array(psf_masses)[:, np.newaxis]
    correct_atomic_numbers_from_mass = np.argmin(
        np.abs(ase.data.atomic_masses_common[np.newaxis, :] - psf_masses_arr), axis=1)
    pdb_ase_atoms.set_atomic_numbers(correct_atomic_numbers_from_mass)
    # print(f"PDB ASE atoms: {pdb_ase_atoms}")

    # Actual masses for COM (Si_mass was misnamed - it held atomic numbers before)
    masses_jax = jnp.array(psf_masses[:total_atoms], dtype=jnp.float32)
    Si_mass = masses_jax  # keep name for compatibility with JAX-MD closure
    Si_mass_sum = Si_mass.sum()
    # print(f"Masses (amu) for JAX-MD: sum={float(Si_mass_sum):.2f}")
    
    Si_mass_expanded = jnp.repeat(Si_mass[:, None], 3, axis=1)
    # print(f"Masses expanded shape: {Si_mass_expanded.shape}")

    # print(f"PyCHARMM coordinates: {coor.get_positions()}")
    # print(f"Ase coordinates: {pdb_ase_atoms.get_positions()}")

    # print(coor.get_positions())
    # Defer simple_physnet_calculator creation when nsteps_ase == 0 to avoid slow first-use JIT
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

    # Load model parameters for the full system
    natoms = len(pdb_ase_atoms)
    assert total_atoms == natoms, (
        f"total_atoms ({total_atoms}) != natoms from PDB ({natoms}). "
        f"atoms_per_monomer_list={atoms_per_monomer_list}"
    )
    params, model = load_model_parameters(epoch_dir, natoms)
    model.natoms = natoms
    print(f"Model loaded: {model}")
    # Get atomic numbers and positions
    Z, R = pdb_ase_atoms.get_atomic_numbers(), pdb_ase_atoms.get_positions()
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


    # Unified calculator supports both uniform and heterogeneous monomer sizes
    calculator_factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_monomer_list,
        N_MONOMERS=n_monomers,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
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
    )
    

    CUTOFF_PARAMS = CutoffParameters(
            ml_cutoff=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
        )

    print(f"Cutoff parameters: {CUTOFF_PARAMS}")

    # Create hybrid calculator (MIC-only: factory uses pbc_cell for PBC, no pbc_map/transform)
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
    )
    if len(calc_result) == 3:
        hybrid_calc, spherical_cutoff_calculator, get_update_fn = calc_result
    else:
        hybrid_calc, spherical_cutoff_calculator = calc_result
        get_update_fn = None

    print(f"Hybrid calculator created: {hybrid_calc}")
    print(f"Spherical cutoff calculator: {spherical_cutoff_calculator}, "
          f"get_update_fn: {get_update_fn}")
    atoms = pdb_ase_atoms

    print(f"ASE atoms: {atoms}")
    atoms.calc = hybrid_calc

    # After: atoms.calc = hybrid_calc
    print(f"PBC status: cell={args.cell}, atoms.pbc={atoms.pbc}, "
        f"calc.do_pbc_map={getattr(hybrid_calc, 'do_pbc_map', 'N/A')}")



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
        print(f"Energy invariance test: E0={E0}, E1={E1}, difference={E1-E0}")
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
        print(f"Initial energy: {hybrid_energy:.6f} eV")
        print(f"Initial forces: {hybrid_forces}")
    

    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
        pycharmm,
        safe_energy_show,
    )
    reset_block()
    nbonds = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################

! Non-bonding parameters
nbonds atom cutnb 10.0  ctofnb 9.0 ctonnb 8.0 -
vswitch NBXMOD 5 -
inbfrq -1 imgfrq -1

shake bonh para sele all end

"""
    pycharmm.lingo.charmm_script(nbonds)
    safe_energy_show()
    pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
    pycharmm.lingo.charmm_script("ENER")
    safe_energy_show()
    # Sync ASE atoms from PyCHARMM so BFGS/ASE MD start from CHARMM-minimized structure
    atoms.set_positions(coor.get_positions())

    def run_heat(): 
        from mmml.interfaces.pycharmmInterface.pycharmmCommands import heat
        pycharmm.lingo.charmm_script(heat)
        atoms.set_positions(coor.get_positions())
        safe_energy_show()
        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
        safe_energy_show()
        pycharmm.lingo.charmm_script("ENER")
        atoms.set_positions(coor.get_positions())
        return atoms

    def run_equilibration():
        from mmml.interfaces.pycharmmInterface.pycharmmCommands import equi
        pycharmm.lingo.charmm_script(equi)
        atoms.set_positions(coor.get_positions())
        safe_energy_show()
        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
        safe_energy_show()
        pycharmm.lingo.charmm_script("ENER")
        atoms.set_positions(coor.get_positions())
        return atoms

    def run_production():
        from mmml.interfaces.pycharmmInterface.pycharmmCommands import production
        pycharmm.lingo.charmm_script(production)
        atoms.set_positions(coor.get_positions())
        safe_energy_show()
        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
        safe_energy_show()
        pycharmm.lingo.charmm_script("ENER")
        atoms.set_positions(coor.get_positions())
        return atoms

    if getattr(args, "charmm_heat", False):
        atoms = run_heat()
    if getattr(args, "charmm_equilibration", False):
        atoms = run_equilibration()
    if getattr(args, "charmm_production", False):
        atoms = run_production()
    
    # Minimize structure if requested
    from mmml.cli.run.ase_runner import run_ase_md
    from mmml.cli.run.jaxmd_runner import set_up_nhc_sim_routine

    def _run_ase_md(atoms, run_index=0, temperature=args.temperature):
        run_ase_md(
            atoms,
            args=args,
            hybrid_calc=hybrid_calc,
            monomer_offsets=monomer_offsets,
            n_monomers=n_monomers,
            atoms_per_monomer_list=atoms_per_monomer_list,
            simple_physnet_calculator=simple_physnet_calculator,
            run_index=run_index,
            temperature=temperature,
        )

    def run_sim_loop(run_sim, sim_key, nsim=1, skip_minimization=False):
        """
        Run the simulation for the given indices and save the trajectory.
        Uses current atoms positions (after ASE MD if run) as initial positions.
        """
        out_positions = []
        out_boxes = []  # NPT: boxes per frame for trajectory cell
        max_is = []
        pos = np.asarray(atoms.get_positions(), dtype=np.float32)
        for i in range(nsim):
            mi, pos, boxes = run_sim(sim_key, R=pos, skip_minimization=skip_minimization)
            out_positions.append(pos)
            out_boxes.append(boxes)
            max_is.append(mi)

        return out_positions, out_boxes, max_is


    sim_key, data_key = jax.random.split(jax.random.PRNGKey(42), 2)
    temperature = args.temperature
    if args.nsteps_ase > 0:
        for i in range(1):
            _run_ase_md(atoms, run_index=i, temperature=temperature)


    if args.nsteps_jaxmd > 0:
        for j in range(1):
            sim_key, data_key = jax.random.split(data_key, 2)
            s = set_up_nhc_sim_routine(
                atoms, args, spherical_cutoff_calculator, get_update_fn,
                CUTOFF_PARAMS, n_monomers, monomer_offsets, Si_mass
            )
            # Skip JAX-MD minimization when ASE already ran (positions are ASE-minimized)
            skip_jaxmd_min = args.nsteps_ase > 0
            out_positions, out_boxes, _ = run_sim_loop(s, sim_key, skip_minimization=skip_jaxmd_min)

            print(f"Out positions: {out_positions}")

            steps_per_frame = getattr(args, "steps_per_recording", None) or (
                25 if (args.ensemble == "npt" and args.cell is not None) else 1000
            )
            for i in range(len(out_positions)):
                traj_filename = f'{args.output_prefix}_md_trajectory_{j}_{i}'
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

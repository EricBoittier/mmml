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
        "--ensemble",
        type=str,
        default="nvt",
        help="Ensemble to run the simulation in (default: nvt).",
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
        import ase
        import ase.calculators.calculator as ase_calc
        import ase.io as ase_io
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md.verlet import VelocityVerlet
        import ase.optimize as ase_opt
        from mmml.pycharmmInterface.import_pycharmm import coor
        from mmml.pycharmmInterface.setupBox import setup_box_generic
        import pandas as pd
        from mmml.pycharmmInterface.import_pycharmm import minimize
        import jax_md
        from jax_md import space, quantity, simulate, partition, units, units
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
    ase_monomer = pdb_ase_atoms[0:args.n_atoms_monomer]
    params, model = load_model_parameters(epoch_dir, args.n_atoms_monomer)
    simple_physnet_calculator = get_ase_calc(params, model, ase_monomer)
    print(f"Simple physnet calculator: {simple_physnet_calculator}")
    ase_monomer.calc = simple_physnet_calculator
    print(f"ASE monomer: {ase_monomer}")
    print(f"ASE monomer energy: {ase_monomer.get_potential_energy()}")
    print(f"ASE monomer forces: {ase_monomer.get_forces()}")
    print(f"ASE monomer positions: {ase_monomer.get_positions()}")
    print(f"ASE monomer cell: {ase_monomer.get_cell()}")
    print(f"ASE monomer pbc: {ase_monomer.get_pbc()}")
    print(f"ASE monomer calculator: {ase_monomer.calc}")
    print(f"ASE monomer calculator type: {type(ase_monomer.calc)}")


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
        MAX_ATOMS_PER_SYSTEM=args.n_monomers * args.n_atoms_monomer,
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





    # Create hybrid calculator (pbc_map/do_pbc_map from factory when cell is set)
    hybrid_calc, spherical_cutoff_calculator = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=args.n_monomers,
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
 
    print(f"Hybrid calculator created: {hybrid_calc}")
    atoms = pdb_ase_atoms

    print(f"ASE atoms: {atoms}")
    atoms.calc = hybrid_calc

    # After: atoms.calc = hybrid_calc
    print(f"PBC status: cell={args.cell}, atoms.pbc={atoms.pbc}, "
        f"calc.do_pbc_map={getattr(hybrid_calc, 'do_pbc_map', 'N/A')}")



    # Test invariance of energy under translation of monomer 0 by a lattice vector (PBC)
    if args.cell is not None:
        E0 = atoms.get_potential_energy()
        a = np.array([float(args.cell), 0.0, 0.0])  # first lattice vector for cubic cell
        g0 = np.where(np.arange(len(atoms)) < (len(atoms) // args.n_monomers))[0]
        R_shift = R.copy()
        R_shift[g0] += a
        atoms.set_positions(R_shift)
        E1 = atoms.get_potential_energy()
        print(f"Energy invariance test: E0={E0}, E1={E1}, difference={E1-E0}")
        assert np.isclose(E0, E1), "Energy invariance test failed"
        atoms.set_positions(R)
    
    # Get initial energy and forces
    hybrid_energy = float(atoms.get_potential_energy())
    hybrid_forces = np.asarray(atoms.get_forces())
    print(f"Initial energy: {hybrid_energy:.6f} eV")
    print(f"Initial forces: {hybrid_forces}")
    

    from mmml.pycharmmInterface.import_pycharmm import (
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
nbonds atom cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
vswitch NBXMOD 5 -
inbfrq -1 imgfrq -1
"""
    pycharmm.lingo.charmm_script(nbonds)
    safe_energy_show()
    pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
    pycharmm.lingo.charmm_script("ENER")
    safe_energy_show()
    from mmml.pycharmmInterface.pycharmmCommands import heat
    # pycharmm.lingo.charmm_script(heat)
    atoms.set_positions(coor.get_positions())
    safe_energy_show()
    pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-2, tolgrd=1e-2)
    safe_energy_show()
    pycharmm.lingo.charmm_script("ENER")

    # Minimize structure if requested
    # if args.minimize_first:
    def wrap_positions_for_pbc(positions):
        """Apply PBC mapping to wrap positions into the cell (molecular wrapping)."""
        if args.cell is None:
            return positions
        pbc_map_fn = getattr(hybrid_calc, "pbc_map", None)
        if pbc_map_fn is None or not getattr(hybrid_calc, "do_pbc_map", False):
            return positions
        R_mapped = pbc_map_fn(jnp.asarray(positions))
        return np.asarray(jax.device_get(R_mapped))

    def minimize_structure(atoms, run_index=0, nsteps=60, fmax=0.0006, charmm=False):

        if charmm:
            pycharmm.minimize.run_abnr(nstep=10000, tolenr=1e-6, tolgrd=1e-6)
            pycharmm.lingo.charmm_script("ENER")
            safe_energy_show()
            atoms.set_positions(coor.get_positions())
            atoms = optimize_as_monomers(atoms, run_index=run_index, nsteps=100, fmax=0.0006)

        traj = ase_io.Trajectory(f'bfgs_{run_index}_{args.output_prefix}_minimized.traj', 'w')
        print("Minimizing structure with hybrid calculator")
        print(f"Running BFGS for {nsteps} steps")
        print(f"Running BFGS with fmax: {fmax}")
        _ = ase_opt.BFGS(atoms, trajectory=traj).run(fmax=fmax, steps=nsteps)
        # Sync with PyCHARMM
        xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(xyz)
        traj.close()
        return atoms
        
    def optimize_as_monomers(atoms, run_index=0, nsteps=60, fmax=0.0006):
        optimized_atoms_positions = np.zeros_like(atoms.get_positions())
        # use ase and the original physnet calculator to optimize the structure, one monomer at a time
        for i in range(args.n_monomers):
            monomer_atoms = atoms[i*args.n_atoms_monomer:(i+1)*args.n_atoms_monomer]
            monomer_atoms.calc = simple_physnet_calculator
            _ = ase_opt.BFGS(monomer_atoms).run(fmax=fmax, steps=nsteps)
            optimized_atoms_positions[i*args.n_atoms_monomer:(i+1)*args.n_atoms_monomer] = monomer_atoms.get_positions()

        atoms.set_positions(optimized_atoms_positions)
        # Wrap positions into cell after monomer optimization (avoids unwrapped coords for PBC)
        if args.cell is not None:
            wrapped = wrap_positions_for_pbc(atoms.get_positions())
            atoms.set_positions(wrapped)
            xyz = pd.DataFrame(wrapped, columns=["x", "y", "z"])
        else:
            xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(xyz)

        return atoms


        
    def run_ase_md(atoms, run_index=0, temperature=args.temperature):
        
        if run_index == 0:
            atoms = optimize_as_monomers(atoms, run_index=run_index, nsteps=100, fmax=0.0006)

        
        print(f"Optimized atoms energy: {atoms.get_potential_energy()}")
        print(f"Optimized atoms forces: {atoms.get_forces()}")

        atoms = minimize_structure(atoms, run_index=run_index,
         nsteps=100 if run_index == 0 else 10, fmax=0.0006 if run_index == 0 else 0.001)
        # Wrap positions into cell after BFGS (avoids unwrapped coords for PBC)
        if args.cell is not None:
            # translate to the center of the cell
            atoms.set_positions(atoms.get_positions() - atoms.get_positions().mean(axis=0))
            wrapped = wrap_positions_for_pbc(atoms.get_positions())
            atoms.set_positions(wrapped)
            xyz = pd.DataFrame(wrapped, columns=["x", "y", "z"])
            coor.set_positions(xyz)

        # Setup MD simulation
        
        timestep_fs = args.timestep
        num_steps = args.nsteps_ase
        ase_atoms = atoms
        
        # Draw initial momenta
        if run_index == 0:
            MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
            Stationary(ase_atoms)  # Remove center of mass translation
            ZeroRotation(ase_atoms)  # Remove rotations

        dt = timestep_fs*ase.units.fs
        print(f"Running ASE MD with timestep: {dt} (ase units)")
        # Initialize Velocity Verlet integrator
        integrator = VelocityVerlet(ase_atoms, timestep=dt)

        # Open trajectory file
        traj_filename = f'{run_index}_{args.output_prefix}_{temperature}K_{num_steps}steps_P{dt}.traj'
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
            # Do NOT wrap positions every timestep: pbc_map applies a discontinuous coordinate
            # transformation. Replacing R with wrapped(R) after integration creates an inconsistent
            # phase-space point (velocities unchanged, positions jumped) and breaks NVE energy
            # conservation. The calculator applies pbc_map internally for forces; integration uses
            # unwrapped coordinates. Wrap only for trajectory output if needed (see traj.write).
            # Save current frame and keep track of energies
            frames[i] = ase_atoms.get_positions()
            potential_energy[i] = ase_atoms.get_potential_energy()
            kinetic_energy[i] = ase_atoms.get_kinetic_energy()
            total_energy[i] = ase_atoms.get_total_energy()
            
            # Check for energy spikes and re-minimize if needed
            if i > 10 and (kinetic_energy[i] > 300 or potential_energy[i] > 0):
                print(f"Energy spike detected at step {i}, re-minimizing...")
                print(f"Kinetic energy: {kinetic_energy[i]:.6f} eV")
                print(f"Potential energy: {potential_energy[i]:.6f} eV")
                print(f"Total energy: {total_energy[i]:.6f} eV")
                print(f"Step: {i}")
                print(f"Breakcount: {breakcount}")
                print(f"Run index: {run_index}")
                print(f"Temperature: {temperature} K")
                print(f"Timestep: {timestep_fs} fs")
                print(f"Number of steps: {num_steps}")
                print(f"Number of atoms: {len(ase_atoms)}")
                print(f"Number of monomers: {args.n_monomers}")
                print(f"Number of atoms per monomer: {args.n_atoms_monomer}")
                print(f"ML cutoff: {args.ml_cutoff} Å")
                print(f"MM switch on: {args.mm_switch_on} Å")
                print(f"MM cutoff: {args.mm_cutoff} Å")
                print(f"Include MM: {args.include_mm}")
                print(f"Skip ML dimers: {args.skip_ml_dimers}")
                print(f"Debug: {args.debug}")
                
                minimize_structure(ase_atoms, run_index=f"{run_index}_{breakcount}_{i}_", nsteps=20 if run_index == 0 else 10, 
                fmax=0.0006 if run_index == 0 else 0.001, charmm=True)
                # assign new velocities
                # MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                cur_eng = ase_atoms.get_potential_energy()
                print(f"Current energy: {cur_eng:.6f} eV")
                Stationary(ase_atoms)
                ZeroRotation(ase_atoms)
                breakcount += 1
            if breakcount > 1:
                print("Maximum number of breaks reached")
                break
            # Occasionally print progress and adjust temperature
            if (i != 0) and (i % args.write_interval == 0):
                if args.cell is not None:
                    traj.write(ase_atoms)

                else:
                    traj.write(ase_atoms)
            if args.ensemble == "nvt":
                if  (i % args.heating_interval == 0):
                    Stationary(ase_atoms)
                    ZeroRotation(ase_atoms)
                    MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                    print(f"Temperature adjusted to: {temperature} K")
            if i % 100 == 0:
                print(f"step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}")

        
        # Close trajectory file
        traj.close()
        print(f"Trajectory saved to: {traj_filename}")
        print("ASE MD simulation complete!")


    def set_up_nhc_sim_routine(atoms, T=args.temperature, dt=5e-3, steps_per_recording=250):
        @jax.jit
        def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
            return spherical_cutoff_calculator(
                atomic_numbers=atomic_numbers,
                positions=positions,
                n_monomers=args.n_monomers,
                cutoff_params=CUTOFF_PARAMS,
                doML=True,
                doMM=args.include_mm,
                doML_dimer=not args.skip_ml_dimers,
                debug=args.debug,
            )


        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
        atomic_numbers = atoms.get_atomic_numbers()
        R = atoms.get_positions()

        @jit
        def jax_md_energy_fn(position, **kwargs):
            # Ensure position is a JAX array
            position = jnp.array(position)
            # l_nbrs = nbrs.update(position)
            result = evaluate_energies_and_forces(
                atomic_numbers=atomic_numbers,
                positions=position,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )
            return result.energy.reshape(-1)[0]
        
        # jax_md_grad_fn = jax.grad(jax_md_energy_fn)

        # evaluate_energies_and_forces
        result = evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=R,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        print(f"Result: {result}")
        init_energy = result.energy.reshape(-1)[0]
        init_forces = result.forces.reshape(-1, 3)
        print(f"Initial energy: {init_energy:.6f} eV")
        print(f"Initial forces: {init_forces}")

        # Use actual cell size when PBC is set; otherwise fallback for non-periodic
        BOXSIZE = float(args.cell) if args.cell is not None else 1000.0
        use_pbc = args.cell is not None
        # Use the calculator's pbc_map for PBC (unwrap→coregister→wrap) to avoid
        # monomer overlap. Our custom wrap_molecules wrapped each COM to [0,L)
        # independently, causing monomers in different images to overlap → E=0, F=nan.
        pbc_map_fn = getattr(atoms.calc, "pbc_map", None) if atoms.calc else None
        print(f"JAX-MD BOXSIZE: {BOXSIZE} Å, PBC: {use_pbc}, pbc_map: {pbc_map_fn is not None}")

        # Energy: pass positions directly; calculator applies pbc_map internally.
        @jit
        def wrapped_energy_fn(position, **kwargs):
            return jax_md_energy_fn(jnp.array(position), **kwargs)

        # Shift: use pbc_map to wrap molecules (keeps monomers intact, coregisters correctly)
        _displacement, _shift_free = space.free()
        if use_pbc and pbc_map_fn is not None:
            def shift(R, dR, **kwargs):
                return pbc_map_fn(R + dR)
            displacement = _displacement
        else:
            shift = _shift_free
            displacement = _displacement

        unwrapped_init_fn, unwrapped_step_fn = jax_md.minimize.fire_descent(
            wrapped_energy_fn, shift, dt_start=0.001, dt_max=0.001
        )
        unwrapped_step_fn = jit(unwrapped_step_fn)


        @jit
        def sim(state):
            def step(i, s):
                return apply_fn(s)

            return lax.fori_loop(0, steps_per_recording, step, state)




        # ========================================================================
        # SIMULATION PARAMETERS (metal units: eV, Å, ps, amu)
        # ========================================================================
        unit = units.metal_unit_system()
        # dt must be in ps: args.timestep is fs, 1 fs = 0.001 ps
        # Use smaller dt for PBC to improve stability (large forces at boundaries)
        dt_fs = args.timestep * (0.5 if use_pbc else 1.0)
        dt = dt_fs * 0.001
        kT = T * unit['temperature']
        steps_per_recording = 25
        rng_key = jax.random.PRNGKey(0)
        print(f"JAX-MD NVE: dt={dt} ps ({dt_fs} fs), kT={kT} ({T} K)")

        # NVE uses same displacement/shift as minimization
        init_fn, apply_fn = simulate.nve(wrapped_energy_fn, shift, dt)
        apply_fn = jit(apply_fn)

        def run_sim(
            key,
            total_steps=args.nsteps_jaxmd,
            steps_per_recording=steps_per_recording,
            R=R,
        ):
            total_records = total_steps // steps_per_recording

            # Translate to center of mass before minimization
            com = jnp.sum(Si_mass[:, None] * R, axis=0) / Si_mass.sum()
            initial_pos = R - com
            fire_state = unwrapped_init_fn(initial_pos)
            fire_positions = []

            # FIRE minimization
            print("*" * 10 + "\nMinimization\n" + "*" * 10)
            NMIN = 100
            for i in range(NMIN):
                fire_positions.append(fire_state.position)
                fire_state = unwrapped_step_fn(fire_state)
                
                if i % (NMIN // 10) == 0:
                    energy = float(wrapped_energy_fn(fire_state.position))
                    max_force = float(jnp.abs(jax.grad(wrapped_energy_fn)(fire_state.position)).max())
                    print(f"{i}/{NMIN}: E={energy:.6f} eV, max|F|={max_force:.6f}")


                # check for nans
                if jnp.isnan(energy):
                    print("NaN energy caught in minimization, using last valid position")
                    break
            # Best position from first minimization (last valid if NaN occurred)
            minimized_pos = fire_state.position
            if jnp.any(~jnp.isfinite(minimized_pos)) and fire_positions:
                minimized_pos = fire_positions[-1]
                print("Using last valid position from first minimization")
            # save pdb
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            ase_io.write(f"{args.output_prefix}_minimized_{current_time}.pdb", atoms)

            # ========================================================================
            # PBC MINIMIZATION (when PBC enabled)
            # ========================================================================
            print("*" * 10 + "\nPBC Minimization\n" + "*" * 10)
            pbc_unwrapped_init_fn, pbc_unwrapped_step_fn = jax_md.minimize.fire_descent(
                wrapped_energy_fn, shift, dt_start=0.001, dt_max=0.001
            )
            pbc_unwrapped_step_fn = jit(pbc_unwrapped_step_fn)
            # Start PBC minimization from best position (valid even if first min hit NaN)
            pbc_fire_state = pbc_unwrapped_init_fn(minimized_pos)
            pbc_fire_positions = []
            
            # Run PBC minimization
            NMIN_PBC = 100  # Fewer steps since we're already close to minimum
            for i in range(NMIN_PBC):
                pbc_fire_positions.append(pbc_fire_state.position)
                pbc_fire_state = pbc_unwrapped_step_fn(pbc_fire_state)
                
                if i % (NMIN_PBC // 10) == 0:
                    energy = float(wrapped_energy_fn(pbc_fire_state.position))
                    max_force = float(jnp.abs(jax.grad(wrapped_energy_fn)(pbc_fire_state.position)).max())
                    print(f"{i}/{NMIN_PBC}: E={energy:.6f} eV, max|F|={max_force:.6f}")
                
                # Check for nans
                if jnp.isnan(energy):
                    print("NaN energy caught in PBC minimization, using last valid position")
                    break
            
            # Save PBC minimized structure
            pbc_current_time = datetime.now().strftime("%H:%M:%S")
            ase_io.write(f"{args.output_prefix}_pbc_minimized_{pbc_current_time}.pdb", atoms)
            print(f"PBC minimization complete. Final energy: {energy:.6f} eV")

            # Use last valid positions if minimization produced NaN
            md_pos = pbc_fire_state.position
            if jnp.any(~jnp.isfinite(md_pos)) and pbc_fire_positions:
                md_pos = pbc_fire_positions[-1]
                print("Warning: NaN in PBC minimization, using last valid position from PBC")
            if jnp.any(~jnp.isfinite(md_pos)) and fire_positions:
                md_pos = fire_positions[-1]
                print("Warning: Using last valid position from first minimization")
            if jnp.any(~jnp.isfinite(md_pos)):
                print("Error: No valid positions for NVE; skipping JAX-MD simulation")
                return 0, jnp.array([]).reshape(0, len(md_pos), 3)

            # NVE init with temperature from metal units (init_fn handles momentum scaling)
            state = init_fn(key, md_pos, kT, mass=Si_mass)
            print(f"Momentum initialized for {T} K")
            nhc_positions = []

            # get energy of initial state
            energy_initial = float(wrapped_energy_fn(state.position))
            print(f"Initial energy: {energy_initial:.6f} eV")

            print("*" * 10 + "\nNVE\n" + "*" * 10)
            print("\t\tTime (ps)\tEnergy (eV)\tTemperature (K)")
            
            # ========================================================================
            # MAIN SIMULATION LOOP
            # ========================================================================
            for i in range(total_records):
                state = sim(state)
                
                # Store current position for trajectory analysis
                nhc_positions.append(state.position)
                    
                # Print progress every 10 steps
                if i % 10 == 0:
                    time_ps = i * steps_per_recording * dt
                    T_curr = jax_md.quantity.temperature(
                        momentum=state.momentum,
                        mass=state.mass
                    ) / unit['temperature']
                    temp = float(T_curr)
                    energy = float(wrapped_energy_fn(state.position))
                    print(f"{time_ps:10.4f}\t{energy:10.4f}\t{temp:10.2f}")

                    # Stop on numerical instability (NaN, Inf, or energy blow-up to 0)
                    if not np.isfinite(energy) or not np.isfinite(temp):
                        print(f"Numerical instability at step {i * steps_per_recording}; stopping.")
                        if len(nhc_positions) > 1:
                            nhc_positions = nhc_positions[:-1]
                            state = type(state)(
                                position=nhc_positions[-1],
                                momentum=state.momentum,
                                mass=state.mass
                            )
                        break
                    if energy >= 0 and energy_initial < 0:
                        print(f"Energy blow-up at step {i * steps_per_recording} (E={energy:.4f}); stopping.")
                        if len(nhc_positions) > 1:
                            nhc_positions = nhc_positions[:-1]
                            state = type(state)(
                                position=nhc_positions[-1],
                                momentum=state.momentum,
                                mass=state.mass
                            )
                        break

            steps_completed = i * steps_per_recording
            print(f"\nSimulated {steps_completed} steps ({steps_completed * dt:.2f} ps)")


            nhc_positions_out = []
            for R in nhc_positions:
                if use_pbc and pbc_map_fn is not None:
                    R = pbc_map_fn(R)
                nhc_positions_out.append(R)
            return steps_completed, jnp.stack(nhc_positions_out)

        return run_sim


    def save_trajectory(out_positions, atoms, filename="nhc_trajectory", format="traj"):
        trajectory = Trajectory(f"{filename}.{format}", "a")
        out_positions = out_positions.reshape(-1, len(atoms),3)
        for R in out_positions:
            atoms.set_positions(R)
            trajectory.write(atoms)
        trajectory.close()


    def run_sim_loop(run_sim, sim_key, nsim=1):
        """
        Run the simulation for the given indices and save the trajectory.
        """
        out_positions = []
        max_is = []
        pos = R
        for i in range(nsim):
            mi, pos = run_sim(sim_key, R=pos)
        out_positions.append(pos)
        max_is.append(mi)

        return out_positions, max_is


    sim_key, data_key = jax.random.split(jax.random.PRNGKey(42), 2)
    temperature = args.temperature
    for i in range(1):
        run_ase_md(atoms, run_index=i, temperature=temperature)


    
    # Main JAXMD simulation loop
    for j in range(1):
        sim_key, data_key = jax.random.split(data_key, 2)
        s = set_up_nhc_sim_routine(atoms, T=temperature)
        out_positions, _ = run_sim_loop(s, sim_key)

        print(f"Out positions: {out_positions}")

        for i in range(len(out_positions)):
            traj_filename = f'{args.output_prefix}_md_trajectory_{j}_{i}.traj'
            print(f"Saving trajectory to: {traj_filename}")
            save_trajectory(out_positions[i], atoms, filename=traj_filename)

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

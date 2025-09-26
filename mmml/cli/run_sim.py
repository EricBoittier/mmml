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

from base import (
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
    
    # Convert ASE masses to JAX-MD units properly
    # ASE masses are in amu, JAX-MD expects masses in internal units
    # The issue is _amu conversion is wrong - let's use a simpler approach
    # For hydrogen: mass should be ~1.66e-27 kg, but we're getting 2.657e-26
    
    # Try using masses directly without _amu conversion
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
    )
    

    CUTOFF_PARAMS = CutoffParameters(
            ml_cutoff=args.ml_cutoff,
            mm_switch_on=args.mm_switch_on,
            mm_cutoff=args.mm_cutoff,
        )

    print(f"Cutoff parameters: {CUTOFF_PARAMS}")

    # Create hybrid calculator
    hybrid_calc, _spherical_cutoff_calculator = calculator_factory(
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
    )
 
    print(f"Hybrid calculator created: {hybrid_calc}")
    atoms = pdb_ase_atoms
    print(f"ASE atoms: {atoms}")
    atoms.calc = hybrid_calc
    
    # Get initial energy and forces
    hybrid_energy = float(atoms.get_potential_energy())
    hybrid_forces = np.asarray(atoms.get_forces())
    print(f"Initial energy: {hybrid_energy:.6f} eV")
    

    from mmml.pycharmmInterface.import_pycharmm import reset_block, pycharmm, reset_block_no_internal
    reset_block()
    reset_block_no_internal()
    reset_block()
    nbonds = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################

! Non-bonding parameters
nbonds atom cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
vswitch NBXMOD 3 -
inbfrq -1 imgfrq -1
"""
    pycharmm.lingo.charmm_script(nbonds)
    pycharmm.energy.show()


    # Minimize structure if requested
    # if args.minimize_first:
    def minimize_structure(atoms, run_index=0, nsteps=60, fmax=0.0006):
        traj = ase_io.Trajectory(f'bfgs_{run_index}_{args.output_prefix}_minimized.traj', 'w')
        print("Minimizing structure with hybrid calculator")
        print(f"Running BFGS for {nsteps} steps")
        print(f"Running BFGS with fmax: {fmax}")
        _ = ase_opt.BFGS(atoms).run(fmax=fmax, steps=nsteps)
        # Sync with PyCHARMM
        xyz = pd.DataFrame(atoms.get_positions() - atoms.get_positions().mean(axis=0), columns=["x", "y", "z"])
        coor.set_positions(xyz)
        traj.write(atoms)
        traj.close()
        return atoms
        
    # atoms = minimize_structure(atoms)

    def run_ase_md(atoms, run_index=0,):
        
        atoms = minimize_structure(atoms, run_index=run_index, nsteps=20 if run_index == 0 else 10, fmax=0.0006 if run_index == 0 else 0.001)

        # Setup MD simulation
        temperature = args.temperature
        timestep_fs = args.timestep
        num_steps = args.nsteps_ase
        ase_atoms = atoms
        
        # Draw initial momenta
        # MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
        # Stationary(ase_atoms)  # Remove center of mass translation
        # ZeroRotation(ase_atoms)  # Remove rotations

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
            # Save current frame and keep track of energies
            frames[i] = ase_atoms.get_positions()
            potential_energy[i] = ase_atoms.get_potential_energy()
            kinetic_energy[i] = ase_atoms.get_kinetic_energy()
            total_energy[i] = ase_atoms.get_total_energy()
            traj.write(ase_atoms)
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

                # pycharmm.lingo.charmm_script("ENER")
                # xyz = pd.DataFrame(ase_atoms.get_positions(), columns=["x", "y", "z"])
                # coor.set_positions(xyz)
                # # pycharmm MM force field minimization
                # # minimize.run_abnr(nstep=10, tolenr=1e-5, tolgrd=1e-5)
                # pycharmm.lingo.charmm_script("ENER")
                # ase_atoms.set_positions(coor.get_positions())
                # _ = ase_opt.BFGS(atoms).run(fmax=0.01, steps=10)
                minimize_structure(ase_atoms, run_index=f"{run_index}_{breakcount}_{i}_", nsteps=20 if run_index == 0 else 10, fmax=0.0006 if run_index == 0 else 0.001)
                # assign new velocities
                # MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                cur_eng = ase_atoms.get_potential_energy()
                print(f"Current energy: {cur_eng:.6f} eV")
                Stationary(ase_atoms)
                ZeroRotation(ase_atoms)
                breakcount += 1
            if breakcount > 4:
                print("Maximum number of breaks reached")
                break
            # Occasionally print progress and adjust temperature
            if i % 10_000 == 0:
                temperature += 1
                Stationary(ase_atoms)
                ZeroRotation(ase_atoms)
                # MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temperature)
                # print(f"Temperature adjusted to: {temperature} K")
            if i % 1 == 0:
                print(f"step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}")

        
        # Close trajectory file
        traj.close()
        print(f"Trajectory saved to: {traj_filename}")
        print("ASE MD simulation complete!")

    for i in range(10):
        run_ase_md(atoms, run_index=i)

    def set_up_nhc_sim_routine(atoms, T=args.temperature, dt=5e-3, steps_per_recording=250):
        @jax.jit
        def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
            return _spherical_cutoff_calculator(
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

        BOXSIZE = float(1000)  # Increased from 45 to 100 Å for proper neighbor list allocation
        
        # Define molecular wrapping function to keep residues together
        def wrap_molecules(positions, n_atoms_per_monomer, n_monomers):
            """
            Wrap molecules to keep residues intact across periodic boundaries.
            
            This function ensures that when atoms cross periodic boundaries, 
            entire residues (monomers) are moved as a unit to maintain molecular
            integrity. This is crucial for energy functions that are sensitive
            to atomic permutations.
            
            Args:
                positions: Array of shape (n_atoms, 3) with atomic positions
                n_atoms_per_monomer: Number of atoms per monomer/residue
                n_monomers: Total number of monomers in the system
                
            Returns:
                Array of wrapped positions with same shape as input
            """
            wrapped_positions = []
            
            # Process each monomer separately
            for i in range(n_monomers):
                start_idx = i * n_atoms_per_monomer
                end_idx = (i + 1) * n_atoms_per_monomer
                monomer_positions = positions[start_idx:end_idx]
                
                # Calculate center of mass of the monomer
                com = jnp.mean(monomer_positions, axis=0)
                
                # Wrap center of mass to box using periodic boundary conditions
                wrapped_com = com - BOXSIZE * jnp.floor(com / BOXSIZE)
                
                # Apply the same translation to all atoms in the monomer
                # This keeps the monomer intact while moving it to the wrapped position
                translation = wrapped_com - com
                wrapped_monomer = monomer_positions + translation
                
                wrapped_positions.append(wrapped_monomer)
            
            return jnp.concatenate(wrapped_positions, axis=0)
        
        # Create wrapped energy function that applies molecular wrapping
        # This preserves momentum conservation while ensuring residue integrity
        @jit
        def wrapped_energy_fn(position, **kwargs):
            """
            Energy function that applies molecular wrapping before calculation.
            
            This function wraps molecular positions to maintain residue integrity
            while preserving the physics of the simulation. The wrapping is applied
            only during energy calculation, not during integration steps, which
            maintains proper momentum conservation.
            
            Args:
                position: Current atomic positions
                **kwargs: Additional arguments passed to the energy function
                
            Returns:
                Energy value with molecular wrapping applied
            """
            # Apply molecular wrapping before energy calculation
            wrapped_position = wrap_molecules(position, args.n_atoms_monomer, args.n_monomers)
            return jax_md_energy_fn(wrapped_position, **kwargs)
        
        displacement, shift = space.free()
        neighbor_fn = partition.neighbor_list(
            displacement, BOXSIZE, 12, format=partition.Dense
        )
        nbrs = neighbor_fn.allocate(R)

        unwrapped_init_fn, unwrapped_step_fn = jax_md.minimize.fire_descent(
            wrapped_energy_fn, shift, dt_start=0.001, dt_max=0.001
        )
        unwrapped_step_fn = jit(unwrapped_step_fn)


        @jit
        def sim(state, nbrs):
            def step(i, state_nbrs):
                state, nbrs = state_nbrs
                nbrs = nbrs.update(state.position)
                state = apply_fn(state, neighbor=nbrs)
                return (state, nbrs)

            return lax.fori_loop(0, steps_per_recording, step, (state, nbrs))




        # ========================================================================
        # SIMULATION PARAMETERS
        # ========================================================================
        # These parameters control the molecular dynamics simulation behavior
        
        timestep = 1e-5  # Time step in ps - reduced for better stability
        T_init = 100  # Initial temperature in Kelvin
        # Reduce initial temperature for momentum generation to account for unit scaling
        T_init_momentum = 300 / 37  # Scale down by the observed ratio
        pressure_init = 1.01325  # Target pressure in bars (atmospheric pressure) 

        chain = 3  # Number of chains in the Nose-Hoover chain.
        chain_steps = 2  # Number of steps per chain.
        sy_steps = 3

        # Dictionary with the NHC settings.
        new_nhc_kwargs = {
            'chain_length': chain, 
            'chain_steps': chain_steps, 
            'sy_steps': sy_steps
        }

        # Convert to metal unit system.
        unit = units.metal_unit_system()

        timestep = timestep * unit['time']
        T_init = T_init * unit['temperature']
        pressure = pressure_init * unit['pressure']

        rng_key = jax.random.PRNGKey(0)

        steps_per_recording = 25
        # Use the correct Boltzmann constant for JAX-MD
        # JAX-MD uses internal units, so we need to be careful about units
        # Let's use the standard value but check if JAX-MD expects different units
        K_B = 8.617333262e-5  # eV/K - standard value
        print(f"Using Boltzmann constant: {K_B} eV/K")
        dt = timestep 
        kT = T_init  # Use scaled temperature for momentum generation
        print(f"kT value for momentum initialization: {kT}")
        print(f"T_init value: {T_init}")
        print(f"unit['temperature']: {unit['temperature']}")

        displacement, shift = space.periodic(BOXSIZE, wrapped=True)
        # init_fn, apply_fn = simulate.nvt_nose_hoover(wrapped_energy_fn, shift, dt, kT)
        init_fn, apply_fn = simulate.nve(wrapped_energy_fn, shift, dt)
        apply_fn = jit(apply_fn)


        def run_sim(
            key, 
            total_steps=args.nsteps_jaxmd, 
            steps_per_recording=1,
            nbrs=nbrs,
            R=R
        ):
            total_records = total_steps // steps_per_recording

            # Center positions before minimization
            initial_pos = R - R.mean(axis=0)
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
                    print("NaN energy caught")
                    print(f"Simulation terminated: E={energy:.4f}eV")
                    
                    # TODO: fix nans, by going back to the previous state?
                    fire_state = unwrapped_step_fn(fire_positions[-1])
                    break
            # save pdb 
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            ase_io.write(f"{args.output_prefix}_minimized_{current_time}.pdb", atoms)

            # ========================================================================
            # PBC MINIMIZATION
            # ========================================================================
            # Additional minimization in periodic boundary conditions to ensure
            # proper molecular packing and eliminate any remaining artifacts
            
            print("*" * 10 + "\nPBC Minimization\n" + "*" * 10)
            
            # Set up PBC minimization using periodic space
            pbc_displacement, pbc_shift = space.periodic(BOXSIZE, wrapped=True)
            pbc_unwrapped_init_fn, pbc_unwrapped_step_fn = jax_md.minimize.fire_descent(
                wrapped_energy_fn, pbc_shift, dt_start=0.001, dt_max=0.001
            )
            pbc_unwrapped_step_fn = jit(pbc_unwrapped_step_fn)
            
            # Start PBC minimization from the current minimized position
            pbc_fire_state = pbc_unwrapped_init_fn(fire_state.position)
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
                    print("NaN energy caught in PBC minimization")
                    print(f"PBC minimization terminated: E={energy:.4f}eV")
                    pbc_fire_state = pbc_unwrapped_step_fn(pbc_fire_positions[-1])
                    break
            
            # Save PBC minimized structure
            pbc_current_time = datetime.now().strftime("%H:%M:%S")
            ase_io.write(f"{args.output_prefix}_pbc_minimized_{pbc_current_time}.pdb", atoms)
            print(f"PBC minimization complete. Final energy: {energy:.6f} eV")

            # NVT simulation
            nbrs = neighbor_fn.allocate(pbc_fire_state.position)
            # TODO: fix nans, by going back to the previous state?
            # TODO: piston mass?
            # For NVE simulations, we need to provide the correct mass units
            # JAX-MD expects masses to match position shape (20, 3) for momentum initialization
            # The init_fn needs expanded mass array to match position dimensions
            state = init_fn(key, pbc_fire_state.position, Si_mass_expanded, neighbor=nbrs)
            
            # Manual momentum initialization to achieve target temperature
            # JAX-MD's init_fn seems to generate momenta that are too large
            # Let's manually initialize momenta for 100 K target temperature
            target_temp = 100.0  # K
            kT_target = K_B * target_temp  # eV
            
            # Generate random momenta with correct temperature
            # p = sqrt(m * kT) * random.normal
            momentum_scale = jnp.sqrt(Si_mass_expanded * kT_target)
            random_momenta = jax.random.normal(key, pbc_fire_state.position.shape)
            scaled_momenta = momentum_scale * random_momenta
            
            # Create state with manually initialized momenta
            state = type(state)(
                position=pbc_fire_state.position,
                momentum=scaled_momenta,
                mass=Si_mass_expanded,
                force=jnp.zeros_like(pbc_fire_state.position)
            )
            print(f"Manually initialized momenta for target temperature: {target_temp} K")
            nhc_positions = []

            # get energy of initial state
            energy_initial = float(wrapped_energy_fn(state.position))
            print(f"Initial energy: {energy_initial:.6f} eV")

            print("*" * 10 + "\nNVT\n" + "*" * 10)
            print("\t\tTime (ps)\tEnergy (eV)\tTemperature (K)")
            
            # ========================================================================
            # MAIN SIMULATION LOOP
            # ========================================================================
            for i in range(total_records):
                # Run simulation steps
                state, nbrs = sim(state, nbrs)
                
                # Store current position for trajectory analysis
                nhc_positions.append(state.position)
                    
                # Print progress every 10 steps
                if i % 10 == 0:
                    time = i * steps_per_recording * dt
                    
                    # Debug: Print momentum and mass info for first few steps
                    if i < 30:
                        print(f"Debug - Step {i}:")
                        print(f"  Momentum shape: {state.momentum.shape}")
                        print(f"  Momentum sample: {state.momentum[0]}")
                        print(f"  Mass shape: {Si_mass_expanded.shape}")
                        print(f"  Mass sample: {Si_mass_expanded[0]}")
                        print(f"  K_B: {K_B}")
                    
                    # Calculate temperature using expanded mass array for proper broadcasting
                    temp_jaxmd = float(quantity.temperature(momentum=state.momentum, mass=Si_mass_expanded) / K_B)
                    
                    # Try manual temperature calculation as alternative
                    # T = (2/3) * <KE> / (N * kB) where KE = 0.5 * p^2 / m
                    # Use the correct mass array for temperature calculation
                    kinetic_energy = 0.5 * jnp.sum(state.momentum**2 / Si_mass_expanded)
                    n_atoms = len(Si_mass)
                    temp_manual = float(2.0 * kinetic_energy / (3.0 * n_atoms * K_B))
                    
                    # Use the manual calculation for now
                    temp = temp_manual
                    
                    if i < 30:  # Debug info
                        print(f"  JAX-MD temp: {temp_jaxmd:.2f} K")
                        print(f"  Manual temp: {temp_manual:.2f} K")
                    
                    # Calculate energy using wrapped positions to maintain residue integrity
                    energy = float(wrapped_energy_fn(state.position, neighbor=nbrs))
                    print(f"{time:10.2f}\t{energy:10.4f}\t{temp:10.2f}")
                    
                    # ========================================================================
                    # ERROR HANDLING AND SIMULATION MONITORING
                    # ========================================================================
                    
                    # Check for NaN energies (indicates numerical instability)
                    if jnp.isnan(energy):
                        print("NaN energy caught")
                        print(f"Simulation terminated: E={energy:.4f}eV")
                        # Only try to restore previous position if we have positions stored
                        if len(nhc_positions) > 1:
                            nhc_positions = nhc_positions[:-1]
                            # Create new state with previous position
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
                nhc_positions_out.append(jax_md.space.transform(box=BOXSIZE, R=R))
            return steps_completed, jnp.stack(nhc_positions_out)

        return run_sim


    def save_trajectory(out_positions, atoms, filename="nhc_trajectory", format="traj"):
        trajectory = Trajectory(f"{filename}.{format}", "a")
        # atoms = ase.Atoms()
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

    
    # Main JAXMD simulation loop
    for j in range(1):
        sim_key, data_key = jax.random.split(data_key, 2)
        s = set_up_nhc_sim_routine(atoms)
        out_positions, _ = run_sim_loop(s, sim_key)

        for i in range(len(out_positions)):
            traj_filename = f'{args.output_prefix}_md_trajectory_{j}_{i}.traj'
            print(f"Saving trajectory to: {traj_filename}")
            save_trajectory(out_positions[i], atoms, filename=traj_filename)

        # atoms = minimize_structure(atoms)

    print("Trajectories saved!")


    print("JAX-MD simulation complete!")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())

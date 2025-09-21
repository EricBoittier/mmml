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
    parser.add_argument(
        "--nsteps",
        type=int,
        default=10000,
        help="Number of steps to run (default: 10000).",
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
    except ModuleNotFoundError as exc:
        sys.exit(f"Required modules not available: {exc}")

    pdbfilename = str(args.pdbfile)
    
    # Setup box and load PDB
    setup_box_generic(pdbfilename, side_length=1000)
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
    # spherical_cutoff_calculator(
    #                     positions=R,
    #                     atomic_numbers=Z,
    #                     n_monomers=self.n_monomers,
    #                     cutoff_params=self.cutoff_params,
    #                     doML=self.doML,
    #                     doMM=self.doMM,
    #                     doML_dimer=self.doML_dimer,
    #                     debug=self.debug,
    print(f"Hybrid calculator created: {hybrid_calc}")
    atoms = pdb_ase_atoms
    print(f"ASE atoms: {atoms}")
    atoms.calc = hybrid_calc
    
    # Get initial energy and forces
    hybrid_energy = float(atoms.get_potential_energy())
    hybrid_forces = np.asarray(atoms.get_forces())
    print(f"Initial energy: {hybrid_energy:.6f} eV")
    
    # Minimize structure if requested
    # if args.minimize_first:
    def minimize_structure(atoms):
        print("Minimizing structure with hybrid calculator")
        _ = ase_opt.BFGS(atoms).run(fmax=0.05, steps=10)
        
        # Sync with PyCHARMM
        xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(xyz)
        print(f"PyCHARMM coordinates after ASE minimization: {coor.show()}")
        
        # Additional PyCHARMM minimization
        minimize.run_abnr(nstep=4000, tolenr=1e-6, tolgrd=1e-6)
        pycharmm.lingo.charmm_script("ENER")
        print(f"PyCHARMM coordinates after PyCHARMM minimization: {coor.show()}")
        
        # Final ASE minimization
        atoms.set_positions(coor.get_positions())
        _ = ase_opt.BFGS(atoms).run(fmax=0.0001, steps=100)
        print("Minimization complete")
        return atoms
        
    atoms = minimize_structure(atoms)

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
    traj_filename = f'{args.output_prefix}_trajectory_{temperature}K_{num_steps}steps.traj'
    traj = ase_io.Trajectory(traj_filename, 'w')

    # Run molecular dynamics
    frames = np.zeros((num_steps, len(ase_atoms), 3))
    potential_energy = np.zeros((num_steps,))
    kinetic_energy = np.zeros((num_steps,))
    total_energy = np.zeros((num_steps,))

    breakcount = 0
    for i in range(1000):
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

    
    # Close trajectory file
    traj.close()
    print(f"Trajectory saved to: {traj_filename}")
    print("ASE MD simulation complete!")

    

    import jax_md
    # JAX-MD imports
    from jax_md import space, smap, energy, quantity, simulate, partition, units
    from ase.units import _amu
    Si_mass = ase_atoms.get_masses() * _amu
    # Si_mass = Si_mass.sum()
    Si_mass = 2.81086E-3
    print(f"Si_mass: {Si_mass}")
    import jax, e3x
    from jax import jit, grad, lax, ops, random
    import jax.numpy as jnp
    from ase.io import Trajectory

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

        TESTIDX = 0
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
        # atomic_numbers = test_data["Z"][TESTIDX]
        # position = R = test_data["R"][TESTIDX]
        atomic_numbers = atoms.get_atomic_numbers()
        R = position = atoms.get_positions()

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
        
        jax_md_grad_fn = jax.grad(jax_md_energy_fn)

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

        BOXSIZE = 30
        displacement, shift = space.free()
        neighbor_fn = partition.neighbor_list(
            displacement, BOXSIZE, 30 / 2, format=partition.Sparse
        )
        nbrs = neighbor_fn.allocate(R)
        unwrapped_init_fn, unwrapped_step_fn = jax_md.minimize.fire_descent(
            jax_md_energy_fn, shift, dt_start=0.001, dt_max=0.001
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

        # Ecatch = test_data["E"].min() * 1.05
        steps_per_recording = 25
        Ecatch = -2000

        K_B = 8.617e-5
        dt = 0.5e-4
        kT = K_B * T

        init_fn, apply_fn = simulate.nvt_nose_hoover(jax_md_energy_fn, shift, dt, kT)
        apply_fn = jit(apply_fn)


        def run_sim(
            key, 
            test_idx, 
            e_catch, 
            t_fact=5, 
            total_steps=args.nsteps, 
            steps_per_recording=100,
            nbrs=nbrs
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
                    energy = float(jax_md_energy_fn(fire_state.position))
                    max_force = float(jnp.abs(jax_md_grad_fn(fire_state.position)).max())
                    print(f"{i}/{NMIN}: E={energy:.6f} eV, max|F|={max_force:.6f}")


                # check for nans
                if jnp.isnan(energy):
                    print("NaN energy caught")
                    print(f"Simulation terminated: E={energy:.4f}eV")
                    break
            # save pdb 
            ase_io.write(f"{args.output_prefix}_minimized.pdb", atoms)

            # NVT simulation
            nbrs = neighbor_fn.allocate(fire_state.position)
            state = init_fn(key, fire_state.position, 2.91086e-3, neighbor=nbrs)
            nhc_positions = []

            # get energy of initial state
            energy_initial = float(jax_md_energy_fn(state.position))
            print(f"Initial energy: {energy_initial:.6f} eV")

            print("*" * 10 + "\nNVT\n" + "*" * 10)
            print("\t\tTime (ps)\tEnergy (eV)\tTemperature (K)")
            
            for i in range(total_records):
                state, nbrs = sim(state, nbrs)
                nhc_positions.append(state.position)
                
                if i % 50 == 0:
                    time = i * steps_per_recording * dt
                    temp = float(quantity.temperature(momentum=state.momentum, mass=Si_mass) / K_B)
                    energy = float(jax_md_energy_fn(state.position, neighbor=nbrs))
                    
                    print(f"{time:10.2f}\t{energy:10.4f}\t{temp:10.2f}")
                    
                    # # Check for simulation stability
                    # if temp > T * t_fact:
                    #     print("Temperature catch reached")
                    #     print(f"Simulation terminated: T={temp:.2f}K, E={energy:.4f}eV")
                    #     break

                    # check for nans
                    if jnp.isnan(energy):
                        print("NaN energy caught")
                        print(f"Simulation terminated: E={energy:.4f}eV")
                        break

                    # check for energy spikes
                    if energy > energy_initial*0.95:
                        print("Energy spike caught")
                        print(f"Simulation terminated: E={energy:.4f}eV")
                        break
                    
                    if energy < energy_initial*1.95:
                        print("Energy catch reached")
                        print(f"Simulation terminated: T={temp:.2f}K, E={energy:.4f}eV")
                        break

            steps_completed = i * steps_per_recording
            print(f"\nSimulated {steps_completed} steps ({steps_completed * dt:.2f} ps)")
            
            return steps_completed, jnp.stack(nhc_positions)

        return run_sim


    def save_trajectory(out_positions, atoms, filename="nhc_trajectory", format="traj"):
        trajectory = Trajectory(f"{filename}.{format}", "a")
        # atoms = ase.Atoms()
        out_positions = out_positions.reshape(-1, len(atoms),3)
        for R in out_positions:
            atoms.set_positions(R)
            trajectory.write(atoms)
        trajectory.close()


    def run_sim_loop(run_sim, sim_key, indices, Ecatch):
        """
        Run the simulation for the given indices and save the trajectory.
        """
        out_positions = []
        max_is = []
        for i in indices:
            print("test data", i)
            mi, pos = run_sim(sim_key, i, Ecatch)
            out_positions.append(pos)
            max_is.append(mi)

        return out_positions, max_is


    sim_key, data_key = jax.random.split(jax.random.PRNGKey(42), 2)

    s = set_up_nhc_sim_routine(atoms)

    for j in range(10):
        out_positions, _ = run_sim_loop(s, sim_key, np.arange(1), -1000)

        for i in range(len(out_positions)):
            traj_filename = f'{args.output_prefix}_md_trajectory_{j}_{i}.traj'
            print(f"Saving trajectory to: {traj_filename}")
            save_trajectory(out_positions[i], atoms, filename=traj_filename)

        atoms = minimize_structure(atoms)

    print("Trajectories saved!")


    print("JAX-MD simulation complete!")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())

# Standard library imports
import os
# Environment setup
os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# If you want to perform simulations in float64 you have to call this before any JAX compuation
# jax.config.update('jax_enable_x64', True)
# Add custom path
import sys
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test")

from itertools import combinations, permutations, product
from pathlib import Path
from typing import Dict, Tuple, List, Any, NamedTuple

# Third-party imports
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array, jit, grad, lax, ops, random
import optax
import orbax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read as ase_read

import ase.calculators.calculator as ase_calc

# JAX-MD imports
from jax_md import space, smap, energy, minimize, quantity, simulate, partition, units

# Local imports
import e3x
import physnetjax
from physnetjax.data.data import prepare_datasets
from physnetjax.training.loss import dipole_calc
from physnetjax.models.model import EF
from physnetjax.training.training import train_model
from physnetjax.data.batches import _prepare_batches as prepare_batches
from physnetjax.calc.helper_mlp import get_ase_calc
from physnetjax.data.read_ase import save_traj_to_npz
from physnetjax.restart.restart import get_last, get_files, get_params_model
from physnetjax.analysis.analysis import plot_stats




# Check JAX configuration
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())


sys.path.append("/pchem-data/meuwly/boittier/home/dcm-lj-data")
from pycharmm_lingo_scripts import script1, script2, script3, load_dcm


orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)




def set_up_model(restart, last=True, n_atoms=16):
    if last:
        lastrestart = get_last(restart)
        params, dimer_model = get_params_model(lastrestart)
    else:
        params, dimer_model = get_params_model(restart)
    dimer_model.natoms = n_atoms
    return params, dimer_model


def set_up_nhc_sim_routine(params, model, test_data, atoms):
    @jax.jit
    def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
        return model.apply(
            params,
            atomic_numbers=atomic_numbers,
            positions=positions,
            dst_idx=dst_idx,
            src_idx=src_idx,
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
        return result["energy"].reshape(-1)[0]
    
    jax_md_grad_fn = jax.grad(jax_md_energy_fn)
    BOXSIZE = 100
    displacement, shift = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement, None, 30 / 2, format=partition.Sparse
    )
    nbrs = neighbor_fn.allocate(R)
    unwrapped_init_fn, unwrapped_step_fn = minimize.fire_descent(
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

    Ecatch = test_data["E"].min() * 1.05
    steps_per_recording = 25

    K_B = 8.617e-5
    dt = 5e-3
    T = 100
    kT = K_B * T

    init_fn, apply_fn = simulate.nvt_nose_hoover(jax_md_energy_fn, shift, dt, kT)
    apply_fn = jit(apply_fn)


    def run_sim(
        key, 
        test_idx, 
        e_catch, 
        t_fact=5, 
        total_steps=100000, 
        steps_per_recording=250
    ):
        total_records = total_steps // steps_per_recording

        # Center positions before minimization
        initial_pos = R - R.mean(axis=0)
        fire_state = unwrapped_init_fn(initial_pos)
        fire_positions = []

        # FIRE minimization
        print("*" * 10 + "\nMinimization\n" + "*" * 10)
        for i in range(10000):
            fire_positions.append(fire_state.position)
            fire_state = unwrapped_step_fn(fire_state)
            
            if i % (10000 // 10) == 0:
                energy = float(jax_md_energy_fn(fire_state.position))
                max_force = float(jnp.abs(jax_md_grad_fn(fire_state.position)).max())
                print(f"{i}/{10000}: E={energy:.6f} eV, max|F|={max_force:.6f}")

        # NVT simulation
        state = init_fn(key, fire_state.position, 2.91086e-3, neighbor=nbrs)
        nhc_positions = []

        print("*" * 10 + "\nNVT\n" + "*" * 10)
        print("\t\tTime (ps)\tEnergy (eV)\tTemperature (K)")
        
        for i in range(total_records):
            state, nbrs = sim(state, nbrs)
            nhc_positions.append(state.position)
            
            if i % 100 == 0:
                time = i * steps_per_recording * dt
                temp = float(quantity.temperature(state.momentum, 2.91086e-3) / K_B)
                energy = float(jax_md_energy_fn(state.position, neighbor=nbrs))
                
                print(f"{time:10.2f}\t{energy:10.4f}\t{temp:10.2f}")
                
                # Check for simulation stability
                if temp > T * t_fact or energy < e_catch:
                    print(f"Simulation terminated: T={temp:.2f}K, E={energy:.4f}eV")
                    break

        steps_completed = i * steps_per_recording
        print(f"\nSimulated {steps_completed} steps ({steps_completed * dt:.2f} ps)")
        
        return steps_completed, jnp.stack(nhc_positions)

    return run_sim


def save_trajectory(out_positions, atoms, filename="nhc_trajectory", format="xyz"):
    trajectory = Trajectory(f"{filename}.{format}", "a")
    for R in out_positions[0]:
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


def args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--restart", type=str, default=None)
    parser.add_argument("--Ecatch", type=float, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sim_key", type=int, default=None)
    parser.add_argument("--n_atoms", type=int, default=16)
    return parser.parse_args()


def check_args(args):
    if args.restart is None:
        raise ValueError("restart is required")
    if args.data_path is None:
        raise ValueError("data_path is required")



def main():
    args = args_parser()
    data = np.load(args.data_path)
    if args.Ecatch is None:
        args.Ecatch = data["E"].min() * 1.05
    if args.sim_key is None:
        args.sim_key = jax.random.PRNGKey(42)
    else:
        args.sim_key = jax.random.PRNGKey(args.sim_key)
    check_args(args)
    params, model = set_up_model(args.restart)
    data = np.load(args.data_path)
    R = data["R"][0]
    Z = data["Z"]
    print("Z", Z.shape)
    print("Z", Z)
    print("R", R.shape)
    print("R", R)
    if len(Z) != args.n_atoms:
        Z = data["Z"][0]
    if len(Z) != len(R):
        raise ValueError("Z and R must have the same length")
    import ase
    atoms = ase.Atoms(Z,R)
    run_sim = set_up_nhc_sim_routine(params, model, data, atoms)
    # run the simulation
    out_positions, max_is = run_sim_loop(run_sim, args.sim_key, args.indices, args.Ecatch)

    print("Trajectories ran from ", max_is.min(), " to ", max_is.max(), " NHC cycles")
    # save the trajectory
    for i in range(len(out_positions)):
        save_trajectory(
            out_positions[i], atoms, filename=f"nhc_trajectory_{i}", format="xyz"
        )

if __name__ == "__main__":
    main()

# Usage
# python jaxmdInterface.py --restart RESTART_FILE --data_path DATA_PATH --n_atoms N_ATOMS --indices INDICES --Ecatch E_CATCH --sim_key SIM_KEY
# example values:
# RESTART_FILE = "restart_file.chk"
# restart = "/pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-81b5843f-e937-48c5-a741-fe74f5312ebf"
# data_path = "/pchem-data/meuwly/boittier/home/asecalcs/sim_t_293.15_k_rho_1044.3_kgperm3_pNone_kPa.npz"
# DATA_PATH = "data.npz"
# N_ATOMS = 16
# INDICES = [0]
# E_CATCH = 0.0001
# SIM_KEY = 42
# # full usage
# python jaxmdInterface.py --restart /pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/dichloromethane-81b5843f-e937-48c5-a741-fe74f5312ebf --data_path /pchem-data/meuwly/boittier/home/asecalcs/sim_t_293.15_k_rho_1044.3_kgperm3_pNone_kPa.npz --n_atoms 16 --indices 0 --sim_key 42
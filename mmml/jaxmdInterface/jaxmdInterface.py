from itertools import combinations, permutations, product
from typing import Dict, Tuple, List, Any, NamedTuple

import jax
# If you want to perform simulations in float64 you have to call this before any JAX compuation
# jax.config.update('jax_enable_x64', True)

import jax
import jax.numpy as jnp
from jax import Array
import os

os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
from jax import jit
import jax.numpy as jnp
import ase.calculators.calculator as ase_calc
# from jax import config
# config.update('jax_enable_x64', True)

# Check JAX configuration
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())

import sys
import e3x
import jax
import numpy as np
import optax
import orbax
from pathlib import Path
import pandas as pd

# Add custom path
sys.path.append("/pchem-data/meuwly/boittier/home/pycharmm_test")
import physnetjax

sys.path.append("/pchem-data/meuwly/boittier/home/dcm-lj-data")
from pycharmm_lingo_scripts import script1, script2, script3, load_dcm

from physnetjax.data.data import prepare_datasets
from physnetjax.training.loss import dipole_calc
from physnetjax.models.model import EF
from physnetjax.training.training import train_model  # from model import dipole_calc
from physnetjax.data.batches import (
    _prepare_batches as prepare_batches,
)  # prepare_batches, prepare_datasets

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

data_key, train_key = jax.random.split(jax.random.PRNGKey(42), 2)

from pathlib import Path

from physnetjax.calc.helper_mlp import get_ase_calc


from physnetjax.data.read_ase import save_traj_to_npz



from jax_md import partition
from jax_md import space
import jax.numpy as np
from jax import random
from jax import jit
from jax import lax
from jax import ops

import time

from jax_md import space, smap, energy, minimize, quantity, simulate

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import jax_md
import numpy as np

from ase.io import read as ase_read
from jax_md import units
from typing import Dict



import time
from jax_md import minimize

from physnetjax.restart.restart import get_last, get_files, get_params_model
from physnetjax.analysis.analysis import plot_stats

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
    atomic_numbers = test_data["Z"][TESTIDX]
    position = R = test_data["R"][TESTIDX]
    @jit
    def jax_md_energy_fn(position, **kwargs):
    l_nbrs = nbrs.update(jnp.array(position))
    _ = evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=position,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
    return _["energy"].reshape(-1)[0]
    jax_md_grad_fn = jax.grad(jax_md_energy_fn)
    BOXSIZE = 100
    # displacement, shift = space.periodic(BOXSIZE, wrapped=False)
    displacement, shift = space.free()
    neighbor_fn = partition.neighbor_list(displacement, None, 30/2, format=partition.Sparse)
    nbrs = neighbor_fn.allocate(R)
    unwrapped_init_fn, unwrapped_step_fn = minimize.fire_descent(jax_md_energy_fn, shift, dt_start=0.001, dt_max=0.001)
    unwrapped_step_fn = jit(unwrapped_step_fn)

    @jit
    def sim(state, nbrs):
    def step(i, state_nbrs):
        state, nbrs = state_nbrs
        nbrs = nbrs.update(state.position)
        return apply_fn(state, neighbor=nbrs), nbrs
    return lax.fori_loop(0, steps_per_recording, step, (state, nbrs))


    Ecatch = test_data["E"].min() * 1.05
    steps_per_recording = 25

    K_B = 8.617e-5
    dt = 5e-3
    T = 100
    kT = K_B * T     

    init_fn, apply_fn = simulate.nvt_nose_hoover(jax_md_energy_fn, shift, dt, kT)
    apply_fn = jit(apply_fn)



    def run_sim(key, TESTIDX, Ecatch, nbrs, TFACT = 5, total_steps = 100000, steps_per_recording = 250):

        total_records = total_steps // steps_per_recording
        # Define the simulation.

        Si_mass = 2.91086E-3

        fire_state = unwrapped_init_fn(R - R.T.mean(axis=1).T)
        fire_positions = []
        
        N = 10000
        print("*"*10)
        print("Minimization")
        print("*"*10)
        for i in range(N):
        fire_positions += [fire_state.position]
        fire_state = jit(unwrapped_step_fn)(fire_state)
        if (i) % int(N//10) == 0:
            print(i, "/",
                    N, 
                    float(jax_md_energy_fn(fire_state.position)), 
                    float(np.abs(np.array(jax_md_grad_fn(fire_state.position))).max()))
        
        state = init_fn(key, fire_state.position, Si_mass, neighbor=nbrs)
        nhc_positions = []
        
        print("*"*10)
        print("NVT")
        print("*"*10)
        # Run the simulation.
        print('\t\tEnergy (eV)\tTemperature (K)')
        for i in range(total_records):
        state, nbrs = sim(state, nbrs)
        nhc_positions += [state.position]
        if (i-1) % 100 == 0:
            iT = float(quantity.temperature(momentum=state.momentum, mass=Si_mass) / K_B)
            iE = float(jax_md_energy_fn(state.position, neighbor=nbrs))
            print(i*steps_per_recording*dt, "ps", 100*i/total_records, "% ={ " , '{:.02f}\t\t\t{:.02f}'.format(
                iE , iT))
            if iT > T*TFACT:
                print("ERROR! bailing!")
                print("T", iT, T*TFACT, "E",  iE, Ecatch)
                break
            if iE < Ecatch:
                print("ERROR! bailing!")
                print("T", iT, T*TFACT, "E",  iE, Ecatch)
                break
            
        print(f"Simulated (NVT, NHC) {i} steps at dt {dt * 1000} (fs)")
        nhc_positions = np.stack(nhc_positions)

        return i*steps_per_recording, nhc_positions

    return run_sim

def save_trajectory(out_positions, atoms, filename="nhc_trajectory", format="xyz"):
    trajectory = Trajectory(f"{filename}.{format}", "a")
    for R in out_positions[0]:
        atoms.set_positions(R)
        trajectory.write( atoms)
    trajectory.close()


def run_sim(indices, Ecatch, nbrs):
    """
    Run the simulation for the given indices and save the trajectory.
    """
    out_positions = []
    max_is = []
    for i in indices:
        print("test data", i)
        mi, pos = run_sim(i, Ecatch, nbrs)[:5]
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

def main():
    args = args_parser()
    data = np.load(args.data_path)
    if args.Ecatch is None:
        args.Ecatch = data["E"].min() * 1.05
    if args.sim_key is None:
        args.sim_key = jax.random.PRNGKey(42)
    else:
        args.sim_key = jax.random.PRNGKey(args.sim_key)

    params, model = set_up_model(args.restart)
    run_sim = set_up_nhc_sim_routine(params, model, data, atoms)
    out_positions, max_is = run_sim(args.sim_key, args.indices, args.Ecatch, nbrs)

    print("Trajectories ran from ", max_is.min(), " to ", max_is.max(), " NHC cycles")
    # save the trajectory
    for i in range(len(out_positions)):
        save_trajectory(out_positions[i], atoms, filename=f"nhc_trajectory_{i}", format="xyz")












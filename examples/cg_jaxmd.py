#!/usr/bin/env python
# coding: utf-8

"""Example script demonstrating a hybrid ML/MM simulation workflow using JAX-MD.

By bypassing ASE and running the optimization and molecular dynamics entirely
within JAX-MD, we achieve significantly faster execution times through full-program
compilation (JIT) and hardware acceleration (GPU/TPU).
"""

from pathlib import Path
import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad

# ASE trajectory, calculator, and Atoms imports
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

# JAX-MD imports
from jax_md import space, minimize, simulate

# PyCHARMM lingo imports
import pycharmm.lingo as lingo

import mmml
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    ensure_pycharmm_loaded,
    CGENFF_PRM,
    pycharmm_loud,
    coor,
)
from mmml.interfaces.pycharmmInterface.trialanine_water_box import build_trialanine_water_box_in_charmm
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    set_charmm_positions,
    setup_nonbonded_only_charmm,
)
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    load_nonbonded_system_from_charmm,
    nonbonded_energy_and_forces,
    _build_pair_indices,
    resolve_nonbonded_excluded_pairs,
)
from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import _nbond_settings_from_cutoffs
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_loud
from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.data.units import KCAL_MOL_TO_EV

# 1. Initialize JAX and PyCHARMM configuration
jax.config.update("jax_enable_x64", True)
ensure_pycharmm_loaded()
pycharmm_loud()

# Path to the pretrained neural network checkpoint parameters
CKPT_PATH = "params_aaa_long_2026-07-04_22-30-27.json"

# 2. Build the initial system in PyCHARMM and minimize
print("--- Building Trialanine Water Box in CHARMM ---")
workdir = Path('/tmp/tria_box')
box = build_trialanine_water_box_in_charmm(n_waters=200, box_side_A=28.0, seed=11, workdir=workdir)

pos = np.asarray(box.positions, dtype=np.float64)
pos = np.random.uniform(-1.0, 1.0, pos.shape) + pos

# translate peptide to the middle of the box (L/2, L/2, L/2)
n_trialanine = 42
peptide_np = np.array(pos[n_trialanine:])
pos[n_trialanine:] -= peptide_np.mean(axis=0)
pos[n_trialanine:] += np.array([box.box_side_A / 2, box.box_side_A / 2, box.box_side_A / 2])

set_charmm_positions(pos)

setup_nonbonded_only_charmm()


for i in range(10):
    # Apply constraint and run steepest descent minimization
    lingo.charmm_script("CONStraint DROPlet FORC 0.01 EXPO 4")
    run_charmm_script_loud("MINI SD 10000")
    lingo.charmm_script("IMAGE")
    lingo.charmm_script("CONStraint DROPlet")
    run_charmm_script_loud("MINI SD 10000")



run_charmm_script_loud("MINI ABNR 10000")

# Retrieve positions and atomic numbers
pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
z = get_Z_from_psf()

# Construct ASE Atoms object to act as a template for trajectory writing
atoms = Atoms(z, pos)
atoms.set_cell(box.cell)
atoms.set_pbc(True)

# Define box size and cell
box_size = float(box.box_side_A)
cell = np.asarray(box.cell, dtype=np.float64)

# 3. Load JAX nonbonded parameters and setup cutoffs
psf_path = box.psf_path
prm_path = CGENFF_PRM
nb_settings = _nbond_settings_from_cutoffs(box.nbond_cutoffs)
nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)

calc = create_calculator_from_checkpoint(CKPT_PATH)
model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))

if model is None or params is None:
    raise ValueError("Could not extract model or params from the loaded calculator.")

# Setup monomer groupings (first 42 atoms: peptide; then groups of 3 for water)
monomer_indices = [np.arange(n_trialanine)]
for i in range(n_trialanine, len(pos), 3):
    monomer_indices.append(np.arange(i, i + 3))

# Pre-convert to JAX arrays for performance
jax_monomer_indices = [jnp.array(idx, dtype=jnp.int32) for idx in monomer_indices]
jax_z = jnp.array(z, dtype=jnp.int32)

# Set up molecule IDs for nonbonded interactions
molecule_id = np.empty(len(pos), dtype=np.int32)
molecule_id[:n_trialanine] = 0
for mol_id, start in enumerate(range(n_trialanine, len(pos), 3), start=1):
    molecule_id[start:start + 3] = mol_id
jax_molecule_id = jnp.array(molecule_id, dtype=jnp.int32)

# Precompute pair list indices for nonbonded interactions to avoid JIT TracerArrayConversionError
print("--- Precomputing nonbonded pair list indices ---")
excluded_pairs = nbond_data.excluded_pairs
if nbond_data.psf_path is not None and nbond_data.psf_bonds is not None:
    excluded_pairs = resolve_nonbonded_excluded_pairs(
        nbond_data.psf_path,
        nbond_data.psf_bonds,
        natom=int(np.asarray(nbond_data.charges).shape[0]),
    )
pair_i, pair_j = _build_pair_indices(pos, cell, excluded_pairs, nb_settings.cutnb)
jax_pair_i = jnp.array(pair_i, dtype=jnp.int32)
jax_pair_j = jnp.array(pair_j, dtype=jnp.int32)

# 5. Define displacement and shift functions for Periodic Boundary Conditions (JAX-MD)
# Using a cubic box setup based on the cell size.
displacement_fn, shift_fn = space.periodic(box_size)

# 6. Build the Hybrid JAX-MD Energy Function
# Using a factory/constructor pattern with jax.vmap to batch identical monomers.
# This prevents JAX from unrolling python loops, enabling fast compilation and execution.
def make_monomer_energy_fn(model, params, jax_z, monomer_indices):
    from collections import defaultdict
    # Group monomer indices by their size (atom count)
    by_size = defaultdict(list)
    for idx in monomer_indices:
        by_size[len(idx)].append(idx)
        
    group_fns = []
    for size, indices_list in by_size.items():
        # Stack indices to shape (N_monomers, size) for batched gathering
        stacked_indices = jnp.stack(indices_list)
        
        # Build pairwise indices for PhysNet sparse model input
        n_atoms = size
        dst_idx, src_idx = np.where(~np.eye(n_atoms, dtype=bool))
        dst_idx = jnp.array(dst_idx, dtype=jnp.int32)
        src_idx = jnp.array(src_idx, dtype=jnp.int32)
        
        # Vectorize model.apply over the batch of monomers (first dimension)
        vmapped_apply = jax.vmap(
            lambda pos, atomic_nums: model.apply(
                params,
                atomic_numbers=atomic_nums,
                positions=pos,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )["energy"],
            in_axes=(0, 0)
        )
        
        # Static atomic numbers for all monomers of this size
        group_z = jax_z[stacked_indices]
        
        def group_energy(r, stacked_idx=stacked_indices, gz=group_z):
            # Gather coordinates: shape (N_monomers, size, 3)
            group_pos = r[stacked_idx]
            energies = vmapped_apply(group_pos, gz)
            return jnp.sum(energies)
            
        group_fns.append(group_energy)
        
    def total_monomer_energy(r):
        return sum(fn(r) for fn in group_fns)
        
    return total_monomer_energy

compute_monomer_energy = make_monomer_energy_fn(model, params, jax_z, jax_monomer_indices)


def hybrid_energy_fn(r: jnp.ndarray) -> jnp.ndarray:
    """Combines intramolecular (ML) and intermolecular (MM via JAX nonbonded) energy."""
    # (A) Intramolecular terms from ML potential
    e_intra = compute_monomer_energy(r)
    
    # (B) Intermolecular nonbonded terms from JAX-MM system
    # We call nonbonded_energy_and_forces but only need the energy.
    terms_raw, _ = nonbonded_energy_and_forces(
        r,
        nbond_data,
        cell,
        nb_settings,
        molecule_id=jax_molecule_id,
        pair_i=jax_pair_i,
        pair_j=jax_pair_j,
    )
    e_inter = terms_raw.get("total", sum(terms_raw.values())) * KCAL_MOL_TO_EV
    
    return e_intra + e_inter

# Create JIT compiled energy and force functions
energy_fn = jit(hybrid_energy_fn)
force_fn = jit(grad(lambda r: -hybrid_energy_fn(r)))

# 7. Structure Minimization with JAX-MD FIRE
print("--- Minimizing System with JAX-MD FIRE ---")
init_r = jnp.array(pos, dtype=jnp.float64)

# Initialize FIRE minimizer with a stable initial step size (0.001 ps = 1 fs)
fire_init, fire_step = minimize.fire_descent(hybrid_energy_fn, shift_fn, dt_start=0.001, dt_max=0.001)
fire_state = fire_init(init_r)

# Perform minimization steps
for step in range(200):
    fire_state = fire_step(fire_state)
    if step % 20 == 0:
        curr_e = energy_fn(fire_state.position)
        print(f"FIRE Step {step:3d} | Energy: {curr_e:.4f} eV")

min_r = fire_state.position
print(f"Minimization complete. Final Energy: {energy_fn(min_r):.4f} eV")

# 8. Molecular Dynamics (NVT Nose-Hoover) with JAX-MD
print("--- Running NVT Nose-Hoover Dynamics with JAX-MD ---")
from jax_md import quantity

# Define simulation conditions
temperature = 200.0  # Kelvin
kb = 8.617333262145e-5  # eV/K (Boltzmann constant in eV/K)
target_temp_ev = temperature * kb
dt_fs = 0.25  # time step in femtoseconds
dt = dt_fs * 0.001  # convert to picoseconds (JAX-MD metal units)

# Setup NHC simulator
# We define particle masses (in AMU)
mass = np.zeros(len(pos))
mass[:n_trialanine] = 12.0  # average mass approximation for peptide
mass[n_trialanine::3] = 16.0  # Oxygen
mass[n_trialanine+1::3] = 1.0  # Hydrogen
mass[n_trialanine+2::3] = 1.0  # Hydrogen
jax_mass = jnp.array(mass, dtype=jnp.float64)

# Initialize simulator state
# nvt_nose_hoover expects positional arguments: energy_or_force_fn, shift_fn, dt, kT
init_fn, step_fn = simulate.nvt_nose_hoover(hybrid_energy_fn, shift_fn, dt, target_temp_ev)
# Using random key to assign initial velocities and initialize NVT Nose-Hoover state
key = jax.random.PRNGKey(42)
state = init_fn(key, min_r, mass=jax_mass)

# JIT-compile the step function for speed
step_fn = jit(step_fn)

# Run dynamics loop
traj_path = "cg_md.traj"
print(f"--- Running dynamics and saving trajectory to {traj_path} ---")
traj = Trajectory(traj_path, "w", atoms)

for step in range(50000):
    state = step_fn(state)
    if step % 100 == 0:
        pos_np = np.asarray(state.position)
        curr_e = float(energy_fn(state.position))
        curr_f = np.asarray(force_fn(state.position))
        
        # Calculate instantaneous kinetic energy and temperature using JAX-MD quantities
        ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
        temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
        print(f"MD Step {step:5d} | Tot Energy: {curr_e + ke:.4f} eV | Temp: {temp:.1f} K")
        
        # Save frame to trajectory with computed energy and forces
        frame = atoms.copy()
        frame.set_positions(pos_np)
        frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
        traj.write(frame)

traj.close()
print(f"JAX-MD Simulation complete! Trajectory saved to {traj_path}")


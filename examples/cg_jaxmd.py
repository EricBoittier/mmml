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
box = build_trialanine_water_box_in_charmm(n_waters=200, box_side_A=38.0, seed=11, workdir=workdir)

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
    lingo.charmm_script("CONStraint DROPlet FORC 0.001 EXPO 4")
    lingo.charmm_script("MINI SD 10000")
    lingo.charmm_script("IMAGE")
    lingo.charmm_script("CONStraint DROPlet")
    lingo.charmm_script("MINI SD 10000")



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
def make_monomer_energy_fn(model, params, jax_z, monomer_indices, displacement_fn):
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
        
        # Vectorized displacement to calculate relative coordinates under PBC
        vmapped_displacement = jax.vmap(
            jax.vmap(displacement_fn, in_axes=(0, None)),
            in_axes=(0, 0)
        )
        
        def group_energy(r, stacked_idx=stacked_indices, gz=group_z):
            # Gather coordinates: shape (N_monomers, size, 3)
            group_pos = r[stacked_idx]
            
            # Unfold coordinate images relative to the first atom of each monomer
            ref_pos = group_pos[:, 0, :]
            displacements = vmapped_displacement(group_pos, ref_pos)
            unfolded_pos = ref_pos[:, None, :] + displacements
            
            energies = vmapped_apply(unfolded_pos, gz)
            return jnp.sum(energies)
            
        group_fns.append(group_energy)
        
    def total_monomer_energy(r):
        return sum(fn(r) for fn in group_fns)
        
    return total_monomer_energy

# Option: Treat peptide-water intermolecular interactions with ML instead of MM
PEPTIDE_WATER_ML = False

def make_peptide_water_ml_energy_fn(model, params, jax_z, peptide_idx, water_indices, displacement_fn):
    # Construct stacked dimer indices of shape (N_waters, 45) containing the peptide and each water
    dimer_indices = [np.concatenate([peptide_idx, idx]) for idx in water_indices]
    stacked_dimer_indices = jnp.stack(dimer_indices)
    
    # 1. Setup dimer evaluation (45 atoms)
    n_atoms_dimer = 45
    dst_idx_dimer, src_idx_dimer = np.where(~np.eye(n_atoms_dimer, dtype=bool))
    dst_idx_dimer = jnp.array(dst_idx_dimer, dtype=jnp.int32)
    src_idx_dimer = jnp.array(src_idx_dimer, dtype=jnp.int32)
    
    vmapped_apply_dimer = jax.vmap(
        lambda pos, atomic_nums: model.apply(
            params,
            atomic_numbers=atomic_nums,
            positions=pos,
            dst_idx=dst_idx_dimer,
            src_idx=src_idx_dimer,
        )["energy"],
        in_axes=(0, 0)
    )
    
    dimer_z = jax_z[stacked_dimer_indices]
    
    # Vectorized displacement to calculate relative coordinates of dimer under PBC relative to first atom
    vmapped_displacement_dimer = jax.vmap(
        jax.vmap(displacement_fn, in_axes=(0, None)),
        in_axes=(0, 0)
    )
    
    # 2. Setup peptide evaluation (42 atoms)
    n_atoms_pep = len(peptide_idx)
    dst_idx_pep, src_idx_pep = np.where(~np.eye(n_atoms_pep, dtype=bool))
    dst_idx_pep = jnp.array(dst_idx_pep, dtype=jnp.int32)
    src_idx_pep = jnp.array(src_idx_pep, dtype=jnp.int32)
    
    pep_z = jax_z[peptide_idx]
    
    vmapped_displacement_pep = jax.vmap(displacement_fn, in_axes=(0, None))
    
    def peptide_energy(r):
        pep_pos = r[peptide_idx]
        ref_pos = pep_pos[0]
        unfolded_pep = ref_pos + vmapped_displacement_pep(pep_pos, ref_pos)
        return model.apply(
            params,
            atomic_numbers=pep_z,
            positions=unfolded_pep,
            dst_idx=dst_idx_pep,
            src_idx=src_idx_pep,
        )["energy"]
        
    n_waters = len(water_indices)
    
    def dimer_energy_fn(r):
        # Evaluate dimer sum (peptide + each water)
        group_pos = r[stacked_dimer_indices]
        ref_pos = group_pos[:, 0, :]
        displacements = vmapped_displacement_dimer(group_pos, ref_pos)
        unfolded_dimer_pos = ref_pos[:, None, :] + displacements
        
        dimer_energies = vmapped_apply_dimer(unfolded_dimer_pos, dimer_z)
        e_dimer_sum = jnp.sum(dimer_energies)
        
        # Evaluate single peptide energy
        e_pep = peptide_energy(r)
        
        # Subtract (N_waters - 1) * E_peptide to get:
        # E_peptide_ML + sum(E_water_i_ML) + sum(E_inter_peptide_water_ML)
        return e_dimer_sum - (n_waters - 1) * e_pep
        
    return dimer_energy_fn

# Configure compute_monomer_energy function based on selection
if PEPTIDE_WATER_ML:
    print("--- Configuring PEPTIDE-WATER interactions with ML (dimer approach) ---")
    peptide_idx = monomer_indices[0]
    water_indices = monomer_indices[1:]
    compute_monomer_energy = make_peptide_water_ml_energy_fn(
        model, params, jax_z, peptide_idx, water_indices, displacement_fn
    )
else:
    print("--- Configuring PEPTIDE-WATER interactions with MM ---")
    compute_monomer_energy = make_monomer_energy_fn(
        model, params, jax_z, jax_monomer_indices, displacement_fn
    )


# Helper function to filter out intramolecular nonbonded terms on the host
def get_intermolecular_pairs(positions, cell_matrix, excluded, cutoff, mol_id):
    pair_i_raw, pair_j_raw = _build_pair_indices(positions, cell_matrix, excluded, cutoff)
    # Exclude pairs where both atoms belong to the same molecule
    inter = mol_id[pair_i_raw] != mol_id[pair_j_raw]
    
    if PEPTIDE_WATER_ML:
        # Also exclude peptide-water pairs: one is 0 (peptide) and the other is > 0 (water)
        is_pep_i = mol_id[pair_i_raw] == 0
        is_pep_j = mol_id[pair_j_raw] == 0
        pep_wat = (is_pep_i & ~is_pep_j) | (is_pep_j & ~is_pep_i)
        inter = inter & ~pep_wat
        
    return pair_i_raw[inter], pair_j_raw[inter]

# Precompute initial pair list
print("--- Precomputing nonbonded pair list indices ---")
excluded_pairs = nbond_data.excluded_pairs
if nbond_data.psf_path is not None and nbond_data.psf_bonds is not None:
    excluded_pairs = resolve_nonbonded_excluded_pairs(
        nbond_data.psf_path,
        nbond_data.psf_bonds,
        natom=int(np.asarray(nbond_data.charges).shape[0]),
    )
pair_i, pair_j = get_intermolecular_pairs(pos, cell, excluded_pairs, nb_settings.cutnb, molecule_id)

# 5. Define displacement and shift functions for Periodic Boundary Conditions (JAX-MD)
# Using a cubic box setup based on the cell size.
displacement_fn, shift_fn = space.periodic(box_size)

def make_hybrid_energy_fn(pi, pj):
    # Keep pi and pj as numpy arrays (np.ndarray) so they are treated as static constants during JIT compilation.
    def hybrid_energy_fn(r) -> jnp.ndarray:
        # (A) Intramolecular terms from ML potential
        e_intra = compute_monomer_energy(r)
        
        # (B) Intermolecular nonbonded terms from JAX-MM system
        # Set molecule_id=None because pair list is already filtered on the host.
        terms_raw, _ = nonbonded_energy_and_forces(
            r,
            nbond_data,
            cell,
            nb_settings,
            molecule_id=None,
            pair_i=pi,
            pair_j=pj,
        )
        e_inter = terms_raw.get("total", sum(terms_raw.values())) * KCAL_MOL_TO_EV
        
        return e_intra + e_inter
    return hybrid_energy_fn

# 7. Minimization and Dynamics Configuration
FIRE_STEPS = 20000
FIRE_PRINT_FREQ = 2000

NVT_TOTAL_STEPS = 50000
NVT_BLOCK_STEPS = 1000

NVE_TOTAL_STEPS = 20000
NVE_BLOCK_STEPS = 10000

# 8. Structure Minimization with JAX-MD FIRE
print("--- Minimizing System with JAX-MD FIRE ---")
init_r = jnp.array(pos, dtype=jnp.float64)

# Create the initial energy and force functions for minimization
initial_energy_fn = make_hybrid_energy_fn(pair_i, pair_j)
energy_fn = jit(initial_energy_fn)
force_fn = jit(grad(lambda r: -initial_energy_fn(r)))

# Initialize FIRE minimizer
fire_init, fire_step = minimize.fire_descent(initial_energy_fn, shift_fn, dt_start=0.001, dt_max=0.001)
fire_state = fire_init(init_r)

# Perform minimization steps
for step in range(FIRE_STEPS):
    fire_state = fire_step(fire_state)
    if step % FIRE_PRINT_FREQ == 0:
        curr_e = energy_fn(fire_state.position)
        print(f"FIRE Step {step:3d} | Energy: {curr_e:.4f} eV")

min_r = fire_state.position
print(f"Minimization complete. Final Energy: {energy_fn(min_r):.4f} eV")

# 9. Molecular Dynamics (NVT Nose-Hoover) with JAX-MD
print("--- Running NVT Nose-Hoover Dynamics with JAX-MD ---")
from jax_md import quantity

# Define simulation conditions
temperature = 200.0  # Kelvin
kb = 8.617333262145e-5  # eV/K (Boltzmann constant in eV/K)
target_temp_ev = temperature * kb
dt_fs = 0.25  # time step in femtoseconds
dt = dt_fs * 0.001  # convert to picoseconds (JAX-MD metal units)

# Setup NHC simulator
mass = np.zeros(len(pos))
mass[:n_trialanine] = 12.0  # average mass approximation for peptide
mass[n_trialanine::3] = 16.0  # Oxygen
mass[n_trialanine+1::3] = 1.0  # Hydrogen
mass[n_trialanine+2::3] = 1.0  # Hydrogen
jax_mass = jnp.array(mass, dtype=jnp.float64)

# Helper function to create JIT-compiled NVT block runner for a specific pair list
def make_nvt_block_runner(pi, pj):
    local_energy_fn = make_hybrid_energy_fn(pi, pj)
    init_fn_local, step_fn_local = simulate.nvt_nose_hoover(local_energy_fn, shift_fn, dt, target_temp_ev)
    step_fn_local = jit(step_fn_local)
    
    @jit
    def run_nvt_block(state, steps=NVT_BLOCK_STEPS):
        def body_fn(i, val_state):
            return step_fn_local(val_state)
        return jax.lax.fori_loop(0, steps, body_fn, state)
        
    return run_nvt_block, init_fn_local, local_energy_fn, jit(grad(lambda r: -local_energy_fn(r)))

# Initialize starting NVT state using the initial pair list
run_nvt_block, init_fn_nvt, energy_fn_nvt, force_fn_nvt = make_nvt_block_runner(pair_i, pair_j)
key = jax.random.PRNGKey(42)
state = init_fn_nvt(key, min_r, mass=jax_mass)

# Run NVT dynamics loop with periodic neighbor list updates
traj_path_nvt = "cg_nvt.traj"
print(f"--- Running NVT dynamics and saving trajectory to {traj_path_nvt} ---")
traj_nvt = Trajectory(traj_path_nvt, "w", atoms)

for step in range(0, NVT_TOTAL_STEPS, NVT_BLOCK_STEPS):
    # Update neighbor list (pair list) on the host based on current positions
    pos_np = np.asarray(state.position)
    pi, pj = get_intermolecular_pairs(pos_np, cell, excluded_pairs, nb_settings.cutnb, molecule_id)
    
    # Retrieve the block runner for these specific pairs
    run_nvt_block, _, energy_fn_nvt, force_fn_nvt = make_nvt_block_runner(pi, pj)
    
    # Run the compiled block of steps
    state = run_nvt_block(state, NVT_BLOCK_STEPS)
    
    # Compute diagnostics
    curr_e = float(energy_fn_nvt(state.position))
    curr_f = np.asarray(force_fn_nvt(state.position))
    ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
    temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
    print(f"NVT Step {step+NVT_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | Temp: {temp:.1f} K")
    
    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(np.asarray(state.position))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_nvt.write(frame)

traj_nvt.close()
print("NVT dynamics complete!")

# 10. Molecular Dynamics (NVE) with JAX-MD to check stability
print("--- Running NVE Dynamics with JAX-MD to check stability ---")

# Helper function to create JIT-compiled NVE block runner for a specific pair list
def make_nve_block_runner(pi, pj):
    local_energy_fn = make_hybrid_energy_fn(pi, pj)
    init_fn_local, step_fn_local = simulate.nve(local_energy_fn, shift_fn, dt)
    step_fn_local = jit(step_fn_local)
    
    @jit
    def run_nve_block(state, steps=NVE_BLOCK_STEPS):
        def body_fn(i, val_state):
            return step_fn_local(val_state)
        return jax.lax.fori_loop(0, steps, body_fn, state)
        
    return run_nve_block, init_fn_local, local_energy_fn, jit(grad(lambda r: -local_energy_fn(r)))

# Initialize NVE simulation state from the final NVT positions and velocities
pos_np_final = np.asarray(state.position)
pi_init, pj_init = get_intermolecular_pairs(pos_np_final, cell, excluded_pairs, nb_settings.cutnb, molecule_id)
run_nve_block, init_fn_nve, energy_fn_nve, force_fn_nve = make_nve_block_runner(pi_init, pj_init)
state_nve = init_fn_nve(key, state.position, target_temp_ev, mass=jax_mass)

traj_path_nve = "cg_nve.traj"
print(f"--- Running NVE dynamics and saving trajectory to {traj_path_nve} ---")
traj_nve = Trajectory(traj_path_nve, "w", atoms)

for step in range(0, NVE_TOTAL_STEPS, NVE_BLOCK_STEPS):
    # Update neighbor list (pair list)
    pos_np = np.asarray(state_nve.position)
    pi, pj = get_intermolecular_pairs(pos_np, cell, excluded_pairs, nb_settings.cutnb, molecule_id)
    
    # Get NVE block runner
    run_nve_block, _, energy_fn_nve, force_fn_nve = make_nve_block_runner(pi, pj)
    
    # Run the compiled block of steps
    state_nve = run_nve_block(state_nve, NVE_BLOCK_STEPS)
    
    # Compute diagnostics to verify energy conservation
    curr_e = float(energy_fn_nve(state_nve.position))
    curr_f = np.asarray(force_fn_nve(state_nve.position))
    ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
    temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
    print(f"NVE Step {step+NVE_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | Temp: {temp:.1f} K")
    
    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(np.asarray(state_nve.position))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_nve.write(frame)

traj_nve.close()
print("NVE dynamics complete!")


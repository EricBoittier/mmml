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
from mmml.interfaces.jaxmdInterface import (
    make_monomer_energy_fn,
    make_peptide_water_ml_energy_fn,
    get_intermolecular_pairs,
)

# 1. Initialize JAX and PyCHARMM configuration
jax.config.update("jax_enable_x64", True)
ensure_pycharmm_loaded()
pycharmm_loud()

# Path to the pretrained neural network checkpoint parameters
CKPT_PATH = "params_aaa_long_2026-07-04_22-30-27.json"

FIRE_STEPS = 500
FIRE_PRINT_FREQ = 100

FIRE_BLOCK_STEPS = 100
NVT_TOTAL_STEPS = 50000
NVT_BLOCK_STEPS = 1000

NVE_TOTAL_STEPS = 20000
NVE_BLOCK_STEPS = 100
FIRE_CYCLES=10
NWATER = 500
BOX_SIDE_A = 30.0

# 2. Build the initial system in PyCHARMM and minimize
print("--- Building Trialanine Water Box in CHARMM ---")
workdir = Path('/tmp/tria_box')
box = build_trialanine_water_box_in_charmm(n_waters=NWATER, box_side_A=BOX_SIDE_A, seed=11, workdir=workdir)

pos = np.asarray(box.positions, dtype=np.float64)
pos = np.random.uniform(-1.0, 1.0, pos.shape) + pos

# translate the entire system so that the peptide is centered in the box (L/2, L/2, L/2)
# while keeping the waters in their relative positions around the peptide
n_trialanine = 42
peptide_center = pos[:n_trialanine].mean(axis=0)
box_center = np.array([box.box_side_A / 2, box.box_side_A / 2, box.box_side_A / 2])
translation = box_center - peptide_center
pos += translation

set_charmm_positions(pos)

setup_nonbonded_only_charmm()


for i in range(10):
    # Apply constraint and run steepest descent minimization
    lingo.charmm_script("CONStraint DROPlet FORC 0.01 EXPO 4")
    lingo.charmm_script("MINI SD 10000")
    # lingo.charmm_script("IMAGE")
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

# Option: Treat peptide-water intermolecular interactions with ML instead of MM
PEPTIDE_WATER_ML = False

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

# Helper function to unfold periodic coordinates to keep molecules contiguous
def unfold_coordinates(positions, L, mon_indices):
    unfolded = np.copy(positions)
    for indices in mon_indices:
        if len(indices) <= 1:
            continue
        ref_pos = positions[indices[0]]
        diff = positions[indices] - ref_pos
        diff = diff - L * np.round(diff / L)
        unfolded[indices] = ref_pos + diff
    return unfolded

# Helper function to check the spatial extent of the peptide (first 42 atoms)
def check_peptide_extent(positions, box_sz):
    pep_pos = np.asarray(positions[:42])
    ref_pos = pep_pos[0]
    diff = pep_pos - ref_pos
    diff = diff - box_sz * np.round(diff / box_sz)
    unfolded_pep = ref_pos + diff
    dists = np.linalg.norm(unfolded_pep[:, None, :] - unfolded_pep[None, :, :], axis=-1)
    return float(np.max(dists))

# Helper function to repair structures using PyCHARMM minimization
def repair_structure_in_charmm(positions):
    print("\n[REPAIR] Temperature spike or NaN detected! Unfolding and repairing structure in CHARMM...")
    # Unfold coordinates first so PyCHARMM does not see split molecules with massive bond lengths
    unfolded_pos = unfold_coordinates(np.asarray(positions), box_size, monomer_indices)
    set_charmm_positions(unfolded_pos)
    
    # Run steep SD and ABNR minimizations in PyCHARMM to resolve overlaps/clashes
    lingo.charmm_script("CONStraint DROPlet FORC 0.01 EXPO 4")
    lingo.charmm_script("MINI SD 100")
    lingo.charmm_script("CONStraint DROPlet")
    lingo.charmm_script("MINI SD 100")
    lingo.charmm_script("MINI ABNR 100")
    # Retrieve repaired positions
    repaired_pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    print("[REPAIR] Structure repaired successfully. Re-initializing state.\n")
    return repaired_pos




# 8. Structure Minimization with JAX-MD FIRE and PyCHARMM Repair Loops
print("--- Minimizing System with JAX-MD FIRE and PyCHARMM Repair Loops (5 cycles) ---")
init_r = jnp.array(pos, dtype=jnp.float64)
pos_current = init_r

# Helper function to create JIT-compiled FIRE block runner for a specific pair list
def make_fire_block_runner(pi, pj):
    local_energy_fn = make_hybrid_energy_fn(pi, pj)
    init_fn_local, step_fn_local = minimize.fire_descent(local_energy_fn, shift_fn, dt_start=0.0001, dt_max=0.001)
    step_fn_local = jit(step_fn_local)
    
    @jit
    def run_fire_block(state, steps=100):
        def body_fn(i, val_state):
            return step_fn_local(val_state)
        return jax.lax.fori_loop(0, steps, body_fn, state)
        
    return run_fire_block, init_fn_local, jit(local_energy_fn), jit(grad(lambda r: -local_energy_fn(r)))


# Open trajectory file for saving minimization path
traj_path_fire = "cg_fire.traj"
print(f"--- Saving minimization trajectory to {traj_path_fire} ---")
traj_fire = Trajectory(traj_path_fire, "w", atoms)

for cycle in range(FIRE_CYCLES):
    print(f"\n--- Minimization Cycle {cycle+1}/{FIRE_CYCLES} ---")
    
    # Initialize starting FIRE state using current coordinates and pair list
    pi_init, pj_init = get_intermolecular_pairs(np.asarray(pos_current), cell, excluded_pairs, nb_settings.cutnb, molecule_id)
    run_fire_block, init_fn_fire, energy_fn_fire, force_fn_fire = make_fire_block_runner(pi_init, pj_init)
    fire_state = init_fn_fire(pos_current)
    
    # Write starting configuration of this cycle to trajectory
    curr_e = float(energy_fn_fire(pos_current))
    curr_f = np.asarray(force_fn_fire(pos_current))
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(pos_current), box_size, monomer_indices))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_fire.write(frame)
    
    # Run FIRE blocks
    for step in range(0, FIRE_STEPS, FIRE_BLOCK_STEPS):
        pos_np = np.asarray(fire_state.position)
        pi, pj = get_intermolecular_pairs(pos_np, cell, excluded_pairs, nb_settings.cutnb, molecule_id)
        run_fire_block, _, energy_fn_fire, force_fn_fire = make_fire_block_runner(pi, pj)
        
        fire_state = run_fire_block(fire_state, FIRE_BLOCK_STEPS)
        
        curr_e = float(energy_fn_fire(fire_state.position))
        curr_f = np.asarray(force_fn_fire(fire_state.position))
        pep_ext = check_peptide_extent(fire_state.position, box_size)
        print(f"Cycle {cycle+1} | FIRE Step {step+FIRE_BLOCK_STEPS:3d} | Energy: {curr_e:.4f} eV | Pep Extent: {pep_ext:.2f} Å")
        
        # Save intermediate configuration to trajectory
        frame = atoms.copy()
        frame.set_positions(unfold_coordinates(np.asarray(fire_state.position), box_size, monomer_indices))
        frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
        traj_fire.write(frame)
        
    # Repair the minimized structures in CHARMM to resolve any close contacts
    pos_current = repair_structure_in_charmm(fire_state.position)

traj_fire.close()
min_r = jnp.array(pos_current, dtype=jnp.float64)
print(f"\nMinimization completed over {FIRE_CYCLES} cycles.")

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
    pep_ext = check_peptide_extent(state.position, box_size)
    
    # Check if a spike occurred, NaN energy, or peptide broke (extent > 15.0 A) and repair if necessary
    if temp > 400.0 or np.isnan(curr_e) or pep_ext > 15.0:
        if pep_ext > 15.0:
            print(f"[REPAIR] Peptide broke! Extent: {pep_ext:.2f} Å (max limit 15.0 Å)")
        repaired_pos = repair_structure_in_charmm(state.position)
        # Re-initialize the state with the repaired coordinates at the target temperature
        state = init_fn_nvt(key, jnp.array(repaired_pos, dtype=jnp.float64), mass=jax_mass)
        # Re-evaluate properties
        curr_e = float(energy_fn_nvt(state.position))
        curr_f = np.asarray(force_fn_nvt(state.position))
        ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
        temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
        pep_ext = check_peptide_extent(repaired_pos, box_size)
        
    print(f"NVT Step {step+NVT_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | Temp: {temp:.1f} K | Peptide Ext: {pep_ext:.2f} Å")
    
    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(state.position), box_size, monomer_indices))
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
    pep_ext = check_peptide_extent(state_nve.position, box_size)
    
    # Check if a spike occurred, NaN energy, or peptide broke (extent > 15.0 A) and repair if necessary
    if temp > 400.0 or np.isnan(curr_e) or pep_ext > 15.0:
        if pep_ext > 15.0:
            print(f"[REPAIR] Peptide broke! Extent: {pep_ext:.2f} Å (max limit 15.0 Å)")
        repaired_pos = repair_structure_in_charmm(state_nve.position)
        # Re-initialize the state with the repaired coordinates
        state_nve = init_fn_nve(key, jnp.array(repaired_pos, dtype=jnp.float64), target_temp_ev, mass=jax_mass)
        # Re-evaluate properties
        curr_e = float(energy_fn_nve(state_nve.position))
        curr_f = np.asarray(force_fn_nve(state_nve.position))
        ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
        temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
        pep_ext = check_peptide_extent(repaired_pos, box_size)
        
    print(f"NVE Step {step+NVE_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | Temp: {temp:.1f} K | Peptide Ext: {pep_ext:.2f} Å")
    
    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(state_nve.position), box_size, monomer_indices))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_nve.write(frame)

traj_nve.close()
print("NVE dynamics complete!")


#!/usr/bin/env python
# coding: utf-8

"""Example script demonstrating a hybrid ML/MM simulation workflow using JAX-MD.

By bypassing ASE and running the optimization and molecular dynamics entirely
within JAX-MD, we achieve significantly faster execution times through full-program
compilation (JIT) and hardware acceleration (GPU/TPU).

GPU Performance Optimizations vs. original:
  1. Zero-retrace pair-list pattern: pair arrays are padded to a fixed MAX_PAIRS
     shape and stored in mutable Python list slots (_pi_ref, _pj_ref). The energy
     function closes over these references; since the shape never changes JAX only
     traces once, eliminating thousands of recompilations.
  2. e14/vdw14 scale arrays precomputed on CPU once per NL update (not inside JIT).
  3. Runner step functions compiled once and reused across all blocks.
  4. value_and_grad: energy + forces computed in a single kernel launch.
  5. Vectorized helpers: unfold_coordinates, get_max_h_x_bond, scale_broken_h_bonds
     now use NumPy broadcasting instead of Python for-loops.
  6. Larger block sizes (FIRE: 200→1000, NVT/NVE: 100→500) to amortize Python
     overhead across more GPU work per outer-loop iteration.
"""

from pathlib import Path
import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad

# Enable persistent JAX compilation cache for fast subsequent execution
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)

# ASE trajectory, calculator, and Atoms imports
from ase import Atoms
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize.fire import FIRE as AseFIRE

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
    charmm_vfswitch_coeffs,
    charmm_fswitch_coeffs,
    _pair_lj_epsilon,
    _pair_vdw_energy,
    _pair_elec_energy,
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
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement
from cg_common import (
    DualTrajectoryWriter,
    load_cg_checkpoint,
    load_cg_config,
    probe_charge_output,
)

# 1. Initialize JAX and PyCHARMM configuration
jax.config.update("jax_enable_x64", True)

# Helper function to parse peptide H-X bonds from the CHARMM PSF file
def parse_peptide_h_x_bonds(psf_path, z_array):
    bonds = []
    in_bonds = False
    with open(psf_path, 'r') as f:
        for line in f:
            if '!NBOND' in line:
                in_bonds = True
                continue
            if in_bonds:
                if not line.strip() or '!' in line:
                    if '!NBOND' not in line:
                        in_bonds = False
                        continue
                parts = line.split()
                for i in range(0, len(parts), 2):
                    idx1 = int(parts[i]) - 1
                    idx2 = int(parts[i+1]) - 1
                    # Peptide atoms are the first 42 atoms
                    if idx1 < 42 and idx2 < 42:
                        z1 = z_array[idx1]
                        z2 = z_array[idx2]
                        # Identify hydrogen (Z=1) to heavy atom (Z>1) bonds
                        if (z1 == 1 and z2 > 1) or (z2 == 1 and z1 > 1):
                            h_idx = idx1 if z1 == 1 else idx2
                            x_idx = idx2 if z1 == 1 else idx1
                            bonds.append((h_idx, x_idx))
    return bonds


def parse_all_peptide_bonds(psf_path):
    bonds = []
    in_bonds = False
    with open(psf_path, 'r') as f:
        for line in f:
            if '!NBOND' in line:
                in_bonds = True
                continue
            if in_bonds:
                if not line.strip() or '!' in line:
                    if '!NBOND' not in line:
                        in_bonds = False
                        continue
                parts = line.split()
                for i in range(0, len(parts), 2):
                    idx1 = int(parts[i]) - 1
                    idx2 = int(parts[i+1]) - 1
                    # Peptide atoms are the first 42 atoms
                    if idx1 < 42 and idx2 < 42:
                        bonds.append((idx1, idx2))
    return bonds


def get_peptide_bond_diagnostics(r, box_size, idx1_arr, idx2_arr, r0_arr):
    r_np = np.asarray(r)
    ri = r_np[idx1_arr]
    rj = r_np[idx2_arr]
    dr = ri - rj
    dr -= box_size * np.round(dr / box_size)
    dists = np.linalg.norm(dr, axis=-1)
    diffs = np.abs(dists - r0_arr)
    return float(diffs.max()), float(diffs.mean())




# Paths to pretrained neural network checkpoint parameters.
script_dir = Path(__file__).parent
_settings = load_cg_config(
    {
        "checkpoint": str(script_dir / "params_test01_2026-07-08_12-58-42.json"),
        "peptide_checkpoint": str(script_dir / "params_test01_2026-07-08_12-58-42.json"),
        "water_checkpoint": str(script_dir / "params_test01_2026-07-08_12-58-42.json"),
        "fire_steps": 1000,
        "fire_print_freq": 1000,
        "fire_block_steps": 100,
        "nvt_total_steps": 1000,
        "nvt_block_steps": 100,
        "nve_total_steps": 2000,
        "nve_block_steps": 500,
        "fire_cycles": 2,
        "n_waters": 100,
        "box_size": 30.0,
        "nl_buffer": 2.0,
        "max_pairs_headroom": 1.05,
        "max_hx_bond_limit": 1.5,
        "nvt_repair_temp_k": 375.0,
        "nve_repair_temp_k": 400.0,
        "seed": 42,
        "temperature": 248.0,
        "dt_fs": 0.5,
        "peptide_water_ml": False,
        "peptide_water_electrostatic_embedding": True,
        "peptide_ml_charge_total_correction": True,
        "water_ml_charge_total_correction": True,
        "peptide_electrostatic_embedding_require_ml_charges": True,
        "water_electrostatic_embedding_require_ml_charges": True,
        "debug": True,
        "start_peptide_traj_path": None,
        "start_peptide_traj_index": 0,
        "constrain_phi_psi": False,
        "phi_target_deg": None,
        "psi_target_deg": None,
        "dihedral_restraint_k_ev": 1.0,
        "phi_central": [14, 16, 18, 24],
        "psi_central": [16, 18, 24, 26],
        "peptide_bond_k_ev": 200.0,
        "workdir": "/tmp/tria_box",
        "output_dir": ".",
        "write_dcd": True,
    },
    description="Trialanine/water direct JAX-MD hybrid example",
)
CKPT_PATH = str(_settings.checkpoint)
PEPTIDE_CKPT_PATH = str(_settings.peptide_checkpoint)
WATER_CKPT_PATH = str(_settings.water_checkpoint)

FIRE_STEPS = int(_settings.fire_steps)
FIRE_PRINT_FREQ = int(_settings.fire_print_freq)
# FIRE_BLOCK_STEPS kept small (100) because FIRE adaptively grows its step size:
# running 1000 steps without checking allows catastrophic divergence before repair.
FIRE_BLOCK_STEPS = int(_settings.fire_block_steps)


<<<<<<< HEAD
NVT_TOTAL_STEPS = int(_settings.nvt_total_steps)
NVT_BLOCK_STEPS = int(_settings.nvt_block_steps)

NVE_TOTAL_STEPS = int(_settings.nve_total_steps)
NVE_BLOCK_STEPS = int(_settings.nve_block_steps)
FIRE_CYCLES = int(_settings.fire_cycles)

NWATER = int(_settings.n_waters)
BOX_SIDE_A = float(_settings.box_size)
NL_BUFFER = float(_settings.nl_buffer)
# Extra headroom fraction for padded pair array (5%)
MAX_PAIRS_HEADROOM = float(_settings.max_pairs_headroom)
MAX_HX_BOND_LIMIT = float(_settings.max_hx_bond_limit)
NVT_REPAIR_TEMP_K = float(_settings.nvt_repair_temp_k)
NVE_REPAIR_TEMP_K = float(_settings.nve_repair_temp_k)
SEED = int(_settings.seed)
=======
NVT_TOTAL_STEPS = 100000
NVT_BLOCK_STEPS = 100

NVE_TOTAL_STEPS = 200000
NVE_BLOCK_STEPS = 500
FIRE_CYCLES = 2

NWATER = 1000
BOX_SIDE_A = 28.0
NL_BUFFER = 2.0
# Extra headroom fraction for padded pair array (5%)
MAX_PAIRS_HEADROOM = 1.15
MAX_HX_BOND_LIMIT = 1.5
NVT_REPAIR_TEMP_K = 375.0
NVE_REPAIR_TEMP_K = 400.0
SEED = 42
>>>>>>> 5ce097cb2 (asdf)
# Define simulation conditions
temperature = float(_settings.temperature)  # Kelvin
kb = 8.617333262145e-5  # eV/K (Boltzmann constant in eV/K)
target_temp_ev = temperature * kb
<<<<<<< HEAD
dt_fs = float(_settings.dt_fs)  # time step in femtoseconds
=======
dt_fs = 0.25  # time step in femtoseconds
>>>>>>> 5ce097cb2 (asdf)
dt = dt_fs * 0.001  # convert to picoseconds (JAX-MD metal units)

# Option: Treat peptide-water intermolecular interactions with ML instead of MM
PEPTIDE_WATER_ML = bool(_settings.peptide_water_ml)
PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING = bool(_settings.peptide_water_electrostatic_embedding)
PEPTIDE_ML_CHARGE_TOTAL_CORRECTION = bool(_settings.peptide_ml_charge_total_correction)
WATER_ML_CHARGE_TOTAL_CORRECTION = bool(_settings.water_ml_charge_total_correction)
PEPTIDE_ELECTROSTATIC_EMBEDDING_REQUIRE_ML_CHARGES = bool(
    _settings.peptide_electrostatic_embedding_require_ml_charges
)
WATER_ELECTROSTATIC_EMBEDDING_REQUIRE_ML_CHARGES = bool(
    _settings.water_electrostatic_embedding_require_ml_charges
)
DEBUG = bool(_settings.debug)

# Optional: start from a peptide-only frame produced by
# scripts/scan_trialanine_phi_psi_pes.py. The frame replaces the first
# n_trialanine coordinates in the solvated box before minimization/dynamics.
START_PEPTIDE_TRAJ_PATH = _settings.start_peptide_traj_path
START_PEPTIDE_TRAJ_INDEX = int(_settings.start_peptide_traj_index)

# Optional PHI/PSI restraints for JAX-MD. If targets are None and a trajectory
# frame is loaded, targets are read from that frame's info when available.
CONSTRAIN_PHI_PSI = bool(_settings.constrain_phi_psi)
PHI_TARGET_DEG = _settings.phi_target_deg
PSI_TARGET_DEG = _settings.psi_target_deg
DIHEDRAL_RESTRAINT_K_EV = float(_settings.dihedral_restraint_k_ev)
PHI_CENTRAL = tuple(_settings.phi_central)  # C1-N2-CA2-C2
PSI_CENTRAL = tuple(_settings.psi_central)  # N2-CA2-C2-N3
PEPTIDE_BOND_K_EV = float(_settings.peptide_bond_k_ev)
OUTPUT_DIR = Path(_settings.output_dir).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ensure_pycharmm_loaded()
pycharmm_loud()



def minimize_peptide_only_in_charmm(sd_steps=10000, abnr_steps=10000):
    """Minimize only the PEPT segment while keeping waters fixed."""
    print("--- Minimizing peptide-only in CHARMM with waters fixed ---")
    try:
        lingo.charmm_script("CONS FIX SELE .NOT. SEGID PEPT END")
        lingo.charmm_script(f"MINI SD NSTEP {sd_steps}")
        lingo.charmm_script(f"MINI ABNR NSTEP {abnr_steps}")
    finally:
        lingo.charmm_script("CONS FIX SELE NONE END")


def minimize_peptide_only_with_ml_calculator(
    positions,
    atomic_numbers,
    peptide_calc,
    workdir,
    n_peptide_atoms=42,
    fmax=0.05,
    steps=500,
):
    """Minimize the peptide subset with the ML ASE calculator and keep solvent fixed."""
    print("--- Minimizing peptide-only with ML ASE calculator ---")
    peptide_atoms = Atoms(
        numbers=np.asarray(atomic_numbers[:n_peptide_atoms], dtype=np.int32),
        positions=np.asarray(positions[:n_peptide_atoms], dtype=np.float64),
    )
    peptide_atoms.calc = peptide_calc
    opt = AseFIRE(
        peptide_atoms,
        logfile=str(Path(workdir) / "peptide_ml_fire.log"),
        trajectory=str(Path(workdir) / "peptide_ml_fire.traj"),
        maxstep=0.03,
    )
    opt.run(fmax=fmax, steps=steps)

    relaxed = np.asarray(positions, dtype=np.float64).copy()
    relaxed[:n_peptide_atoms] = peptide_atoms.get_positions()
    return relaxed


def load_peptide_start_frame(traj_path, frame_index=0):
    """Read a peptide-only ASE trajectory frame and return positions plus PHI/PSI info."""
    if traj_path is None:
        return None, {}
    frame = ase_read(str(traj_path), index=frame_index)
    positions = np.asarray(frame.get_positions(), dtype=np.float64)
    if positions.shape[0] != n_trialanine:
        raise ValueError(
            f"Expected peptide trajectory frame with {n_trialanine} atoms, "
            f"got {positions.shape[0]} from {traj_path}"
        )
    return positions, dict(frame.info)


# 2. Build the initial system in PyCHARMM and minimize
print("--- Building Trialanine Water Box in CHARMM ---")
workdir = Path(_settings.workdir).expanduser()
box = build_trialanine_water_box_in_charmm(n_waters=NWATER,
    box_side_A=BOX_SIDE_A, seed=SEED, workdir=workdir
    )

from mmml.interfaces.pycharmmInterface.peptide_builder import infer_charge_and_spin_from_psf
pep_charge, pep_spin = infer_charge_and_spin_from_psf(box.psf_path)
print(f"Inferred peptide charge={pep_charge}, spin multiplicity={pep_spin} from PSF.")

pos = np.asarray(box.positions, dtype=np.float64)
pos = np.random.uniform(-0.1, 0.1, pos.shape) + pos

# Translate the entire system so that the peptide is centered in the box
n_trialanine = 42
peptide_center = pos[:n_trialanine].mean(axis=0)
box_center = np.array([box.box_side_A / 2, box.box_side_A / 2, box.box_side_A / 2])
translation = box_center - peptide_center
pos += translation / 2.0

start_peptide_pos, start_peptide_info = load_peptide_start_frame(
    START_PEPTIDE_TRAJ_PATH, START_PEPTIDE_TRAJ_INDEX
)
if start_peptide_pos is not None:
    print(
        f"--- Loading peptide start frame {START_PEPTIDE_TRAJ_INDEX} "
        f"from {START_PEPTIDE_TRAJ_PATH} ---"
    )
    pos[:n_trialanine] = start_peptide_pos + (box_center - start_peptide_pos.mean(axis=0))
    if PHI_TARGET_DEG is None and "actual_phi_deg" in start_peptide_info:
        PHI_TARGET_DEG = float(start_peptide_info["actual_phi_deg"])
    if PSI_TARGET_DEG is None and "actual_psi_deg" in start_peptide_info:
        PSI_TARGET_DEG = float(start_peptide_info["actual_psi_deg"])

if CONSTRAIN_PHI_PSI:
    print(
        "--- JAX-MD PHI/PSI restraints enabled: "
        f"phi={PHI_TARGET_DEG}, psi={PSI_TARGET_DEG}, "
        f"k={DIHEDRAL_RESTRAINT_K_EV} eV/rad^2 ---"
    )

z = get_Z_from_psf()

peptide_calc, peptide_model, peptide_params = load_cg_checkpoint(PEPTIDE_CKPT_PATH)

set_charmm_positions(pos)

minimize_peptide_only_in_charmm()

pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
pos = minimize_peptide_only_with_ml_calculator(pos, z, peptide_calc, workdir, n_trialanine)
set_charmm_positions(pos)

lingo.charmm_script("MINI SD NSTEP 10000")
lingo.charmm_script("MINI ABNR NSTEP 10000")

# Retrieve positions and atomic numbers
pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
z = get_Z_from_psf()
pep_h_x_bonds = parse_peptide_h_x_bonds(box.psf_path, z)
print(f"Parsed {len(pep_h_x_bonds)} peptide H-X bonds from PSF.")
pep_all_bonds = parse_all_peptide_bonds(box.psf_path)
print(f"Parsed {len(pep_all_bonds)} peptide bonds from PSF.")

# Precompute H-X bond index arrays for vectorized operations
h_idx_arr = np.array([b[0] for b in pep_h_x_bonds], dtype=np.int32)
x_idx_arr = np.array([b[1] for b in pep_h_x_bonds], dtype=np.int32)

pep_bond_idx1_arr = np.array([b[0] for b in pep_all_bonds], dtype=np.int32)
pep_bond_idx2_arr = np.array([b[1] for b in pep_all_bonds], dtype=np.int32)

# Construct ASE Atoms object to act as a template for trajectory writing
atoms = Atoms(z, pos)
atoms.set_cell(box.cell)
atoms.set_pbc(True)

# Define box size and cell
box_size = float(box.box_side_A)
cell = np.asarray(box.cell, dtype=np.float64)

# Convert indices to JAX arrays and measure initial bond lengths
h_idx_jax = jnp.array(h_idx_arr, dtype=jnp.int32)
x_idx_jax = jnp.array(x_idx_arr, dtype=jnp.int32)
pep_bond_idx1_jax = jnp.array(pep_bond_idx1_arr, dtype=jnp.int32)
pep_bond_idx2_jax = jnp.array(pep_bond_idx2_arr, dtype=jnp.int32)

pos_init = np.asarray(pos)
r0_list = []
for h_idx, x_idx in pep_h_x_bonds:
    dr = pos_init[h_idx] - pos_init[x_idx]
    dr -= box_size * np.round(dr / box_size)
    r0 = np.linalg.norm(dr)
    r0_list.append(r0)
r0_jax = jnp.array(r0_list, dtype=jnp.float64)
print(f"--- Measured {len(r0_list)} equilibrium H-X bond lengths for flat-bottom restraints ---")

r0_pep_list = []
for idx1, idx2 in pep_all_bonds:
    dr = pos_init[idx1] - pos_init[idx2]
    dr -= box_size * np.round(dr / box_size)
    r0 = np.linalg.norm(dr)
    r0_pep_list.append(r0)
r0_pep_jax = jnp.array(r0_pep_list, dtype=jnp.float64)
print(f"--- Measured {len(r0_pep_list)} equilibrium peptide bond lengths for constraints ---")


# 3. Load JAX nonbonded parameters and setup cutoffs
psf_path = box.psf_path
prm_path = CGENFF_PRM
nb_settings = _nbond_settings_from_cutoffs(box.nbond_cutoffs)
nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)

water_calc, water_model, water_params = load_cg_checkpoint(WATER_CKPT_PATH)

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

# Precompute stacked monomer index arrays grouped by size for vectorized unfolding
from collections import defaultdict
_mon_by_size = defaultdict(list)
for idx in monomer_indices:
    _mon_by_size[len(idx)].append(idx)
# List of (stacked_indices, size) tuples — one entry per unique monomer size
_mon_stacked_groups = [(np.stack(lst), sz) for sz, lst in _mon_by_size.items()]

# 5. Define displacement and shift functions for Periodic Boundary Conditions
displacement_fn, shift_fn = space.periodic(box_size)

# Configure compute_monomer_energy function based on selection
if PEPTIDE_WATER_ML:
    raise NotImplementedError(
        "PEPTIDE_WATER_ML requires a dedicated peptide-water dimer checkpoint; "
        "the current script has separate peptide and water monomer checkpoints."
    )
else:
    print("--- Configuring PEPTIDE-WATER interactions with MM ---")
    monomer_charges = {42: float(pep_charge), 3: 0.0}
    monomer_spins = {42: float(pep_spin), 3: 1.0}
    compute_peptide_energy = make_monomer_energy_fn(
        peptide_model, peptide_params, jax_z, [jax_monomer_indices[0]], displacement_fn,
        charges=monomer_charges, spins=monomer_spins
    )
    compute_water_energy = make_monomer_energy_fn(
        water_model, water_params, jax_z, jax_monomer_indices[1:], displacement_fn,
        charges=monomer_charges, spins=monomer_spins
    )

    def compute_monomer_energy(r):
        return compute_peptide_energy(r) + compute_water_energy(r)

pep_idx_jax = jax_monomer_indices[0]
pep_z_jax = jax_z[pep_idx_jax]
pep_n_atoms = int(n_trialanine)
pep_dst_idx_np, pep_src_idx_np = np.where(~np.eye(pep_n_atoms, dtype=bool))
pep_dst_idx_jax = jnp.array(pep_dst_idx_np, dtype=jnp.int32)
pep_src_idx_jax = jnp.array(pep_src_idx_np, dtype=jnp.int32)
pep_ref_charge_total = float(np.asarray(nbond_data.charges[:n_trialanine]).sum())

water_stacked_idx_jax = jnp.array(np.stack(monomer_indices[1:]), dtype=jnp.int32)
water_n_atoms = 3
water_dst_idx_np, water_src_idx_np = np.where(~np.eye(water_n_atoms, dtype=bool))
water_dst_idx_jax = jnp.array(water_dst_idx_np, dtype=jnp.int32)
water_src_idx_jax = jnp.array(water_src_idx_np, dtype=jnp.int32)
water_z_jax = jax_z[water_stacked_idx_jax]
water_flat_idx_jax = water_stacked_idx_jax.reshape(-1)
water_ref_charge_total = float(np.asarray(nbond_data.charges[n_trialanine:n_trialanine + water_n_atoms]).sum())

probe_charge_output(
    peptide_model,
    peptide_params,
    np.asarray(z[:n_trialanine]),
    np.asarray(pos[:n_trialanine]),
    charge=float(pep_charge),
    spin=float(pep_spin),
    label="peptide checkpoint",
)
probe_charge_output(
    water_model,
    water_params,
    np.asarray(z[n_trialanine:n_trialanine + water_n_atoms]),
    np.asarray(pos[n_trialanine:n_trialanine + water_n_atoms]),
    charge=0.0,
    spin=1.0,
    label="water checkpoint",
)


def compute_peptide_ml_charges(r):
    """Return geometry-dependent peptide atomic charges from the peptide ML model."""
    pep_pos = r[pep_idx_jax]
    ref_pos = pep_pos[0]
    unfolded = ref_pos + jax.vmap(displacement_fn, in_axes=(0, None))(pep_pos, ref_pos)
    centered = unfolded - jnp.mean(unfolded, axis=0, keepdims=True)
    is_pep_spooky = (
        hasattr(peptide_model, "charges")
        and hasattr(peptide_model, "total_charge")
    ) or "spooky" in str(type(peptide_model)).lower()

    if is_pep_spooky:
        outputs = peptide_model.apply(
            peptide_params,
            atomic_numbers=pep_z_jax,
            positions=centered,
            charges=jnp.full((pep_z_jax.shape[0], 1), pep_charge, dtype=jnp.float32),
            spins=jnp.full((pep_z_jax.shape[0], 1), pep_spin, dtype=jnp.float32),
            dst_idx=pep_dst_idx_jax,
            src_idx=pep_src_idx_jax,
            compute_forces=False,
        )
    else:
        outputs = peptide_model.apply(
            peptide_params,
            atomic_numbers=pep_z_jax,
            positions=centered,
            dst_idx=pep_dst_idx_jax,
            src_idx=pep_src_idx_jax,
            compute_forces=False,
        )
    if "charges_as_mono" in outputs:
        q_pep = jnp.asarray(outputs["charges_as_mono"], dtype=jnp.float64).reshape(-1)[:pep_n_atoms]
    elif "charges" in outputs:
        q_pep = jnp.asarray(outputs["charges"], dtype=jnp.float64).reshape(-1)[:pep_n_atoms]
    else:
        if PEPTIDE_ELECTROSTATIC_EMBEDDING_REQUIRE_ML_CHARGES:
            raise ValueError(
                "PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING requires a peptide checkpoint "
                "whose model output includes 'charges_as_mono' or 'charges'."
            )
        q_pep = _q_jax[:n_trialanine]
    if PEPTIDE_ML_CHARGE_TOTAL_CORRECTION:
        q_pep = q_pep + (pep_ref_charge_total - jnp.sum(q_pep)) / pep_n_atoms
    return q_pep


def compute_water_ml_charges(r):
    """Return geometry-dependent water atomic charges from the water ML model."""
    group_pos = r[water_stacked_idx_jax]
    ref_pos = group_pos[:, 0, :]
    displacements = jax.vmap(
        jax.vmap(displacement_fn, in_axes=(0, None)),
        in_axes=(0, 0),
    )(group_pos, ref_pos)
    unfolded = ref_pos[:, None, :] + displacements
    centered = unfolded - jnp.mean(unfolded, axis=1, keepdims=True)

    is_water_spooky = (
        hasattr(water_model, "charges")
        and hasattr(water_model, "total_charge")
    ) or "spooky" in str(type(water_model)).lower()

    if is_water_spooky:
        outputs = jax.vmap(
            lambda pos, atomic_nums: water_model.apply(
                water_params,
                atomic_numbers=atomic_nums,
                positions=pos,
                charges=jnp.full((3, 1), 0.0, dtype=jnp.float32),
                spins=jnp.full((3, 1), 1.0, dtype=jnp.float32),
                dst_idx=water_dst_idx_jax,
                src_idx=water_src_idx_jax,
                compute_forces=False,
            ),
            in_axes=(0, 0),
        )(centered, water_z_jax)
    else:
        outputs = jax.vmap(
            lambda pos, atomic_nums: water_model.apply(
                water_params,
                atomic_numbers=atomic_nums,
                positions=pos,
                dst_idx=water_dst_idx_jax,
                src_idx=water_src_idx_jax,
                compute_forces=False,
            ),
            in_axes=(0, 0),
        )(centered, water_z_jax)

    if "charges_as_mono" in outputs:
        q_water = jnp.asarray(outputs["charges_as_mono"], dtype=jnp.float64).reshape((-1, water_n_atoms))
    elif "charges" in outputs:
        q_water = jnp.asarray(outputs["charges"], dtype=jnp.float64).reshape((-1, water_n_atoms))
    else:
        if WATER_ELECTROSTATIC_EMBEDDING_REQUIRE_ML_CHARGES:
            raise ValueError(
                "PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING requires a water checkpoint "
                "whose model output includes 'charges_as_mono' or 'charges'."
            )
        q_water = _q_jax[water_flat_idx_jax].reshape((-1, water_n_atoms))

    if WATER_ML_CHARGE_TOTAL_CORRECTION:
        q_water = q_water + (water_ref_charge_total - jnp.sum(q_water, axis=1, keepdims=True)) / water_n_atoms
    return q_water.reshape(-1)

# Precompute initial pair list
print("--- Precomputing nonbonded pair list indices ---")
excluded_pairs = nbond_data.excluded_pairs
if nbond_data.psf_path is not None and nbond_data.psf_bonds is not None:
    excluded_pairs = resolve_nonbonded_excluded_pairs(
        nbond_data.psf_path,
        nbond_data.psf_bonds,
        natom=int(np.asarray(nbond_data.charges).shape[0]),
    )

# ─────────────────────────────────────────────────────────────────────────────
# GPU OPTIMIZATION: Padded pair-list infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_e14_vdw14(pair_i_np, pair_j_np):
    """Precompute per-pair 1-4 scaling arrays on the CPU.

    Called on every neighbor-list update (cheap — runs on CPU once per NL rebuild).
    Returns JAX arrays of fixed shape equal to MAX_PAIRS with padding zeros.
    """
    n = len(pair_i_np)
    e14 = np.ones(n, dtype=np.float64)
    vdw14 = np.ones(n, dtype=np.float64)
    for k, (i, j) in enumerate(zip(pair_i_np, pair_j_np)):
        if (int(i), int(j)) in nbond_data.e14_pairs:
            e14[k] = nb_settings.e14fac
            vdw14[k] = nb_settings.vdw14fac
    return e14, vdw14


from scipy.spatial import cKDTree

def _get_inter_pairs_np(pos_np):
    """Compute intermolecular pair list on host using cKDTree with periodic boundaries."""
    boxsize = np.diag(cell)
    cutoff = nb_settings.cutnb + NL_BUFFER
    # cKDTree requires coordinates to be inside the [0, boxsize] domain
    wrapped_pos = np.mod(pos_np, boxsize)
    tree = cKDTree(wrapped_pos, boxsize=boxsize)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    
    # Filter intermolecular pairs
    i_arr = pairs[:, 0]
    j_arr = pairs[:, 1]
    inter = molecule_id[i_arr] != molecule_id[j_arr]
    
    if PEPTIDE_WATER_ML:
        is_pep_i = molecule_id[i_arr] == 0
        is_pep_j = molecule_id[j_arr] == 0
        pep_wat = (is_pep_i & ~is_pep_j) | (is_pep_j & ~is_pep_i)
        inter = inter & ~pep_wat
        
    return i_arr[inter], j_arr[inter]


# Sample the pair count from the initial structure to size the padded arrays
_pi0, _pj0 = _get_inter_pairs_np(pos)
MAX_PAIRS = int(len(_pi0) * MAX_PAIRS_HEADROOM) + 16  # add 16 as safety margin
print(f"--- Initial pair count: {len(_pi0)}, MAX_PAIRS buffer: {MAX_PAIRS} ---")


def _pad_pairs(pi_np, pj_np):
    """Pad pair index arrays to MAX_PAIRS with (0,0) sentinel entries.

    Sentinel pairs (both index 0) contribute zero energy because same-atom pairs
    have zero charge product and the VDW self-energy is zero (r→0 handled by cutoff
    mask which evaluates False for r=0 < c2of, but since atom 0 contributes to
    real pairs we use a zero-mask column instead).
    """
    n = len(pi_np)
    assert n <= MAX_PAIRS, (
        f"Pair count {n} exceeds MAX_PAIRS={MAX_PAIRS}. "
        "Increase MAX_PAIRS_HEADROOM or decrease NL_BUFFER."
    )
    pi_pad = np.zeros(MAX_PAIRS, dtype=np.int32)
    pj_pad = np.zeros(MAX_PAIRS, dtype=np.int32)
    mask = np.zeros(MAX_PAIRS, dtype=np.float64)   # 1.0 for real pairs, 0.0 for padding
    pi_pad[:n] = pi_np
    pj_pad[:n] = pj_np
    mask[:n] = 1.0
    return pi_pad, pj_pad, mask


def _pad_e14_vdw14(e14_np, vdw14_np):
    """Pad 1-4 scaling arrays to MAX_PAIRS with 1.0 (neutral scaling)."""
    e14_pad = np.ones(MAX_PAIRS, dtype=np.float64)
    vdw14_pad = np.ones(MAX_PAIRS, dtype=np.float64)
    n = len(e14_np)
    e14_pad[:n] = e14_np
    vdw14_pad[:n] = vdw14_np
    return e14_pad, vdw14_pad


# ─────────────────────────────────────────────────────────────────────────────
# Mutable reference slots for pair data — JAX sees fixed shapes → traces once
# ─────────────────────────────────────────────────────────────────────────────
# We close over these Python list slots in the JIT-compiled energy function.
# Updating _pi_ref[0] etc. before each block is a Python-level mutation that
# does NOT trigger JAX re-tracing because the array shape is unchanged.

_pi_ref = [jnp.zeros(MAX_PAIRS, dtype=jnp.int32)]
_pj_ref = [jnp.zeros(MAX_PAIRS, dtype=jnp.int32)]
_mask_ref = [jnp.zeros(MAX_PAIRS, dtype=jnp.float64)]
_e14_ref = [jnp.ones(MAX_PAIRS, dtype=jnp.float64)]
_vdw14_ref = [jnp.ones(MAX_PAIRS, dtype=jnp.float64)]

# Precompute static per-atom arrays used inside the energy function
_q_jax = jnp.asarray(nbond_data.charges, dtype=jnp.float64)
_eps_jax = jnp.asarray(nbond_data.epsilon, dtype=jnp.float64)
_rmin_jax = jnp.asarray(nbond_data.rmin, dtype=jnp.float64)
_cell_jax = jnp.asarray(cell if cell.ndim == 2 else np.diag(cell), dtype=jnp.float64)
_vfswitch = charmm_vfswitch_coeffs(nb_settings)
_fswitch = charmm_fswitch_coeffs(nb_settings)
_c2of = nb_settings.c2ofnb

if PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING:
    print(
        "--- Electrostatic embedding enabled: peptide-water Coulomb uses "
        "fluctuating ML charges; water-water Coulomb and all MM LJ remain unchanged ---"
    )
    print(
        f"--- Peptide ML charges corrected to total charge {pep_ref_charge_total:.6f} e ---"
        if PEPTIDE_ML_CHARGE_TOTAL_CORRECTION
        else "--- Peptide ML charges are used without total-charge correction ---"
    )
    print(
        f"--- Water ML charges corrected to total charge {water_ref_charge_total:.6f} e per water ---"
        if WATER_ML_CHARGE_TOTAL_CORRECTION
        else "--- Water ML charges are used without total-charge correction ---"
    )
    if getattr(peptide_model, "charges", True) is False:
        raise ValueError(
            f"Peptide checkpoint {PEPTIDE_CKPT_PATH!r} has charges=False and cannot "
            "provide fluctuating charges for electrostatic embedding."
        )
    if getattr(water_model, "charges", True) is False:
        raise ValueError(
            f"Water checkpoint {WATER_CKPT_PATH!r} has charges=False and cannot "
            "provide fluctuating charges for electrostatic embedding."
        )


def _intermolecular_charge_products(r, pi, pj, e14):
    """Pair charge products, with ML charges only for peptide-water pairs."""
    qq_mm = _q_jax[pi] * _q_jax[pj]
    if not PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING:
        return qq_mm * e14 / nb_settings.eps

    q_ml = _q_jax.at[:n_trialanine].set(compute_peptide_ml_charges(r))
    q_ml = q_ml.at[water_flat_idx_jax].set(compute_water_ml_charges(r))
    is_peptide_i = pi < n_trialanine
    is_peptide_j = pj < n_trialanine
    is_peptide_water = jnp.logical_xor(is_peptide_i, is_peptide_j)
    qq = jnp.where(is_peptide_water, q_ml[pi] * q_ml[pj], qq_mm)
    return qq * e14 / nb_settings.eps


def update_pair_refs(pos_np):
    """Rebuild pair list on CPU and update mutable reference slots.

    This is the ONLY function that touches the pair data between blocks.
    Because array shapes are constant, the compiled energy_fn is never re-traced.
    """
    pi_np, pj_np = _get_inter_pairs_np(pos_np)
    e14_np, vdw14_np = _precompute_e14_vdw14(pi_np, pj_np)

    pi_pad, pj_pad, mask = _pad_pairs(pi_np, pj_np)
    e14_pad, vdw14_pad = _pad_e14_vdw14(e14_np, vdw14_np)

    _pi_ref[0] = jnp.array(pi_pad, dtype=jnp.int32)
    _pj_ref[0] = jnp.array(pj_pad, dtype=jnp.int32)
    _mask_ref[0] = jnp.array(mask, dtype=jnp.float64)
    _e14_ref[0] = jnp.array(e14_pad, dtype=jnp.float64)
    _vdw14_ref[0] = jnp.array(vdw14_pad, dtype=jnp.float64)


# Populate refs with the initial pair list
update_pair_refs(pos)


def _dihedral_angle_rad(r, atom_indices):
    """Signed dihedral angle in radians for four atom indices."""
    p0 = r[atom_indices[0]]
    p1 = r[atom_indices[1]]
    p2 = r[atom_indices[2]]
    p3 = r[atom_indices[3]]

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / jnp.linalg.norm(b1)
    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1

    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)
    return jnp.arctan2(y, x)


def _periodic_angle_delta_rad(angle, target):
    """Smallest signed angle difference angle-target in radians."""
    return jnp.arctan2(jnp.sin(angle - target), jnp.cos(angle - target))


def _phi_psi_restraint_energy(r):
    if not CONSTRAIN_PHI_PSI:
        return 0.0
    if PHI_TARGET_DEG is None or PSI_TARGET_DEG is None:
        return 0.0

    phi = _dihedral_angle_rad(r, PHI_CENTRAL)
    psi = _dihedral_angle_rad(r, PSI_CENTRAL)
    phi_target = jnp.deg2rad(float(PHI_TARGET_DEG))
    psi_target = jnp.deg2rad(float(PSI_TARGET_DEG))
    d_phi = _periodic_angle_delta_rad(phi, phi_target)
    d_psi = _periodic_angle_delta_rad(psi, psi_target)
    return 0.5 * DIHEDRAL_RESTRAINT_K_EV * (d_phi * d_phi + d_psi * d_psi)


@jit
def hybrid_energy_fn(r, pi=None, pj=None, mask=None, e14=None, vdw14=None) -> jnp.ndarray:
    """Single JIT-compiled hybrid ML/MM energy function.

    Takes padded pair arrays as keyword arguments. Since their shapes (MAX_PAIRS)
    are constant, JAX compiles the simulators exactly once and reuses the graph
    when new array values are passed in.
    """
    # (A) Intramolecular terms from ML potential
    e_intra = compute_monomer_energy(r)

    # (B) Intermolecular MM nonbonded terms with padded pair list
    ri = r[pi]
    rj = r[pj]
    disp = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri, rj)
    
    # CRITICAL: Prevent zero displacement for padding pairs (mask=0)
    # and overlapping real pairs to avoid NaN gradients from jnp.linalg.norm at 0.
    disp_sq = jnp.sum(disp**2, axis=-1)
    safe_disp_sq = jnp.where(mask > 0.5, disp_sq, 1.0)
    dist = jnp.sqrt(jnp.maximum(safe_disp_sq, 1e-12))
    within_ctof = (dist * dist < _c2of) * mask   # 0.0 for padding and out-of-cutoff

    # CRITICAL: Clamp distances for padding pairs (mask=0) to a safe non-zero value
    # before passing to VDW/elec kernels.
    safe_dist = jnp.where(mask > 0.5, dist, 1.0)

    ep = _pair_lj_epsilon(_eps_jax[pi], _eps_jax[pj])
    sig = _rmin_jax[pi] + _rmin_jax[pj]
    qq = _intermolecular_charge_products(r, pi, pj, e14)

    vdw = _pair_vdw_energy(safe_dist, ep, sig, nb_settings, _vfswitch, use_jax_pme_dispersion=False)
    vdw = vdw * vdw14
    elec = _pair_elec_energy(safe_dist, qq, nb_settings, _fswitch)

    vdw = jnp.where(within_ctof, vdw, 0.0)
    elec = jnp.where(within_ctof, elec, 0.0)
    e_inter = (jnp.sum(vdw) + jnp.sum(elec)) * KCAL_MOL_TO_EV

    # (C) Flat-bottom harmonic restraints on peptide H-X bonds
    # Kicks in only if bonds stretch more than 0.08 Å beyond equilibrium (r0_jax).
    ri_pep = r[h_idx_jax]
    rj_pep = r[x_idx_jax]
    disp_pep = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri_pep, rj_pep)
    disp_pep_sq = jnp.sum(disp_pep**2, axis=-1)
    dist_pep = jnp.sqrt(jnp.maximum(disp_pep_sq, 1e-12))
    excess = jnp.maximum(dist_pep - (r0_jax + 0.08), 0.0)
    e_restraint = jnp.sum(0.5 * 100.0 * jnp.square(excess))

    # (D) Harmonic restraints on ALL peptide bonds
    ri_all_pep = r[pep_bond_idx1_jax]
    rj_all_pep = r[pep_bond_idx2_jax]
    disp_all_pep = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri_all_pep, rj_all_pep)
    disp_all_pep_sq = jnp.sum(disp_all_pep**2, axis=-1)
    dist_all_pep = jnp.sqrt(jnp.maximum(disp_all_pep_sq, 1e-12))
    e_pep_bond_restraints = jnp.sum(0.5 * PEPTIDE_BOND_K_EV * jnp.square(dist_all_pep - r0_pep_jax))

    e_dihedral_restraint = _phi_psi_restraint_energy(r)

    return e_intra + e_inter + e_restraint + e_pep_bond_restraints + e_dihedral_restraint



@jit
def _debug_ml_energy(r):
    """JIT-compiled ML intramolecular energy only (for diagnostics)."""
    return compute_monomer_energy(r)


@jit
def _debug_mm_energy_components(r, pi, pj, mask, e14, vdw14):
    """JIT-compiled intermolecular LJ and electrostatic energies."""
    ri = r[pi]; rj = r[pj]
    disp = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri, rj)
    # CRITICAL: Prevent zero displacement for padding pairs (mask=0)
    # and overlapping real pairs to avoid NaN gradients from jnp.linalg.norm at 0.
    disp_sq = jnp.sum(disp**2, axis=-1)
    safe_disp_sq = jnp.where(mask > 0.5, disp_sq, 1.0)
    dist = jnp.sqrt(jnp.maximum(safe_disp_sq, 1e-12))
    within_ctof = (dist * dist < _c2of) * mask
    safe_dist = jnp.where(mask > 0.5, dist, 1.0)
    ep = _pair_lj_epsilon(_eps_jax[pi], _eps_jax[pj])
    sig = _rmin_jax[pi] + _rmin_jax[pj]
    qq = _intermolecular_charge_products(r, pi, pj, e14)
    vdw = _pair_vdw_energy(safe_dist, ep, sig, nb_settings, _vfswitch, use_jax_pme_dispersion=False) * vdw14
    elec = _pair_elec_energy(safe_dist, qq, nb_settings, _fswitch)
    vdw = jnp.where(within_ctof, vdw, 0.0)
    elec = jnp.where(within_ctof, elec, 0.0)
    return jnp.array(
        [jnp.sum(vdw) * KCAL_MOL_TO_EV, jnp.sum(elec) * KCAL_MOL_TO_EV]
    )


@jit
def _debug_mm_energy(r, pi, pj, mask, e14, vdw14):
    """JIT-compiled total intermolecular MM energy (for diagnostics)."""
    return jnp.sum(_debug_mm_energy_components(r, pi, pj, mask, e14, vdw14))


def diagnose_energy(r, label=""):
    """Print energy components to identify NaN source."""
    pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
    e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]

    r_np = np.asarray(r)
    if not np.isfinite(r_np).all():
        n_bad = int((~np.isfinite(r_np)).any(axis=-1).sum())
        print(f"[DIAG{label}] ⚠ Positions: {n_bad} atoms with NaN/Inf — energy will be NaN")
        return
    e_ml = float(_debug_ml_energy(r))
    e_lj, e_elec = map(
        float, _debug_mm_energy_components(r, pi, pj, mask, e14, vdw14)
    )
    e_mm = e_lj + e_elec
    e_tot = e_ml + e_mm
    ml_ok = "✓" if np.isfinite(e_ml) else "✗ NaN/Inf"
    mm_ok = "✓" if np.isfinite(e_mm) else "✗ NaN/Inf"
    print(f"[DIAG{label}] E_ML={e_ml:.4f} eV {ml_ok} | "
          f"E_LJ={e_lj:.4f} eV | E_elec={e_elec:.4f} eV | "
          f"E_MM={e_mm:.4f} eV {mm_ok} | "
          f"E_tot={e_tot:.4f} eV | Pairs_real={int((_mask_ref[0]>0.5).sum())}")
    if PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING:
        q_pep = np.asarray(compute_peptide_ml_charges(r))
        q_water = np.asarray(compute_water_ml_charges(r)).reshape(-1, water_n_atoms)
        print(
            f"[DIAG{label}] q_pep: min={q_pep.min():.4f}, max={q_pep.max():.4f}, "
            f"sum={q_pep.sum():.6f} e | q_water mean by site="
            f"{np.mean(q_water, axis=0)} e | max |water sum|="
            f"{np.max(np.abs(np.sum(q_water, axis=1))):.3e} e"
        )


def run_force_and_nl_diagnostics(r, pi, pj, mask, e14, vdw14, cycle, step):
    """Detailed diagnostics of forces and neighbor lists, creating plots."""
    import matplotlib
    matplotlib.use('Agg') # Ensure non-interactive backend is used
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import networkx as nx
    
    r_np = np.asarray(r)
    pi_np = np.asarray(pi)
    pj_np = np.asarray(pj)
    mask_np = np.asarray(mask)
    
    print(f"\n=== FORCE & NEIGHBOR LIST DIAGNOSTICS (Cycle {cycle}, Step {step}) ===")
    print(f"Positions: min={r_np.min():.3f}, max={r_np.max():.3f}, mean={r_np.mean():.3f}, std={r_np.std():.3f}")
    
    # helper for atom names
    def get_atom_name(idx):
        if 'atoms' in globals() and idx < len(atoms):
            return f"{atoms.get_chemical_symbols()[idx]}{idx}"
        return f"Atom{idx}"

    # 1. Energy Components
    e_ml = float(_debug_ml_energy(r))
    e_mm = float(_debug_mm_energy(r, pi, pj, mask, e14, vdw14))
    print(f"Energies: E_ML={e_ml:.4f} eV | E_MM={e_mm:.4f} eV | E_tot={e_ml+e_mm:.4f} eV")

    # 2. Forces
    f_tot_mag = np.zeros(1)
    f_ml_mag = np.zeros(1)
    f_mm_mag = np.zeros(1)
    f_res_mag = np.zeros(1)
    
    f_tot = None
    f_ml = None
    f_mm = None
    
    try:
        grad_tot = jax.grad(lambda r_: hybrid_energy_fn(r_, pi, pj, mask, e14, vdw14))(r)
        f_tot = -np.asarray(grad_tot)
        f_tot_mag = np.linalg.norm(f_tot, axis=-1)
        print(f"Total Forces: min={f_tot_mag.min():.3f}, max={f_tot_mag.max():.3f}, mean={f_tot_mag.mean():.3f}, std={f_tot_mag.std():.3f}")
        nans_tot = np.isnan(f_tot).sum()
        if nans_tot > 0:
            print(f"  ⚠ WARNING: {nans_tot} NaNs detected in Total Forces!")
    except Exception as e:
        print(f"  Failed to compute Total Forces gradient: {e}")
        
    try:
        grad_ml = jax.grad(lambda r_: compute_monomer_energy(r_))(r)
        f_ml = -np.asarray(grad_ml)
        f_ml_mag = np.linalg.norm(f_ml, axis=-1)
        print(f"ML Monomer Forces: min={f_ml_mag.min():.3f}, max={f_ml_mag.max():.3f}, mean={f_ml_mag.mean():.3f}, std={f_ml_mag.std():.3f}")
        nans_ml = np.isnan(f_ml).sum()
        if nans_ml > 0:
            print(f"  ⚠ WARNING: {nans_ml} NaNs detected in ML Forces!")
    except Exception as e:
        print(f"  Failed to compute ML Forces gradient: {e}")

    try:
        grad_mm = jax.grad(lambda r_: _debug_mm_energy(r_, pi, pj, mask, e14, vdw14))(r)
        f_mm = -np.asarray(grad_mm)
        f_mm_mag = np.linalg.norm(f_mm, axis=-1)
        print(f"MM Nonbonded Forces: min={f_mm_mag.min():.3f}, max={f_mm_mag.max():.3f}, mean={f_mm_mag.mean():.3f}, std={f_mm_mag.std():.3f}")
        nans_mm = np.isnan(f_mm).sum()
        if nans_mm > 0:
            print(f"  ⚠ WARNING: {nans_mm} NaNs detected in MM Forces!")
    except Exception as e:
        print(f"  Failed to compute MM Forces gradient: {e}")

    try:
        def restraint_energy_fn(r_):
            ri_pep = r_[h_idx_jax]
            rj_pep = r_[x_idx_jax]
            disp_pep = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri_pep, rj_pep)
            disp_pep_sq = jnp.sum(disp_pep**2, axis=-1)
            dist_pep = jnp.sqrt(jnp.maximum(disp_pep_sq, 1e-12))
            excess = jnp.maximum(dist_pep - (r0_jax + 0.08), 0.0)
            e_h_x = jnp.sum(0.5 * 100.0 * jnp.square(excess))

            ri_all_pep = r_[pep_bond_idx1_jax]
            rj_all_pep = r_[pep_bond_idx2_jax]
            disp_all_pep = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri_all_pep, rj_all_pep)
            disp_all_pep_sq = jnp.sum(disp_all_pep**2, axis=-1)
            dist_all_pep = jnp.sqrt(jnp.maximum(disp_all_pep_sq, 1e-12))
            e_pep_bonds = jnp.sum(0.5 * PEPTIDE_BOND_K_EV * jnp.square(dist_all_pep - r0_pep_jax))
            
            return e_h_x + e_pep_bonds

        grad_res = jax.grad(restraint_energy_fn)(r)
        f_res = -np.asarray(grad_res)
        f_res_mag = np.linalg.norm(f_res, axis=-1)
        print(f"Restraint Forces: min={f_res_mag.min():.3f}, max={f_res_mag.max():.3f}, mean={f_res_mag.mean():.3f}, std={f_res_mag.std():.3f}")
        nans_res = np.isnan(f_res).sum()
        if nans_res > 0:
            print(f"  ⚠ WARNING: {nans_res} NaNs detected in Restraint Forces!")
    except Exception as e:
        print(f"  Failed to compute Restraint Forces gradient: {e}")

    # 3. Analyze distances for neighbor list pairs
    active_idx = np.where(mask_np > 0.5)[0]
    dists = np.array([])
    if len(active_idx) > 0:
        active_pi = pi_np[active_idx]
        active_pj = pj_np[active_idx]
        ri_active = r_np[active_pi]
        rj_active = r_np[active_pj]
        diff = rj_active - ri_active
        box_sz = np.diag(cell) if cell.ndim == 2 else np.diag(np.diag(cell))
        diff -= box_sz * np.round(diff / box_sz)
        dists = np.linalg.norm(diff, axis=-1)
        
        print(f"Active NL Pairs Count: {len(active_idx)}")
        print(f"Active Pair Distances: min={dists.min():.3f} Å, max={dists.max():.3f} Å, mean={dists.mean():.3f} Å")
        
        close_idx = np.where(dists < 1.2)[0]
        if len(close_idx) > 0:
            print(f"  ⚠ WARNING: {len(close_idx)} pairs are extremely close (< 1.2 Å)!")
            sorted_close = np.argsort(dists[close_idx])
            for idx in sorted_close[:5]:
                p_idx = close_idx[idx]
                print(f"    Atom {active_pi[p_idx]} ({get_atom_name(active_pi[p_idx])}) - Atom {active_pj[p_idx]} ({get_atom_name(active_pj[p_idx])}): dist = {dists[p_idx]:.3f} Å")

    # 4. Generate Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot A: Forces per Atom
    if f_tot is not None:
        axes[0, 0].plot(f_tot_mag, label="Total Force", alpha=0.7)
    if f_ml is not None:
        axes[0, 0].plot(f_ml_mag, label="ML Force", alpha=0.7)
    if f_mm is not None:
        axes[0, 0].plot(f_mm_mag, label="MM Force", alpha=0.7)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel("Atom Index")
    axes[0, 0].set_ylabel("Force Magnitude (eV/Å)")
    axes[0, 0].set_title("Forces per Atom (Log scale)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    # Plot B: Distance matrix log-scale for first 100 atoms
    n_sub = min(100, len(r_np))
    sub_pos = r_np[:n_sub]
    diff_matrix = sub_pos[:, None, :] - sub_pos[None, :, :]
    box_sz = np.diag(cell) if cell.ndim == 2 else np.diag(np.diag(cell))
    diff_matrix -= box_sz * np.round(diff_matrix / box_sz)
    dist_matrix = np.linalg.norm(diff_matrix, axis=-1)
    dist_matrix_safe = np.where(dist_matrix > 0, dist_matrix, 1e-5)
    
    im = axes[0, 1].matshow(dist_matrix_safe, norm=LogNorm(vmin=1e-1, vmax=dist_matrix_safe.max()), cmap='viridis')
    fig.colorbar(im, ax=axes[0, 1], label="Distance (Å)")
    axes[0, 1].set_title(f"Distance Matrix (First {n_sub} Atoms, Log)")

    # Plot C: Adjacency matrix of neighbor list (first 100 atoms)
    adj = np.zeros((n_sub, n_sub))
    if len(active_idx) > 0:
        sub_pairs = (active_pi < n_sub) & (active_pj < n_sub)
        for i, j in zip(active_pi[sub_pairs], active_pj[sub_pairs]):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    
    im_adj = axes[1, 0].matshow(adj, cmap='binary')
    axes[1, 0].set_title(f"Neighbor List Adjacency (First {n_sub} Atoms)")

    # Plot D: Peptide Neighbor List Graph Layout
    if len(active_idx) > 0:
        G = nx.Graph()
        pep_atoms = list(range(42))
        G.add_nodes_from(pep_atoms)
        
        pep_pairs = (active_pi < 42) | (active_pj < 42)
        pep_active_pi = active_pi[pep_pairs]
        pep_active_pj = active_pj[pep_pairs]
        pep_dists = dists[pep_pairs]
        
        close_threshold = 4.5
        for i, j, d in zip(pep_active_pi, pep_active_pj, pep_dists):
            if d < close_threshold:
                G.add_edge(int(i), int(j), weight=float(d))
        
        nodes = list(G.nodes())
        pos_layout = nx.spring_layout(G, seed=42)
        node_colors = ['lightblue' if n < 42 else 'salmon' for n in nodes]
        node_sizes = [150 if n < 42 else 50 for n in nodes]
        
        nx.draw_networkx_nodes(G, pos_layout, ax=axes[1, 1], node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=0.5)
        pep_labels = {n: f"{atoms.get_chemical_symbols()[n]}{n}" if ('atoms' in globals() and n < len(atoms)) else f"Atom{n}" for n in nodes if n < 42}
        nx.draw_networkx_labels(G, pos_layout, labels=pep_labels, ax=axes[1, 1], font_size=8)
        nx.draw_networkx_edges(G, pos_layout, ax=axes[1, 1], alpha=0.2, edge_color='gray')
        
        axes[1, 1].set_title(f"Peptide NL Connections (< {close_threshold} Å)")
        axes[1, 1].axis('off')
        
    plt.tight_layout()
    plot_name = f"diagnostics_cycle_{cycle}_step_{step}.png"
    plt.savefig(plot_name, dpi=150)
    print(f"  Generated diagnostics plot: {plot_name}")
    plt.close()
    print("===============================================================\n")





# ─────────────────────────────────────────────────────────────────────────────
# Vectorized helper functions (no Python loops)
# ─────────────────────────────────────────────────────────────────────────────

def unfold_coordinates(positions, L, mon_stacked_groups):
    """Vectorized coordinate unfolding under PBC.

    Uses precomputed _mon_stacked_groups (list of (stacked_idx, size)) to avoid
    per-monomer Python iteration. Each group is handled with NumPy broadcasting.
    """
    unfolded = np.array(positions)
    for stacked, _sz in mon_stacked_groups:
        # stacked: (N_monomers, size) index array
        coords = unfolded[stacked]           # (N, size, 3)
        ref = coords[:, 0:1, :]             # (N, 1, 3)
        diff = coords - ref
        diff -= L * np.round(diff / L)
        unfolded[stacked] = ref + diff
    return unfolded


def get_max_h_x_bond(positions, box_sz, h_idx, x_idx):
    """Vectorized max H-X bond length computation under PBC."""
    if len(h_idx) == 0:
        return 0.0
    pos_np = np.asarray(positions)
    diff = pos_np[h_idx] - pos_np[x_idx]      # (N_bonds, 3)
    diff -= box_sz * np.round(diff / box_sz)
    return float(np.linalg.norm(diff, axis=-1).max())


def scale_broken_h_bonds(positions, box_sz, h_idx, x_idx, threshold=1.3, target=1.02):
    """Vectorized H-X bond repair: scales back bonds longer than threshold."""
    new_pos = np.array(np.asarray(positions))
    if len(h_idx) == 0:
        return new_pos, False
    diff = new_pos[h_idx] - new_pos[x_idx]    # (N_bonds, 3)
    diff -= box_sz * np.round(diff / box_sz)
    dists = np.linalg.norm(diff, axis=-1)      # (N_bonds,)
    broken = dists > threshold
    if broken.any():
        dirs = diff[broken] / dists[broken, np.newaxis]
        new_pos[h_idx[broken]] = new_pos[x_idx[broken]] + dirs * target
    return new_pos, bool(broken.any())


# Helper function to repair structures using PyCHARMM minimization
def repair_structure_in_charmm(positions):
    print("\n[REPAIR] Temperature spike or NaN detected! Unfolding and repairing structure in CHARMM...")
    pos_np = np.asarray(positions)
    if not np.isfinite(pos_np).all():
        n_bad = int((~np.isfinite(pos_np)).any(axis=-1).sum())
        raise RuntimeError(
            f"[REPAIR] Positions contain NaN/Inf on {n_bad} atoms — "
            "cannot pass to CHARMM (would segfault). "
            "The simulation exploded before the repair point. "
            "Consider reducing dt, FIRE_BLOCK_STEPS, or dt_start/dt_max."
        )
    unfolded_pos = unfold_coordinates(pos_np, box_size, _mon_stacked_groups)
    set_charmm_positions(unfolded_pos)
    lingo.charmm_script("MINI SD NSTEP 1000")
    lingo.charmm_script("MINI ABNR NSTEP 1000")
    repaired_pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    print("[REPAIR] Structure repaired successfully. Re-initializing state.\n")
    return repaired_pos


def choose_finite_repair_positions(candidate, previous_block, last_good, label):
    """Pick finite coordinates for repair, preferring the latest viable state."""
    candidates = [
        ("current", candidate),
        ("pre-block", previous_block),
        ("last-good", last_good),
    ]
    for source, positions in candidates:
        pos_np = np.asarray(positions)
        if pos_np.shape[-1] == 3 and np.isfinite(pos_np).all():
            if source != "current":
                print(
                    f"[{label}] Current positions are non-finite; "
                    f"repairing from {source} finite checkpoint instead."
                )
            return pos_np.copy(), source
    raise RuntimeError(
        f"[{label}] No finite coordinate checkpoint is available for repair. "
        "The integrator state and saved checkpoints are all non-finite."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build simulation runner functions ONCE — never rebuilt inside loops
# ─────────────────────────────────────────────────────────────────────────────

# FIRE minimization
print("--- Building FIRE minimizer (compiled once) ---")
_init_fn_fire, _step_fn_fire = minimize.fire_descent(
    hybrid_energy_fn, shift_fn, dt_start=0.0001, dt_max=0.001
)
_step_fn_fire = jit(_step_fn_fire)

@jit
def run_fire_block(state, pi, pj, mask, e14, vdw14):
    """Run FIRE_BLOCK_STEPS steps of FIRE (compiled once)."""
    def body_fn(i, s):
        return _step_fn_fire(s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
    return jax.lax.fori_loop(0, FIRE_BLOCK_STEPS, body_fn, state)


@jit
def run_fire_block_repair(state, pi, pj, mask, e14, vdw14):
    """Run 200 steps of FIRE for post-repair minimization (compiled once)."""
    def body_fn(i, s):
        return _step_fn_fire(s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
    return jax.lax.fori_loop(0, 200, body_fn, state)


# NVT Nose-Hoover
print("--- Building NVT NHC simulator (compiled once) ---")
from jax_md import quantity

mass = np.zeros(len(pos))
mass[:n_trialanine] = 12.0   # average mass approximation for peptide
mass[n_trialanine::3] = 16.0  # Oxygen
mass[n_trialanine+1::3] = 1.0  # Hydrogen
mass[n_trialanine+2::3] = 1.0  # Hydrogen
jax_mass = jnp.array(mass, dtype=jnp.float64)

_init_fn_nvt, _step_fn_nvt = simulate.nvt_nose_hoover(hybrid_energy_fn, shift_fn, dt, target_temp_ev)
_step_fn_nvt = jit(_step_fn_nvt)

@jit
def run_nvt_block(state, pi, pj, mask, e14, vdw14):
    def body_fn(i, s):
        return _step_fn_nvt(s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
    return jax.lax.fori_loop(0, NVT_BLOCK_STEPS, body_fn, state)


# NVE
print("--- Building NVE simulator (compiled once) ---")
_init_fn_nve, _step_fn_nve = simulate.nve(hybrid_energy_fn, shift_fn, dt)
_step_fn_nve = jit(_step_fn_nve)

@jit
def run_nve_block(state, pi, pj, mask, e14, vdw14):
    def body_fn(i, s):
        return _step_fn_nve(s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
    return jax.lax.fori_loop(0, NVE_BLOCK_STEPS, body_fn, state)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Structure Minimization with JAX-MD FIRE and PyCHARMM Repair Loops
# ─────────────────────────────────────────────────────────────────────────────
print("--- Minimizing System with JAX-MD FIRE and PyCHARMM Repair Loops ---")
init_r = jnp.array(pos, dtype=jnp.float64)
pos_current = init_r

traj_path_fire = str(OUTPUT_DIR / "cg_fire.traj")
print(f"--- Saving minimization trajectory to {traj_path_fire} ---")
traj_fire = DualTrajectoryWriter(
    traj_path_fire,
    atoms,
    write_dcd=bool(_settings.write_dcd),
    dt_ps=dt,
    steps_per_frame=FIRE_BLOCK_STEPS,
)

for cycle in range(FIRE_CYCLES):
    print(f"\n--- Minimization Cycle {cycle+1}/{FIRE_CYCLES} ---")

    # Update pair list from current coordinates (CPU, once per cycle start)
    update_pair_refs(np.asarray(pos_current))
    pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
    e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]

    # --- Pre-FIRE diagnostic: print energy components before any step ---
    diagnose_energy(pos_current, label=f" Cycle{cycle+1} init")
    if DEBUG:
        run_force_and_nl_diagnostics(pos_current, pi, pj, mask, e14, vdw14, cycle+1, 0)

    fire_state = _init_fn_fire(pos_current, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)

    # Write starting configuration of this cycle to trajectory
    curr_f = np.asarray(fire_state.force)
    curr_e = float(hybrid_energy_fn(pos_current, pi, pj, mask, e14, vdw14))
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(pos_current), box_size, _mon_stacked_groups))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_fire.write(frame)

    # Run FIRE blocks — pair list rebuilt every FIRE_BLOCK_STEPS steps
    nan_detected = False
    last_good_pos = np.asarray(pos_current)   # checkpoint: last positions with finite energy
    for step in range(0, FIRE_STEPS, FIRE_BLOCK_STEPS):
        # Save current positions before the block so we can recover if FIRE diverges
        pos_before_block = np.asarray(fire_state.position)

        # Update pair list (CPU side only; no JAX re-trace)
        update_pair_refs(pos_before_block)
        pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
        e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]

        fire_state = run_fire_block(fire_state, pi, pj, mask, e14, vdw14)

        curr_f = np.asarray(fire_state.force)
        curr_e = float(hybrid_energy_fn(fire_state.position, pi, pj, mask, e14, vdw14))
        max_bond = get_max_h_x_bond(fire_state.position, box_size, h_idx_arr, x_idx_arr)
        max_dev, mean_dev = get_peptide_bond_diagnostics(fire_state.position, box_size, pep_bond_idx1_arr, pep_bond_idx2_arr, r0_pep_list)
        print(f"Cycle {cycle+1} | FIRE Step {step+FIRE_BLOCK_STEPS:4d} | "
              f"Energy: {curr_e:.4f} eV | Max H-X Bond: {max_bond:.2f} Å | "
              f"Max/Mean Pep Bond Dev: {max_dev:.4f}/{mean_dev:.4f} Å")


        if not np.isfinite(curr_e) or not np.isfinite(max_bond):
            print(f"[FIRE] NaN/Inf energy detected at step {step+FIRE_BLOCK_STEPS} — "
                  f"reverting to last-good positions (before this block) and repairing")
            # Diagnose using pre-block positions (which are still finite)
            good_pos_jax = jnp.array(pos_before_block, dtype=jnp.float64)
            diagnose_energy(good_pos_jax,
                            label=f" Cycle{cycle+1} pre-step{step+FIRE_BLOCK_STEPS}")
            if DEBUG:
                print("  [DEBUG] Running diagnostics on last good positions before the NaN step:")
                run_force_and_nl_diagnostics(good_pos_jax, pi, pj, mask, e14, vdw14, cycle+1, step)
                print("  [DEBUG] Running diagnostics on the NaN positions (to see which forces are NaNs):")
                try:
                    run_force_and_nl_diagnostics(fire_state.position, pi, pj, mask, e14, vdw14, cycle+1, step+FIRE_BLOCK_STEPS)
                except Exception as e_diag:
                    print(f"Failed to run diagnostics on NaN state: {e_diag}")
            # Re-initialize FIRE from pre-block (finite) positions for CHARMM repair.
            # FireDescentState is a plain dataclass so we use _init_fn_fire, not .replace().
            update_pair_refs(pos_before_block)
            pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
            e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
            fire_state = _init_fn_fire(good_pos_jax, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
            nan_detected = True
            break

        last_good_pos = pos_before_block
        frame = atoms.copy()
        frame.set_positions(unfold_coordinates(np.asarray(fire_state.position), box_size, _mon_stacked_groups))
        frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
        traj_fire.write(frame)

    # Repair the minimized structures in CHARMM at the end of every cycle
    # Use fire_state.position (reverted to pre-NaN if divergence occurred)
    pos_current = jnp.array(repair_structure_in_charmm(fire_state.position), dtype=jnp.float64)

traj_fire.close()
min_r = pos_current
print(f"\nMinimization completed over {FIRE_CYCLES} cycles.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Molecular Dynamics (NVT Nose-Hoover)
# ─────────────────────────────────────────────────────────────────────────────
print("--- Running NVT Nose-Hoover Dynamics with JAX-MD ---")

key = jax.random.PRNGKey(42)

# Initialize starting NVT state
update_pair_refs(np.asarray(min_r))
pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
state = _init_fn_nvt(key, min_r, mass=jax_mass, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)

traj_path_nvt = str(OUTPUT_DIR / "cg_nvt.traj")
print(f"--- Running NVT dynamics and saving trajectory to {traj_path_nvt} ---")
traj_nvt = DualTrajectoryWriter(
    traj_path_nvt,
    atoms,
    write_dcd=bool(_settings.write_dcd),
    dt_ps=dt,
    steps_per_frame=NVT_BLOCK_STEPS,
)
last_good_nvt_pos = np.asarray(min_r, dtype=np.float64)

for step in range(0, NVT_TOTAL_STEPS, NVT_BLOCK_STEPS):
    pos_before_block = np.asarray(state.position, dtype=np.float64)
    if np.isfinite(pos_before_block).all():
        last_good_nvt_pos = pos_before_block.copy()
    else:
        print("[NVT] Non-finite positions detected before block; reverting to last-good checkpoint.")
        pos_before_block = last_good_nvt_pos.copy()
        update_pair_refs(pos_before_block)
        pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
        e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
        state = _init_fn_nvt(key, jnp.array(pos_before_block, dtype=jnp.float64),
                             mass=jax_mass, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)

    # Update pair list on CPU (no JAX re-trace — shapes are fixed)
    update_pair_refs(pos_before_block)
    pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
    e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]

    # Run the compiled block of steps
    state = run_nvt_block(state, pi, pj, mask, e14, vdw14)

    # Compute diagnostics directly from state and JITted hybrid_energy_fn
    curr_f = np.asarray(state.force)
    curr_e = float(hybrid_energy_fn(state.position, pi, pj, mask, e14, vdw14))
    ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
    temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
    max_bond = get_max_h_x_bond(state.position, box_size, h_idx_arr, x_idx_arr)

    # Check for instability and repair
    unstable_nvt = (
        (not np.isfinite(curr_e))
        or (not np.isfinite(temp))
        or (not np.isfinite(max_bond))
        or temp > NVT_REPAIR_TEMP_K
        or max_bond > MAX_HX_BOND_LIMIT
        or (not np.isfinite(np.asarray(state.position)).all())
    )
    if unstable_nvt:
        if temp > NVT_REPAIR_TEMP_K:
            print(f"[REPAIR] NVT temperature exceeded repair threshold: "
                  f"{temp:.1f} K > {NVT_REPAIR_TEMP_K:.1f} K")
        if max_bond > MAX_HX_BOND_LIMIT:
            print(f"[REPAIR] Peptide H-X bond broke! Max bond length: {max_bond:.2f} Å "
                  f"(max limit {MAX_HX_BOND_LIMIT} Å)")
        repair_input_pos, repair_source = choose_finite_repair_positions(
            state.position, pos_before_block, last_good_nvt_pos, "NVT REPAIR"
        )
        # 1. Scale back stretched H-X bonds
        scaled_pos, _ = scale_broken_h_bonds(
            repair_input_pos, box_size, h_idx_arr, x_idx_arr
        )
        # 2. Minimize in PyCHARMM
        print(f"[REPAIR] NVT repair input source: {repair_source}")
        repaired_pos = repair_structure_in_charmm(scaled_pos)
        # 3. Post-repair JAX FIRE minimization
        print("[REPAIR] Running post-repair JAX FIRE minimization (200 steps)...")
        repaired_jax = jnp.array(repaired_pos, dtype=jnp.float64)
        update_pair_refs(repaired_pos)
        pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
        e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
        fire_state_rep = _init_fn_fire(repaired_jax, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
        fire_state_rep = run_fire_block_repair(fire_state_rep, pi, pj, mask, e14, vdw14)
        final_min_pos = jnp.array(fire_state_rep.position, dtype=jnp.float64)

        # Re-initialize NVT state
        update_pair_refs(np.asarray(final_min_pos))
        pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
        e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
        state = _init_fn_nvt(key, final_min_pos, mass=jax_mass, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
        # Re-evaluate diagnostics
        curr_f = np.asarray(state.force)
        curr_e = float(hybrid_energy_fn(state.position, pi, pj, mask, e14, vdw14))
        ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
        temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
        max_bond = get_max_h_x_bond(state.position, box_size, h_idx_arr, x_idx_arr)
        if np.isfinite(np.asarray(state.position)).all() and np.isfinite(curr_e):
            last_good_nvt_pos = np.asarray(state.position, dtype=np.float64).copy()
    else:
        last_good_nvt_pos = np.asarray(state.position, dtype=np.float64).copy()

    max_dev, mean_dev = get_peptide_bond_diagnostics(state.position, box_size, pep_bond_idx1_arr, pep_bond_idx2_arr, r0_pep_list)
    print(f"NVT Step {step+NVT_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | "
          f"Temp: {temp:.1f} K | Max H-X Bond: {max_bond:.2f} Å | "
          f"Max/Mean Pep Bond Dev: {max_dev:.4f}/{mean_dev:.4f} Å")


    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(state.position), box_size, _mon_stacked_groups))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_nvt.write(frame)

traj_nvt.close()
print("NVT dynamics complete!")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Molecular Dynamics (NVE) to check stability
# ─────────────────────────────────────────────────────────────────────────────
print("--- Running NVE Dynamics with JAX-MD to check stability ---")

# Initialize NVE state from final NVT positions
update_pair_refs(np.asarray(state.position))
pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
state_nve = _init_fn_nve(key, state.position, target_temp_ev, mass=jax_mass, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)

traj_path_nve = str(OUTPUT_DIR / "cg_nve.traj")
print(f"--- Running NVE dynamics and saving trajectory to {traj_path_nve} ---")
traj_nve = DualTrajectoryWriter(
    traj_path_nve,
    atoms,
    write_dcd=bool(_settings.write_dcd),
    dt_ps=dt,
    steps_per_frame=NVE_BLOCK_STEPS,
)
last_good_nve_pos = np.asarray(state.position, dtype=np.float64)

# Measure initial NVE total energy baseline for conservation checks
init_e = float(hybrid_energy_fn(state_nve.position, pi, pj, mask, e14, vdw14))
init_ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
initial_nve_energy = init_e + init_ke
print(f"Initial NVE Energy Baseline: {initial_nve_energy:.6f} eV (Potential: {init_e:.6f} eV, Kinetic: {init_ke:.6f} eV)")


for step in range(0, NVE_TOTAL_STEPS, NVE_BLOCK_STEPS):
    pos_before_block = np.asarray(state_nve.position, dtype=np.float64)
    if np.isfinite(pos_before_block).all():
        last_good_nve_pos = pos_before_block.copy()
    else:
        print("[NVE] Non-finite positions detected before block; reverting to last-good checkpoint.")
        pos_before_block = last_good_nve_pos.copy()
        update_pair_refs(pos_before_block)
        pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
        e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
        state_nve = _init_fn_nve(key, jnp.array(pos_before_block, dtype=jnp.float64),
                                 target_temp_ev, mass=jax_mass,
                                 pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)

    # Update pair list on CPU (no JAX re-trace)
    update_pair_refs(pos_before_block)
    pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
    e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]

    # Run the compiled block of steps
    state_nve = run_nve_block(state_nve, pi, pj, mask, e14, vdw14)

    # Compute diagnostics
    curr_f = np.asarray(state_nve.force)
    curr_e = float(hybrid_energy_fn(state_nve.position, pi, pj, mask, e14, vdw14))
    ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
    temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
    max_bond = get_max_h_x_bond(state_nve.position, box_size, h_idx_arr, x_idx_arr)

    # Check for instability and repair
    unstable_nve = (
        (not np.isfinite(curr_e))
        or (not np.isfinite(temp))
        or (not np.isfinite(max_bond))
        or temp > NVE_REPAIR_TEMP_K
        or max_bond > MAX_HX_BOND_LIMIT
        or (not np.isfinite(np.asarray(state_nve.position)).all())
    )
    if unstable_nve:
        if temp > NVE_REPAIR_TEMP_K:
            print(f"[REPAIR] NVE temperature exceeded repair threshold: "
                  f"{temp:.1f} K > {NVE_REPAIR_TEMP_K:.1f} K")
        if max_bond > MAX_HX_BOND_LIMIT:
            print(f"[REPAIR] Peptide H-X bond broke! Max bond length: {max_bond:.2f} Å "
                  f"(max limit {MAX_HX_BOND_LIMIT} Å)")
        repair_input_pos, repair_source = choose_finite_repair_positions(
            state_nve.position, pos_before_block, last_good_nve_pos, "NVE REPAIR"
        )
        # 1. Scale back stretched H-X bonds
        scaled_pos, _ = scale_broken_h_bonds(
            repair_input_pos, box_size, h_idx_arr, x_idx_arr
        )
        # 2. Minimize in PyCHARMM
        print(f"[REPAIR] NVE repair input source: {repair_source}")
        repaired_pos = repair_structure_in_charmm(scaled_pos)
        # 3. Post-repair JAX FIRE minimization
        print("[REPAIR] Running post-repair JAX FIRE minimization (200 steps)...")
        repaired_jax = jnp.array(repaired_pos, dtype=jnp.float64)
        update_pair_refs(repaired_pos)
        pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
        e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
        fire_state_rep = _init_fn_fire(repaired_jax, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
        fire_state_rep = run_fire_block_repair(fire_state_rep, pi, pj, mask, e14, vdw14)
        final_min_pos = jnp.array(fire_state_rep.position, dtype=jnp.float64)

        # Re-initialize NVE state
        update_pair_refs(np.asarray(final_min_pos))
        pi = _pi_ref[0]; pj = _pj_ref[0]; mask = _mask_ref[0]
        e14 = _e14_ref[0]; vdw14 = _vdw14_ref[0]
        state_nve = _init_fn_nve(key, final_min_pos, target_temp_ev, mass=jax_mass, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14)
        # Re-evaluate diagnostics
        curr_f = np.asarray(state_nve.force)
        curr_e = float(hybrid_energy_fn(state_nve.position, pi, pj, mask, e14, vdw14))
        ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
        temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
        max_bond = get_max_h_x_bond(state_nve.position, box_size, h_idx_arr, x_idx_arr)
        if np.isfinite(np.asarray(state_nve.position)).all() and np.isfinite(curr_e):
            last_good_nve_pos = np.asarray(state_nve.position, dtype=np.float64).copy()
            initial_nve_energy = curr_e + ke
            print(f"[REPAIR] Reset NVE Energy Baseline post-repair to: {initial_nve_energy:.6f} eV")
    else:
        last_good_nve_pos = np.asarray(state_nve.position, dtype=np.float64).copy()

    tot_energy = curr_e + ke
    energy_drift = tot_energy - initial_nve_energy
    max_dev, mean_dev = get_peptide_bond_diagnostics(state_nve.position, box_size, pep_bond_idx1_arr, pep_bond_idx2_arr, r0_pep_list)
    print(f"NVE Step {step+NVE_BLOCK_STEPS:5d} | Tot Energy: {tot_energy:.4f} eV | Drift: {energy_drift:.6f} eV | "
          f"Temp: {temp:.1f} K | Max H-X Bond: {max_bond:.2f} Å | "
          f"Max/Mean Pep Bond Dev: {max_dev:.4f}/{mean_dev:.4f} Å")


    # Save frame to trajectory
    frame = atoms.copy()
    frame.set_positions(unfold_coordinates(np.asarray(state_nve.position), box_size, _mon_stacked_groups))
    frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
    traj_nve.write(frame)

traj_nve.close()
print("NVE dynamics complete!")

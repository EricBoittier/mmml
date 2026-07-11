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
# pyrefly: ignore [missing-import]
from cg_common import DualTrajectoryWriter

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

# 1. Initialize JAX and PyCHARMM configuration
jax.config.update("jax_enable_x64", True)
ensure_pycharmm_loaded()
pycharmm_loud()

# Helper function to parse peptide H-X bonds from the CHARMM PSF file
def parse_peptide_h_x_bonds(psf_path, z_array, n_peptide_atoms):
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
                    if idx1 < n_peptide_atoms and idx2 < n_peptide_atoms:
                        z1 = z_array[idx1]
                        z2 = z_array[idx2]
                        # Identify hydrogen (Z=1) to heavy atom (Z>1) bonds
                        if (z1 == 1 and z2 > 1) or (z2 == 1 and z1 > 1):
                            h_idx = idx1 if z1 == 1 else idx2
                            x_idx = idx2 if z1 == 1 else idx1
                            bonds.append((h_idx, x_idx))
    return bonds


def parse_all_peptide_bonds(psf_path, n_peptide_atoms):
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
                    if idx1 < n_peptide_atoms and idx2 < n_peptide_atoms:
                        bonds.append((idx1, idx2))
    return bonds


def get_peptide_bond_diagnostics(r, box_size, idx1_arr, idx2_arr, r0_arr):
    if len(idx1_arr) == 0:
        return 0.0, 0.0
    r_np = np.asarray(r)
    ri = r_np[idx1_arr]
    rj = r_np[idx2_arr]
    dr = ri - rj
    dr -= box_size * np.round(dr / box_size)
    dists = np.linalg.norm(dr, axis=-1)
    diffs = np.abs(dists - r0_arr)
    return float(diffs.max()), float(diffs.mean())




# Default configuration parameters
DEFAULTS = {
    "checkpoint": str(Path(__file__).parent / "sppoky-epoch-0010_params.json"),
    "peptide_checkpoint": None,
    "water_checkpoint": None,

    "sequence": "ALA ALA ALA ALA ALA ALA",  # default is trialanine
    "first_patch": "ACE",
    "last_patch": "CT3",
    "peptide_patches": None,
    "initial_peptide_pdb": None,
    "pdb_id": None,
    "pdb_chain": None,
    "n_waters": 500,
    "box_size": 28.0,
    "seed": 42,
    "temperature": 198.0,
    "dt_fs": 0.5,

    "use_ml_intramolecular": True,
    "peptide_water_ml": True,
    "peptide_water_ml_cutoff_A": 8.0,
    "peptide_water_ml_switch_width_A": 1.0,
    "peptide_water_ml_headroom": 1.5,
    "peptide_water_ml_core_vdw": True,
    "peptide_water_ml_core_cutoff_A": 3.0,
    "peptide_water_ml_core_switch_width_A": 0.5,
    "peptide_water_electrostatic_embedding": False,
    "peptide_ml_charge_total_correction": False,
    "water_ml_charge_total_correction": False,
    "peptide_electrostatic_embedding_require_ml_charges": False,
    "water_electrostatic_embedding_require_ml_charges": False,
    "electrostatics_damping_sigma": None,

    "fire_steps": 10000,
    "fire_block_steps": 1000,
    "nvt_total_steps": 8000000,
    "nvt_block_steps": 1000,
    "nve_total_steps": 8000000,
    "nve_block_steps": 5000,
    "fire_cycles": 2,

    "nl_buffer": 2.0,
    "max_pairs_headroom": 1.15,
    "max_hx_bond_limit": 1.5,
    "max_bond_dev_limit": 0.3,
    "nvt_repair_temp_k": 375.0,
    "nve_repair_temp_k": 400.0,
    "debug": True,
    "debug_expensive_ml_force_components": False,

    "start_peptide_traj_path": None,
    "start_peptide_traj_index": 0,

    "constrain_phi_psi": False,
    "phi_target_deg": None,
    "psi_target_deg": None,
    "dihedral_restraint_k_ev": 1.0,
    "peptide_bond_k_ev": 0.0,

    # --- Steered MD / Jarzynski (end-to-end distance) ---
    # Anchors default to the terminal carbon of each end group of RESI TRIA:
    #   CAY (ACE cap methyl C, 0-based atom 0) and CAT (CT3 cap methyl C, 0-based atom 38).
    "smd_enable": False,
    "smd_atom_i": 0,           # N-terminal end-group carbon (CAY)
    "smd_atom_j": 38,          # C-terminal end-group carbon (CAT)
    "smd_k_ev_a2": 5.0,        # spring constant, eV/Å^2
    "smd_v_a_per_ps": 1.0,     # pull rate (lambda velocity), Å/ps
    "smd_d0_a": None,          # start distance; None -> initial CV value
    "smd_pull_steps": None,    # steps per pull; None -> nvt_total_steps
    "smd_equil_steps": None,   # restrained equilibration steps at d0; None -> nvt_block_steps
    "smd_n_pulls": 20,         # number of independent pulling trajectories
    "smd_pmf_bins": 60,        # histogram bins for Hummer-Szabo PMF

    "write_dcd": True,
    "output_dir": ".",
    "workdir": "/tmp/tria_box",
    "nvt_temp_schedule": None,
}

# Import load_cg_config from cg_common
# pyrefly: ignore [missing-import]
from cg_common import load_cg_config, parse_temp_schedule

# Load configuration
config = load_cg_config(DEFAULTS, description="Hybrid JAX-MD ML/MM Simulator")

# Apply CLI short overrides
import sys
if "--short" in sys.argv:
    config.fire_steps = 10
    config.fire_block_steps = 10
    config.nvt_total_steps = 10
    config.nvt_block_steps = 10
    config.nve_total_steps = 10
    config.nve_block_steps = 10

# Set fallbacks
if config.peptide_checkpoint is None:
    config.peptide_checkpoint = config.checkpoint
if config.water_checkpoint is None:
    config.water_checkpoint = config.checkpoint

# Extract settings to local variables to maintain compatibility with the rest of the script
CKPT_PATH = config.checkpoint
PEPTIDE_CKPT_PATH = config.peptide_checkpoint
WATER_CKPT_PATH = config.water_checkpoint

FIRE_STEPS = config.fire_steps
FIRE_BLOCK_STEPS = config.fire_block_steps
NVT_TOTAL_STEPS = config.nvt_total_steps
NVT_BLOCK_STEPS = config.nvt_block_steps
NVE_TOTAL_STEPS = config.nve_total_steps
NVE_BLOCK_STEPS = config.nve_block_steps
FIRE_CYCLES = config.fire_cycles

NWATER = config.n_waters
BOX_SIDE_A = config.box_size
NL_BUFFER = config.nl_buffer
MAX_PAIRS_HEADROOM = config.max_pairs_headroom
MAX_HX_BOND_LIMIT = config.max_hx_bond_limit
NVT_REPAIR_TEMP_K = config.nvt_repair_temp_k
NVE_REPAIR_TEMP_K = config.nve_repair_temp_k
SEED = config.seed

temperature = config.temperature
kb = 8.617333262145e-5  # eV/K
target_temp_ev = temperature * kb
dt_fs = config.dt_fs
dt = dt_fs * 0.001

PEPTIDE_WATER_ML = config.peptide_water_ml
PEPTIDE_WATER_ML_CUTOFF_A = config.peptide_water_ml_cutoff_A
PEPTIDE_WATER_ML_SWITCH_WIDTH_A = config.peptide_water_ml_switch_width_A
PEPTIDE_WATER_ML_HEADROOM = config.peptide_water_ml_headroom
PEPTIDE_WATER_ML_CORE_VDW = config.peptide_water_ml_core_vdw
PEPTIDE_WATER_ML_CORE_CUTOFF_A = config.peptide_water_ml_core_cutoff_A
PEPTIDE_WATER_ML_CORE_SWITCH_WIDTH_A = config.peptide_water_ml_core_switch_width_A
USE_ML_INTRAMOLECULAR = config.use_ml_intramolecular
PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING = config.peptide_water_electrostatic_embedding
PEPTIDE_ML_CHARGE_TOTAL_CORRECTION = config.peptide_ml_charge_total_correction
WATER_ML_CHARGE_TOTAL_CORRECTION = config.water_ml_charge_total_correction
PEPTIDE_ELECTROSTATIC_EMBEDDING_REQUIRE_ML_CHARGES = config.peptide_electrostatic_embedding_require_ml_charges
WATER_ELECTROSTATIC_EMBEDDING_REQUIRE_ML_CHARGES = config.water_electrostatic_embedding_require_ml_charges
DEBUG = config.debug
DEBUG_EXPENSIVE_ML_FORCE_COMPONENTS = config.debug_expensive_ml_force_components
MAX_BOND_DEV_LIMIT = config.max_bond_dev_limit

START_PEPTIDE_TRAJ_PATH = config.start_peptide_traj_path
START_PEPTIDE_TRAJ_INDEX = config.start_peptide_traj_index

CONSTRAIN_PHI_PSI = config.constrain_phi_psi
PHI_TARGET_DEG = config.phi_target_deg
PSI_TARGET_DEG = config.psi_target_deg
DIHEDRAL_RESTRAINT_K_EV = config.dihedral_restraint_k_ev
PHI_CENTRAL = (14, 16, 18, 24)
PSI_CENTRAL = (16, 18, 24, 26)
PEPTIDE_BOND_K_EV = config.peptide_bond_k_ev

SMD_ENABLE      = config.smd_enable
SMD_ATOM_I      = int(config.smd_atom_i)
SMD_ATOM_J      = int(config.smd_atom_j)
SMD_K_EV_A2     = float(config.smd_k_ev_a2)
SMD_V_A_PER_PS  = float(config.smd_v_a_per_ps)
SMD_D0_A        = config.smd_d0_a
SMD_PULL_STEPS  = config.smd_pull_steps
SMD_EQUIL_STEPS = config.smd_equil_steps
SMD_N_PULLS     = int(config.smd_n_pulls)
SMD_PMF_BINS    = int(config.smd_pmf_bins)

WRITE_DCD = config.write_dcd
OUTPUT_DIR = config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)



NVT_TEMP_SCHEDULE = config.nvt_temp_schedule
if NVT_TEMP_SCHEDULE is not None:
    nvt_temp_func = parse_temp_schedule(NVT_TEMP_SCHEDULE, NVT_TOTAL_STEPS)
else:
    nvt_temp_func = lambda step: temperature


def minimize_peptide_only_in_charmm(sd_steps=1000, abnr_steps=1000):
    """Minimize only the PEPT segment while keeping waters fixed."""
    if not HAS_PEPTIDE:
        print("--- Skipping peptide-only CHARMM minimization: no peptide present ---")
        return
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
    if n_peptide_atoms <= 0:
        return np.asarray(positions, dtype=np.float64).copy()
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
    if not HAS_PEPTIDE:
        raise ValueError("start_peptide_traj_path is only valid when a peptide is present.")
    frame = ase_read(str(traj_path), index=frame_index)
    positions = np.asarray(frame.get_positions(), dtype=np.float64)
    if positions.shape[0] != n_trialanine:
        raise ValueError(
            f"Expected peptide trajectory frame with {n_trialanine} atoms, "
            f"got {positions.shape[0]} from {traj_path}"
        )
    return positions, dict(frame.info)


# 2. Build the initial system in PyCHARMM and minimize
workdir = Path(config.workdir)
from mmml.interfaces.pycharmmInterface.peptide_builder import parse_sequence
sequence_residues = parse_sequence(config.sequence)
SOLVENT_ONLY = len(sequence_residues) == 0

if config.pdb_id is not None and SOLVENT_ONLY:
    raise ValueError("pdb_id cannot be used with an empty solvent-only sequence.")

if config.pdb_id is not None:
    from mmml.interfaces.pycharmmInterface.peptide_builder import prepare_rcsb_peptide_input

    if config.first_patch == "ACE" and config.last_patch == "CT3":
        config.first_patch = None
        config.last_patch = None

    prepared_pdb = prepare_rcsb_peptide_input(
        config.pdb_id,
        chain_id=config.pdb_chain,
        workdir=workdir,
        seg_name="PEPT",
    )
    config.sequence = prepared_pdb["sequence"]
    if config.initial_peptide_pdb is None:
        config.initial_peptide_pdb = str(prepared_pdb["chain_pdb_path"])
    config.peptide_patches = [
        *(config.peptide_patches or []),
        *prepared_pdb["peptide_patches"],
    ]
    print(
        f"--- Prepared RCSB {config.pdb_id.upper()} "
        f"chain {prepared_pdb['chain_id'] or 'blank'}: "
        f"{len(config.sequence)} residues, "
        f"{len(prepared_pdb['peptide_patches'])} DISU patches ---"
    )
    sequence_residues = parse_sequence(config.sequence)

if SOLVENT_ONLY:
    if NWATER <= 0:
        raise ValueError("Solvent-only mode requires n_waters > 0.")
    if START_PEPTIDE_TRAJ_PATH is not None:
        raise ValueError("start_peptide_traj_path cannot be used in solvent-only mode.")
    if CONSTRAIN_PHI_PSI:
        raise ValueError("constrain_phi_psi cannot be used in solvent-only mode.")
    print(f"--- Building TIP3 solvent-only box with {NWATER} waters in CHARMM ---")
    from mmml.interfaces.pycharmmInterface.tip3_liquid_box import build_tip3_liquid_box_in_charmm

    box = build_tip3_liquid_box_in_charmm(
        n_waters=NWATER,
        box_side_A=BOX_SIDE_A,
        seed=SEED,
        workdir=workdir,
    )
elif NWATER > 0:
    seq_clean = " ".join(sequence_residues)
    use_trialanine_template = (
        seq_clean == "ALA ALA ALA"
        and config.first_patch == "ACE"
        and config.last_patch == "CT3"
        and not config.peptide_patches
        and config.initial_peptide_pdb is None
    )
    if use_trialanine_template:
        print("--- Building Trialanine Water Box in CHARMM ---")
        box = build_trialanine_water_box_in_charmm(n_waters=NWATER,
            box_side_A=BOX_SIDE_A, seed=SEED, workdir=workdir
            )
    else:
        print(f"--- Building Solvated Peptide ({config.sequence}) with {NWATER} waters in CHARMM ---")
        from mmml.interfaces.pycharmmInterface.peptide_builder import (
            build_peptide_in_charmm,
            solvate_peptide_in_charmm,
        )
        peptide = build_peptide_in_charmm(
            sequence=config.sequence,
            first_patch=config.first_patch,
            last_patch=config.last_patch,
            peptide_patches=config.peptide_patches,
            initial_pdb_path=config.initial_peptide_pdb,
            seed=SEED,
            workdir=workdir,
            minimize=True,
        )
        box = solvate_peptide_in_charmm(
            peptide,
            box_side_A=BOX_SIDE_A,
            n_waters=NWATER,
            seed=SEED,
            workdir=workdir,
        )
else:
    print(f"--- Building Gas-Phase Peptide ({config.sequence}) in CHARMM ---")
    from mmml.interfaces.pycharmmInterface.peptide_builder import build_peptide_in_charmm, SolvatedPeptideBox, PbcNbondCutoffs

    build_result = build_peptide_in_charmm(
        sequence=config.sequence,
        first_patch=config.first_patch,
        last_patch=config.last_patch,
        peptide_patches=config.peptide_patches,
        initial_pdb_path=config.initial_peptide_pdb,
        seed=SEED,
        workdir=workdir,
        minimize=True,
    )

    # Setup dummy PbcNbondCutoffs for nonbonded settings in gas phase
    nbond_cutoffs = PbcNbondCutoffs(
        cubic_box_side_A=BOX_SIDE_A,
        cutnb=15.0,
        cutim=15.0,
        ctonnb=10.0,
        ctofnb=12.0,
        ctexnb=999.0,
    )
    box = SolvatedPeptideBox(
        positions=np.asarray(build_result.positions, dtype=np.float64),
        psf_path=build_result.psf_path,
        pdb_path=build_result.pdb_path,
        box_side_A=BOX_SIDE_A,
        n_peptide_atoms=build_result.n_atoms,
        n_waters=0,
        nbond_cutoffs=nbond_cutoffs,
        sequence=sequence_residues,
    )

pos = np.asarray(box.positions, dtype=np.float64)
pos = np.random.uniform(-0.1, 0.1, pos.shape) + pos

box_center = np.array([box.box_side_A / 2, box.box_side_A / 2, box.box_side_A / 2])
n_trialanine = int(getattr(box, "n_peptide_atoms", 0 if SOLVENT_ONLY else 42))
HAS_PEPTIDE = n_trialanine > 0

if HAS_PEPTIDE:
    from mmml.interfaces.pycharmmInterface.peptide_builder import infer_charge_and_spin_from_psf
    pep_charge, pep_spin = infer_charge_and_spin_from_psf(box.psf_path)
    print(f"Inferred peptide charge={pep_charge}, spin multiplicity={pep_spin} from PSF.")
    peptide_center = pos[:n_trialanine].mean(axis=0)
    pos += box_center - peptide_center
else:
    pep_charge, pep_spin = 0, 1.0
    print("No peptide present; configuring solvent-only simulation.")
    if PEPTIDE_WATER_ML:
        print("--- Disabling peptide_water_ml: no peptide-water pairs exist ---")
        PEPTIDE_WATER_ML = False
    if PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING:
        print("--- Disabling peptide-water electrostatic embedding: no peptide-water pairs exist ---")
        PEPTIDE_WATER_ELECTROSTATIC_EMBEDDING = False

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

if USE_ML_INTRAMOLECULAR and HAS_PEPTIDE:
    peptide_calc = create_calculator_from_checkpoint(
        PEPTIDE_CKPT_PATH,
        electrostatics_damping_sigma=config.electrostatics_damping_sigma,
    )
    peptide_model = getattr(peptide_calc, "model", getattr(peptide_calc, "_mmml_physnet_model", None))
    peptide_params = getattr(peptide_calc, "params", getattr(peptide_calc, "_mmml_physnet_params", None))

    if peptide_model is None or peptide_params is None:
        raise ValueError("Could not extract model or params from the loaded peptide calculator.")
else:
    peptide_calc = None
    peptide_model = None
    peptide_params = None

set_charmm_positions(pos)

minimize_peptide_only_in_charmm()

if USE_ML_INTRAMOLECULAR:
    pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
    pos = minimize_peptide_only_with_ml_calculator(pos, z, peptide_calc, workdir, n_trialanine)
    set_charmm_positions(pos)

lingo.charmm_script("MINI SD NSTEP 10000")
lingo.charmm_script("MINI ABNR NSTEP 10000")

# Retrieve positions and atomic numbers
pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
z = get_Z_from_psf()
pep_h_x_bonds = parse_peptide_h_x_bonds(box.psf_path, z, n_trialanine)
print(f"Parsed {len(pep_h_x_bonds)} peptide H-X bonds from PSF.")
pep_all_bonds = parse_all_peptide_bonds(box.psf_path, n_trialanine)
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

# Precompute excluded pairs early for intramolecular setup
excluded_pairs = nbond_data.excluded_pairs
if nbond_data.psf_path is not None and nbond_data.psf_bonds is not None:
    excluded_pairs = resolve_nonbonded_excluded_pairs(
        nbond_data.psf_path,
        nbond_data.psf_bonds,
        natom=int(np.asarray(nbond_data.charges).shape[0]),
    )

from mmml.interfaces.pycharmmInterface.mm_system_energy import load_bonded_system_from_psf
from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
    bonded_energy_components,
    KCAL_MOL_TO_EV,
)
# Load full system CGenFF bonded topology/parameters (all atoms)
bonded_system = load_bonded_system_from_psf(psf_path, pos, prm_file=prm_path)

# Set up static peptide intramolecular nonbonded pairs if we are mocking the ML potential
if not USE_ML_INTRAMOLECULAR:
    _pi_list = []
    _pj_list = []
    _e14_list = []
    _vdw14_list = []
    for _i in range(n_trialanine):
        for _j in range(_i + 1, n_trialanine):
            _pair = (_i, _j)
            if _pair in excluded_pairs:
                continue
            _pi_list.append(_i)
            _pj_list.append(_j)
            if _pair in nbond_data.e14_pairs:
                _e14_list.append(nb_settings.e14fac)
                _vdw14_list.append(nb_settings.vdw14fac)
            else:
                _e14_list.append(1.0)
                _vdw14_list.append(1.0)

    pep_intra_pi = jnp.array(_pi_list, dtype=jnp.int32)
    pep_intra_pj = jnp.array(_pj_list, dtype=jnp.int32)
    pep_intra_e14 = jnp.array(_e14_list, dtype=jnp.float64)
    pep_intra_vdw14 = jnp.array(_vdw14_list, dtype=jnp.float64)
else:
    pep_intra_pi = None
    pep_intra_pj = None
    pep_intra_e14 = None
    pep_intra_vdw14 = None

if USE_ML_INTRAMOLECULAR:
    water_calc = create_calculator_from_checkpoint(
        WATER_CKPT_PATH,
        electrostatics_damping_sigma=config.electrostatics_damping_sigma,
    )
    water_model = getattr(water_calc, "model", getattr(water_calc, "_mmml_physnet_model", None))
    water_params = getattr(water_calc, "params", getattr(water_calc, "_mmml_physnet_params", None))

    if water_model is None or water_params is None:
        raise ValueError("Could not extract model or params from the loaded water calculator.")
else:
    water_calc = None
    water_model = None
    water_params = None

# Setup monomer groupings: optional peptide first, then groups of 3 for water.
monomer_indices = [np.arange(n_trialanine)] if HAS_PEPTIDE else []
for i in range(n_trialanine, len(pos), 3):
    monomer_indices.append(np.arange(i, i + 3))

# Pre-convert to JAX arrays for performance
jax_monomer_indices = [jnp.array(idx, dtype=jnp.int32) for idx in monomer_indices]
jax_z = jnp.array(z, dtype=jnp.int32)

# Set up molecule IDs for nonbonded interactions
molecule_id = np.empty(len(pos), dtype=np.int32)
molecule_id[:n_trialanine] = 0
first_water_mol_id = 1 if HAS_PEPTIDE else 0
for mol_id, start in enumerate(range(n_trialanine, len(pos), 3), start=first_water_mol_id):
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
    if not USE_ML_INTRAMOLECULAR:
        raise ValueError("PEPTIDE_WATER_ML requires use_ml_intramolecular=True.")
    print(
        "--- Configuring PEPTIDE-WATER interactions with foundation ML model "
        "(peptide-water dimers) ---"
    )
    print(
        "--- Peptide-water ML interaction uses smooth energy/force switch: "
        f"cutoff={PEPTIDE_WATER_ML_CUTOFF_A} Å, "
        f"width={PEPTIDE_WATER_ML_SWITCH_WIDTH_A} Å ---"
    )
    if PEPTIDE_WATER_ML_CORE_VDW:
        print(
            "--- Peptide-water ML uses additive repulsive CHARMM LJ core: "
            f"cutoff={PEPTIDE_WATER_ML_CORE_CUTOFF_A} Å, "
            f"width={PEPTIDE_WATER_ML_CORE_SWITCH_WIDTH_A} Å ---"
        )
    dimer_size = n_trialanine + 3
    ml_charges = {n_trialanine: float(pep_charge), 3: 0.0, dimer_size: float(pep_charge)}
    ml_spins = {n_trialanine: float(pep_spin), 3: 1.0, dimer_size: float(pep_spin)}
    compute_peptide_water_ml_energy = make_peptide_water_ml_energy_fn(
        peptide_model,
        peptide_params,
        jax_z,
        jax_monomer_indices[0],
        jax_monomer_indices[1:],
        displacement_fn,
        charges=ml_charges,
        spins=ml_spins,
        interaction_cutoff_A=PEPTIDE_WATER_ML_CUTOFF_A,
        interaction_switch_width_A=PEPTIDE_WATER_ML_SWITCH_WIDTH_A,
    )
    compute_peptide_monomer_energy = make_monomer_energy_fn(
        peptide_model,
        peptide_params,
        jax_z,
        [jax_monomer_indices[0]],
        displacement_fn,
        charges=ml_charges,
        spins=ml_spins,
    )
    compute_water_monomer_energy = make_monomer_energy_fn(
        peptide_model,
        peptide_params,
        jax_z,
        jax_monomer_indices[1:],
        displacement_fn,
        charges=ml_charges,
        spins=ml_spins,
    )

    def compute_monomer_energy(r, pw_slots=None, pw_mask=None):
        return compute_peptide_water_ml_energy(r, pw_slots, pw_mask)

    def compute_ml_bonded_energy(r, pw_slots=None, pw_mask=None):
        return compute_peptide_monomer_energy(r) + compute_water_monomer_energy(r)

    def compute_ml_nonbonded_energy(r, pw_slots=None, pw_mask=None):
        return compute_monomer_energy(r, pw_slots, pw_mask) - compute_ml_bonded_energy(r)
else:
    print("--- Configuring PEPTIDE-WATER interactions with MM ---")
    if USE_ML_INTRAMOLECULAR:
        monomer_charges = {3: 0.0}
        monomer_spins = {3: 1.0}
        if HAS_PEPTIDE:
            monomer_charges[n_trialanine] = float(pep_charge)
            monomer_spins[n_trialanine] = float(pep_spin)
            compute_peptide_energy = make_monomer_energy_fn(
                peptide_model, peptide_params, jax_z, [jax_monomer_indices[0]], displacement_fn,
                charges=monomer_charges, spins=monomer_spins
            )
            water_monomer_indices = jax_monomer_indices[1:]
        else:
            compute_peptide_energy = None
            water_monomer_indices = jax_monomer_indices
        compute_water_energy = make_monomer_energy_fn(
            water_model, water_params, jax_z, water_monomer_indices, displacement_fn,
            charges=monomer_charges, spins=monomer_spins
        )

        def compute_monomer_energy(r, pw_slots=None, pw_mask=None):
            e_water = compute_water_energy(r)
            if compute_peptide_energy is None:
                return e_water
            return compute_peptide_energy(r) + e_water

        def compute_ml_bonded_energy(r, pw_slots=None, pw_mask=None):
            return compute_monomer_energy(r, pw_slots, pw_mask)

        def compute_ml_nonbonded_energy(r, pw_slots=None, pw_mask=None):
            return jnp.asarray(0.0, dtype=jnp.float64)
    else:
        @jit
        def compute_peptide_intra_nb_energy(r):
            ri = r[pep_intra_pi]
            rj = r[pep_intra_pj]
            disp = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri, rj)
            disp_sq = jnp.sum(disp**2, axis=-1)
            dist = jnp.sqrt(jnp.maximum(disp_sq, 1e-12))

            ep = _pair_lj_epsilon(_eps_jax[pep_intra_pi], _eps_jax[pep_intra_pj])
            sig = _rmin_jax[pep_intra_pi] + _rmin_jax[pep_intra_pj]
            qq = _q_jax[pep_intra_pi] * _q_jax[pep_intra_pj]

            vdw = _pair_vdw_energy(dist, ep, sig, nb_settings, _vfswitch, use_jax_pme_dispersion=False)
            vdw = vdw * pep_intra_vdw14
            elec = _pair_elec_energy(dist, qq, nb_settings, _fswitch)
            elec = elec * pep_intra_e14

            within_ctof = dist * dist < _c2of
            vdw = jnp.where(within_ctof, vdw, 0.0)
            elec = jnp.where(within_ctof, elec, 0.0)

            return (jnp.sum(vdw) + jnp.sum(elec)) * KCAL_MOL_TO_EV

        @jit
        def compute_monomer_energy(r, pw_slots=None, pw_mask=None):
            bonded_comp = bonded_energy_components(
                r,
                bonded_system.topology,
                bonded_system.bonded,
                displacement_fn,
                urey_k=bonded_system.urey_k,
                urey_r0=bonded_system.urey_r0,
                include_cmap=True,
            )
            e_bonded = bonded_comp["total"] * KCAL_MOL_TO_EV
            e_pep_intra_nb = compute_peptide_intra_nb_energy(r)
            return e_bonded + e_pep_intra_nb

        def compute_ml_bonded_energy(r, pw_slots=None, pw_mask=None):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def compute_ml_nonbonded_energy(r, pw_slots=None, pw_mask=None):
            return jnp.asarray(0.0, dtype=jnp.float64)

pep_n_atoms = int(n_trialanine)
if HAS_PEPTIDE:
    pep_idx_jax = jax_monomer_indices[0]
    pep_z_jax = jax_z[pep_idx_jax]
    pep_dst_idx_np, pep_src_idx_np = np.where(~np.eye(pep_n_atoms, dtype=bool))
    pep_dst_idx_jax = jnp.array(pep_dst_idx_np, dtype=jnp.int32)
    pep_src_idx_jax = jnp.array(pep_src_idx_np, dtype=jnp.int32)
else:
    pep_idx_jax = jnp.zeros(0, dtype=jnp.int32)
    pep_z_jax = jnp.zeros(0, dtype=jnp.int32)
    pep_dst_idx_jax = jnp.zeros(0, dtype=jnp.int32)
    pep_src_idx_jax = jnp.zeros(0, dtype=jnp.int32)
pep_ref_charge_total = float(np.asarray(nbond_data.charges[:n_trialanine]).sum())

water_indices = monomer_indices[1:] if HAS_PEPTIDE else monomer_indices
if len(water_indices) > 0:
    water_stacked_idx_jax = jnp.array(np.stack(water_indices), dtype=jnp.int32)
    water_n_atoms = 3
    water_dst_idx_np, water_src_idx_np = np.where(~np.eye(water_n_atoms, dtype=bool))
    water_dst_idx_jax = jnp.array(water_dst_idx_np, dtype=jnp.int32)
    water_src_idx_jax = jnp.array(water_src_idx_np, dtype=jnp.int32)
    water_z_jax = jax_z[water_stacked_idx_jax]
    water_flat_idx_jax = water_stacked_idx_jax.reshape(-1)
    water_ref_charge_total = float(np.asarray(nbond_data.charges[n_trialanine:n_trialanine + water_n_atoms]).sum())
else:
    water_stacked_idx_jax = jnp.zeros((0, 3), dtype=jnp.int32)
    water_n_atoms = 3
    water_dst_idx_jax = jnp.zeros((0, 2), dtype=jnp.int32)
    water_src_idx_jax = jnp.zeros((0, 2), dtype=jnp.int32)
    water_z_jax = jnp.zeros((0, 3), dtype=jnp.int32)
    water_flat_idx_jax = jnp.zeros(0, dtype=jnp.int32)
    water_ref_charge_total = 0.0


def compute_peptide_ml_charges(r):
    """Return geometry-dependent peptide atomic charges from the peptide ML model."""
    if not HAS_PEPTIDE:
        return jnp.zeros(0, dtype=jnp.float64)
    if not USE_ML_INTRAMOLECULAR:
        return _q_jax[:n_trialanine]
    pep_pos = r[pep_idx_jax]
    ref_pos = pep_pos[0]
    unfolded = ref_pos + jax.vmap(displacement_fn, in_axes=(0, None))(pep_pos, ref_pos)
    centered = unfolded - jnp.mean(unfolded, axis=0, keepdims=True)
    # is_pep_spooky = (
    #     hasattr(peptide_model, "charges")
    #     and hasattr(peptide_model, "total_charge")
    # ) or "spooky" in str(type(peptide_model)).lower()
    import inspect

    def _accepts_kwarg(model, name):
        try:
            return name in inspect.signature(model.__call__).parameters
        except Exception:
            return False
    is_pep_spooky = (
        _accepts_kwarg(peptide_model, "charges")
        and _accepts_kwarg(peptide_model, "spins")
    )

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
    if not USE_ML_INTRAMOLECULAR:
        return _q_jax[water_flat_idx_jax].reshape((-1, water_n_atoms))
    group_pos = r[water_stacked_idx_jax]
    ref_pos = group_pos[:, 0, :]
    displacements = jax.vmap(
        jax.vmap(displacement_fn, in_axes=(0, None)),
        in_axes=(0, 0),
    )(group_pos, ref_pos)
    unfolded = ref_pos[:, None, :] + displacements
    centered = unfolded - jnp.mean(unfolded, axis=1, keepdims=True)

    # is_water_spooky = (
    #     hasattr(water_model, "charges")
    #     and hasattr(water_model, "total_charge")
    # ) or "spooky" in str(type(water_model)).lower()
    import inspect
    def _accepts_kwarg(model, name):
        try:
            return name in inspect.signature(model.__call__).parameters
        except Exception:
            return False

    is_water_spooky = (
        _accepts_kwarg(water_model, "charges")
        and _accepts_kwarg(water_model, "spins")
    )

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
# excluded_pairs is already resolved and precomputed at setup

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

_pw_water_indices_np = (
    np.stack(monomer_indices[1:]).astype(np.int32)
    if PEPTIDE_WATER_ML and HAS_PEPTIDE and len(monomer_indices) > 1
    else np.zeros((0, 3), dtype=np.int32)
)


def _get_peptide_water_active_slots_np(pos_np):
    """Return water slot ids whose atoms are inside the peptide-water ML skin."""
    n_waters_local = int(_pw_water_indices_np.shape[0])
    if not PEPTIDE_WATER_ML or not HAS_PEPTIDE or n_waters_local == 0:
        return np.zeros(0, dtype=np.int32)
    if PEPTIDE_WATER_ML_CUTOFF_A is None:
        return np.arange(n_waters_local, dtype=np.int32)

    active_radius = float(PEPTIDE_WATER_ML_CUTOFF_A) + float(NL_BUFFER)
    boxsize = np.diag(cell)
    pep_pos = np.asarray(pos_np[:n_trialanine], dtype=np.float64)
    water_pos = np.asarray(pos_np[_pw_water_indices_np], dtype=np.float64)
    diff = water_pos[:, None, :, :] - pep_pos[None, :, None, :]
    diff -= boxsize * np.round(diff / boxsize)
    min_dist = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=(1, 2)))
    return np.where(min_dist < active_radius)[0].astype(np.int32)


def _peptide_water_distance_stats_np(pos_np, slots_np=None):
    """Return min/mean active peptide-water atom distances under MIC."""
    if not PEPTIDE_WATER_ML or not HAS_PEPTIDE or len(_pw_water_indices_np) == 0:
        return float("nan"), float("nan")
    water_indices_np = _pw_water_indices_np if slots_np is None else _pw_water_indices_np[slots_np]
    if len(water_indices_np) == 0:
        return float("nan"), float("nan")
    boxsize = np.diag(cell)
    pep_pos = np.asarray(pos_np[:n_trialanine], dtype=np.float64)
    water_pos = np.asarray(pos_np[water_indices_np], dtype=np.float64)
    diff = water_pos[:, None, :, :] - pep_pos[None, :, None, :]
    diff -= boxsize * np.round(diff / boxsize)
    min_by_water = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=(1, 2)))
    return float(min_by_water.min()), float(min_by_water.mean())


_pw_slots0 = _get_peptide_water_active_slots_np(pos)
_pw_min0, _pw_mean0 = _peptide_water_distance_stats_np(pos, _pw_slots0)
if PEPTIDE_WATER_ML and HAS_PEPTIDE and len(_pw_water_indices_np) > 0:
    if PEPTIDE_WATER_ML_CUTOFF_A is None:
        MAX_PW_ML_WATERS = len(_pw_water_indices_np)
    else:
        MAX_PW_ML_WATERS = min(
            len(_pw_water_indices_np),
            max(16, int(len(_pw_slots0) * float(PEPTIDE_WATER_ML_HEADROOM)) + 16),
        )
else:
    MAX_PW_ML_WATERS = 1
print(
    f"--- Initial peptide-water ML active waters: {len(_pw_slots0)}, "
    f"MAX_PW_ML_WATERS buffer: {MAX_PW_ML_WATERS}, "
    f"min/mean active PW distance: {_pw_min0:.3f}/{_pw_mean0:.3f} Å ---"
)


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


def _pad_peptide_water_slots(slots_np):
    """Pad active peptide-water ML water slots to fixed JAX shape."""
    n = len(slots_np)
    assert n <= MAX_PW_ML_WATERS, (
        f"Peptide-water ML active water count {n} exceeds "
        f"MAX_PW_ML_WATERS={MAX_PW_ML_WATERS}. Increase "
        "peptide_water_ml_headroom or nl_buffer, or reduce peptide_water_ml_cutoff_A."
    )
    slots_pad = np.zeros(MAX_PW_ML_WATERS, dtype=np.int32)
    mask = np.zeros(MAX_PW_ML_WATERS, dtype=np.float64)
    slots_pad[:n] = slots_np
    mask[:n] = 1.0
    return slots_pad, mask


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
_pw_slots_ref = [jnp.zeros(MAX_PW_ML_WATERS, dtype=jnp.int32)]
_pw_mask_ref = [jnp.zeros(MAX_PW_ML_WATERS, dtype=jnp.float64)]

# Precompute static per-atom arrays used inside the energy function
_q_jax = jnp.asarray(nbond_data.charges, dtype=jnp.float64)
_eps_jax = jnp.asarray(nbond_data.epsilon, dtype=jnp.float64)
_rmin_jax = jnp.asarray(nbond_data.rmin, dtype=jnp.float64)
_cell_jax = jnp.asarray(cell if cell.ndim == 2 else np.diag(cell), dtype=jnp.float64)
_pw_water_indices_jax = jnp.asarray(_pw_water_indices_np, dtype=jnp.int32)
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
    pw_slots_np = _get_peptide_water_active_slots_np(pos_np)

    pi_pad, pj_pad, mask = _pad_pairs(pi_np, pj_np)
    e14_pad, vdw14_pad = _pad_e14_vdw14(e14_np, vdw14_np)
    pw_slots_pad, pw_mask = _pad_peptide_water_slots(pw_slots_np)

    _pi_ref[0] = jnp.array(pi_pad, dtype=jnp.int32)
    _pj_ref[0] = jnp.array(pj_pad, dtype=jnp.int32)
    _mask_ref[0] = jnp.array(mask, dtype=jnp.float64)
    _e14_ref[0] = jnp.array(e14_pad, dtype=jnp.float64)
    _vdw14_ref[0] = jnp.array(vdw14_pad, dtype=jnp.float64)
    _pw_slots_ref[0] = jnp.array(pw_slots_pad, dtype=jnp.int32)
    _pw_mask_ref[0] = jnp.array(pw_mask, dtype=jnp.float64)


def _current_energy_refs():
    return (
        _pi_ref[0],
        _pj_ref[0],
        _mask_ref[0],
        _e14_ref[0],
        _vdw14_ref[0],
        _pw_slots_ref[0],
        _pw_mask_ref[0],
    )


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
    if not HAS_PEPTIDE:
        return 0.0
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


def end_to_end_distance(r):
    """MIC end-to-end distance (Å) between the two SMD anchor atoms."""
    disp = mic_displacement(r[SMD_ATOM_I], r[SMD_ATOM_J], _cell_jax)
    return jnp.sqrt(jnp.sum(disp * disp) + 1e-12)


def smd_bias_energy(r, lambda_t):
    """Moving harmonic restraint V = 1/2 k (d(r) - lambda_t)^2 on the end-to-end CV (eV)."""
    d = end_to_end_distance(r)
    return 0.5 * SMD_K_EV_A2 * jnp.square(d - lambda_t)


def _smooth_cutoff_weight(dist, cutoff, width):
    width = float(max(0.0, min(float(width), float(cutoff))))
    if width <= 0.0:
        return jnp.where(dist < cutoff, 1.0, 0.0)
    switch_on = float(cutoff) - width
    t = jnp.clip((dist - switch_on) / width, 0.0, 1.0)
    smoothstep = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    return 1.0 - smoothstep


def compute_peptide_water_core_vdw_energy(r, pw_slots=None, pw_mask=None):
    """Short-range repulsive CHARMM LJ wall for ML peptide-water pairs."""
    if not PEPTIDE_WATER_ML_CORE_VDW or not PEPTIDE_WATER_ML or not HAS_PEPTIDE:
        return jnp.asarray(0.0, dtype=jnp.float64)
    if _pw_water_indices_jax.shape[0] == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)

    if pw_slots is None:
        active_water_indices = _pw_water_indices_jax
        active_mask = jnp.ones((_pw_water_indices_jax.shape[0],), dtype=jnp.float64)
    else:
        active_water_indices = _pw_water_indices_jax[pw_slots]
        active_mask = jnp.asarray(pw_mask, dtype=jnp.float64)

    pep_pos = r[:n_trialanine]
    water_pos = r[active_water_indices]
    disp = water_pos[:, None, :, :] - pep_pos[None, :, None, :]
    box_diag = jnp.diag(_cell_jax)
    disp = disp - box_diag * jnp.round(disp / box_diag)
    dist_sq = jnp.sum(disp * disp, axis=-1)
    dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-12))

    pep_idx = jnp.arange(n_trialanine, dtype=jnp.int32)
    ep = _pair_lj_epsilon(
        _eps_jax[pep_idx][None, :, None],
        _eps_jax[active_water_indices][:, None, :],
    )
    sig = _rmin_jax[pep_idx][None, :, None] + _rmin_jax[active_water_indices][:, None, :]
    sig_r6 = (sig / jnp.maximum(dist, 1e-10)) ** 6
    vdw_full = ep * (sig_r6 * sig_r6 - 2.0 * sig_r6)
    repulsive = jnp.maximum(vdw_full, 0.0)
    weights = _smooth_cutoff_weight(
        dist,
        PEPTIDE_WATER_ML_CORE_CUTOFF_A,
        PEPTIDE_WATER_ML_CORE_SWITCH_WIDTH_A,
    )
    return jnp.sum(active_mask[:, None, None] * weights * repulsive) * KCAL_MOL_TO_EV


@jit
def hybrid_energy_fn(
    r,
    pi=None,
    pj=None,
    mask=None,
    e14=None,
    vdw14=None,
    pw_slots=None,
    pw_mask=None,
    smd_lambda=None,
    **unused_kwargs,
) -> jnp.ndarray:
    """Single JIT-compiled hybrid ML/MM energy function.

    Takes padded pair arrays as keyword arguments. Since their shapes (MAX_PAIRS)
    are constant, JAX compiles the simulators exactly once and reuses the graph
    when new array values are passed in.
    """
    # (A) Intramolecular terms from ML potential
    e_intra = compute_monomer_energy(r, pw_slots, pw_mask)
    e_pw_core = compute_peptide_water_core_vdw_energy(r, pw_slots, pw_mask)

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

    if HAS_PEPTIDE:
        # (C) Flat-bottom harmonic restraints on peptide H-X bonds.
        ri_pep = r[h_idx_jax]
        rj_pep = r[x_idx_jax]
        disp_pep = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri_pep, rj_pep)
        disp_pep_sq = jnp.sum(disp_pep**2, axis=-1)
        dist_pep = jnp.sqrt(jnp.maximum(disp_pep_sq, 1e-12))
        excess = jnp.maximum(dist_pep - (r0_jax + 0.08), 0.0)
        e_restraint = jnp.sum(0.5 * 100.0 * jnp.square(excess))

        # (D) Harmonic restraints on ALL peptide bonds.
        ri_all_pep = r[pep_bond_idx1_jax]
        rj_all_pep = r[pep_bond_idx2_jax]
        disp_all_pep = jax.vmap(lambda a, b: mic_displacement(a, b, _cell_jax))(ri_all_pep, rj_all_pep)
        disp_all_pep_sq = jnp.sum(disp_all_pep**2, axis=-1)
        dist_all_pep = jnp.sqrt(jnp.maximum(disp_all_pep_sq, 1e-12))
        e_pep_bond_restraints = jnp.sum(0.5 * PEPTIDE_BOND_K_EV * jnp.square(dist_all_pep - r0_pep_jax))
    else:
        e_restraint = 0.0
        e_pep_bond_restraints = 0.0

    e_dihedral_restraint = _phi_psi_restraint_energy(r)

    # (E) Steered-MD moving harmonic restraint on the end-to-end distance CV.
    # Only added when a lambda center is supplied (SMD runs); plain NVT/NVE pass None.
    if smd_lambda is None:
        e_smd = jnp.asarray(0.0, dtype=jnp.float64)
    else:
        e_smd = smd_bias_energy(r, smd_lambda)

    return (e_intra + e_pw_core + e_inter + e_restraint
            + e_pep_bond_restraints + e_dihedral_restraint + e_smd)



@jit
def _debug_ml_energy(r, pw_slots=None, pw_mask=None):
    """JIT-compiled total ML energy contribution (for diagnostics)."""
    return compute_monomer_energy(r, pw_slots, pw_mask)


@jit
def _debug_ml_energy_components(r, pw_slots=None, pw_mask=None):
    """JIT-compiled ML bonded and nonbonded interaction components."""
    e_bonded = compute_ml_bonded_energy(r, pw_slots, pw_mask)
    e_nonbonded = compute_ml_nonbonded_energy(r, pw_slots, pw_mask)
    return jnp.array([e_bonded, e_nonbonded])


@jit
def _debug_pw_core_energy(r, pw_slots=None, pw_mask=None):
    """JIT-compiled short-range peptide-water core energy."""
    return compute_peptide_water_core_vdw_energy(r, pw_slots, pw_mask)


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
    pw_slots = _pw_slots_ref[0]; pw_mask = _pw_mask_ref[0]

    r_np = np.asarray(r)
    if not np.isfinite(r_np).all():
        n_bad = int((~np.isfinite(r_np)).any(axis=-1).sum())
        print(f"[DIAG{label}] ⚠ Positions: {n_bad} atoms with NaN/Inf — energy will be NaN")
        return
    e_ml_bonded, e_ml_nonbonded = map(float, _debug_ml_energy_components(r, pw_slots, pw_mask))
    e_ml = e_ml_bonded + e_ml_nonbonded
    e_pw_core = float(_debug_pw_core_energy(r, pw_slots, pw_mask))
    e_lj, e_elec = map(
        float, _debug_mm_energy_components(r, pi, pj, mask, e14, vdw14)
    )
    e_mm = e_lj + e_elec
    e_tot = e_ml + e_pw_core + e_mm
    ml_ok = "✓" if np.isfinite(e_ml) else "✗ NaN/Inf"
    mm_ok = "✓" if np.isfinite(e_mm) else "✗ NaN/Inf"
    active_pw_slots_np = np.asarray(_pw_slots_ref[0])[np.asarray(_pw_mask_ref[0]) > 0.5]
    pw_min_dist, pw_mean_dist = _peptide_water_distance_stats_np(r_np, active_pw_slots_np)
    print(f"[DIAG{label}] E_ML={e_ml:.4f} eV {ml_ok} "
          f"(bonded={e_ml_bonded:.4f}, nonbonded={e_ml_nonbonded:.4f}) | "
          f"E_PWcore={e_pw_core:.4f} eV | "
          f"E_LJ={e_lj:.4f} eV | E_elec={e_elec:.4f} eV | "
          f"E_MM={e_mm:.4f} eV {mm_ok} | "
          f"E_tot={e_tot:.4f} eV | Pairs_real={int((_mask_ref[0]>0.5).sum())} | "
          f"PW_ML_active={int((_pw_mask_ref[0]>0.5).sum())}/{MAX_PW_ML_WATERS} | "
          f"PW_min/mean={pw_min_dist:.3f}/{pw_mean_dist:.3f} Å")
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
    pw_slots = _pw_slots_ref[0]
    pw_mask = _pw_mask_ref[0]
    
    print(f"\n=== FORCE & NEIGHBOR LIST DIAGNOSTICS (Cycle {cycle}, Step {step}) ===")
    print(f"Positions: min={r_np.min():.3f}, max={r_np.max():.3f}, mean={r_np.mean():.3f}, std={r_np.std():.3f}")
    
    # helper for atom names
    def get_atom_name(idx):
        if 'atoms' in globals() and idx < len(atoms):
            return f"{atoms.get_chemical_symbols()[idx]}{idx}"
        return f"Atom{idx}"

    def get_pair_class(i, j):
        if molecule_id[i] == molecule_id[j]:
            return "intramolecular"
        if HAS_PEPTIDE and ((i < n_trialanine) ^ (j < n_trialanine)):
            return "peptide-water"
        return "water-water"

    # 1. Energy Components
    e_ml_bonded, e_ml_nonbonded = map(float, _debug_ml_energy_components(r, pw_slots, pw_mask))
    e_ml = e_ml_bonded + e_ml_nonbonded
    e_pw_core = float(_debug_pw_core_energy(r, pw_slots, pw_mask))
    e_mm = float(_debug_mm_energy(r, pi, pj, mask, e14, vdw14))
    active_pw_slots_np = np.asarray(pw_slots)[np.asarray(pw_mask) > 0.5]
    pw_min_dist, pw_mean_dist = _peptide_water_distance_stats_np(r_np, active_pw_slots_np)
    print(
        f"Energies: E_ML={e_ml:.4f} eV "
        f"(bonded={e_ml_bonded:.4f}, nonbonded={e_ml_nonbonded:.4f}) | "
        f"E_PWcore={e_pw_core:.4f} eV | "
        f"E_MM={e_mm:.4f} eV | E_tot={e_ml+e_pw_core+e_mm:.4f} eV"
    )
    print(f"Peptide-water active min/mean distance: {pw_min_dist:.3f}/{pw_mean_dist:.3f} Å")

    # 2. Forces
    f_tot_mag = np.zeros(1)
    f_ml_mag = np.zeros(1)
    f_ml_bonded_mag = np.zeros(1)
    f_ml_nonbonded_mag = np.zeros(1)
    f_pw_core_mag = np.zeros(1)
    f_mm_mag = np.zeros(1)
    f_res_mag = np.zeros(1)
    
    f_tot = None
    f_ml = None
    f_ml_bonded = None
    f_ml_nonbonded = None
    f_pw_core = None
    f_mm = None
    
    try:
        grad_tot = jax.grad(
            lambda r_: hybrid_energy_fn(r_, pi, pj, mask, e14, vdw14, pw_slots, pw_mask)
        )(r)
        f_tot = -np.asarray(grad_tot)
        f_tot_mag = np.linalg.norm(f_tot, axis=-1)
        print(f"Total Forces: min={f_tot_mag.min():.3f}, max={f_tot_mag.max():.3f}, mean={f_tot_mag.mean():.3f}, std={f_tot_mag.std():.3f}")
        nans_tot = np.isnan(f_tot).sum()
        if nans_tot > 0:
            print(f"  ⚠ WARNING: {nans_tot} NaNs detected in Total Forces!")
    except Exception as e:
        print(f"  Failed to compute Total Forces gradient: {e}")
        
    if DEBUG_EXPENSIVE_ML_FORCE_COMPONENTS:
        try:
            grad_ml = jax.grad(lambda r_: compute_monomer_energy(r_, pw_slots, pw_mask))(r)
            f_ml = -np.asarray(grad_ml)
            f_ml_mag = np.linalg.norm(f_ml, axis=-1)
            print(f"ML Total Forces: min={f_ml_mag.min():.3f}, max={f_ml_mag.max():.3f}, mean={f_ml_mag.mean():.3f}, std={f_ml_mag.std():.3f}")
            nans_ml = np.isnan(f_ml).sum()
            if nans_ml > 0:
                print(f"  ⚠ WARNING: {nans_ml} NaNs detected in ML Forces!")
        except Exception as e:
            print(f"  Failed to compute ML Forces gradient: {e}")
    else:
        print("ML Total Forces: skipped (set debug_expensive_ml_force_components=True to compute)")

    try:
        grad_ml_bonded = jax.grad(lambda r_: compute_ml_bonded_energy(r_, pw_slots, pw_mask))(r)
        f_ml_bonded = -np.asarray(grad_ml_bonded)
        f_ml_bonded_mag = np.linalg.norm(f_ml_bonded, axis=-1)
        print(
            f"ML Bonded Forces: min={f_ml_bonded_mag.min():.3f}, "
            f"max={f_ml_bonded_mag.max():.3f}, mean={f_ml_bonded_mag.mean():.3f}, "
            f"std={f_ml_bonded_mag.std():.3f}"
        )
        nans_ml_bonded = np.isnan(f_ml_bonded).sum()
        if nans_ml_bonded > 0:
            print(f"  ⚠ WARNING: {nans_ml_bonded} NaNs detected in ML bonded forces!")
    except Exception as e:
        print(f"  Failed to compute ML bonded force gradient: {e}")

    if DEBUG_EXPENSIVE_ML_FORCE_COMPONENTS:
        try:
            grad_ml_nonbonded = jax.grad(lambda r_: compute_ml_nonbonded_energy(r_, pw_slots, pw_mask))(r)
            f_ml_nonbonded = -np.asarray(grad_ml_nonbonded)
            f_ml_nonbonded_mag = np.linalg.norm(f_ml_nonbonded, axis=-1)
            print(
                f"ML Nonbonded Forces: min={f_ml_nonbonded_mag.min():.3f}, "
                f"max={f_ml_nonbonded_mag.max():.3f}, mean={f_ml_nonbonded_mag.mean():.3f}, "
                f"std={f_ml_nonbonded_mag.std():.3f}"
            )
            nans_ml_nonbonded = np.isnan(f_ml_nonbonded).sum()
            if nans_ml_nonbonded > 0:
                print(f"  ⚠ WARNING: {nans_ml_nonbonded} NaNs detected in ML nonbonded forces!")
        except Exception as e:
            print(f"  Failed to compute ML nonbonded force gradient: {e}")
    else:
        print("ML Nonbonded Forces: skipped (set debug_expensive_ml_force_components=True to compute)")

    try:
        grad_pw_core = jax.grad(lambda r_: compute_peptide_water_core_vdw_energy(r_, pw_slots, pw_mask))(r)
        f_pw_core = -np.asarray(grad_pw_core)
        f_pw_core_mag = np.linalg.norm(f_pw_core, axis=-1)
        print(
            f"Peptide-Water Core Forces: min={f_pw_core_mag.min():.3f}, "
            f"max={f_pw_core_mag.max():.3f}, mean={f_pw_core_mag.mean():.3f}, "
            f"std={f_pw_core_mag.std():.3f}"
        )
        nans_pw_core = np.isnan(f_pw_core).sum()
        if nans_pw_core > 0:
            print(f"  ⚠ WARNING: {nans_pw_core} NaNs detected in peptide-water core forces!")
    except Exception as e:
        print(f"  Failed to compute peptide-water core force gradient: {e}")

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
            if not HAS_PEPTIDE:
                return 0.0
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
        print(f"Active Peptide-Water ML Waters: {int((np.asarray(pw_mask) > 0.5).sum())}/{MAX_PW_ML_WATERS}")
        print(f"Active Pair Distances: min={dists.min():.3f} Å, max={dists.max():.3f} Å, mean={dists.mean():.3f} Å")
        
        close_idx = np.where(dists < 1.2)[0]
        if len(close_idx) > 0:
            print(f"  ⚠ WARNING: {len(close_idx)} pairs are extremely close (< 1.2 Å)!")
            sorted_close = np.argsort(dists[close_idx])
            for idx in sorted_close[:5]:
                p_idx = close_idx[idx]
                i_atom = int(active_pi[p_idx])
                j_atom = int(active_pj[p_idx])
                print(
                    f"    Atom {i_atom} ({get_atom_name(i_atom)}) - "
                    f"Atom {j_atom} ({get_atom_name(j_atom)}): "
                    f"dist = {dists[p_idx]:.3f} Å [{get_pair_class(i_atom, j_atom)}]"
                )

    # 4. Generate Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot A: Forces per Atom
    if f_tot is not None:
        axes[0, 0].plot(f_tot_mag, label="Total Force", alpha=0.7)
    if f_ml is not None:
        axes[0, 0].plot(f_ml_mag, label="ML Force", alpha=0.7)
    if f_ml_bonded is not None:
        axes[0, 0].plot(f_ml_bonded_mag, label="ML Bonded Force", alpha=0.7)
    if f_ml_nonbonded is not None:
        axes[0, 0].plot(f_ml_nonbonded_mag, label="ML Nonbonded Force", alpha=0.7)
    if f_pw_core is not None:
        axes[0, 0].plot(f_pw_core_mag, label="PW Core Force", alpha=0.7)
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
        pep_atoms = list(range(n_trialanine))
        G.add_nodes_from(pep_atoms)
        
        pep_pairs = (active_pi < n_trialanine) | (active_pj < n_trialanine)
        pep_active_pi = active_pi[pep_pairs]
        pep_active_pj = active_pj[pep_pairs]
        pep_dists = dists[pep_pairs]
        
        close_threshold = 4.5
        for i, j, d in zip(pep_active_pi, pep_active_pj, pep_dists):
            if d < close_threshold:
                G.add_edge(int(i), int(j), weight=float(d))
        
        nodes = list(G.nodes())
        pos_layout = nx.spring_layout(G, seed=42)
        node_colors = ['lightblue' if n < n_trialanine else 'salmon' for n in nodes]
        node_sizes = [150 if n < n_trialanine else 50 for n in nodes]
        
        nx.draw_networkx_nodes(G, pos_layout, ax=axes[1, 1], node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=0.5)
        pep_labels = {n: f"{atoms.get_chemical_symbols()[n]}{n}" if ('atoms' in globals() and n < len(atoms)) else f"Atom{n}" for n in nodes if n < n_trialanine}
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
# Vectorized helper functions (no pure Python loops)
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
    print("\n[REPAIR] Minimizing structure in PyCHARMM...")
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
def run_fire_block(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask):
    """Run FIRE_BLOCK_STEPS steps of FIRE (compiled once)."""
    def body_fn(i, s):
        return _step_fn_fire(
            s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
            pw_slots=pw_slots, pw_mask=pw_mask,
        )
    return jax.lax.fori_loop(0, FIRE_BLOCK_STEPS, body_fn, state)


@jit
def run_fire_block_repair(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask):
    """Run 200 steps of FIRE for post-repair minimization (compiled once)."""
    def body_fn(i, s):
        return _step_fn_fire(
            s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
            pw_slots=pw_slots, pw_mask=pw_mask,
        )
    return jax.lax.fori_loop(0, 200, body_fn, state)


def _repair_candidate_metrics(label, positions):
    """Evaluate repaired candidate quality under the current MMML potential."""
    pos_np = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(pos_np).all():
        return {
            "label": label,
            "positions": pos_np,
            "finite": False,
            "energy": float("inf"),
            "max_force": float("inf"),
            "mean_force": float("inf"),
        }
    update_pair_refs(pos_np)
    pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
    r_jax = jnp.array(pos_np, dtype=jnp.float64)
    energy = float(hybrid_energy_fn(r_jax, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
    force = -np.asarray(
        jax.grad(lambda r: hybrid_energy_fn(r, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))(r_jax)
    )
    force_norm = np.linalg.norm(force, axis=-1)
    finite = np.isfinite(energy) and np.isfinite(force_norm).all()
    return {
        "label": label,
        "positions": pos_np,
        "finite": bool(finite),
        "energy": energy if finite else float("inf"),
        "max_force": float(force_norm.max()) if finite else float("inf"),
        "mean_force": float(force_norm.mean()) if finite else float("inf"),
    }


def _run_jax_mmml_fire_repair(positions):
    """Run the MMML FIRE repair stage from supplied coordinates."""
    pos_np = np.asarray(positions, dtype=np.float64)
    update_pair_refs(pos_np)
    pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
    fire_state = _init_fn_fire(
        jnp.array(pos_np, dtype=jnp.float64),
        pi=pi,
        pj=pj,
        mask=mask,
        e14=e14,
        vdw14=vdw14,
        pw_slots=pw_slots,
        pw_mask=pw_mask,
    )
    fire_state = run_fire_block_repair(fire_state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask)
    return np.asarray(fire_state.position, dtype=np.float64)


def run_staged_repair_for_jaxmd(positions, label):
    """Run MM(system), ML(peptide), and MMML FIRE repair; return best candidate."""
    candidates = []

    print(f"[REPAIR] {label}: running MM(system) CHARMM minimization...")
    mm_pos = repair_structure_in_charmm(positions)
    candidates.append(_repair_candidate_metrics("MM(system)", mm_pos))

    if USE_ML_INTRAMOLECULAR and peptide_calc is not None:
        print(f"[REPAIR] {label}: running ML(peptide) minimization...")
        ml_pos = minimize_peptide_only_with_ml_calculator(
            mm_pos, z, peptide_calc, workdir, n_trialanine
        )
        candidates.append(_repair_candidate_metrics("ML(peptide)", ml_pos))
    else:
        print(f"[REPAIR] {label}: skipping ML(peptide) minimization (ML disabled).")
        ml_pos = mm_pos

    print(f"[REPAIR] {label}: running MMML(system) JAX FIRE minimization (200 steps)...")
    mmml_pos = _run_jax_mmml_fire_repair(ml_pos)
    candidates.append(_repair_candidate_metrics("MMML(system) FIRE", mmml_pos))

    finite_candidates = [candidate for candidate in candidates if candidate["finite"]]
    if not finite_candidates:
        raise RuntimeError(f"[REPAIR] {label}: all staged repair candidates are non-finite.")

    print(f"[REPAIR] {label}: candidate ranking by max force, then energy:")
    for candidate in sorted(candidates, key=lambda item: (not item["finite"], item["max_force"], item["energy"])):
        status = "finite" if candidate["finite"] else "non-finite"
        print(
            f"[REPAIR]   {candidate['label']}: {status}, "
            f"E={candidate['energy']:.6f} eV, "
            f"max|F|={candidate['max_force']:.6f} eV/A, "
            f"mean|F|={candidate['mean_force']:.6f} eV/A"
        )

    best = min(finite_candidates, key=lambda item: (item["max_force"], item["energy"]))
    print(f"[REPAIR] {label}: selected {best['label']} for JAX-MD restart.")
    return jnp.array(best["positions"], dtype=jnp.float64)


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
def run_nvt_block(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask, target_temp_ev):
    def body_fn(i, s):
        return _step_fn_nvt(
            s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
            pw_slots=pw_slots, pw_mask=pw_mask, kT=target_temp_ev,
        )
    return jax.lax.fori_loop(0, NVT_BLOCK_STEPS, body_fn, state)


# NVE
print("--- Building NVE simulator (compiled once) ---")
_init_fn_nve, _step_fn_nve = simulate.nve(hybrid_energy_fn, shift_fn, dt)
_step_fn_nve = jit(_step_fn_nve)

@jit
def run_nve_block(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask):
    def body_fn(i, s):
        return _step_fn_nve(
            s, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
            pw_slots=pw_slots, pw_mask=pw_mask,
        )
    return jax.lax.fori_loop(0, NVE_BLOCK_STEPS, body_fn, state)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Structure Minimization with JAX-MD FIRE and PyCHARMM Repair Loops
# ─────────────────────────────────────────────────────────────────────────────
print("--- Minimizing System with JAX-MD FIRE and PyCHARMM Repair Loops ---")
init_r = jnp.array(pos, dtype=jnp.float64)
pos_current = init_r

traj_path_fire = os.path.join(OUTPUT_DIR, "cg_fire.traj")
print(f"--- Saving minimization trajectory to {traj_path_fire} and its .dcd counterpart ---")
traj_fire = DualTrajectoryWriter(traj_path_fire, atoms, dt_ps=0.0, steps_per_frame=FIRE_BLOCK_STEPS)

for cycle in range(FIRE_CYCLES):
    print(f"\n--- Minimization Cycle {cycle+1}/{FIRE_CYCLES} ---")

    # Update pair list from current coordinates (CPU, once per cycle start)
    update_pair_refs(np.asarray(pos_current))
    pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()

    # --- Pre-FIRE diagnostic: print energy components before any step ---
    diagnose_energy(pos_current, label=f" Cycle{cycle+1} init")
    if DEBUG:
        run_force_and_nl_diagnostics(pos_current, pi, pj, mask, e14, vdw14, cycle+1, 0)

    fire_state = _init_fn_fire(
        pos_current, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
        pw_slots=pw_slots, pw_mask=pw_mask,
    )

    # Write starting configuration of this cycle to trajectory
    curr_f = np.asarray(fire_state.force)
    curr_e = float(hybrid_energy_fn(pos_current, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
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
        pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()

        fire_state = run_fire_block(fire_state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask)

        curr_f = np.asarray(fire_state.force)
        curr_e = float(hybrid_energy_fn(fire_state.position, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
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
            pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
            fire_state = _init_fn_fire(
                good_pos_jax, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
                pw_slots=pw_slots, pw_mask=pw_mask,
            )
            nan_detected = True
            break

        last_good_pos = pos_before_block
        frame = atoms.copy()
        frame.set_positions(unfold_coordinates(np.asarray(fire_state.position), box_size, _mon_stacked_groups))
        frame.calc = SinglePointCalculator(frame, energy=curr_e, forces=curr_f)
        traj_fire.write(frame)

    # Print energy breakdown after fire cycle completes
    diagnose_energy(fire_state.position, label=f" Cycle {cycle+1} completed")

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

md_key = jax.random.PRNGKey(SEED)


def next_md_key():
    """Return a fresh PRNG key for velocity-generating MD initializers."""
    global md_key
    md_key, subkey = jax.random.split(md_key)
    return subkey

# Initialize starting NVT state
update_pair_refs(np.asarray(min_r))
pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
init_temp_k = nvt_temp_func(0)
init_temp_ev = init_temp_k * kb
state = _init_fn_nvt(
    next_md_key(), min_r, mass=jax_mass, pi=pi, pj=pj, mask=mask,
    e14=e14, vdw14=vdw14, pw_slots=pw_slots, pw_mask=pw_mask, kT=init_temp_ev,
)

traj_path_nvt = os.path.join(OUTPUT_DIR, "cg_nvt.traj")
print(f"--- Running NVT dynamics and saving trajectory to {traj_path_nvt} and its .dcd counterpart ---")
traj_nvt = DualTrajectoryWriter(traj_path_nvt, atoms, dt_ps=dt, steps_per_frame=NVT_BLOCK_STEPS)
last_good_nvt_pos = np.asarray(min_r, dtype=np.float64)

for step in range(0, NVT_TOTAL_STEPS, NVT_BLOCK_STEPS):
    curr_temp_k = nvt_temp_func(step)
    curr_temp_ev = curr_temp_k * kb

    pos_before_block = np.asarray(state.position, dtype=np.float64)
    if np.isfinite(pos_before_block).all():
        last_good_nvt_pos = pos_before_block.copy()
    else:
        print(f"[NVT] Non-finite positions detected before block; reverting to last-good checkpoint. (T={curr_temp_k:.1f} K)")
        pos_before_block = last_good_nvt_pos.copy()
        update_pair_refs(pos_before_block)
        pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
        state = _init_fn_nvt(next_md_key(), jnp.array(pos_before_block, dtype=jnp.float64),
                             mass=jax_mass, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
                             pw_slots=pw_slots, pw_mask=pw_mask, kT=curr_temp_ev)

    # Update pair list on CPU (no JAX re-trace — shapes are fixed)
    update_pair_refs(pos_before_block)
    pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()

    # Run the compiled block of steps with current target temperature
    state = run_nvt_block(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask, curr_temp_ev)

    # Compute diagnostics directly from state and JITted hybrid_energy_fn
    curr_f = np.asarray(state.force)
    curr_e = float(hybrid_energy_fn(state.position, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
    ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
    temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
    max_bond = get_max_h_x_bond(state.position, box_size, h_idx_arr, x_idx_arr)
    max_dev, mean_dev = get_peptide_bond_diagnostics(state.position, box_size, pep_bond_idx1_arr, pep_bond_idx2_arr, r0_pep_list)

    # Check for instability and repair
    repair_threshold = max(NVT_REPAIR_TEMP_K, curr_temp_k + 100.0)
    unstable_nvt = (
        (not np.isfinite(curr_e))
        or (not np.isfinite(temp))
        or (not np.isfinite(max_bond))
        or temp > repair_threshold
        or max_bond > MAX_HX_BOND_LIMIT
        or (not np.isfinite(np.asarray(state.position)).all())
        or (max_dev > MAX_BOND_DEV_LIMIT)
    )
    if unstable_nvt:
        if temp > repair_threshold:
            print(f"[REPAIR] NVT temperature exceeded repair threshold: "
                  f"{temp:.1f} K > {repair_threshold:.1f} K")
        if max_bond > MAX_HX_BOND_LIMIT:
            print(f"[REPAIR] Peptide H-X bond broke! Max bond length: {max_bond:.2f} Å "
                  f"(max limit {MAX_HX_BOND_LIMIT} Å)")
        if max_dev > MAX_BOND_DEV_LIMIT:
            print(f"[REPAIR] Peptide bond deviation exceeded repair threshold: "
                  f"{max_dev:.4f} Å > {MAX_BOND_DEV_LIMIT:.4f} Å")
        repair_input_pos, repair_source = choose_finite_repair_positions(
            state.position, pos_before_block, last_good_nvt_pos, "NVT REPAIR"
        )
        # 1. Scale back stretched H-X bonds
        scaled_pos, _ = scale_broken_h_bonds(
            repair_input_pos, box_size, h_idx_arr, x_idx_arr
        )
        # 2. Run staged MM/ML/MMML repair and select the best finite candidate
        print(f"[REPAIR] NVT repair input source: {repair_source}")
        final_min_pos = run_staged_repair_for_jaxmd(scaled_pos, "NVT")

        # Re-initialize NVT state
        update_pair_refs(np.asarray(final_min_pos))
        pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
        state = _init_fn_nvt(
            next_md_key(), final_min_pos, mass=jax_mass, pi=pi, pj=pj, mask=mask,
            e14=e14, vdw14=vdw14, pw_slots=pw_slots, pw_mask=pw_mask, kT=curr_temp_ev,
        )
        # Re-evaluate diagnostics
        curr_f = np.asarray(state.force)
        curr_e = float(hybrid_energy_fn(state.position, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
        ke = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
        temp = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / kb)
        max_bond = get_max_h_x_bond(state.position, box_size, h_idx_arr, x_idx_arr)
        max_dev, mean_dev = get_peptide_bond_diagnostics(state.position, box_size, pep_bond_idx1_arr, pep_bond_idx2_arr, r0_pep_list)
        if np.isfinite(np.asarray(state.position)).all() and np.isfinite(curr_e):
            last_good_nvt_pos = np.asarray(state.position, dtype=np.float64).copy()
    else:
        last_good_nvt_pos = np.asarray(state.position, dtype=np.float64).copy()

    print(f"NVT Step {step+NVT_BLOCK_STEPS:5d} | Tot Energy: {curr_e + ke:.4f} eV | "
          f"Temp: {temp:.1f} K (target {curr_temp_k:.1f} K) | Max H-X Bond: {max_bond:.2f} Å | "
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
pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
state_nve = _init_fn_nve(
    next_md_key(), state.position, target_temp_ev, mass=jax_mass, pi=pi, pj=pj,
    mask=mask, e14=e14, vdw14=vdw14, pw_slots=pw_slots, pw_mask=pw_mask,
)

traj_path_nve = os.path.join(OUTPUT_DIR, "cg_nve.traj")
print(f"--- Running NVE dynamics and saving trajectory to {traj_path_nve} and its .dcd counterpart ---")
traj_nve = DualTrajectoryWriter(traj_path_nve, atoms, dt_ps=dt, steps_per_frame=NVE_BLOCK_STEPS)
last_good_nve_pos = np.asarray(state.position, dtype=np.float64)

# Measure initial NVE total energy baseline for conservation checks
init_e = float(hybrid_energy_fn(state_nve.position, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
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
        pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
        state_nve = _init_fn_nve(next_md_key(), jnp.array(pos_before_block, dtype=jnp.float64),
                                 target_temp_ev, mass=jax_mass,
                                 pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
                                 pw_slots=pw_slots, pw_mask=pw_mask)

    # Update pair list on CPU (no JAX re-trace)
    update_pair_refs(pos_before_block)
    pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()

    # Run the compiled block of steps
    state_nve = run_nve_block(state_nve, pi, pj, mask, e14, vdw14, pw_slots, pw_mask)

    # Compute diagnostics
    curr_f = np.asarray(state_nve.force)
    curr_e = float(hybrid_energy_fn(state_nve.position, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
    ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
    temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
    max_bond = get_max_h_x_bond(state_nve.position, box_size, h_idx_arr, x_idx_arr)
    max_dev, mean_dev = get_peptide_bond_diagnostics(state_nve.position, box_size, pep_bond_idx1_arr, pep_bond_idx2_arr, r0_pep_list)

    # Check for instability and repair
    unstable_nve = (
        (not np.isfinite(curr_e))
        or (not np.isfinite(temp))
        or (not np.isfinite(max_bond))
        or temp > NVE_REPAIR_TEMP_K
        or max_bond > MAX_HX_BOND_LIMIT
        or (not np.isfinite(np.asarray(state_nve.position)).all())
        or (max_dev > MAX_BOND_DEV_LIMIT)
    )
    if unstable_nve:
        if temp > NVE_REPAIR_TEMP_K:
            print(f"[REPAIR] NVE temperature exceeded repair threshold: "
                  f"{temp:.1f} K > {NVE_REPAIR_TEMP_K:.1f} K")
        if max_bond > MAX_HX_BOND_LIMIT:
            print(f"[REPAIR] Peptide H-X bond broke! Max bond length: {max_bond:.2f} Å "
                  f"(max limit {MAX_HX_BOND_LIMIT} Å)")
        if max_dev > MAX_BOND_DEV_LIMIT:
            print(f"[REPAIR] Peptide bond deviation exceeded repair threshold: "
                  f"{max_dev:.4f} Å > {MAX_BOND_DEV_LIMIT:.4f} Å")
        repair_input_pos, repair_source = choose_finite_repair_positions(
            state_nve.position, pos_before_block, last_good_nve_pos, "NVE REPAIR"
        )
        # 1. Scale back stretched H-X bonds
        scaled_pos, _ = scale_broken_h_bonds(
            repair_input_pos, box_size, h_idx_arr, x_idx_arr
        )
        # 2. Run staged MM/ML/MMML repair and select the best finite candidate
        print(f"[REPAIR] NVE repair input source: {repair_source}")
        final_min_pos = run_staged_repair_for_jaxmd(scaled_pos, "NVE")

        # Re-initialize NVE state
        update_pair_refs(np.asarray(final_min_pos))
        pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
        state_nve = _init_fn_nve(
            next_md_key(), final_min_pos, target_temp_ev, mass=jax_mass, pi=pi, pj=pj,
            mask=mask, e14=e14, vdw14=vdw14, pw_slots=pw_slots, pw_mask=pw_mask,
        )
        # Re-evaluate diagnostics
        curr_f = np.asarray(state_nve.force)
        curr_e = float(hybrid_energy_fn(state_nve.position, pi, pj, mask, e14, vdw14, pw_slots, pw_mask))
        ke = float(quantity.kinetic_energy(momentum=state_nve.momentum, mass=state_nve.mass))
        temp = float(quantity.temperature(momentum=state_nve.momentum, mass=state_nve.mass) / kb)
        max_bond = get_max_h_x_bond(state_nve.position, box_size, h_idx_arr, x_idx_arr)
        max_dev, mean_dev = get_peptide_bond_diagnostics(state_nve.position, box_size, pep_bond_idx1_arr, pep_bond_idx2_arr, r0_pep_list)
        if np.isfinite(np.asarray(state_nve.position)).all() and np.isfinite(curr_e):
            last_good_nve_pos = np.asarray(state_nve.position, dtype=np.float64).copy()
            initial_nve_energy = curr_e + ke
            print(f"[REPAIR] Reset NVE Energy Baseline post-repair to: {initial_nve_energy:.6f} eV")
    else:
        last_good_nve_pos = np.asarray(state_nve.position, dtype=np.float64).copy()

    tot_energy = curr_e + ke
    energy_drift = tot_energy - initial_nve_energy
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


# ─────────────────────────────────────────────────────────────────────────────
# 11. Steered MD + Jarzynski / Hummer-Szabo PMF along the end-to-end distance
# ─────────────────────────────────────────────────────────────────────────────
if SMD_ENABLE and HAS_PEPTIDE:
    print("--- Steered MD (Jarzynski / Hummer-Szabo, end-to-end distance) ---")
    print(f"--- CV anchors: atom {SMD_ATOM_I} <-> atom {SMD_ATOM_J} "
          f"(0-based; terminal carbons CAY/CAT) ---")

    smd_pull_steps  = int(SMD_PULL_STEPS)  if SMD_PULL_STEPS  is not None else NVT_TOTAL_STEPS
    smd_equil_steps = int(SMD_EQUIL_STEPS) if SMD_EQUIL_STEPS is not None else NVT_BLOCK_STEPS
    block_steps     = NVT_BLOCK_STEPS
    n_blocks        = smd_pull_steps // block_steps
    block_dt        = block_steps * dt                 # ps per block
    lam_dot         = SMD_V_A_PER_PS                    # Å/ps

    smd_base_pos = np.asarray(min_r, dtype=np.float64)
    d0 = float(SMD_D0_A) if SMD_D0_A is not None \
         else float(end_to_end_distance(jnp.asarray(smd_base_pos)))
    d_target = d0 + lam_dot * (n_blocks * block_dt)
    lam_prof = d0 + lam_dot * (np.arange(1, n_blocks + 1) * block_dt)  # lambda at each block end
    print(f"--- Pulling {SMD_N_PULLS} trajectories: d0={d0:.3f} -> {d_target:.3f} Å, "
          f"k={SMD_K_EV_A2} eV/Å², v={lam_dot} Å/ps, {n_blocks} blocks/pull ---")

    _smd_init, _smd_step = simulate.nvt_nose_hoover(hybrid_energy_fn, shift_fn, dt, target_temp_ev)

    def run_smd_block(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask, kT, lam):
        def body(_, st):
            return _smd_step(st, pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
                             pw_slots=pw_slots, pw_mask=pw_mask, kT=kT, smd_lambda=lam)
        return jax.lax.fori_loop(0, block_steps, body, state)
    run_smd_block = jit(run_smd_block)

    z_prof = np.zeros((SMD_N_PULLS, n_blocks), dtype=np.float64)  # CV per block
    W_prof = np.zeros((SMD_N_PULLS, n_blocks), dtype=np.float64)  # cumulative work per block

    for pull in range(SMD_N_PULLS):
        # Fresh velocities + restrained equilibration at fixed lambda = d0.
        update_pair_refs(smd_base_pos)
        pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
        lam0 = jnp.asarray(d0, dtype=jnp.float64)
        state = _smd_init(next_md_key(), jnp.asarray(smd_base_pos), mass=jax_mass,
                          pi=pi, pj=pj, mask=mask, e14=e14, vdw14=vdw14,
                          pw_slots=pw_slots, pw_mask=pw_mask, kT=target_temp_ev,
                          smd_lambda=lam0)
        for _ in range(max(1, smd_equil_steps // block_steps)):
            update_pair_refs(np.asarray(state.position))
            pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
            state = run_smd_block(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask,
                                  target_temp_ev, lam0)

        # Pull: advance lambda one block at a time, accumulating external work.
        W = 0.0
        for m in range(n_blocks):
            lam_end = float(lam_prof[m])
            update_pair_refs(np.asarray(state.position))
            pi, pj, mask, e14, vdw14, pw_slots, pw_mask = _current_energy_refs()
            state = run_smd_block(state, pi, pj, mask, e14, vdw14, pw_slots, pw_mask,
                                  target_temp_ev, jnp.asarray(lam_end, dtype=jnp.float64))
            z = float(end_to_end_distance(state.position))
            # dW = (dV/dlambda) dlambda = k (lambda - z) * (v * block_dt)
            W += SMD_K_EV_A2 * (lam_end - z) * (lam_dot * block_dt)
            z_prof[pull, m] = z
            W_prof[pull, m] = W
        print(f"[SMD] pull {pull+1}/{SMD_N_PULLS}: final d={z_prof[pull, -1]:.3f} Å, "
              f"W_total={W_prof[pull, -1]:.4f} eV")

    # --- Hummer-Szabo weighted-histogram PMF reconstruction (PNAS 2001) ---
    beta = 1.0 / (kb * temperature)
    N = SMD_N_PULLS

    # Per-block Jarzynski normaliser A_t = <exp(-beta W_t)>, computed stably.
    Wt = W_prof                              # (N, n_blocks)
    wmin_t = Wt.min(axis=0, keepdims=True)   # per-block shift for stability
    expW = np.exp(-beta * (Wt - wmin_t))     # (N, n_blocks)
    logA_t = (np.log(expW.mean(axis=0)) - beta * wmin_t[0])   # log <exp(-beta W_t)>, per block
    inv_logA_t = -logA_t                     # 1/A_t = exp(inv_logA_t) in log space

    # Histogram grid for the PMF.
    z_lo = min(d0, float(z_prof.min())) - 0.5
    z_hi = max(d_target, float(z_prof.max())) + 0.5
    edges = np.linspace(z_lo, z_hi, SMD_PMF_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Numerator: sum_t (1/A_t) * (1/N) sum_i exp(-beta W_it) * delta(z - z_it)
    # done in a numerically stable way per (i, t) via log-weights.
    logw = -beta * Wt - np.log(N) + inv_logA_t[None, :]     # (N, n_blocks) log weight per sample
    logw_max = logw.max()
    w = np.exp(logw - logw_max)                             # shifted weights
    num = np.zeros(SMD_PMF_BINS, dtype=np.float64)
    bins = np.clip(np.digitize(z_prof.ravel(), edges) - 1, 0, SMD_PMF_BINS - 1)
    np.add.at(num, bins, w.ravel())

    # Denominator: sum_t (1/A_t) * exp(-beta V(z, lambda_t)) for each bin center z.
    Vz = 0.5 * SMD_K_EV_A2 * (centers[:, None] - lam_prof[None, :]) ** 2   # (bins, n_blocks)
    logden = inv_logA_t[None, :] - beta * Vz                              # (bins, n_blocks)
    logden_max = logden.max()
    den = np.exp(logden - logden_max).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmf_ev = -(1.0 / beta) * (np.log(num) - logw_max - (np.log(den) + logden_max))
    valid = np.isfinite(pmf_ev)
    if valid.any():
        pmf_ev = pmf_ev - np.nanmin(pmf_ev[valid])   # zero the minimum

    out_npz = os.path.join(OUTPUT_DIR, "smd_jarzynski.npz")
    np.savez(
        out_npz,
        z_profile=z_prof, work_profile=W_prof, lambda_profile=lam_prof,
        pmf_centers=centers, pmf_ev=pmf_ev,
        k_ev_a2=SMD_K_EV_A2, v_a_per_ps=lam_dot, temperature_k=temperature,
        atom_i=SMD_ATOM_I, atom_j=SMD_ATOM_J,
    )
    print(f"[SMD] Wrote work profiles + Hummer-Szabo PMF to {out_npz}")
    print("[SMD] PMF (eV) vs end-to-end distance (Å):")
    for c, g in zip(centers, pmf_ev):
        if np.isfinite(g):
            print(f"    d={c:6.3f} Å   G={g:8.4f} eV")

    # Endpoint Jarzynski ΔF for reference (stable exponential average of total work).
    Wtot = W_prof[:, -1]
    wmin = Wtot.min()
    dF = -(1.0 / beta) * (np.log(np.mean(np.exp(-beta * (Wtot - wmin))))) + wmin
    print(f"[SMD] Endpoint Jarzynski ΔF({d0:.3f} -> {d_target:.3f} Å) = {dF:.4f} eV "
          f"(mean W={Wtot.mean():.4f} eV, {N} pulls)")
    print("Steered MD complete!")

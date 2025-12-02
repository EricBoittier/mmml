#!/usr/bin/env python
# coding: utf-8
"""
MM/ML Hybrid Simulation Setup Script

This script demonstrates how to:
1. Load training/validation data
2. Setup PyCHARMM system for MM contributions
3. Load ML model from checkpoint
4. Optimize LJ parameters and cutoff parameters
5. Initialize simulations from data batches
6. Run energy/force calculations and structure minimization

OUTLINE:
--------
1. Setup: Environment, imports, mock CLI arguments
2. Load Data: Prepare training/validation datasets and batches
3. Load Model: Load ML model from checkpoint (JSON/pickle/orbax)
4. Setup Calculator: Create hybrid ML/MM calculator factory
5. Setup PyCHARMM: Initialize PyCHARMM system (required for MM)
6. Atom Reordering: Reorder atoms to match PyCHARMM ordering
7. Optimize Parameters: Fit LJ scaling factors and cutoff parameters
8. Initialize Simulations: Create ASE Atoms objects from batches
9. Run Calculations: Compute energies/forces and minimize structures
"""

import mmml
import ase
import os
from pathlib import Path
import argparse
import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

# ========================================================================
# SECTION 1: ENVIRONMENT SETUP
# ========================================================================
# Set environment variables for JAX/GPU
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check JAX configuration
devices = jax.local_devices()
print(f"JAX devices: {devices}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# ========================================================================
# SECTION 2: IMPORTS
# ========================================================================
from mmml.cli.base import (
    load_model_parameters,
    resolve_checkpoint_paths,
    setup_ase_imports,
    setup_mmml_imports,
)
from mmml.pycharmmInterface import import_pycharmm
import pycharmm
import pycharmm.ic as ic
import pycharmm.psf as psf
import pycharmm.energy as energy
from mmml.pycharmmInterface.mmml_calculator import setup_calculator, CutoffParameters
from mmml.physnetjax.physnetjax.data.data import prepare_datasets
from mmml.physnetjax.physnetjax.data.batches import prepare_batches_jit
from mmml.pycharmmInterface.setupBox import setup_box_generic
from mmml.pycharmmInterface import setupRes, setupBox
from mmml.pycharmmInterface.import_pycharmm import (
    reset_block, 
    coor, 
    reset_block_no_internal, 
    pycharmm_quiet, 
    pycharmm_verbose
)
from mmml.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM
from mmml.utils.hybrid_optimization import (
    extract_lj_parameters_from_calculator,
    fit_hybrid_potential_to_training_data_jax,
    fit_lj_parameters_to_training_data_jax,
    fit_hybrid_parameters_iteratively,
    expand_scaling_parameters_to_full_set,
)
from mmml.utils.simulation_utils import (
    reorder_atoms_to_match_pycharmm,
    initialize_simulation_from_batch,
    initialize_multiple_simulations,
)

# Setup ASE imports
Atoms = setup_ase_imports()
CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()

# Additional imports for simulation
import ase.io as ase_io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
import ase.optimize as ase_opt
from ase.visualize import view as view_atoms
from ase.visualize.plot import plot_atoms

# ========================================================================
# SECTION 3: MOCK CLI ARGUMENTS
# ========================================================================
# Create a mock args object that mimics CLI arguments
# This allows the script to follow the same structure as run_sim.py

class MockArgs:
    """Mock CLI arguments following run_sim.py structure"""
    def __init__(self):
        # Paths
        self.pdbfile = None  # Will be created from valid_data if needed
        self.checkpoint = None  # Will be set below (can be Path or str)

        # System parameters
        self.n_monomers = 2
        self.n_atoms_monomer = 10
        self.atoms_per_monomer = 10  # Alias for compatibility

        # Calculator parameters
        self.ml_cutoff = 1.0
        self.mm_switch_on = 6.0
        self.mm_cutoff = 1.0
        self.include_mm = True
        self.skip_ml_dimers = False
        self.debug = True

        # MD simulation parameters
        self.temperature = 210.0
        self.timestep = 0.1
        self.nsteps_jaxmd = 100_000
        self.nsteps_ase = 10000
        self.ensemble = "nvt"
        self.heating_interval = 500
        self.write_interval = 100
        self.energy_catch = 0.5

        # Output
        self.output_prefix = "md_simulation"
        self.cell = None  # No PBC by default

        # Validation
        self.validate = False

# Create mock args object
args = MockArgs()

# System parameters (can be overridden)
ATOMS_PER_MONOMER = args.n_atoms_monomer
N_MONOMERS = args.n_monomers

print(f"Mock args created:")
print(f"  n_monomers: {args.n_monomers}")
print(f"  n_atoms_monomer: {args.n_atoms_monomer}")
print(f"  ml_cutoff: {args.ml_cutoff}")
print(f"  mm_switch_on: {args.mm_switch_on}")
print(f"  mm_cutoff: {args.mm_cutoff}")

# ========================================================================
# SECTION 4: LOAD DATA AND PREPARE BATCHES
# ========================================================================
# Load training/validation data and prepare batches for simulations

# Initialize random key for data loading
data_key = jax.random.PRNGKey(42)

# Data file path
data_file = "/pchem-data/meuwly/boittier/home/mmml/mmml/data/fixed-acetone-only_MP2_21000.npz"

print(f"\nLoading data from: {data_file}")

# Prepare datasets
train_data, valid_data = prepare_datasets(
    data_key, 
    10500,  # num_train
    10500,  # num_valid
    [data_file], 
    natoms=ATOMS_PER_MONOMER * N_MONOMERS
)

# Prepare batches for validation data (used to initialize simulations)
valid_batches = prepare_batches_jit(data_key, valid_data, 1, num_atoms=ATOMS_PER_MONOMER * N_MONOMERS)
train_batches = prepare_batches_jit(data_key, train_data, 1, num_atoms=ATOMS_PER_MONOMER * N_MONOMERS)

print(f"Loaded {len(valid_data['R'])} validation samples")
print(f"Prepared {len(valid_batches)} validation batches")
print(f"Each batch contains {len(valid_batches[0]['R'])} atoms")

# ========================================================================
# SECTION 5: LOAD MODEL AND SETUP CALCULATOR
# ========================================================================
# Load ML model from checkpoint and create calculator factory

# Checkpoint path
uid = "test-84aa02d9-e329-46c4-b12c-f55e6c9a2f94"
SCICORE = Path("/pchem-data/meuwly/boittier/home/")
RESTART = SCICORE / "ckpts" / f"{uid}" / "epoch-6595" 
args.checkpoint = RESTART  # Keep as Path object (resolve_checkpoint_paths handles both str and Path)

def load_model_parameters_json(epoch_dir, natoms, use_orbax=False):
    """
    Load model parameters from checkpoint using JSON (no orbax/pickle required).

    This function tries to load checkpoints from JSON files first, then falls back
    to pickle if needed. JSON is preferred for portability.

    Args:
        epoch_dir: Path to checkpoint epoch directory
        natoms: Number of atoms
        use_orbax: If True, try orbax first (default: False)

    Returns:
        params, model: Model parameters and model instance
    """
    from mmml.physnetjax.physnetjax.models.model import EF
    import json
    import pickle

    epoch_dir = Path(epoch_dir)

    # Try orbax first if requested
    if use_orbax:
        try:
            from mmml.physnetjax.physnetjax.restart.restart import get_params_model
            params, model = get_params_model(str(epoch_dir), natoms=natoms)
            if model is not None:
                print("✓ Loaded checkpoint using orbax")
                return params, model
        except Exception as e:
            print(f"Warning: orbax loading failed: {e}")
            print("Falling back to JSON/pickle-based loading...")

    # Helper function to convert JSON-serialized arrays back to JAX arrays
    def json_to_jax(obj):
        """Recursively convert JSON lists to JAX arrays."""
        if isinstance(obj, dict):
            return {k: json_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Check if it's a nested list (array)
            if len(obj) > 0 and isinstance(obj[0], (list, int, float)):
                arr = jnp.array(obj)
                return arr
            else:
                return [json_to_jax(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return obj
        else:
            return obj

    # Try JSON-based loading first (preferred)
    json_candidates = [
        epoch_dir / "params.json",
        epoch_dir / "best_params.json",
        epoch_dir / "checkpoint.json",
        epoch_dir / "final_params.json",
    ]

    params = None
    params_source = None

    # Try JSON files first
    for json_path in json_candidates:
        if json_path.exists():
            print(f"Loading parameters from JSON: {json_path}")
            try:
                with open(json_path, 'r') as f:
                    checkpoint_data = json.load(f)

                # Extract params
                if isinstance(checkpoint_data, dict):
                    params_data = checkpoint_data.get('params') or checkpoint_data.get('ema_params') or checkpoint_data
                else:
                    params_data = checkpoint_data

                # Convert JSON arrays back to JAX arrays
                params = json_to_jax(params_data)
                params_source = "json"
                break
            except Exception as e:
                print(f"  Failed to load from {json_path}: {e}")
                continue

    # Fall back to pickle if JSON not found
    if params is None:
        pickle_candidates = [
            epoch_dir / "params.pkl",
            epoch_dir / "best_params.pkl",
            epoch_dir / "checkpoint.pkl",
            epoch_dir / "final_params.pkl",
        ]

        for pkl_path in pickle_candidates:
            if pkl_path.exists():
                print(f"Loading parameters from pickle: {pkl_path}")
                with open(pkl_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                # Extract params
                if isinstance(checkpoint_data, dict):
                    params = checkpoint_data.get('params') or checkpoint_data.get('ema_params') or checkpoint_data
                else:
                    params = checkpoint_data
                params_source = "pickle"
                break

    if params is None:
        all_candidates = [str(p) for p in json_candidates + [
            epoch_dir / "params.pkl",
            epoch_dir / "best_params.pkl",
            epoch_dir / "checkpoint.pkl",
            epoch_dir / "final_params.pkl",
        ]]
        raise FileNotFoundError(
            f"Could not find parameters in {epoch_dir}.\n"
            f"Tried JSON: {[str(p) for p in json_candidates]}\n"
            f"Tried pickle: {[str(p) for p in pickle_candidates if p.exists()]}\n"
            f"Please ensure checkpoint files exist."
        )

    # Load model config (prefer JSON)
    config_candidates = [
        epoch_dir / "model_config.json",
        epoch_dir.parent / "model_config.json",
        epoch_dir / "model_config.pkl",
        epoch_dir.parent / "model_config.pkl",
    ]

    model_kwargs = {}
    for config_path in config_candidates:
        if config_path.exists():
            print(f"Loading model config from: {config_path}")
            try:
                if config_path.suffix == '.json':
                    with open(config_path, 'r') as f:
                        model_kwargs = json.load(f)
                else:
                    with open(config_path, 'rb') as f:
                        model_kwargs = pickle.load(f)
                break
            except Exception as e:
                print(f"  Warning: Failed to load config from {config_path}: {e}")
                continue

    # If no config found, use defaults
    if not model_kwargs:
        print("Warning: No model config found, using defaults")
        model_kwargs = {
            'features': 64,
            'cutoff': 8.0,
            'max_degree': 2,
            'num_iterations': 3,
        }

    # Set natoms
    model_kwargs['natoms'] = natoms

    # Create model
    model = EF(**model_kwargs)
    model.natoms = natoms

    print(f"✓ Loaded checkpoint using {params_source} (no orbax required)")
    print(f"  Model: {model}")

    return params, model

# Resolve checkpoint paths
if args.checkpoint is not None:
    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)
    print(f"\nCheckpoint base dir: {base_ckpt_dir}")
    print(f"Checkpoint epoch dir: {epoch_dir}")
else:
    raise ValueError("Checkpoint path must be provided via args.checkpoint")

# Load model parameters
natoms = ATOMS_PER_MONOMER * N_MONOMERS

# Try JSON-based loading first (preferred, no orbax/pickle required)
try:
    params, model = load_model_parameters_json(epoch_dir, natoms, use_orbax=False)
    print(f"Model loaded using JSON/pickle: {model}")
except Exception as e:
    print(f"JSON/pickle-based loading failed: {e}")
    print("Trying orbax-based loading (requires GPU environment)...")
    try:
        params, model = load_model_parameters(epoch_dir, natoms)
        model.natoms = natoms
        print(f"Model loaded using orbax: {model}")
    except Exception as e2:
        raise RuntimeError(
            f"Failed to load model with all methods:\n"
            f"  JSON/pickle: {e}\n"
            f"  Orbax: {e2}\n"
            f"Make sure checkpoint files exist in {epoch_dir}\n"
            f"Preferred format: JSON files (params.json, model_config.json)"
        )

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
    cell=args.cell,
)

# Create cutoff parameters
CUTOFF_PARAMS = CutoffParameters(
    ml_cutoff=args.ml_cutoff,
    mm_switch_on=args.mm_switch_on,
    mm_cutoff=args.mm_cutoff,
)
print(f"\nCutoff parameters: {CUTOFF_PARAMS}")

# ========================================================================
# SECTION 6: SETUP PYCHARMM SYSTEM
# ========================================================================
# IMPORTANT: PyCHARMM system must be initialized BEFORE creating calculators
# that use MM contributions, otherwise charges won't be available
#
# This generates residues in PyCHARMM and builds the structure.
# The atom ordering from PyCHARMM will be used to reorder valid_data batch atoms.

# Generate residues in PyCHARMM
# For N_MONOMERS=2, we generate "ACO ACO" (two acetone molecules)
# Adjust the residue string based on N_MONOMERS and your system
residue_string = " ".join(["ACO"] * N_MONOMERS)
print(f"\nGenerating {N_MONOMERS} residues: {residue_string}")

try:
    # Generate residues (this creates the PSF structure)
    setupRes.generate_residue(residue_string)
    print("Residues generated successfully")

    # Build the structure using internal coordinates
    ic.build()
    print("Structure built using internal coordinates")

    # Show coordinates
    coor.show()

    # Get PyCHARMM atom ordering information
    # This will be used to reorder valid_data batch atoms
    pycharmm_atypes = np.array(psf.get_atype())[:N_MONOMERS * ATOMS_PER_MONOMER]
    pycharmm_resids = np.array(psf.get_res())[:N_MONOMERS * ATOMS_PER_MONOMER]
    pycharmm_iac = np.array(psf.get_iac())[:N_MONOMERS * ATOMS_PER_MONOMER]

    print(f"PyCHARMM atom types: {pycharmm_atypes}")
    print(f"PyCHARMM residue IDs: {pycharmm_resids}")
    print(f"PyCHARMM has {len(pycharmm_atypes)} atoms")

    # View PyCHARMM state
    mmml.pycharmmInterface.import_pycharmm.view_pycharmm_state()

except Exception as e:
    print(f"Warning: Could not initialize PyCHARMM system: {e}")
    print("You may need to adjust residue names/numbers")
    print("MM contributions will be disabled if PyCHARMM is not initialized")
    if args.include_mm:
        print("Setting include_mm=False since PyCHARMM initialization failed")
        args.include_mm = False
    pycharmm_atypes = None
    pycharmm_resids = None
    pycharmm_iac = None

# Setup PyCHARMM non-bonded parameters
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
pycharmm_quiet()

# ========================================================================
# SECTION 7: ATOM REORDERING (if needed)
# ========================================================================
# The atoms from valid_data batches may need to be reordered to match 
# PyCHARMM's atom ordering. This section demonstrates manual reordering.
# The initialize_simulation_from_batch function can also handle this automatically.

# Example: Manual reordering based on energy minimization
# This is a hardcoded reordering found through trial and error
# In practice, use reorder_atoms_to_match_pycharmm() function
final_reorder = [3, 0, 1, 2, 7, 8, 9, 4, 5, 6, 
                 13, 10, 11, 12, 17, 18, 19, 14, 15, 16]

# Apply reordering to train batches
train_batches_copy = train_batches.copy()
for i in range(len(train_batches)):
    train_batches_copy[i]["R"] = train_batches[i]["R"][final_reorder]
    train_batches_copy[i]["Z"] = train_batches[i]["Z"][final_reorder]
    train_batches_copy[i]["F"] = train_batches[i]["F"][final_reorder]

print(f"\nApplied atom reordering to {len(train_batches_copy)} batches")

# ========================================================================
# SECTION 8: OPTIMIZE PARAMETERS
# ========================================================================
# Optimize LJ scaling factors and cutoff parameters to better match training data

# Step 1: Extract base LJ parameters (do this once, after calculator_factory is created)
print("\n" + "=" * 60)
print("Extracting LJ parameters")
print("=" * 60)
lj_params = extract_lj_parameters_from_calculator(ATOMS_PER_MONOMER=10, N_MONOMERS=2)
print(f"LJ parameters extracted: {list(lj_params.keys())}")

# Step 2: Iteratively optimize LJ and cutoff parameters
# This alternates between optimizing LJ parameters and cutoff parameters,
# using the results from each step to improve the next
print("\n" + "=" * 60)
print("ITERATIVE OPTIMIZATION: Alternating LJ and Cutoff Optimization")
print("=" * 60)
print("This will alternate between:")
print("  1. Optimizing LJ parameters (using current cutoffs)")
print("  2. Optimizing cutoff parameters (using current LJ parameters)")
print("This iterative approach leads to better overall optimization.")

# Choose optimization mode:
# Option A: Iterative optimization (recommended)
USE_ITERATIVE = True  # Set to False to use separate modes

if USE_ITERATIVE:
    result_iterative = fit_hybrid_parameters_iteratively(
        train_batches=train_batches_copy,
        base_calculator_factory=calculator_factory,
        model=model,
        model_params=params,
        atc_epsilons=lj_params["atc_epsilons"],
        atc_rmins=lj_params["atc_rmins"],
        atc_qs=lj_params["atc_qs"],
        at_codes=lj_params["at_codes"],
        pair_idx_atom_atom=lj_params["pair_idx_atom_atom"],
        cutoff_params=CUTOFF_PARAMS,
        args=args,
        n_iterations=100,  # Number of alternating iterations
        n_samples=100,
        min_com_distance=4.0,  # Filter out samples with COM distance < 3.5 Å (large force errors)
        energy_weight=1.0,
        force_weight=10.0,
        lj_learning_rate=0.0051,
        cutoff_learning_rate=0.0051,
        lj_n_iterations=10,  # Iterations per LJ optimization step
        cutoff_n_iterations=10,  # Iterations per cutoff optimization step
        convergence_threshold=1e-3,  # Stop early if loss improvement < 0.1%
        verbose=True,
    )
    
    opt_ep_scale_lj = result_iterative["ep_scale"]
    opt_sig_scale_lj = result_iterative["sig_scale"]
    # Convert JAX arrays to Python floats for CutoffParameters (required for hashability)
    CUTOFF_PARAMS = CutoffParameters(
        ml_cutoff=float(result_iterative["ml_cutoff"]),
        mm_switch_on=float(result_iterative["mm_switch_on"]),
        mm_cutoff=float(result_iterative["mm_cutoff"]),
    )
    
    # Store cutoff results for later use (fix NameError)
    result_cutoff = {
        "ml_cutoff": result_iterative["ml_cutoff"],
        "mm_switch_on": result_iterative["mm_switch_on"],
        "mm_cutoff": result_iterative["mm_cutoff"],
    }
    
    print(f"\nFinal optimized parameters:")
    print(f"  ep_scale: {opt_ep_scale_lj}")
    print(f"  sig_scale: {opt_sig_scale_lj}")
    print(f"  Cutoff parameters: {CUTOFF_PARAMS}")
    print(f"  Loss history: {result_iterative['loss_history']}")

else:
    # Option B: Separate optimization modes (original approach)
    # Step 2a: Optimize LJ parameters only
    print("\n" + "=" * 60)
    print("MODE 1: Optimizing LJ parameters only")
    print("=" * 60)
    result_lj = fit_hybrid_potential_to_training_data_jax(
        train_batches=train_batches_copy,
        base_calculator_factory=calculator_factory,
        model=model,
        model_params=params,
        atc_epsilons=lj_params["atc_epsilons"],
        atc_rmins=lj_params["atc_rmins"],
        atc_qs=lj_params["atc_qs"],
        at_codes=lj_params["at_codes"],
        pair_idx_atom_atom=lj_params["pair_idx_atom_atom"],
        cutoff_params=CUTOFF_PARAMS,
        args=args,
        optimize_mode="lj_only",
        n_samples=20,
        energy_weight=1.0,
        force_weight=1.0,
        learning_rate=0.01,
        n_iterations=100,
        verbose=True
    )

    opt_ep_scale_lj = result_lj["ep_scale"]
    opt_sig_scale_lj = result_lj["sig_scale"]
    print(f"\nOptimized LJ scales:")
    print(f"  ep_scale: {opt_ep_scale_lj}")
    print(f"  sig_scale: {opt_sig_scale_lj}")

    # Step 2b: Optimize cutoff parameters (using optimized LJ parameters)
    print("\n" + "=" * 60)
    print("MODE 2: Optimizing cutoff parameters")
    print("=" * 60)
    print("Using optimized LJ parameters from MODE 1 for better starting loss")
    result_cutoff = fit_hybrid_potential_to_training_data_jax(
        train_batches=train_batches_copy,
        base_calculator_factory=calculator_factory,
        model=model,
        model_params=params,
        atc_epsilons=lj_params["atc_epsilons"],
        atc_rmins=lj_params["atc_rmins"],
        atc_qs=lj_params["atc_qs"],
        at_codes=lj_params["at_codes"],
        pair_idx_atom_atom=lj_params["pair_idx_atom_atom"],
        cutoff_params=CUTOFF_PARAMS,
        optimize_mode="cutoff_only",
        initial_ep_scale=opt_ep_scale_lj,  # Use optimized LJ parameters from MODE 1
        initial_sig_scale=opt_sig_scale_lj,  # Use optimized LJ parameters from MODE 1
        initial_ml_cutoff=1.0,
        initial_mm_switch_on=6.0,
        initial_mm_cutoff=1.0,
        n_samples=20,
        learning_rate=0.01,
        n_iterations=100,
        verbose=True
    )

    # Update cutoff parameters with optimized values
    # Convert JAX arrays to Python floats for CutoffParameters (required for hashability)
    CUTOFF_PARAMS = CutoffParameters(
        ml_cutoff=float(result_cutoff["ml_cutoff"]), 
        mm_switch_on=float(result_cutoff["mm_switch_on"]), 
        mm_cutoff=float(result_cutoff["mm_cutoff"])
    )
    print(f"\nOptimized cutoff parameters: {CUTOFF_PARAMS}")

# Step 4: Create calculator with optimized parameters
print("\n" + "=" * 60)
print("Creating optimized calculator")
print("=" * 60)
# Expand to full parameter set
full_ep_scale, full_sig_scale = expand_scaling_parameters_to_full_set(
   opt_ep_scale_lj, opt_sig_scale_lj, lj_params
)

calculator_factory_lj_optimized = setup_calculator(
    ATOMS_PER_MONOMER=args.n_atoms_monomer,
    N_MONOMERS=args.n_monomers,
    ml_cutoff_distance=CUTOFF_PARAMS.ml_cutoff,
    mm_switch_on=CUTOFF_PARAMS.mm_switch_on,
    mm_cutoff=CUTOFF_PARAMS.mm_cutoff,
    doML=True,
    doMM=args.include_mm,
    doML_dimer=not args.skip_ml_dimers,
    debug=True,
    model_restart_path=base_ckpt_dir,
    MAX_ATOMS_PER_SYSTEM=natoms,
    ml_energy_conversion_factor=1,
    ml_force_conversion_factor=1,
    cell=args.cell,
    ep_scale=np.array(full_ep_scale),
    sig_scale=np.array(full_sig_scale),
)
print("Optimized calculator created")

# ========================================================================
# SECTION 9: INITIALIZE SIMULATIONS FROM BATCHES
# ========================================================================
# Initialize ASE Atoms objects from data batches for running simulations

print("\n" + "=" * 60)
print("Initializing simulations from batches")
print("=" * 60)

# Initialize first simulation from batch 0
# FIXED: Added missing CUTOFF_PARAMS argument
atoms, hybrid_calc = initialize_simulation_from_batch(
    train_batches_copy[0], 
    calculator_factory_lj_optimized, 
    CUTOFF_PARAMS,  # FIXED: Was missing
    args
)

# Enable verbose output for debugging
hybrid_calc.verbose = True
atoms.calc = hybrid_calc

print(f"Initialized simulation:")
print(f"  Number of atoms: {len(atoms)}")
print(f"  Calculator: {type(hybrid_calc)}")

# Initialize multiple simulations (optional)
simulations = initialize_multiple_simulations(
    train_batches_copy[:2], 
    calculator_factory_lj_optimized, 
    CUTOFF_PARAMS, 
    args
)
print(f"\nInitialized {len(simulations)} simulations from batches")

# ========================================================================
# SECTION 10: RUN CALCULATIONS
# ========================================================================
# Compute energies/forces and compare with reference data

print("\n" + "=" * 60)
print("Running energy/force calculations")
print("=" * 60)

# Example: Calculate energy and forces for multiple configurations
for i, b in enumerate(train_batches_copy[:10]):
    if b["N"] == 20:  # Only process batches with correct number of atoms
        atoms.set_positions(b["R"])
        f_true = b["F"]
        f_calc = atoms.get_forces()
        
        # Plot comparison (optional)
        plt.scatter(f_true.flatten(), f_calc.flatten(), alpha=0.5, label=f"Batch {i}")
        print(f"Batch {i}: Max force diff = {np.abs(f_true - f_calc).max():.6f}")

plt.xlabel("Reference Forces")
plt.ylabel("Calculated Forces")
plt.title("Force Comparison")
plt.legend()
plt.show()

# Get calculator results for analysis
if hasattr(atoms.calc, 'results'):
    print(f"\nCalculator results keys: {list(atoms.calc.results.keys())}")

# Example: Calculate energy and forces for first simulation
if len(simulations) > 0:
    atoms_example, calc_example = simulations[0]
    _energy = atoms_example.get_potential_energy()
    _forces = atoms_example.get_forces()
    print(f"\nExample simulation:")
    print(f"  Energy: {_energy:.6f} eV")
    print(f"  Forces shape: {_forces.shape}")
    print(f"  Max force magnitude: {np.abs(_forces).max():.6f} eV/Å")

# ========================================================================
# SECTION 11: STRUCTURE MINIMIZATION
# ========================================================================
# Minimize structures using BFGS optimizer

def minimize_structure(atoms, run_index=0, nsteps=60, fmax=0.0006, charmm=False, calculator=None):
    """
    Minimize structure using BFGS optimizer (from run_sim.py)

    Args:
        atoms: ASE Atoms object (must have calculator set, or provide calculator)
        run_index: Index for trajectory file naming
        nsteps: Maximum number of optimization steps
        fmax: Force convergence criterion
        charmm: If True, run CHARMM minimization first
        calculator: Optional calculator to set if atoms doesn't have one
    """
    # Ensure calculator is set
    if atoms.calc is None:
        if calculator is not None:
            atoms.calc = calculator
        else:
            # Try to create calculator from atoms
            Z = atoms.get_atomic_numbers()
            R = atoms.get_positions()
            try:
                calc, _ = calculator_factory(
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
                atoms.calc = calc
                print("  Created calculator for minimization")
            except Exception as e:
                raise RuntimeError(f"Cannot minimize: atoms has no calculator and cannot create one: {e}")

    if charmm:
        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-6, tolgrd=1e-6)
        pycharmm.lingo.charmm_script("ENER")
        pycharmm.energy.show()
        atoms.set_positions(coor.get_positions())

    traj = ase_io.Trajectory(f'bfgs_{run_index}_{args.output_prefix}_minimized.traj', 'w')
    print("Minimizing structure with hybrid calculator")
    print(f"Running BFGS for {nsteps} steps")
    print(f"Running BFGS with fmax: {fmax}")
    _ = ase_opt.BFGS(atoms, trajectory=traj).run(fmax=fmax, steps=nsteps)
    # Sync with PyCHARMM
    xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
    coor.set_positions(xyz)
    traj.write(atoms)
    traj.close()
    return atoms

# Example: Minimize the first simulation
if len(simulations) > 0:
    atoms_to_minimize, calc_to_minimize = simulations[0]
    atoms_to_minimize = atoms_to_minimize.copy()
    atoms_to_minimize.calc = calc_to_minimize
    print("\nRunning minimization...")
    # Uncomment to run minimization:
    # atoms_minimized = minimize_structure(atoms_to_minimize, run_index=0, nsteps=100, fmax=0.0006)

# ========================================================================
# SECTION 12: SUMMARY
# ========================================================================
print("\n" + "=" * 60)
print("Simulation Setup Complete")
print("=" * 60)
print(f"Number of simulations initialized: {len(simulations)}")
print(f"Number of atoms per simulation: {ATOMS_PER_MONOMER * N_MONOMERS}")
print(f"Number of monomers: {N_MONOMERS}")
print(f"Atoms per monomer: {ATOMS_PER_MONOMER}")
print(f"ML cutoff: {CUTOFF_PARAMS.ml_cutoff} Å")
print(f"MM switch on: {CUTOFF_PARAMS.mm_switch_on} Å")
print(f"MM cutoff: {CUTOFF_PARAMS.mm_cutoff} Å")
print(f"Valid data batches available: {len(valid_batches)}")
print("=" * 60)
print("\nTo run MD simulations, use the helper functions or refer to run_sim.py")
print("Note: Residue numbers may need adjustment based on your system")


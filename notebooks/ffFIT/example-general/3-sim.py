#!/usr/bin/env python
# coding: utf-8

# ## Outline
# 
# This notebook is meant to detail setting up the MM/ML simulations 
# 
# ## fitting the LJs terms

# In[1]:


import mmml
import ase

import os
from pathlib import Path
import argparse
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check JAX configuration
devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())


# # Setup: Mock CLI Arguments (following run_sim.py structure)
# 
# This cell creates a mock args object that mimics the CLI arguments from `run_sim.py`.
# This allows the notebook to follow the same structure as the script.

# In[91]:


# Import required modules (following run_sim.py structure)
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
from mmml.pycharmmInterface.import_pycharmm import reset_block, coor
from mmml.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM

# Setup ASE imports
Atoms = setup_ase_imports()
CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()

# Additional imports for simulation
import ase.io as ase_io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
import ase.optimize as ase_opt


# In[92]:


# ========================================================================
# MOCK CLI ARGUMENTS (spoofing run_sim.py CLI)
# ========================================================================
# Create a mock args object that mimics the CLI arguments from run_sim.py
# This allows the notebook to follow the same structure as the script

class MockArgs:
    """Mock CLI arguments following run_sim.py structure"""
    def __init__(self):
        # Paths
        self.pdbfile = None  # Will be created from valid_data if needed
        self.checkpoint = Path(RESTART) if 'RESTART' in globals() else None

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

# Override with notebook-specific values if needed
if 'ATOMS_PER_MONOMER' in globals():
    args.n_atoms_monomer = ATOMS_PER_MONOMER
    args.atoms_per_monomer = ATOMS_PER_MONOMER
if 'N_MONOMERS' in globals():
    args.n_monomers = N_MONOMERS

print(f"Mock args created:")
print(f"  n_monomers: {args.n_monomers}")
print(f"  n_atoms_monomer: {args.n_atoms_monomer}")
print(f"  ml_cutoff: {args.ml_cutoff}")
print(f"  mm_switch_on: {args.mm_switch_on}")
print(f"  mm_cutoff: {args.mm_cutoff}")


# In[93]:


# System parameters (can be overridden by args)
ATOMS_PER_MONOMER = args.n_atoms_monomer
N_MONOMERS = args.n_monomers


# # Load Data and Prepare Batches (following run_sim.py structure)
# 
# This cell loads the validation data and prepares batches that will be used to initialize simulations.
# Note: The residue numbers in the PDB/PSF may need to be adjusted based on the actual system.

# In[94]:


# ========================================================================
# LOAD DATA AND PREPARE BATCHES (following run_sim.py structure)
# ========================================================================

# Initialize random key for data loading
if 'data_key' not in globals():
    data_key = jax.random.PRNGKey(42)


data_file = "/pchem-data/meuwly/boittier/home/mmml/mmml/data/fixed-acetone-only_MP2_21000.npz"

print(f"Loading data from: {data_file}")

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


# In[95]:


# Additional utility imports (if needed)
from ase.visualize.plot import plot_atoms


# In[96]:


# Additional PyCHARMM imports (already imported in cell 3, but kept for reference)
from mmml.pycharmmInterface import setupRes, setupBox


# In[97]:


# ========================================================================
# LOAD MODEL AND SETUP CALCULATOR (following run_sim.py structure)
# ========================================================================
uid = "test-84aa02d9-e329-46c4-b12c-f55e6c9a2f94"
SCICORE = Path('/scicore/home/meuwly/boitti0000/')
SCICORE = Path("/pchem-data/meuwly/boittier/home/")
RESTART=str(SCICORE / "ckpts" / f"{uid}" / "epoch-5450" / "json_checkpoint")

# ========================================================================
# JSON-BASED CHECKPOINT LOADER (no orbax/pickle required)
# ========================================================================
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

    # If no config found, try to extract from checkpoint directory structure
    if not model_kwargs:
        print("Warning: No model config found, using defaults")
        # Try to infer from directory name or use defaults
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
    print(f"Checkpoint base dir: {base_ckpt_dir}")
    print(f"Checkpoint epoch dir: {epoch_dir}")
else:
    # Fallback if RESTART is defined
    if 'RESTART' in globals():
        base_ckpt_dir = Path(RESTART)
        epoch_dir = base_ckpt_dir
    else:
        raise ValueError("Checkpoint path must be provided via args.checkpoint or RESTART variable")

# Load model parameters (using JSON-based loader to avoid orbax/pickle requirement)
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

# Setup calculator factory (following run_sim.py)
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
print(f"Cutoff parameters: {CUTOFF_PARAMS}")


# ## Fit Lennard-Jones Parameters to Training Data
# 
# Before running simulations, we can optimize the LJ parameters (epsilon and sigma scaling factors) to better match the training dataset. This fits only the MM part of the hybrid potential.
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

# In[ ]:


# ========================================================================
# IMPORT OPTIMIZATION FUNCTIONS FROM UTILITY MODULE
# ========================================================================
# All optimization functions have been moved to mmml.utils.hybrid_optimization
# Import them here for use in the notebook

from mmml.utils.hybrid_optimization import (
    extract_lj_parameters_from_calculator,
    fit_hybrid_potential_to_training_data_jax,
    fit_lj_parameters_to_training_data_jax,
)

get_ipython().run_line_magic('pinfo', 'extract_lj_parameters_from_calculator')
get_ipython().run_line_magic('pinfo', 'fit_hybrid_potential_to_training_data_jax')


# # Initialize Simulations from valid_data Batches
# 
# This section initializes simulations using positions and atomic numbers from `valid_data` batches.
# Each batch can be used to create an ASE Atoms object and run a simulation.

# In[10]:


# ========================================================================
# SETUP Pycharmm SYSTEM FIRST (required before MM contributions)
# ========================================================================
# IMPORTANT: PyCHARMM system must be initialized BEFORE creating calculators
# that use MM contributions, otherwise charges won't be available
#
# This generates residues in PyCHARMM and builds the structure.
# The atom ordering from PyCHARMM will be used to reorder valid_data batch atoms.

# Clear CHARMM state
# CLEAR_CHARMM()
# reset_block()

                                                                                                                                                                                                    # Generate residues in PyCHARMM
# For N_MONOMERS=2, we generate "ACO ACO" (two acetone molecules)
# Adjust the residue string based on N_MONOMERS and your system
residue_string = " ".join(["ACO"] * N_MONOMERS)
print(f"Generating {N_MONOMERS} residues: {residue_string}")

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


# In[11]:


mmml.pycharmmInterface.import_pycharmm.view_pycharmm_state()


# In[12]:


Z, R = valid_data["Z"][10], valid_data["R"][10]

ase_atoms = ase.Atoms(Z,R)
import pandas as pd
from ase.visualize import view as view_atoms
view_atoms(ase_atoms, viewer="x3d")


# In[13]:


R_ = R.copy()
R


# In[14]:


coor.set_positions(pd.DataFrame(R, columns=["x", "y", "z"]))
mmml.pycharmmInterface.import_pycharmm.view_pycharmm_state()


# In[15]:


pycharmm_atypes = psf.get_atype()
pycharmm_resids = np.array(psf.get_resid())


# In[16]:


pycharmm_atypes, pycharmm_resids


# In[17]:


_ = coor.show()
_


# In[18]:


from mmml.pycharmmInterface.import_pycharmm import reset_block, pycharmm, reset_block_no_internal, pycharmm_quiet, pycharmm_verbose
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


# In[19]:


pycharmm_quiet()


# In[20]:


def test_configuration(reordered):
    coor.set_positions(pd.DataFrame(R[reordered], columns=["x", "y", "z"]))
    coor.get_positions()
    # energy.show()
    energy_df = energy.get_energy()
    return float(energy_df[["IMPR", "ANGL", "BOND", "UREY", "DIHE"]].to_numpy().sum())

import itertools

# Example: 20-atom array = first 10 in block A, next 10 in block B
atoms = list(range(20))  # replace with your actual atom indices / objects

n = 10  # size of each block
block1 = atoms[:n]
block2 = atoms[n:2*n]
res = []




# In[21]:


test_configuration(np.arange(20))


# In[24]:


# min_res = 10000000
# reordered_found = None
# for perm in itertools.permutations(range(n)):
#     # Apply the same permutation to both blocks
#     new_block1 = [block1[i] for i in perm]
#     new_block2 = [block2[i] for i in perm]

#     # Recombine into a 20-length array
#     reordered = new_block1 + new_block2
#     res = test_configuration(reordered)
#     # print(res)
#     if min_res > res:
#         reordered_found = reordered
#         min_res = res
    # min_res, reordered_found
# reordered_found


# In[ ]:





# In[ ]:





# In[25]:


def val_configuration(reordered):
    coor.set_positions(pd.DataFrame(R[reordered], columns=["x", "y", "z"]))
    coor.get_positions()
    # energy.show()
    energy_df = energy.get_energy()
    return energy_df[["IMPR", "ANGL", "BOND", "UREY", "DIHE"]]


# In[26]:


val_configuration(np.arange(20))


# In[27]:


Z


# In[28]:


pycharmm_atypes


# In[29]:


final_reorder = [3, 0, 1, 2, 7, 8, 9, 4, 5, 6, 
                 13, 10, 11, 12, 17, 18, 19, 14, 15, 16]
final_reorder

val_configuration(final_reorder)


# In[30]:


mmml.pycharmmInterface.import_pycharmm.view_pycharmm_state()


# # Setup PyCHARMM System (REQUIRED before MM contributions)
# 
# **IMPORTANT**: The PyCHARMM system must be initialized BEFORE creating calculators that use MM contributions. 
# 
# This cell:
# 1. Generates residues using `setupRes.generate_residue()` (e.g., "ACO ACO" for two acetone molecules)
# 2. Builds the structure using `ic.build()`
# 3. Gets the atom ordering from PyCHARMM
# 
# **Note on atom reordering**: The atoms from `valid_data` batches may need to be reordered to match PyCHARMM's atom ordering. 
# The `reorder_atoms_to_match_pycharmm()` function handles this, but you may need to customize it based on your system.
# 
# - Residue names (e.g., "ACO" for acetone) must match your system
# - The number of residues should match `N_MONOMERS`
# - If PyCHARMM initialization fails, MM contributions will be automatically disabled

# In[ ]:





# # Initialize Multiple Simulations from valid_data Batches
# 
# This cell demonstrates how to initialize multiple simulations from different batches.
# Each simulation can be run independently.

# In[31]:


from mmml.utils.simulation_utils import (
    reorder_atoms_to_match_pycharmm,
    initialize_simulation_from_batch,
    initialize_multiple_simulations,
)
# initialize_simulation_from_batch?
# Initialize first simulation from batch 0
# atoms, hybrid_calc = initialize_simulation_from_batch(valid_batches[0], calculator_factory, CUTOFF_PARAMS, args)


# In[32]:


# Step 1: Extract base LJ parameters (do this once, after calculator_factory is created)
lj_params = extract_lj_parameters_from_calculator(ATOMS_PER_MONOMER=10, N_MONOMERS=2 )
lj_params


# In[33]:


# Prepare batches for validation data (used to initialize simulations)
valid_batches = prepare_batches_jit(data_key, valid_data, 1, num_atoms=ATOMS_PER_MONOMER * N_MONOMERS)
train_batches = prepare_batches_jit(data_key, train_data, 1, num_atoms=ATOMS_PER_MONOMER * N_MONOMERS)


# In[34]:


type(train_batches)


# In[35]:


train_batches_copy = train_batches.copy()


# In[36]:


for i in range(len(train_batches)):
    train_batches_copy[i]["R"] = train_batches[i]["R"][final_reorder]
    train_batches_copy[i]["Z"] = train_batches[i]["Z"][final_reorder]
    train_batches_copy[i]["F"] = train_batches[i]["F"][final_reorder]


# In[37]:


# ========================================================================
# MODE 1: Optimize LJ parameters only
# ========================================================================
print("=" * 60)
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
    cutoff_params=CUTOFF_PARAMS,  # Optional
    args=args,  # Optional
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



# In[38]:


opt_ep_scale_lj, opt_sig_scale_lj


# In[39]:


result_lj


# In[40]:


# # ========================================================================
# # MODE 2: Optimize ML parameters only
# # ========================================================================
# print("\n" + "=" * 60)
# print("MODE 2: Optimizing ML parameters only")
# print("=" * 60)

# result_ml = fit_hybrid_potential_to_training_data_jax(
#     train_batches=train_batches_copy,
#     base_calculator_factory=calculator_factory,
#     model=model,
#     model_params=params,
#     atc_epsilons=lj_params["atc_epsilons"],
#     atc_rmins=lj_params["atc_rmins"],
#     atc_qs=lj_params["atc_qs"],
#     at_codes=lj_params["at_codes"],
#     pair_idx_atom_atom=lj_params["pair_idx_atom_atom"],
#     optimize_mode="ml_only",
#     n_samples=20,
#     energy_weight=1.0,
#     force_weight=1.0,
#     learning_rate=0.001,  # Lower LR for ML params
#     n_iterations=100,
#     verbose=True
# )

# opt_ml_params = result_ml["ml_params"]



# In[ ]:


# # ========================================================================
# # MODE 3: Optimize both ML and LJ parameters together
# # ========================================================================
# print("\n" + "=" * 60)
# print("MODE 3: Optimizing both ML and LJ parameters together")
# print("=" * 60)

# result_both = fit_hybrid_potential_to_training_data_jax(
#     train_batches=train_batches_copy,
#     base_calculator_factory=calculator_factory,
#     model=model,
#     model_params=params,
#     atc_epsilons=lj_params["atc_epsilons"],
#     atc_rmins=lj_params["atc_rmins"],
#     atc_qs=lj_params["atc_qs"],
#     at_codes=lj_params["at_codes"],
#     pair_idx_atom_atom=lj_params["pair_idx_atom_atom"],
#     optimize_mode="both",
#     n_samples=20,
#     energy_weight=1.0,
#     force_weight=1.0,
#     learning_rate=0.01,
#     n_iterations=100,
#     verbose=True
# )

# opt_ml_params_both = result_both["ml_params"]
# opt_ep_scale_both = result_both["ep_scale"]
# opt_sig_scale_both = result_both["sig_scale"]



# In[41]:


result = fit_hybrid_potential_to_training_data_jax(
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
    initial_ml_cutoff=1.0,  # optional
    initial_mm_switch_on=6.0,  # optional
    initial_mm_cutoff=1.0,  # optional
    n_samples=20,
    learning_rate=0.01,
    n_iterations=100,
    verbose=True
)


# In[42]:


# result


# In[43]:


CUTOFF_PARAMS = CutoffParameters(ml_cutoff=result["ml_cutoff"], mm_switch_on=result["mm_switch_on"], mm_cutoff=result["mm_cutoff"])


# In[68]:


# # ========================================================================
# # Use optimized parameters in subsequent calculations
# # ========================================================================
# # For LJ-only optimization:
from mmml.utils.hybrid_optimization import expand_scaling_parameters_to_full_set
# Expand to full parameter set
full_ep_scale, full_sig_scale = expand_scaling_parameters_to_full_set(
   opt_ep_scale_lj, opt_sig_scale_lj, lj_params
)  # Shape: (163,) - all types


calculator_factory_lj_optimized = setup_calculator(
    ATOMS_PER_MONOMER=args.n_atoms_monomer,
    N_MONOMERS=args.n_monomers,
    ml_cutoff_distance=result["ml_cutoff"],
    mm_switch_on=result["mm_switch_on"],
    mm_cutoff=result["mm_cutoff"],
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


# In[69]:


get_ipython().run_line_magic('pinfo', 'initialize_simulation_from_batch')


# In[86]:


from mmml.utils.simulation_utils import (
    reorder_atoms_to_match_pycharmm,
    initialize_simulation_from_batch,
    initialize_multiple_simulations,
)
# initialize_simulation_from_batch?
# Initialize first simulation from batch 0
atoms, hybrid_calc = initialize_simulation_from_batch(train_batches_copy[0], calculator_factory_lj_optimized, , args)


# In[90]:


dir(args)


# In[78]:


hybrid_calc.verbose = True
atoms.calc = hybrid_calc


# In[79]:


get_ipython().run_line_magic('pinfo', 'hybrid_calc')


# In[80]:


import matplotlib.pyplot as plt

for b in train_batches_copy[:10]:
    if b["N"] == 20:
        atoms.set_positions(b["R"])
        f_true, f_calc = b["F"], atoms.get_forces()
        # print(f_true)
        # print(f_calc)
        # print(f_true - f_calc)
        print("sssss")
        plt.scatter(f_true.flatten(), f_calc.flatten())

plt.show()


# In[81]:


atoms.calc.results


# In[82]:


atoms.get_forces()


# In[83]:


atoms.get_atomic_numbers()


# In[84]:


pycharmm_atypes


# In[85]:


mmml.pycharmmInterface.import_pycharmm.view_pycharmm_state()


# In[65]:


get_ipython().run_line_magic('pinfo', 'initialize_multiple_simulations')


# In[85]:


# Initialize multiple simulations
# Adjust n_simulations as needed
simulations = initialize_multiple_simulations(train_batches_copy[:2], calculator_factory_lj_optimized, CUTOFF_PARAMS, args)
print(f"\nInitialized {len(simulations)} simulations from valid_data batches")


# # Example: Run a Simple Energy Calculation
# 
# This demonstrates how to use the initialized simulations.

# In[92]:


                # ========================================================================
# EXAMPLE: RUN ENERGY CALCULATIONS
# ========================================================================

# Example: Calculate energy for the first simulation
if len(simulations) > 0:
    atoms_example, calc_example = simulations[0]
    _energy = atoms_example.get_potential_energy()
    _forces = atoms_example.get_forces()
    print(f"Example simulation energy: {_energy:.6f} eV")
    print(f"Example simulation forces shape: {_forces.shape}")
    print(f"Max force magnitude: {np.abs(_forces).max():.6f} eV/Å")
else:
    print("No simulations initialized. Check batch data and system parameters.")


# In[68]:


from ase.visualize import view
view(atoms_example, viewer="x3d")


# In[69]:


forces


# In[78]:


train_batches_copy[0]["F"]


# In[79]:


train_batches_copy[0]["E"]


# # Next Steps: Running MD Simulations
# 
# To run MD simulations following `run_sim.py`, you can:
# 1. Use the `minimize_structure` function from run_sim.py
# 2. Use the `run_ase_md` function for ASE-based MD
# 3. Use JAX-MD for more advanced simulations
# 
# See `run_sim.py` for complete MD simulation setup.

# In[74]:


# ========================================================================
# HELPER FUNCTIONS (from run_sim.py)
# ========================================================================
# These functions can be copied from run_sim.py for running MD simulations

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
    import pandas as pd
    xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
    coor.set_positions(xyz)
    traj.write(atoms)
    traj.close()
    return atoms

# Example: Minimize the first simulation
if len(simulations) > 0:
    # Get atoms and calculator from the simulation
    atoms_to_minimize, calc_to_minimize = simulations[0]
    # Create a copy but preserve the calculator
    atoms_to_minimize = atoms_to_minimize.copy()
    atoms_to_minimize.calc = calc_to_minimize  # Ensure calculator is set
    print("Running minimization...")
    print("Note: Calculator is preserved from the initialized simulation")
    # Uncomment to run minimization:
    atoms_minimized = minimize_structure(atoms_to_minimize, run_index=0, nsteps=100, fmax=0.0006)


# In[77]:


view_atoms(atoms_minimized, viewer="x3d")


# In[82]:


atoms_minimized.get_potential_energy()


# In[80]:


atoms_minimized.get_forces()


# # Notes on Residue Numbers and Atom Ordering
# 
# When setting up PyCHARMM simulations:
# 
# **Residue Setup:**
# - Use `setupRes.generate_residue("ACO ACO")` to generate residues (for 2 acetone molecules)
# - Use `ic.build()` to build the structure
# - The number of residues should match `N_MONOMERS`
# 
# **Atom Ordering:**
# - PyCHARMM has a specific atom ordering based on residue and atom type
# - The `valid_data` batch atoms **must be reordered** to match PyCHARMM's ordering
# - The `reorder_atoms_to_match_pycharmm()` function tries different orderings and selects the one that **minimizes CHARMM internal energy** (`energy.get_term_by_name("INTE")`)
# - Common swaps tested: indices 0↔3, 10↔13, and combinations
# - The function automatically finds the best ordering by energy minimization
# 
# **To customize reordering:**
# 1. Add more swap patterns to the `candidate_orderings` list in `reorder_atoms_to_match_pycharmm()`
# 2. The function will automatically test all candidates and select the one with minimum INTE energy
# 3. Example swaps: `fix_idxs[0] = _fix_idxs[3]; fix_idxs[3] = _fix_idxs[0]` (swap 0↔3)
# 4. The energy-based selection ensures the correct ordering is found automatically
# 

# In[71]:


# ========================================================================
# SUMMARY
# ========================================================================
print("=" * 60)
print("Simulation Setup Complete")
print("=" * 60)
print(f"Number of simulations initialized: {len(simulations)}")
print(f"Number of atoms per simulation: {ATOMS_PER_MONOMER * N_MONOMERS}")
print(f"Number of monomers: {N_MONOMERS}")
print(f"Atoms per monomer: {ATOMS_PER_MONOMER}")
print(f"ML cutoff: {args.ml_cutoff} Å")
print(f"MM switch on: {args.mm_switch_on} Å")
print(f"MM cutoff: {args.mm_cutoff} Å")
print(f"Valid data batches available: {len(valid_batches)}")
print("=" * 60)
print("\nTo run MD simulations, use the helper functions or refer to run_sim.py")
print("Note: Residue numbers may need adjustment based on your system")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[73]:


R = valid_batches[0]["R"]
Z = valid_batches[0]["Z"]
R,Z


# In[ ]:





# In[ ]:





# In[ ]:





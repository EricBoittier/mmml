#!/usr/bin/env python
# coding: utf-8
"""
Setup Cluster of 10 Molecules with Optimized Parameters

This script:
1. Loads optimized LJ and cutoff parameters from p3_sim_FIXED.py
2. Generates a cluster of 10 molecules (from a reference dimer structure)
3. Sets up the calculator with optimized parameters for the cluster
4. Saves the cluster structure

Usage:
    python setup_cluster_10molecules.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import json
import ase
import pandas as pd
from ase import Atoms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for JAX/GPU
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"
N_MOLS_IN_CLUSTER = 50
# ========================================================================
# IMPORTS
# ========================================================================
from mmml.cli.base import setup_ase_imports, setup_mmml_imports
from mmml.pycharmmInterface.mmml_calculator import setup_calculator, CutoffParameters
from mmml.utils.simulation_utils import initialize_simulation_from_batch, reorder_atoms_to_match_pycharmm

# Setup ASE imports
Atoms = setup_ase_imports()
CutoffParameters, ev2kcalmol, setup_calculator, get_ase_calc = setup_mmml_imports()

# PyCHARMM imports (for LJ parameter extraction)
try:
    import pycharmm
    import pycharmm.ic as ic
    import pycharmm.generate as gen  # Note: it's pycharmm.generate, not pycharmm.gen
    import pycharmm.psf as psf
    from mmml.pycharmmInterface import setupRes
    from mmml.pycharmmInterface.import_pycharmm import (
        reset_block, 
        coor, 
        reset_block_no_internal, 
        pycharmm_quiet
    )
    PYCHARMM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyCHARMM not available: {e}")
    PYCHARMM_AVAILABLE = False

print("=" * 70)
print("CLUSTER OF {N_MOLS_IN_CLUSTER} MOLECULES SETUP")
print("=" * 70)

# ========================================================================
# STEP 1: LOAD OPTIMIZED PARAMETERS
# ========================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading optimized parameters")
print("=" * 70)

params_file = Path("optimized_parameters/optimized_lj_cutoff_params.json")
if not params_file.exists():
    raise FileNotFoundError(
        f"Optimized parameters file not found: {params_file}\n"
        f"Please run p3_sim_FIXED.py first to generate optimized parameters."
    )

with open(params_file, 'r') as f:
    saved_params = json.load(f)

# Extract parameters
opt_ep_scale = np.array(saved_params["ep_scale"])
opt_sig_scale = np.array(saved_params["sig_scale"])
cutoff_params = CutoffParameters(
    ml_cutoff=saved_params["ml_cutoff"],
    mm_switch_on=saved_params["mm_switch_on"],
    mm_cutoff=saved_params["mm_cutoff"],
)
checkpoint_path = Path(saved_params["checkpoint_path"])
n_monomers_ref = saved_params.get("n_monomers", 2)
n_atoms_monomer = saved_params.get("n_atoms_monomer", 10)

print(f"✓ Loaded optimized parameters from: {params_file}")
print(f"  ML cutoff: {cutoff_params.ml_cutoff} Å")
print(f"  MM switch on: {cutoff_params.mm_switch_on} Å")
print(f"  MM cutoff: {cutoff_params.mm_cutoff} Å")
print(f"  Checkpoint path: {checkpoint_path}")

# ========================================================================
# STEP 2: LOAD MODEL (for calculator setup)
# ========================================================================
print("\n" + "=" * 70)
print("STEP 2: Loading ML model")
print("=" * 70)

# Load model using the same approach as p3_sim_FIXED.py
from mmml.cli.base import resolve_checkpoint_paths
from mmml.physnetjax.physnetjax.restart.restart import get_params_model

base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(checkpoint_path)
print(f"  Base checkpoint dir: {base_ckpt_dir}")
print(f"  Epoch dir: {epoch_dir}")

# For cluster of 10 molecules
n_cluster_molecules = N_MOLS_IN_CLUSTER
n_atoms_cluster = n_cluster_molecules * n_atoms_monomer

try:
    params, model = get_params_model(str(epoch_dir), natoms=n_atoms_cluster)
    model.natoms = n_atoms_cluster
    print(f"✓ Loaded model for {n_cluster_molecules} molecules ({n_atoms_cluster} atoms)")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("  You may need to load it manually or adjust the checkpoint path")
    params = None
    model = None

# ========================================================================
# STEP 3: LOAD PRE-PACKED CLUSTER (50 MOLECULES)
# ========================================================================
print("\n" + "=" * 70)
print("STEP 3: Loading pre-packed cluster (50 molecules)")
print("=" * 70)

from pathlib import Path
import ase.io

pdb_path = Path("pdb/init-packmol.pdb")
print(f"  Loading cluster from: {pdb_path}")
cluster_atoms = ase.io.read(pdb_path)
cluster_positions = cluster_atoms.get_positions()
cluster_Z = cluster_atoms.get_atomic_numbers()

assert len(cluster_atoms) == n_atoms_cluster, (
    f"Expected {n_atoms_cluster} atoms, got {len(cluster_atoms)}"
)

print(f"✓ Loaded cluster of {n_cluster_molecules} molecules")
print(f"  Total atoms: {len(cluster_atoms)}")
print(f"  Cluster dimensions: {np.max(cluster_positions, axis=0) - np.min(cluster_positions, axis=0)}")

# Save cluster structure (for reference)
cluster_output_file = Path("cluster_50molecules.xyz")
ase.io.write(str(cluster_output_file), cluster_atoms)
print(f"  Saved cluster structure to: {cluster_output_file}")

# ========================================================================
# STEP 4: LOAD LJ PARAMETERS (FROM SAVED FILE OR PYCHARMM)
# ========================================================================
print("\n" + "=" * 70)
print("STEP 4: Loading LJ parameters")
print("=" * 70)

# LJ parameters are per atom type, not per molecule count
# So parameters saved for 2 molecules work for 10 molecules
# We'll use saved parameters if available (preferred), otherwise extract from PyCHARMM

lj_params_base = None

# First, try to use saved parameters (preferred - no PyCHARMM needed)
if "lj_params" in saved_params:
    print("  Using LJ parameters from saved file...")
    lj_params_base = saved_params["lj_params"].copy()
    # Convert lists back to numpy arrays if needed
    array_keys = ["atc_epsilons", "atc_rmins", "atc_qs", "pair_idx_atom_atom", "unique_iac_codes", "iac_to_param_idx"]
    for key in array_keys:
        if key in lj_params_base and lj_params_base[key] is not None:
            if isinstance(lj_params_base[key], list):
                lj_params_base[key] = np.array(lj_params_base[key])
            elif isinstance(lj_params_base[key], (jnp.ndarray, np.ndarray)):
                lj_params_base[key] = np.array(lj_params_base[key])

    # Expand pair_idx_atom_atom from base (2-monomer) system to full cluster if available
    try:
        if "pair_idx_atom_atom" in lj_params_base and lj_params_base["pair_idx_atom_atom"] is not None:
            base_pairs = np.array(lj_params_base["pair_idx_atom_atom"])
            if base_pairs.ndim == 2 and base_pairs.shape[1] == 2:
                expanded_pairs = []
                for a in range(n_cluster_molecules):
                    for b in range(a + 1, n_cluster_molecules):
                        offset_a = a * n_atoms_monomer
                        offset_b = b * n_atoms_monomer
                        for i0, i1 in base_pairs:
                            if i0 < n_atoms_monomer:
                                mapped_i0 = offset_a + i0
                            else:
                                mapped_i0 = offset_b + (i0 - n_atoms_monomer)
                            if i1 < n_atoms_monomer:
                                mapped_i1 = offset_a + i1
                            else:
                                mapped_i1 = offset_b + (i1 - n_atoms_monomer)
                            expanded_pairs.append([mapped_i0, mapped_i1])
                lj_params_base["pair_idx_atom_atom_expanded"] = np.array(expanded_pairs, dtype=int)
                print(f"  ✓ Expanded pair_idx_atom_atom to cluster size: {len(expanded_pairs)} pairs")
    except Exception as e_expand:
        print(f"  ⚠ Could not expand pair_idx_atom_atom: {e_expand}")
    print("  ✓ Using LJ parameters from saved file")
    print(f"     Note: Parameters were saved for 2 molecules but work for any number")
else:
    # Fallback: Extract from PyCHARMM (requires initialization)
    print("  Saved LJ parameters not found, will extract from PyCHARMM...")
    if not PYCHARMM_AVAILABLE:
        raise RuntimeError(
            "Cannot proceed without LJ parameters.\n"
            "Options:\n"
            "  1. Re-run p3_sim_FIXED.py to generate saved parameters\n"
            "  2. Ensure PyCHARMM is available for parameter extraction"
        )
    
    # Initialize PyCHARMM for 2 monomers (base system - number doesn't matter for parameter extraction)
    base_n_monomers = 2
    residue_string = " ".join(["ACO"] * base_n_monomers)
    print(f"  Initializing PyCHARMM with {base_n_monomers} residues for parameter extraction...")
    
    try:
        # Generate residues (this creates the PSF structure)
        setupRes.generate_residue(residue_string)
        print("  ✓ Residues generated successfully")
        
        # Build the structure using internal coordinates
        ic.build()
        print("  ✓ Structure built using internal coordinates")
        
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
        print("  ✓ PyCHARMM non-bonded parameters set")
        
        # Extract parameters
        lj_params_base = extract_lj_parameters_from_calculator(
            ATOMS_PER_MONOMER=n_atoms_monomer,
            N_MONOMERS=base_n_monomers
        )
        print("  ✓ Base LJ parameters extracted from PyCHARMM")
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract LJ parameters from PyCHARMM: {e}")

# ========================================================================
# STEP 5: SETUP PYCHARMM FOR CLUSTER (FOR SIMULATIONS)
# ========================================================================
print("\n" + "=" * 70)
print("STEP 5: Setting up PyCHARMM for 10-molecule cluster")
print("=" * 70)

# Set up PyCHARMM for the actual cluster size (10 molecules)
# This is needed for running simulations, but doesn't affect parameter loading
if PYCHARMM_AVAILABLE:
    try:
        # Will store permutation back to model order (for ML force remapping)
        ml_reorder_indices = None

        # Clear any previous PyCHARMM state FIRST
        print("  Clearing previous PyCHARMM state...")
        from mmml.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM
        CLEAR_CHARMM()
        print("  ✓ PyCHARMM state cleared")
        
        # Read RTF and parameter files (same as p3_sim_FIXED.py)
        from mmml.pycharmmInterface.import_pycharmm import read, CGENFF_RTF, CGENFF_PRM, settings
        
        print("  Reading RTF and parameter files...")
        read.rtf(CGENFF_RTF)
        # Set bomb level to -2 to suppress NBFIX warnings when reading CGENFF parameters
        bl = settings.set_bomb_level(-2)
        wl = settings.set_warn_level(-2)
        read.prm(CGENFF_PRM)
        # Restore bomb level and warn level
        settings.set_bomb_level(bl)
        settings.set_warn_level(wl)
        pycharmm.lingo.charmm_script("bomlev 0")
        print("  ✓ RTF and parameter files read")
        
        # Generate sequence string for 10 acetone molecules
        # Using read.sequence_string and gen.new_segment approach (as shown in example)
        
        print("  Generating sequence for 10 acetone molecules...")
        # Create sequence string: "ACO ACO ACO ..." (10 times)
        sequence_string = " ".join(["ACO"] * n_cluster_molecules)
        print(f"  Sequence string: {sequence_string[:50]}... (truncated)")
        
        # Read the sequence
        read.sequence_string(sequence_string)
        print("  ✓ Sequence read")
        
        # Generate segment (using single segment for all molecules)
        print("  Generating segment...")
        gen.new_segment(seg_name='CLUSTER', setup_ic=True)
        print("  ✓ Segment generated")
        
        # Fill IC parameters
        print("  Filling IC parameters...")
        ic.prm_fill(replace_all=True)
        print("  ✓ IC parameters filled")
        
        # Build structure using internal coordinates
        print("  Building structure from internal coordinates...")
        ic.build()
        print("  ✓ Structure built from IC")
        
        # Verify PyCHARMM has the expected number of atoms
        pycharmm_n_atoms = len(psf.get_atype())
        print(f"  PyCHARMM PSF has {pycharmm_n_atoms} atoms")
        
        if pycharmm_n_atoms < n_atoms_cluster:
            print(f"  ⚠ Warning: PyCHARMM has {pycharmm_n_atoms} atoms, expected {n_atoms_cluster}")
            print(f"     This may cause calculator initialization to fail")
        else:
            print(f"  ✓ PyCHARMM has sufficient atoms: {pycharmm_n_atoms} >= {n_atoms_cluster}")
        
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

        _read_pdb_file = "pdb/init-packmol.pdb"
        mol = ase.io.read(_read_pdb_file)
        print(mol)
        print(mol.get_chemical_symbols())
        # coor.set_positions expects a pandas DataFrame with x/y/z columns
        init_pos = pd.DataFrame(mol.get_positions(), columns=["x", "y", "z"])
        coor.set_positions(init_pos)

        print("Running energy minimization to check PyCHARMM initialization...")

        XYZ = coor.get_positions()
        print(f"XYZ: {XYZ}")
        print(f"XYZ shape: {XYZ.shape}")
        print(f"XYZ[:10]: {XYZ[:10]}")
        print(f"XYZ[-10:]: {XYZ[-10:]}")
        print(f"XYZ min: {np.min(XYZ)}")
        print(f"XYZ max: {np.max(XYZ)}")
        print(f"XYZ mean: {np.mean(XYZ)}")
        print(f"XYZ std: {np.std(XYZ)}")
        # set XYZ to random positions
        XYZ = np.random.random(XYZ.shape)
        XYZ = pd.DataFrame(XYZ, columns=["x", "y", "z"])
        coor.set_positions(XYZ)
        print(f"XYZ: {XYZ}")
        print(f"XYZ shape: {XYZ.shape}")
        print(f"XYZ[:10]: {XYZ[:10]}")
        print(f"XYZ[-10:]: {XYZ[-10:]}")
        print(f"XYZ min: {np.min(XYZ)}")
        print(f"XYZ max: {np.max(XYZ)}")

        pycharmm.minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
        pycharmm.lingo.charmm_script("ENER")
        pycharmm.energy.show()
        print("Energy minimization completed")
        XYZ = coor.get_positions()
        cluster_atoms.set_positions(XYZ)
        pycharmm.lingo.charmm_script(nbonds)
        pycharmm_quiet()
        print("  ✓ PyCHARMM non-bonded parameters set for cluster")
        
        # Final verification
        pycharmm_atypes = np.array(psf.get_atype())
        pycharmm_charges = np.array(psf.get_charges())
        pycharmm_iac = np.array(psf.get_iac())
        pycharmm_resids = np.array(psf.get_res())
        # print(f"  ✓ PyCHARMM fully initialized:")
        # print(f"     Atoms: {len(pycharmm_atypes)}")
        # print(f"     Charges: {len(pycharmm_charges)}")
        # print(f"     IAC codes: {len(pycharmm_iac)}")
        # print(f"     Residue IDs: {len(pycharmm_resids)}")
        
        # # ========================================================================
        # # REORDER ATOMS TO MATCH PYCHARMM ORDERING
        # # ========================================================================
        # print("\n  Reordering cluster atoms to match PyCHARMM ordering...")
        
        # # IMPORTANT: PyCHARMM generates the PSF with atoms in a specific order based on the RTF.
        # # We need to reorder our cluster positions to match PyCHARMM's PSF order.
        # # First, try automatic reordering which tests multiple patterns and finds the best match
        # try:
        #     cluster_positions_reordered, cluster_Z_reordered, reorder_indices_auto = reorder_atoms_to_match_pycharmm(
        #         R=cluster_positions,
        #         Z=cluster_Z,
        #         pycharmm_atypes=pycharmm_atypes[:n_atoms_cluster],
        #         pycharmm_resids=pycharmm_resids[:n_atoms_cluster],
        #         ATOMS_PER_MONOMER=n_atoms_monomer,
        #         N_MONOMERS=n_cluster_molecules,
        #     )
        #     cluster_positions = cluster_positions_reordered
        #     cluster_Z = cluster_Z_reordered
        #     print(f"  ✓ Automatic reordering found pattern (first 20 indices: {reorder_indices_auto[:20]})")
        #     try:
        #         inv = np.argsort(reorder_indices_auto)
        #         if len(inv) == len(cluster_positions):
        #             ml_reorder_indices = inv
        #     except Exception:
        #         pass
            
        # except Exception as e_auto:
        #     print(f"  ⚠ Automatic reordering failed, trying manual pattern: {e_auto}")
        #     # Fallback to manual pattern from p3_sim_FIXED.py
        #     # Pattern for 2 molecules: [3, 0, 1, 2, 7, 8, 9, 4, 5, 6, 13, 10, 11, 12, 17, 18, 19, 14, 15, 16]
        #     # Per molecule (10 atoms): [3, 0, 1, 2, 7, 8, 9, 4, 5, 6]
        #     monomer_reorder_pattern = [3, 0, 1, 2, 7, 8, 9, 4, 5, 6]
            
        #     # Extend to 10 molecules
        #     print(f"  Applying manual reordering pattern per molecule: {monomer_reorder_pattern}")
        #     full_reorder_indices = []
        #     for mol_idx in range(n_cluster_molecules):
        #         mol_start = mol_idx * n_atoms_monomer
        #         for atom_idx_in_mol in monomer_reorder_pattern:
        #             full_reorder_indices.append(mol_start + atom_idx_in_mol)
            
        #     print(f"  Full reordering pattern (first 20 indices): {full_reorder_indices[:20]}")
        #     print(f"  Full reordering pattern (last 20 indices): {full_reorder_indices[-20:]}")
            
        #     # Apply reordering
        #     cluster_positions = cluster_positions[full_reorder_indices]
        #     cluster_Z = cluster_Z[full_reorder_indices]
        #     print(f"  ✓ Applied manual reordering pattern")
            
        #     # Update PyCHARMM coordinates
        #     coor.set_positions(pd.DataFrame(cluster_positions, columns=["x", "y", "z"]))
        
        # Reordering disabled: keep PDB order as-is
        print("  Reordering skipped; keeping PDB atom order.")
        ml_reorder_indices = None
        
        # Verify the reordering worked by checking summed internal energy (no INTE)
        try:
            import pycharmm.energy as energy
            energy.get_energy()

            # Sum available internal terms
            preferred_terms = ["BOND", "ANGLE", "DIHE", "IMPR", "UREY", "CMAP"]
            term_names = energy.get_term_names()
            term_statuses = energy.get_term_statuses()

            total_internal = 0.0
            terms_used = []
            any_term = False

            for term in preferred_terms:
                if term in term_names:
                    idx = term_names.index(term)
                    if term_statuses[idx]:
                        try:
                            val = energy.get_term_by_name(term)
                            if np.isfinite(val):
                                total_internal += val
                                terms_used.append(f"{term}={val:.6f}")
                                any_term = True
                        except Exception:
                            continue

            # Fallback: use any active term except INTE if none were added
            if not any_term:
                for name, status in zip(term_names, term_statuses):
                    if status and name != "INTE":
                        try:
                            val = energy.get_term_by_name(name)
                            if np.isfinite(val):
                                total_internal += val
                                terms_used.append(f"{name}={val:.6f}")
                                any_term = True
                        except Exception:
                            continue

            if any_term:
                terms_str = " ".join(terms_used)
                print(f"  ✓ PyCHARMM internal energy (sum) after reordering: {total_internal:.6f} kcal/mol")
                print(f"    Components: {terms_str}")
                if total_internal > 100:
                    print(f"  ⚠ WARNING: High internal energy ({total_internal:.2f} kcal/mol) suggests ordering may still be incorrect")
                    print(f"     Large forces may occur - check atom mapping")
            else:
                print("  ⚠ Could not find active internal energy terms to report (excluding INTE)")

        except Exception as e_energy:
            print(f"  ⚠ Could not check PyCHARMM energy: {e_energy}")
        
    except Exception as e:
        print(f"  ✗ ERROR: Could not initialize PyCHARMM for cluster: {e}")
        import traceback
        traceback.print_exc()
        print(f"     Calculator initialization WILL fail if MM is enabled")
        print(f"     The fallback will create an ML-only calculator")
        PYCHARMM_INITIALIZED = False
else:
    print("  ⚠ PyCHARMM not available - cluster simulations will use ML only")
    PYCHARMM_INITIALIZED = False

# Set flag to indicate PyCHARMM status
if PYCHARMM_AVAILABLE:


    try:
        pycharmm_n_atoms_check = len(psf.get_atype())
        PYCHARMM_INITIALIZED = (pycharmm_n_atoms_check >= n_atoms_cluster)
        if PYCHARMM_INITIALIZED:
            print(f"  ✓ PyCHARMM initialization verified: {pycharmm_n_atoms_check} atoms >= {n_atoms_cluster} required")
    except Exception as e:
        print(f"  ⚠ Could not verify PyCHARMM initialization: {e}")
        PYCHARMM_INITIALIZED = False
else:
    PYCHARMM_INITIALIZED = False

# Validate atom-type codes (IAC) against LJ params using iac_to_param_idx mapping
mapped_param_indices = None
if PYCHARMM_INITIALIZED and lj_params_base is not None and "at_codes" in lj_params_base and "iac_to_param_idx" in lj_params_base:
    try:
        iac_to_param_idx = lj_params_base.get("iac_to_param_idx", {})
        at_codes_saved = np.array(lj_params_base["at_codes"])
        # Use one-monomer atom codes (first n_atoms_monomer) and tile to cluster
        at_codes_monomer = at_codes_saved[:n_atoms_monomer]
        expected_cluster = np.tile(at_codes_monomer, n_cluster_molecules)
        pycharmm_iac = np.array(psf.get_iac())[:len(expected_cluster)]
        # First pass: map PyCHARMM IACs to param indices using the saved mapping
        mapped = []
        for iac in pycharmm_iac:
            key = str(int(iac))
            mapped.append(iac_to_param_idx.get(key, None))
        mapped_param_indices = np.array(mapped, dtype=object)

        # If many missing, auto-build a mapping from all monomers' IAC codes to saved atom codes
        n_missing = np.sum(mapped_param_indices == None)
        if n_missing > 0:
            auto_map = {}
            for i in range(len(pycharmm_iac)):
                key = str(int(pycharmm_iac[i]))
                val = int(expected_cluster[i])  # expected atom code at this position
                if key in auto_map and auto_map[key] != val:
                    print(f"  ⚠ Auto-map conflict for IAC {key}: existing {auto_map[key]} vs new {val}")
                else:
                    auto_map[key] = val
            # Merge auto_map over existing mapping (without overwriting existing entries)
            merged_map = dict(iac_to_param_idx)
            for k, v in auto_map.items():
                if k not in merged_map:
                    merged_map[k] = v
            # Recompute mapped_param_indices with merged mapping
            mapped = []
            for iac in pycharmm_iac:
                key = str(int(iac))
                mapped.append(merged_map.get(key, None))
            mapped_param_indices = np.array(mapped, dtype=object)
            lj_params_base["iac_to_param_idx_autogenerated"] = auto_map
            iac_to_param_idx = merged_map
            # Final missing check after auto-map
            n_missing = np.sum(mapped_param_indices == None)
            if n_missing > 0:
                print(f"  ⚠ After auto-mapping, still {n_missing} IAC codes unmapped; LJ scaling may be wrong for those atoms.")

        # Require same length
        if len(expected_cluster) != len(mapped_param_indices):
            print(f"  ⚠ Atom-type length mismatch: expected {len(expected_cluster)}, mapped {len(mapped_param_indices)}; skipping detailed check")
        else:
            mismatch_mask = (expected_cluster != mapped_param_indices)
            n_mismatch = np.sum(mismatch_mask | (mapped_param_indices == None))
            if n_mismatch == 0:
                print("  ✓ Atom type codes match saved LJ parameter atom codes for all atoms (via iac_to_param_idx)")
            else:
                bad_idx = np.where(mismatch_mask | (mapped_param_indices == None))[0]
                sample = bad_idx[:10]
                print(f"  ⚠ Atom-type code mismatch for {n_mismatch} atoms (showing first {len(sample)}):")
                for idx in sample:
                    print(f"     atom {idx}: expected {expected_cluster[idx]}, got {mapped_param_indices[idx]}")
                print("     Large type mismatches will lead to wrong LJ scaling and large forces.")
        # Store expanded at_codes for downstream use if needed
        lj_params_base["at_codes_expanded"] = expected_cluster
    except Exception as e:
        print(f"  ⚠ Atom-type validation failed: {e}")

# Build per-atom codes for calculator (prefer mapped indices, else expanded codes)
at_codes_for_calc = None
try:
    if mapped_param_indices is not None and len(mapped_param_indices) == n_atoms_cluster:
        mapped_numeric = np.array([-1 if v is None else int(v) for v in mapped_param_indices], dtype=int)
        at_codes_for_calc = mapped_numeric
    elif lj_params_base is not None and "at_codes_expanded" in lj_params_base:
        at_codes_for_calc = np.array(lj_params_base["at_codes_expanded"], dtype=int)
except Exception:
    pass

print(f"at_codes_for_calc: {at_codes_for_calc}")
print("--------------------------------")
print(f"n_atoms_cluster: {n_atoms_cluster}")
Z = [round(_/2) for _ in psf.get_amass()]
cluster_atoms.set_atomic_numbers(Z)
print(f"Z: {Z}")
print("--------------------------------")
print(f"psf.get_atype(): {psf.get_atype()}")
print("cluster_atoms.get_atomic_numbers(): {cluster_atoms.get_atomic_numbers()}")
print("cluster_atoms.get_atomic_positions(): {cluster_atoms.get_atomic_positions()}")
print("cluster_atoms.get_atomic_numbers(): {cluster_atoms.get_atomic_numbers()}")
print("cluster_atoms.get_atomic_positions(): {cluster_atoms.get_atomic_positions()}")


print("--------------------------------")
print(f"pycharmm_iac: {pycharmm_iac}")
print("--------------------------------")
print(f"mapped_param_indices: {mapped_param_indices}")
print("--------------------------------")
print(f"lj_params_base: {lj_params_base}")
print("--------------------------------")
print(f"at_codes_expanded: {lj_params_base['at_codes_expanded']}")
print("--------------------------------")
print(f"at_codes_for_calc: {at_codes_for_calc}")
print("--------------------------------")

# ========================================================================
# STEP 6: SETUP CALCULATOR FOR CLUSTER
# ========================================================================
print("\n" + "=" * 70)
print("STEP 6: Setting up calculator for cluster")
print("=" * 70)

# Note: For a cluster, we need to expand the LJ scaling parameters
# Since we optimized for 2 molecules, we need to expand for all pairs in 10 molecules
# For simplicity, we'll use the same scaling factors for all pairs
from mmml.utils.hybrid_optimization import expand_scaling_parameters_to_full_set, extract_lj_parameters_from_calculator


# Get full expanded scaling parameters
# The saved file already contains full_ep_scale and full_sig_scale which are expanded
# for all atom types. These work for any number of molecules since they're per atom type.
print("  Loading expanded scaling parameters...")
print(f"  Checking saved params keys: {list(saved_params.keys())}")

if "full_ep_scale" in saved_params and "full_sig_scale" in saved_params:
    print("  Using pre-expanded scaling parameters from saved file...")
    full_ep_scale_cluster = np.array(saved_params["full_ep_scale"])
    full_sig_scale_cluster = np.array(saved_params["full_sig_scale"])
    print(f"  ✓ Loaded expanded scaling parameters (shape: {full_ep_scale_cluster.shape})")
else:
    # If full sets are not saved, use the fitted scales per atom type.
    # IMPORTANT: ep_scale/sig_scale are per atom type (same length as atc_epsilons), not per-atom.
    # Ensure we have LJ params to determine the number of atom types.
    if lj_params_base is None or "atc_epsilons" not in lj_params_base:
        raise ValueError("LJ parameters missing; cannot align ep_scale/sig_scale to atom types.")
    n_types = len(lj_params_base["atc_epsilons"])
    print("  Pre-expanded parameters not found, aligning fitted scales to atom types...")
    if opt_ep_scale.shape[0] < n_types:
        raise ValueError(f"ep_scale length {opt_ep_scale.shape[0]} < required atom types {n_types}")
    if opt_sig_scale.shape[0] < n_types:
        raise ValueError(f"sig_scale length {opt_sig_scale.shape[0]} < required atom types {n_types}")
    full_ep_scale_cluster = np.array(opt_ep_scale[:n_types])
    full_sig_scale_cluster = np.array(opt_sig_scale[:n_types])
    print(f"  ✓ Using fitted ep/sig scales for {n_types} atom types")

# --------------------------------------------------------------------
# Override cutoffs if saved values are unphysical
# --------------------------------------------------------------------
saved_ml_cut = float(saved_params.get("ml_cutoff", cutoff_params.ml_cutoff))
saved_mm_on = float(saved_params.get("mm_switch_on", cutoff_params.mm_switch_on))
saved_mm_cut = float(saved_params.get("mm_cutoff", cutoff_params.mm_cutoff))


# Create calculator factory for cluster
print("  Creating calculator factory...")
calculator_factory_cluster = setup_calculator(
    ATOMS_PER_MONOMER=n_atoms_monomer,
    N_MONOMERS=n_cluster_molecules,
    ml_cutoff_distance=cutoff_params.ml_cutoff,
    mm_switch_on=cutoff_params.mm_switch_on,
    mm_cutoff=cutoff_params.mm_cutoff,
    doML=True,
    doMM=True,
    doML_dimer=True,
    debug=True,
    model_restart_path=base_ckpt_dir,
    MAX_ATOMS_PER_SYSTEM=n_atoms_monomer * 2,
    cell=None,  # No PBC
    ep_scale=np.array(full_ep_scale_cluster),
    sig_scale=np.array(full_sig_scale_cluster),
    at_codes_override=at_codes_for_calc,
)

print(f"✓ Calculator factory created for {n_cluster_molecules} molecules")

# ========================================================================
# STEP 7: INITIALIZE CALCULATOR AND TEST
# ========================================================================
print("\n" + "=" * 70)
print("STEP 7: Initializing calculator and testing")
print("=" * 70)

# Create a mock args object (minimal)
class MockArgs:
    def __init__(self):
        self.n_monomers = n_cluster_molecules
        self.n_atoms_monomer = n_atoms_monomer
        self.include_mm = True
        self.skip_ml_dimers = False
        self.debug = False

args_cluster = MockArgs()

# Initialize calculator
# Only try MM if PyCHARMM was successfully initialized
doMM = True  # Default to True, but will be disabled if PyCHARMM not initialized

try:
    doMM_for_calc = doMM and PYCHARMM_INITIALIZED
    if doMM and not PYCHARMM_INITIALIZED:
        print("  ⚠ PyCHARMM not properly initialized - creating calculator with MM disabled")
    
    # Ensure we're using the reordered positions (from Step 5)
    # The reordering should have updated cluster_positions and cluster_Z
    print(f"  Using cluster positions shape: {cluster_positions.shape}, Z shape: {cluster_Z.shape}")
    
    calc, _ = calculator_factory_cluster(
        atomic_numbers=cluster_Z,
        atomic_positions=cluster_positions,
        n_monomers=n_cluster_molecules,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=True,  # Only enable MM if PyCHARMM is ready
        doML_dimer=True,
        backprop=False,
        debug=False,

    )
    
    cluster_atoms.calc = calc
    
    # Test calculation
    print("  Testing energy/force calculation...")
    energy = cluster_atoms.get_potential_energy()
    forces = cluster_atoms.get_forces()
    
    print(f"✓ Calculator initialized and tested successfully")
    print(f"  Cluster energy: {energy:.6f} eV")
    print(f"  Forces shape: {forces.shape}")
    
    # Check for very large forces (indicating potential issues)
    max_force = np.abs(forces).max()
    mean_force = np.abs(forces).mean()
    print(f"  Max force magnitude: {max_force:.6f} eV/Å")
    print(f"  Mean force magnitude: {mean_force:.6f} eV/Å")
    
    # Check for problematic forces
    large_force_threshold = 100.0  # eV/Å
    large_forces = np.abs(forces) > large_force_threshold
    n_large = np.sum(large_forces)
    if n_large > 0:
        print(f"  ⚠ WARNING: {n_large} atoms have forces > {large_force_threshold} eV/Å")
        print(f"     This may indicate atom ordering issues or unphysical configurations")
        print(f"     Large force atoms: {np.where(large_forces)[0]}")
        
        # Check if these are in the first few atoms (often indicates ordering issues)
        if np.any(large_forces[:10]):
            print(f"     Some of the first 10 atoms have large forces - atom reordering may have failed")
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(forces)):
        nan_count = np.sum(~np.isfinite(forces))
        print(f"  ⚠ WARNING: {nan_count} NaN/Inf values found in forces!")
    
except Exception as e:
    print(f"  ⚠ Warning: Calculator initialization failed: {e}")
    print(f"  This might be due to:")
    print(f"    1. PyCHARMM not properly initialized (check Step 5 output)")
    print(f"    2. Scaling parameter array size mismatch")
    print(f"    3. Empty arrays in LJ parameter extraction")
    print(f"  ")
    print(f"  You can try:")
    print(f"    - Re-run the script to ensure PyCHARMM is set up correctly")
    print(f"    - Or create calculator with doMM=False if MM is not needed")
    
    # Try creating calculator with MM disabled as fallback
    print(f"  ")
    print(f"  Attempting fallback: Creating calculator with MM disabled...")
    try:
        calculator_factory_cluster_ml_only = setup_calculator(
            ATOMS_PER_MONOMER=n_atoms_monomer,
            N_MONOMERS=n_cluster_molecules,
            ml_cutoff_distance=cutoff_params.ml_cutoff,
            mm_switch_on=cutoff_params.mm_switch_on,
            mm_cutoff=cutoff_params.mm_cutoff,
            doML=True,
            doMM=False,  # Disable MM to avoid PyCHARMM issues
            doML_dimer=True,
            debug=False,
            model_restart_path=base_ckpt_dir,
            MAX_ATOMS_PER_SYSTEM=n_atoms_cluster*2,
            cell=None,
            at_codes_override=at_codes_for_calc,
        )
        
        calc_ml_only, _ = calculator_factory_cluster_ml_only(
            atomic_numbers=cluster_Z,
            atomic_positions=cluster_positions,
            n_monomers=n_cluster_molecules,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
            backprop=True,
            debug=False,
        )
        
        cluster_atoms.calc = calc_ml_only
        energy = cluster_atoms.get_potential_energy()
        forces = cluster_atoms.get_forces()
        
        print(f"  ✓ Fallback successful: Calculator created with ML only (MM disabled)")
        print(f"     Cluster energy: {energy:.6f} eV")
        print(f"     Forces shape: {forces.shape}")
        print(f"     Note: MM contributions are disabled")
        
    except Exception as e2:
        print(f"  ✗ Fallback also failed: {e2}")
        print(f"     Please check PyCHARMM setup and parameter files")

# Save cluster with calculator info
cluster_info = {
    "n_molecules": n_cluster_molecules,
    "n_atoms": n_atoms_cluster,
    "n_atoms_per_molecule": n_atoms_monomer,
    "cutoff_params": {
        "ml_cutoff": float(cutoff_params.ml_cutoff),
        "mm_switch_on": float(cutoff_params.mm_switch_on),
        "mm_cutoff": float(cutoff_params.mm_cutoff),
    },
    "cluster_file": str(cluster_output_file),
    "checkpoint_path": str(checkpoint_path),
}

cluster_info_file = Path("cluster_10molecules_info.json")
with open(cluster_info_file, 'w') as f:
    json.dump(cluster_info, f, indent=2)

print(f"✓ Saved cluster info to: {cluster_info_file}")

print("\n" + "=" * 70)
print("CLUSTER SETUP COMPLETE")
print("=" * 70)
print(f"\nCluster structure saved to: {cluster_output_file}")
print(f"Cluster info saved to: {cluster_info_file}")
print(f"\nYou can now use cluster_atoms for MD simulations:")
print(f"  from ase.md.verlet import VelocityVerlet")
print(f"  from ase import units")
print(f"  dyn = VelocityVerlet(cluster_atoms, timestep=0.5 * units.fs)")
print(f"  dyn.run(1000)")

################## VALIDATION

# assert that the pycharmm atom types and the at_codes_for_calc are the same
for i in range(len(cluster_atoms)):
    print(f"Atom {i}: {cluster_atoms.get_atomic_numbers()[i]} == {at_codes_for_calc[i]}")
    if cluster_atoms.get_atomic_numbers()[i] != at_codes_for_calc[i]:
        print(f"Atom {i} has atomic number {cluster_atoms.get_atomic_numbers()[i]} but at_codes_for_calc has {at_codes_for_calc[i]}")




#print forces per atom
print("Forces per atom:")
for i in range(len(cluster_atoms)):
    print(f"Atom {i}: {cluster_atoms.get_forces()[i]}")

# get the calculator and validate the indices, and components are all sensible
calc = cluster_atoms.calc
results = calc.results
for k in results:
    print(f"Result {k}: {results[k]}")

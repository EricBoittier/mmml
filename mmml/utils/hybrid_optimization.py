"""
Hybrid MM/ML Potential Optimization Utilities

This module provides JAX-native optimization functions for fitting hybrid potential parameters.
Supports three optimization modes:
1. "ml_only": Optimize ML model parameters only
2. "lj_only": Optimize LJ scaling parameters (ep_scale, sig_scale) only
3. "both": Optimize both ML and LJ parameters together
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from itertools import product


def extract_lj_parameters_from_calculator(
    ATOMS_PER_MONOMER: int,
    N_MONOMERS: int,
) -> Dict[str, np.ndarray]:
    """
    Extract base LJ parameters and indices from the calculator setup.
    
    This should be called once after calculator_factory is created to extract
    the base parameters that will be scaled during optimization.
    
    IMPORTANT: This function extracts parameters ONLY for atom types that are
    actually used in the system (based on at_codes from the PSF). This ensures
    we only optimize the parameters that are needed.
    
    Args:
        ATOMS_PER_MONOMER: Number of atoms per monomer
        N_MONOMERS: Number of monomers
    
    Returns:
        dict with keys:
            atc_epsilons: Base epsilon values for each atom type (only used types)
            atc_rmins: Base rmin values for each atom type (only used types)
            atc_qs: Charges for each atom type (only used types)
            at_codes: Atom type codes for each atom in the system (mapped to parameter indices)
            pair_idx_atom_atom: Pair indices for atom-atom interactions
            atc: List of atom type names actually used in the system
            iac_to_param_idx: Mapping from IAC codes to parameter array indices
    """
    import pycharmm.param as param
    from mmml.pycharmmInterface.import_pycharmm import psf, CGENFF_RTF, CGENFF_PRM, read, settings, reset_block
    
    # Get atom type codes from PSF (these are IAC codes, 1-indexed in PyCHARMM)
    try:
        iac_codes_raw = np.array(psf.get_iac())
    except Exception as e:
        raise RuntimeError(
            f"Failed to get IAC codes from PSF. Make sure PyCHARMM is initialized "
            f"and the PSF structure is built before calling this function. Error: {e}"
        )
    
    n_atoms_expected = N_MONOMERS * ATOMS_PER_MONOMER
    if len(iac_codes_raw) < n_atoms_expected:
        raise RuntimeError(
            f"PSF has {len(iac_codes_raw)} atoms, but expected {n_atoms_expected} "
            f"(N_MONOMERS={N_MONOMERS} * ATOMS_PER_MONOMER={ATOMS_PER_MONOMER}). "
            f"Make sure PyCHARMM is fully initialized with the correct number of atoms."
        )
    
    iac_codes_raw = iac_codes_raw[:n_atoms_expected]
    
    # Convert IAC codes to 0-indexed if needed
    if len(iac_codes_raw) > 0 and np.min(iac_codes_raw) > 0:
        iac_codes = iac_codes_raw - 1
    else:
        iac_codes = iac_codes_raw
    
    # Get all atom types from parameter file
    try:
        atc_all = param.get_atc()
    except Exception as e:
        raise RuntimeError(
            f"Failed to get atom types from PyCHARMM. Make sure parameters are loaded. Error: {e}"
        )
    
    if len(atc_all) == 0:
        raise RuntimeError(
            "No atom types found in PyCHARMM parameter file. Make sure CGENFF parameters are loaded."
        )
    
    # Load CGENFF parameters (this should match what's in setup_calculator)
    reset_block()
    read.rtf(CGENFF_RTF)
    bl = settings.set_bomb_level(-2)
    wl = settings.set_warn_level(-2)
    read.prm(CGENFF_PRM)
    settings.set_bomb_level(bl)
    settings.set_warn_level(wl)
    
    # Extract parameters from parameter file
    cgenff_rtf = open(CGENFF_RTF).readlines()
    atc_all = param.get_atc()  # Refresh after reading
    cgenff_params_dict_q = {}
    for _ in cgenff_rtf:
        if _.startswith("ATOM"):
            _, atomname, at, q = _.split()[:4]
            try:
                cgenff_params_dict_q[at] = float(q)
            except:
                cgenff_params_dict_q[at] = float(q.split("!")[0])
    
    cgenff_params_dict = {}
    for p in open(CGENFF_PRM).readlines():
        if len(p) > 5 and len(p.split()) > 4 and p.split()[1] == "0.0" and p[0] != "!":
            res, _, ep, sig = p.split()[:4]
            cgenff_params_dict[res] = (float(ep), float(sig))
    
    # Find which atom types are actually used in the system
    # IAC codes index into atc_all, so we need to find unique atom types used
    unique_iac_codes = np.unique(iac_codes)
    unique_iac_codes = unique_iac_codes[unique_iac_codes >= 0]  # Remove negative indices
    unique_iac_codes = unique_iac_codes[unique_iac_codes < len(atc_all)]  # Remove out-of-bounds
    
    if len(unique_iac_codes) == 0:
        raise RuntimeError(
            f"No valid atom types found! IAC codes: {iac_codes}, "
            f"unique (after filtering): {np.unique(iac_codes)}, "
            f"atc_all length: {len(atc_all)}. "
            f"This usually means:\n"
            f"1. PyCHARMM PSF is not properly initialized\n"
            f"2. IAC codes are out of bounds (max IAC: {np.max(iac_codes) if len(iac_codes) > 0 else 'N/A'}, "
            f"atc_all length: {len(atc_all)})\n"
            f"3. Atom reordering hasn't been done yet\n"
            f"Make sure to call this function AFTER PyCHARMM initialization and atom reordering."
        )
    
    # Get the actual atom type names used in the system
    atc_used = [atc_all[int(iac)] for iac in unique_iac_codes]
    
    # Create mapping from IAC code to parameter index (for the reduced parameter set)
    iac_to_param_idx = {int(iac): idx for idx, iac in enumerate(unique_iac_codes)}
    
    # Map at_codes to parameter indices (for the reduced set)
    at_codes = np.array([iac_to_param_idx.get(int(iac), 0) for iac in iac_codes])
    
    # Extract base parameters ONLY for used atom types
    atc_epsilons = np.array([cgenff_params_dict.get(_, (0.0, 0.0))[0] for _ in atc_used])
    atc_rmins = np.array([cgenff_params_dict.get(_, (0.0, 0.0))[1] for _ in atc_used])
    atc_qs = np.array([cgenff_params_dict_q.get(_, 0.0) for _ in atc_used])
    
    # Compute pair indices (matching the calculator setup)
    from mmml.pycharmmInterface.mmml_calculator import dimer_permutations
    pair_idxs_product = np.array([(a,b) for a,b in list(product(np.arange(ATOMS_PER_MONOMER), repeat=2))])
    dimer_perms = np.array(dimer_permutations(N_MONOMERS))
    pair_idxs_np = dimer_perms * ATOMS_PER_MONOMER
    pair_idx_atom_atom = pair_idxs_np[:, None, :] + pair_idxs_product[None,...]
    pair_idx_atom_atom = pair_idx_atom_atom.reshape(-1, 2)
    
    return {
        "atc_epsilons": atc_epsilons,
        "atc_rmins": atc_rmins,
        "atc_qs": atc_qs,
        "at_codes": at_codes,  # Now mapped to parameter indices
        "pair_idx_atom_atom": pair_idx_atom_atom,
        "atc": atc_used,  # List of atom type names actually used
        "iac_to_param_idx": iac_to_param_idx,  # Mapping for reference
        "unique_iac_codes": unique_iac_codes,  # Original IAC codes (0-indexed) for used types
    }


def expand_scaling_parameters_to_full_set(
    optimized_ep_scale: np.ndarray,
    optimized_sig_scale: np.ndarray,
    lj_params: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand optimized scaling parameters from reduced set to full parameter set.
    
    The optimization only optimizes parameters for atom types actually used in the system.
    However, the calculator expects scaling parameters for ALL atom types in the parameter file.
    This function expands the optimized parameters to the full set.
    
    Args:
        optimized_ep_scale: Optimized epsilon scaling factors (shape: n_used_types)
        optimized_sig_scale: Optimized sigma scaling factors (shape: n_used_types)
        lj_params: Dictionary from extract_lj_parameters_from_calculator, must contain:
            - "unique_iac_codes": IAC codes (0-indexed) for used atom types
            - "iac_to_param_idx": Mapping from IAC codes to parameter indices
    
    Returns:
        full_ep_scale: Expanded epsilon scaling factors (shape: n_all_types)
        full_sig_scale: Expanded sigma scaling factors (shape: n_all_types)
    """
    import pycharmm.param as param
    
    # Get all atom types from parameter file
    try:
        atc_all = param.get_atc()
    except Exception as e:
        raise RuntimeError(
            f"Failed to get atom types from PyCHARMM. Make sure parameters are loaded. Error: {e}"
        )
    
    n_all_types = len(atc_all)
    
    # Initialize full arrays with ones (default scaling)
    full_ep_scale = np.ones(n_all_types)
    full_sig_scale = np.ones(n_all_types)
    
    # Get IAC codes for used types
    if "unique_iac_codes" not in lj_params:
        raise ValueError(
            "lj_params must contain 'unique_iac_codes'. "
            "Make sure you're using the latest version of extract_lj_parameters_from_calculator."
        )
    
    unique_iac_codes = lj_params["unique_iac_codes"]
    
    # Map optimized values to full parameter set
    # unique_iac_codes are 0-indexed IAC codes that index into atc_all
    for i, iac_code in enumerate(unique_iac_codes):
        iac_int = int(iac_code)
        if 0 <= iac_int < n_all_types:
            full_ep_scale[iac_int] = float(optimized_ep_scale[i])
            full_sig_scale[iac_int] = float(optimized_sig_scale[i])
    
    return full_ep_scale, full_sig_scale


def create_hybrid_fitting_factory(
    base_calculator_factory,
    model,
    model_params,
    atc_epsilons: np.ndarray,
    atc_rmins: np.ndarray,
    atc_qs: np.ndarray,
    at_codes: np.ndarray,
    pair_idx_atom_atom: np.ndarray,
    cutoff_params,
    optimize_mode: str = "lj_only",
    args=None,
):
    """
    Create a factory function that computes hybrid energy/forces with differentiable parameters.
    
    This factory supports four modes:
    - "ml_only": Only ML parameters are optimized (model_params)
    - "lj_only": Only LJ scaling parameters are optimized (ep_scale, sig_scale)
    - "cutoff_only": Only cutoff parameters are optimized (ml_cutoff, mm_switch_on, mm_cutoff)
    - "both": Both ML and LJ parameters are optimized together
    
    Args:
        base_calculator_factory: The base calculator factory (from setup_calculator)
        model: ML model instance
        model_params: ML model parameters (JAX PyTree)
        atc_epsilons: Base epsilon values for each atom type (numpy array)
        atc_rmins: Base rmin values for each atom type (numpy array)
        atc_qs: Charges for each atom type (numpy array)
        at_codes: Atom type codes for each atom in the system
        pair_idx_atom_atom: Pair indices for atom-atom interactions
        cutoff_params: Cutoff parameters
        optimize_mode: "ml_only", "lj_only", "cutoff_only", or "both"
        args: Arguments object (needed for calculator factory calls)
    
    Returns:
        compute_energy_forces: Function that computes energy/forces with given parameters
    """
    def compute_energy_forces(R, Z, params_dict):
        """
        Compute hybrid MM/ML energy and forces with differentiable parameters.
        
        Args:
            R: Positions (n_atoms, 3)
            Z: Atomic numbers (n_atoms,)
            params_dict: Dictionary with keys:
                - "ml_params": ML model parameters (if optimize_mode includes "ml")
                - "ep_scale": Epsilon scaling factors (if optimize_mode includes "lj")
                - "sig_scale": Sigma scaling factors (if optimize_mode includes "lj")
                - "ml_cutoff": ML cutoff distance (if optimize_mode == "cutoff_only")
                - "mm_switch_on": MM switch-on distance (if optimize_mode == "cutoff_only")
                - "mm_cutoff": MM cutoff distance (if optimize_mode == "cutoff_only")
        
        Returns:
            E: Total energy (scalar)
            F: Forces (n_atoms, 3)
        """
        # Extract parameters based on mode
        if optimize_mode in ["ml_only", "both"]:
            current_ml_params = params_dict.get("ml_params", model_params)
        else:
            current_ml_params = model_params
        
        if optimize_mode in ["lj_only", "both"]:
            ep_scale = params_dict.get("ep_scale", jnp.ones(len(atc_epsilons)))
            sig_scale = params_dict.get("sig_scale", jnp.ones(len(atc_rmins)))
        else:
            # Use provided scaling if available (e.g., from previous optimization), otherwise default to 1.0
            ep_scale = params_dict.get("ep_scale", jnp.ones(len(atc_epsilons)))
            sig_scale = params_dict.get("sig_scale", jnp.ones(len(atc_rmins)))
        
        # Extract cutoff parameters
        if optimize_mode == "cutoff_only":
            ml_cutoff_val = params_dict.get("ml_cutoff", cutoff_params.ml_cutoff if cutoff_params else 2.0)
            mm_switch_on_val = params_dict.get("mm_switch_on", cutoff_params.mm_switch_on if cutoff_params else 5.0)
            mm_cutoff_val = params_dict.get("mm_cutoff", cutoff_params.mm_cutoff if cutoff_params else 1.0)
        else:
            # Use fixed cutoff values
            ml_cutoff_val = cutoff_params.ml_cutoff if cutoff_params else 2.0
            mm_switch_on_val = cutoff_params.mm_switch_on if cutoff_params else 5.0
            mm_cutoff_val = cutoff_params.mm_cutoff if cutoff_params else 1.0
        
        # Skip MM computation if optimize_mode is "ml_only" (MM will be precomputed and added separately)
        if optimize_mode == "ml_only":
            # For ML-only, we only compute ML contributions (MM is precomputed)
            mm_energy = jnp.array(0.0)
            mm_forces = jnp.zeros_like(R)
        else:
            # Compute MM contributions with LJ parameters
            at_ep = -1 * jnp.abs(jnp.array(atc_epsilons)) * ep_scale
            at_rm = jnp.array(atc_rmins) * sig_scale
            at_q = jnp.array(atc_qs)
            
            # Validate at_codes indices are within bounds
            # at_codes should already be mapped to parameter indices (0-indexed)
            # Also ensure we don't have empty arrays
            if len(at_codes) == 0:
                # Return zero energy/forces if no atoms
                return jnp.array(0.0), jnp.zeros_like(R)
            
            # Ensure parameter arrays are not empty
            if len(at_ep) == 0 or len(at_rm) == 0 or len(at_q) == 0:
                return jnp.array(0.0), jnp.zeros_like(R)
            
            # Convert at_codes to JAX array and validate
            # at_codes should already be 0-indexed and mapped to parameter indices
            at_codes_arr = np.array(at_codes)
            
            # Clamp at_codes to valid range (0 to len-1)
            # This ensures we don't have out-of-bounds indices
            at_codes_safe = jnp.clip(jnp.array(at_codes_arr), 0, len(at_ep) - 1)
            
            rmins_per_system = jnp.take(at_rm, at_codes_safe)
            epsilons_per_system = jnp.take(at_ep, at_codes_safe)
            q_per_system = jnp.take(at_q, at_codes_safe)
            
            # Validate pair indices are within bounds
            n_atoms = len(R)
            if len(pair_idx_atom_atom) == 0:
                return jnp.array(0.0), jnp.zeros_like(R)
            
            # Clamp pair indices to valid range
            pair_idx_safe = jnp.clip(pair_idx_atom_atom, 0, n_atoms - 1)
            
            rm_a = jnp.take(rmins_per_system, pair_idx_safe[:, 0])
            rm_b = jnp.take(rmins_per_system, pair_idx_safe[:, 1])
            ep_a = jnp.take(epsilons_per_system, pair_idx_safe[:, 0])
            ep_b = jnp.take(epsilons_per_system, pair_idx_safe[:, 1])
            q_a = jnp.take(q_per_system, pair_idx_safe[:, 0])
            q_b = jnp.take(q_per_system, pair_idx_safe[:, 1])
            
            pair_rm = (rm_a + rm_b)
            pair_ep = (ep_a * ep_b) ** 0.5
            pair_qq = q_a * q_b
            
            displacements = R[pair_idx_safe[:, 0]] - R[pair_idx_safe[:, 1]]
            distances = jnp.linalg.norm(displacements, axis=1)
            
            def lennard_jones(r, sig, ep):
                r6 = (sig / r) ** 6
                return ep * (r6 ** 2 - 2 * r6)
            
            coulombs_constant = 3.32063711e2
            coulomb_epsilon = 1e-10
            def coulomb(r, qq, constant=coulombs_constant, eps=coulomb_epsilon):
                r_safe = jnp.maximum(r, eps)
                return -constant * qq / r_safe
            
            vdw_energies = lennard_jones(distances, pair_rm, pair_ep)
            coulomb_energies = coulomb(distances, pair_qq)
            mm_pair_energies = vdw_energies + coulomb_energies
            mm_energy = mm_pair_energies.sum()
            
            if hasattr(mm_energy, 'shape') and mm_energy.shape != ():
                mm_energy = jnp.sum(mm_energy) if mm_energy.size > 0 else jnp.array(0.0)
            
            def mm_energy_fn(R_pos):
                disp = R_pos[pair_idx_safe[:, 0]] - R_pos[pair_idx_safe[:, 1]]
                dist = jnp.linalg.norm(disp, axis=1)
                vdw = lennard_jones(dist, pair_rm, pair_ep)
                coul = coulomb(dist, pair_qq)
                energy_sum = (vdw + coul).sum()
                if hasattr(energy_sum, 'shape') and energy_sum.shape != ():
                    energy_sum = jnp.sum(energy_sum)
                return energy_sum
            
            mm_forces = -jax.grad(mm_energy_fn)(R)
        
        # Compute ML contributions using model
        # IMPORTANT: For ML optimization, we need to use model.apply directly with updated params
        # The calculator factory uses fixed params, so we bypass it when optimizing ML
        # For LJ-only optimization with precomputed ML, we skip ML computation entirely
        # For ML-only optimization, we compute ML with updated params (MM is precomputed)
        try:
            # Skip ML if optimize_mode is "lj_only" (ML will be precomputed and added separately)
            if optimize_mode == "lj_only":
                # For LJ-only, we only compute MM contributions (ML is precomputed)
                ml_energy = jnp.array(0.0)
                ml_forces = jnp.zeros_like(R)
            elif optimize_mode in ["ml_only", "both"] and model is not None and hasattr(model, 'apply'):
                # For ML optimization, use model.apply directly with updated parameters
                # We need to prepare batches in the same format as the calculator expects
                n_monomers_val = args.n_monomers if args and hasattr(args, 'n_monomers') else 2
                skip_ml_dimers = args.skip_ml_dimers if args and hasattr(args, 'skip_ml_dimers') else False
                
                # Prepare batches manually (matching calculator's format)
                from mmml.pycharmmInterface.mmml_calculator import (
                    prepare_batches_md,
                    dimer_permutations,
                    indices_of_monomer,
                    indices_of_pairs,
                )
                
                # Determine ATOMS_PER_MONOMER from the system
                ATOMS_PER_MONOMER_val = len(Z) // n_monomers_val if n_monomers_val > 0 else len(Z)
                
                # Prepare monomer and dimer indices (matching calculator logic)
                all_monomer_idxs = []
                for a in range(1, n_monomers_val + 1):
                    idxs = indices_of_monomer(a, n_atoms=ATOMS_PER_MONOMER_val, n_mol=n_monomers_val)
                    all_monomer_idxs.append(jnp.array(idxs))
                
                all_dimer_idxs = []
                if not skip_ml_dimers:
                    for a, b in dimer_permutations(n_monomers_val):
                        idxs = indices_of_pairs(a + 1, b + 1, n_atoms=ATOMS_PER_MONOMER_val, n_mol=n_monomers_val)
                        all_dimer_idxs.append(jnp.array(idxs))
                
                # Prepare batch data
                max_atoms = max(ATOMS_PER_MONOMER_val, 2 * ATOMS_PER_MONOMER_val)
                SPATIAL_DIMS = 3
                
                # Monomer positions and atomic numbers
                monomer_positions = jnp.zeros((n_monomers_val, max_atoms, SPATIAL_DIMS))
                for i, idxs in enumerate(all_monomer_idxs):
                    monomer_positions = monomer_positions.at[i, :ATOMS_PER_MONOMER_val].set(
                        R[idxs]
                    )
                monomer_atomic = jnp.zeros((n_monomers_val, max_atoms), dtype=jnp.int32)
                for i, idxs in enumerate(all_monomer_idxs):
                    monomer_atomic = monomer_atomic.at[i, :ATOMS_PER_MONOMER_val].set(
                        Z[idxs]
                    )
                
                # Dimer positions and atomic numbers
                n_dimers = len(all_dimer_idxs)
                dimer_positions = jnp.zeros((n_dimers, max_atoms, SPATIAL_DIMS))
                if n_dimers > 0:
                    for i, idxs in enumerate(all_dimer_idxs):
                        dimer_positions = dimer_positions.at[i, :2 * ATOMS_PER_MONOMER_val].set(
                            R[idxs]
                        )
                    dimer_atomic = jnp.zeros((n_dimers, max_atoms), dtype=jnp.int32)
                    for i, idxs in enumerate(all_dimer_idxs):
                        dimer_atomic = dimer_atomic.at[i, :2 * ATOMS_PER_MONOMER_val].set(
                            Z[idxs]
                        )
                else:
                    dimer_atomic = jnp.zeros((0, max_atoms), dtype=jnp.int32)
                
                # Combine monomer and dimer data
                batch_data = {
                    "R": jnp.concatenate([monomer_positions, dimer_positions]) if n_dimers > 0 else monomer_positions,
                    "Z": jnp.concatenate([monomer_atomic, dimer_atomic]) if n_dimers > 0 else monomer_atomic,
                    "N": jnp.concatenate([
                        jnp.full((n_monomers_val,), ATOMS_PER_MONOMER_val),
                        jnp.full((n_dimers,), 2 * ATOMS_PER_MONOMER_val)
                    ]) if n_dimers > 0 else jnp.full((n_monomers_val,), ATOMS_PER_MONOMER_val),
                }
                
                BATCH_SIZE = n_monomers_val + n_dimers
                batches = prepare_batches_md(batch_data, batch_size=BATCH_SIZE, num_atoms=max_atoms)[0]
                
                # Use model.apply directly with updated params
                ml_output = model.apply(
                    current_ml_params,
                    atomic_numbers=batches["Z"],
                    positions=batches["R"],
                    dst_idx=batches["dst_idx"],
                    src_idx=batches["src_idx"],
                    batch_segments=batches["batch_segments"],
                    batch_size=BATCH_SIZE,
                    batch_mask=batches["batch_mask"],
                    atom_mask=batches["atom_mask"]
                )
                
                # Extract energy and forces from model output
                ml_energy_raw = ml_output.get("energy", jnp.array(0.0))
                ml_forces_raw = ml_output.get("forces", jnp.zeros((len(R), 3)))
                
                # Handle energy (might be per-config or total)
                ml_energy = jnp.asarray(ml_energy_raw)
                if ml_energy.shape != ():
                    ml_energy = jnp.sum(ml_energy) if ml_energy.size > 0 else jnp.array(0.0)
                
                # Handle forces - map back to original atom order
                # Forces from model are for batched system (monomers + dimers)
                # We need to extract just the monomer forces and map them back
                ml_forces_batched = jnp.asarray(ml_forces_raw)
                
                # Extract forces for monomers only (first n_monomers * ATOMS_PER_MONOMER atoms)
                n_monomer_atoms = n_monomers_val * ATOMS_PER_MONOMER_val
                if ml_forces_batched.shape[0] >= n_monomer_atoms:
                    ml_forces = ml_forces_batched[:n_monomer_atoms]
                else:
                    # If forces are shorter, pad or use what we have
                    ml_forces = jnp.zeros((len(R), 3))
                    ml_forces = ml_forces.at[:ml_forces_batched.shape[0]].set(ml_forces_batched)
                    
            else:
                # For LJ-only optimization or when model is None, use calculator
                import ase
                n_monomers_val = args.n_monomers if args and hasattr(args, 'n_monomers') else 2
                calc, _ = base_calculator_factory(
                    atomic_numbers=Z,
                    atomic_positions=R,
                    n_monomers=n_monomers_val,
                    cutoff_params=cutoff_params,
                    doML=True,
                    doMM=False,
                    doML_dimer=not (args.skip_ml_dimers if args else False),
                    backprop=True,
                    debug=False,
                    energy_conversion_factor=1,
                    force_conversion_factor=1,
                )
                
                atoms = ase.Atoms(Z, R)
                atoms.calc = calc
                ml_energy_raw = atoms.get_potential_energy()
                ml_forces_raw = atoms.get_forces()
                
                ml_energy = jnp.asarray(ml_energy_raw)
                if ml_energy.shape != ():
                    ml_energy = jnp.sum(ml_energy) if ml_energy.size > 0 else jnp.array(0.0)
                
                ml_forces = jnp.asarray(ml_forces_raw)
        except Exception as e:
            # If ML computation fails, set to zero (allows MM-only fitting)
            ml_energy = jnp.array(0.0)
            ml_forces = jnp.zeros_like(R)
        
        # Apply cutoff-dependent switching if optimizing cutoffs
        if optimize_mode == "cutoff_only":
            # Import switching functions
            from mmml.pycharmmInterface.mmml_calculator import ml_switch_simple, mm_switch_simple
            
            # Ensure cutoff parameters are valid (positive, finite)
            ml_cutoff_val = jnp.maximum(ml_cutoff_val, 0.1)  # Minimum 0.1 Å
            mm_switch_on_val = jnp.maximum(mm_switch_on_val, ml_cutoff_val + 0.1)  # Must be > ml_cutoff
            mm_cutoff_val = jnp.maximum(mm_cutoff_val, 0.1)  # Minimum 0.1 Å
            
            # Ensure all are finite
            ml_cutoff_val = jnp.where(jnp.isfinite(ml_cutoff_val), ml_cutoff_val, 2.0)
            mm_switch_on_val = jnp.where(jnp.isfinite(mm_switch_on_val), mm_switch_on_val, 5.0)
            mm_cutoff_val = jnp.where(jnp.isfinite(mm_cutoff_val), mm_cutoff_val, 1.0)
            
            # Calculate pair distances for switching (apply switching per-pair, not globally)
            # This is more accurate and gives better gradients
            n_atoms = len(R)
            if len(pair_idx_atom_atom) > 0:
                pair_distances = jnp.linalg.norm(
                    R[pair_idx_atom_atom[:, 0]] - R[pair_idx_atom_atom[:, 1]], 
                    axis=1
                )
                
                # Apply switching per-pair and average the scales
                # This ensures gradients flow through all pairs, not just a single distance
                ml_scales_per_pair = ml_switch_simple(pair_distances, ml_cutoff_val, mm_switch_on_val)
                mm_scales_per_pair = mm_switch_simple(pair_distances, mm_switch_on_val, mm_cutoff_val)
                
                # Average scales across all pairs (weighted by distance to emphasize close pairs)
                # Use inverse distance weighting to emphasize short-range pairs where switching matters most
                weights = 1.0 / (pair_distances + 1e-6)  # Add small epsilon to avoid division by zero
                weights = weights / (jnp.sum(weights) + 1e-10)  # Normalize
                
                ml_scale = jnp.sum(ml_scales_per_pair * weights)
                mm_scale = jnp.sum(mm_scales_per_pair * weights)
                
                # Also compute a simple average as fallback
                ml_scale_avg = jnp.mean(ml_scales_per_pair)
                mm_scale_avg = jnp.mean(mm_scales_per_pair)
                
                # Use weighted average, but ensure we have gradients
                # Blend weighted and unweighted to ensure gradients flow
                ml_scale = 0.5 * ml_scale + 0.5 * ml_scale_avg
                mm_scale = 0.5 * mm_scale + 0.5 * mm_scale_avg
            else:
                # Fallback: use average distance from origin
                switching_distance = jnp.mean(jnp.linalg.norm(R, axis=1))
                switching_distance = jnp.maximum(switching_distance, 0.1)
                switching_distance = jnp.where(jnp.isfinite(switching_distance), switching_distance, 5.0)
                
                # Apply ML switching (1.0 at short range, tapers to 0.0 at mm_switch_on)
                ml_scale = ml_switch_simple(switching_distance, ml_cutoff_val, mm_switch_on_val)
                # Apply MM switching (0.0 at short range, ramps to 1.0 at mm_switch_on + mm_cutoff)
                mm_scale = mm_switch_simple(switching_distance, mm_switch_on_val, mm_cutoff_val)
            
            # Ensure scales are finite and in valid range [0, 1]
            ml_scale = jnp.clip(jnp.where(jnp.isfinite(ml_scale), ml_scale, 1.0), 0.0, 1.0)
            mm_scale = jnp.clip(jnp.where(jnp.isfinite(mm_scale), mm_scale, 0.0), 0.0, 1.0)
            
            # Scale energies and forces by switching functions
            ml_energy = ml_energy * ml_scale
            ml_forces = ml_forces * ml_scale
            mm_energy = mm_energy * mm_scale
            mm_forces = mm_forces * mm_scale
            
            # Ensure results are finite
            ml_energy = jnp.where(jnp.isfinite(ml_energy), ml_energy, 0.0)
            mm_energy = jnp.where(jnp.isfinite(mm_energy), mm_energy, 0.0)
            ml_forces = jnp.where(jnp.isfinite(ml_forces), ml_forces, 0.0)
            mm_forces = jnp.where(jnp.isfinite(mm_forces), mm_forces, 0.0)
        
        total_energy = ml_energy + mm_energy
        total_forces = ml_forces + mm_forces
        
        return total_energy, total_forces
    
    return compute_energy_forces


def fit_hybrid_potential_to_training_data_jax(
    train_batches: List[Dict],
    base_calculator_factory,
    model,
    model_params,
    atc_epsilons: np.ndarray,
    atc_rmins: np.ndarray,
    atc_qs: np.ndarray,
    at_codes: np.ndarray,
    pair_idx_atom_atom: np.ndarray,
    cutoff_params=None,
    args=None,
    optimize_mode: str = "lj_only",  # "ml_only", "lj_only", "cutoff_only", or "both"
    initial_ep_scale: Optional[np.ndarray] = None,
    initial_sig_scale: Optional[np.ndarray] = None,
    initial_ml_cutoff: Optional[float] = None,
    initial_mm_switch_on: Optional[float] = None,
    initial_mm_cutoff: Optional[float] = None,
    n_samples: Optional[int] = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    learning_rate: float = 0.01,
    n_iterations: int = 100,
    verbose: bool = True,
    n_monomers: Optional[int] = None,
    skip_ml_dimers: bool = False,
) -> Dict[str, Any]:
    """
    Fit hybrid potential parameters to training data using JAX optimization.
    
    Supports four optimization modes:
    1. "ml_only": Optimize ML model parameters only
    2. "lj_only": Optimize LJ scaling parameters (ep_scale, sig_scale) only
    3. "cutoff_only": Optimize cutoff parameters (ml_cutoff, mm_switch_on, mm_cutoff) only
    4. "both": Optimize both ML and LJ parameters together
    
    Args:
        train_batches: List of training batches (from prepare_batches_jit)
        base_calculator_factory: Base calculator factory (from setup_calculator)
        model: ML model instance
        model_params: Initial ML model parameters (JAX PyTree)
        atc_epsilons: Base epsilon values for each atom type (numpy array)
        atc_rmins: Base rmin values for each atom type (numpy array)
        atc_qs: Charges for each atom type (numpy array)
        at_codes: Atom type codes for each atom in the system (numpy array)
        pair_idx_atom_atom: Pair indices for atom-atom interactions (numpy array)
        cutoff_params: Cutoff parameters (optional, can be None)
        args: Arguments object (optional, for calculator factory calls)
        optimize_mode: "ml_only", "lj_only", "cutoff_only", or "both"
        n_monomers: Number of monomers (optional, extracted from args if provided)
        skip_ml_dimers: Whether to skip ML dimers (optional, extracted from args if provided)
        initial_ep_scale: Initial epsilon scaling factors (array, defaults to ones)
        initial_sig_scale: Initial sigma scaling factors (array, defaults to ones)
        initial_ml_cutoff: Initial ML cutoff distance (float, defaults to cutoff_params.ml_cutoff)
        initial_mm_switch_on: Initial MM switch-on distance (float, defaults to cutoff_params.mm_switch_on)
        initial_mm_cutoff: Initial MM cutoff distance (float, defaults to cutoff_params.mm_cutoff)
        n_samples: Number of training samples to use (if None, uses all)
        energy_weight: Weight for energy loss term
        force_weight: Weight for force loss term
        learning_rate: Learning rate for optimization
        n_iterations: Number of optimization iterations
        verbose: Print progress
    
    Returns:
        result_dict: Dictionary with optimized parameters:
            - "ml_params": Optimized ML parameters (if mode includes "ml")
            - "ep_scale": Optimized epsilon scaling factors (if mode includes "lj")
            - "sig_scale": Optimized sigma scaling factors (if mode includes "lj")
            - "ml_cutoff": Optimized ML cutoff (if mode == "cutoff_only")
            - "mm_switch_on": Optimized MM switch-on (if mode == "cutoff_only")
            - "mm_cutoff": Optimized MM cutoff (if mode == "cutoff_only")
            - "loss_history": History of loss values
    """
    if optimize_mode not in ["ml_only", "lj_only", "cutoff_only", "both"]:
        raise ValueError(f"optimize_mode must be 'ml_only', 'lj_only', 'cutoff_only', or 'both', got {optimize_mode}")
    
    # Convert inputs to JAX arrays
    atc_epsilons_jax = jnp.array(atc_epsilons)
    atc_rmins_jax = jnp.array(atc_rmins)
    atc_qs_jax = jnp.array(atc_qs)
    at_codes_jax = jnp.array(at_codes)
    pair_idx_atom_atom_jax = jnp.array(pair_idx_atom_atom)
    
    n_atom_types = len(atc_epsilons)
    
    # Initialize parameters based on mode
    params = {}
    
    if optimize_mode in ["ml_only", "both"]:
        params["ml_params"] = model_params
    
    if optimize_mode in ["lj_only", "both"]:
        if initial_ep_scale is None:
            initial_ep_scale = jnp.ones(n_atom_types)
        else:
            initial_ep_scale = jnp.array(initial_ep_scale)
        
        if initial_sig_scale is None:
            initial_sig_scale = jnp.ones(n_atom_types)
        else:
            initial_sig_scale = jnp.array(initial_sig_scale)
        
        params["ep_scale"] = initial_ep_scale
        params["sig_scale"] = initial_sig_scale
    
    if optimize_mode == "cutoff_only":
        if initial_ml_cutoff is None:
            initial_ml_cutoff = cutoff_params.ml_cutoff if cutoff_params else 2.0
        if initial_mm_switch_on is None:
            initial_mm_switch_on = cutoff_params.mm_switch_on if cutoff_params else 5.0
        if initial_mm_cutoff is None:
            initial_mm_cutoff = cutoff_params.mm_cutoff if cutoff_params else 1.0
        
        params["ml_cutoff"] = jnp.array(initial_ml_cutoff)
        params["mm_switch_on"] = jnp.array(initial_mm_switch_on)
        params["mm_cutoff"] = jnp.array(initial_mm_cutoff)
        
        # Store optimized LJ parameters if provided (from previous optimization)
        # These will be used in the loss function to get better starting loss
        optimized_ep_scale = None
        optimized_sig_scale = None
        if initial_ep_scale is not None:
            optimized_ep_scale = jnp.array(initial_ep_scale)
        if initial_sig_scale is not None:
            optimized_sig_scale = jnp.array(initial_sig_scale)
    
    # Extract n_monomers and skip_ml_dimers from args if provided, otherwise use defaults
    if args is not None:
        n_monomers_val = getattr(args, 'n_monomers', n_monomers) if n_monomers is None else n_monomers
        skip_ml_dimers_val = getattr(args, 'skip_ml_dimers', skip_ml_dimers) if hasattr(args, 'skip_ml_dimers') else skip_ml_dimers
    else:
        n_monomers_val = n_monomers if n_monomers is not None else 2
        skip_ml_dimers_val = skip_ml_dimers
    
    # Create a minimal args-like object if needed
    class MinimalArgs:
        def __init__(self, n_monomers, skip_ml_dimers):
            self.n_monomers = n_monomers
            self.skip_ml_dimers = skip_ml_dimers
    
    args_for_factory = args if args is not None else MinimalArgs(n_monomers_val, skip_ml_dimers_val)
    
    # Select training samples
    if n_samples is None:
        n_samples = len(train_batches)
    n_samples = min(n_samples, len(train_batches))
    selected_batches = train_batches[:n_samples]
    
    if verbose:
        print(f"Fitting hybrid potential using {n_samples} training samples")
        print(f"  Optimization mode: {optimize_mode}")
        print(f"  Number of atom types: {n_atom_types}")
        if optimize_mode in ["lj_only", "both"]:
            print(f"  Initial ep_scale: {initial_ep_scale}")
            print(f"  Initial sig_scale: {initial_sig_scale}")
        if optimize_mode == "cutoff_only":
            print(f"  Initial ml_cutoff: {initial_ml_cutoff}")
            print(f"  Initial mm_switch_on: {initial_mm_switch_on}")
            print(f"  Initial mm_cutoff: {initial_mm_cutoff}")
        print(f"  Energy weight: {energy_weight}, Force weight: {force_weight}")
        print(f"  Learning rate: {learning_rate}, Iterations: {n_iterations}")
    
    # Precompute contributions that don't change during optimization
    precomputed_ml = None
    precomputed_mm = None
    
    if optimize_mode == "lj_only":
        # Precompute ML contributions (they don't change for LJ-only optimization)
        if verbose:
            print("  Precomputing ML contributions (constant for LJ-only optimization)...")
        
        # Create a factory that only computes ML (no MM)
        compute_ml_only = create_hybrid_fitting_factory(
            base_calculator_factory,
            model,
            model_params,
            atc_epsilons_jax,
            atc_rmins_jax,
            atc_qs_jax,
            at_codes_jax,
            pair_idx_atom_atom_jax,
            cutoff_params,
            optimize_mode="ml_only",  # Only compute ML
            args=args_for_factory,
        )
        
        # Precompute ML energies and forces for all training samples
        # We'll compute them in the same structure as training_data will be created
        precomputed_ml = []
        
        # First, prepare training data structure to match indices
        temp_training_data = []
        for batch in selected_batches:
            R = batch["R"]
            Z = batch["Z"]
            
            if R.ndim == 3:
                # Multiple configurations in batch
                for config_idx in range(R.shape[0]):
                    temp_training_data.append({
                        "R": jnp.array(R[config_idx]),
                        "Z": jnp.array(Z[config_idx]),
                    })
            else:
                # Single configuration
                temp_training_data.append({
                    "R": jnp.array(R),
                    "Z": jnp.array(Z),
                })
        
        # Now precompute ML for each training data entry
        for data in temp_training_data:
            R_config = data["R"]
            Z_config = data["Z"]
            # Use dummy params (ML params don't matter since we're using fixed model_params)
            dummy_params = {"ml_params": model_params}
            ml_e, ml_f = compute_ml_only(R_config, Z_config, dummy_params)
            precomputed_ml.append({
                "energies": ml_e,
                "forces": ml_f,
            })
        
        if verbose:
            print(f"  ✓ ML contributions precomputed for {len(precomputed_ml)} configurations")
    
    elif optimize_mode == "ml_only":
        # Precompute MM contributions (they don't change for ML-only optimization)
        if verbose:
            print("  Precomputing MM contributions (constant for ML-only optimization)...")
        
        # Create a factory that only computes MM (no ML)
        compute_mm_only = create_hybrid_fitting_factory(
            base_calculator_factory,
            model,
            model_params,
            atc_epsilons_jax,
            atc_rmins_jax,
            atc_qs_jax,
            at_codes_jax,
            pair_idx_atom_atom_jax,
            cutoff_params,
            optimize_mode="lj_only",  # Only compute MM (LJ params fixed)
            args=args_for_factory,
        )
        
        # Precompute MM energies and forces for all training samples
        # We'll compute them in the same structure as training_data will be created
        precomputed_mm = []
        
        # First, prepare training data structure to match indices
        temp_training_data = []
        for batch in selected_batches:
            R = batch["R"]
            Z = batch["Z"]
            
            if R.ndim == 3:
                # Multiple configurations in batch
                for config_idx in range(R.shape[0]):
                    temp_training_data.append({
                        "R": jnp.array(R[config_idx]),
                        "Z": jnp.array(Z[config_idx]),
                    })
            else:
                # Single configuration
                temp_training_data.append({
                    "R": jnp.array(R),
                    "Z": jnp.array(Z),
                })
        
        # Now precompute MM for each training data entry
        for data in temp_training_data:
            R_config = data["R"]
            Z_config = data["Z"]
            # Use default LJ scaling (1.0) since we're not optimizing LJ
            dummy_params = {"ep_scale": jnp.ones(len(atc_epsilons_jax)), "sig_scale": jnp.ones(len(atc_rmins_jax))}
            mm_e, mm_f = compute_mm_only(R_config, Z_config, dummy_params)
            precomputed_mm.append({
                "energies": mm_e,
                "forces": mm_f,
            })
        
        if verbose:
            print(f"  ✓ MM contributions precomputed for {len(precomputed_mm)} configurations")
    
    # Create the differentiable factory (for MM contributions or full hybrid)
    compute_energy_forces = create_hybrid_fitting_factory(
        base_calculator_factory,
        model,
        model_params,
        atc_epsilons_jax,
        atc_rmins_jax,
        atc_qs_jax,
        at_codes_jax,
        pair_idx_atom_atom_jax,
        cutoff_params,
        optimize_mode=optimize_mode,
        args=args_for_factory,
    )
    
    # Prepare training data
    def ensure_scalar_energy(e):
        if e is None:
            return None
        e_arr = jnp.asarray(e)
        if e_arr.shape != ():
            if e_arr.size > 0:
                e_arr = e_arr.ravel()[0] if e_arr.size == 1 else jnp.sum(e_arr)
            else:
                e_arr = jnp.array(0.0)
        return e_arr
    
    training_data = []
    for batch in selected_batches:
        R = batch["R"]
        Z = batch["Z"]
        E_ref = batch.get("E", None)
        F_ref = batch.get("F", None)
        
        if R.ndim == 3:
            for config_idx in range(R.shape[0]):
                E_ref_config = None
                if E_ref is not None:
                    E_ref_raw = E_ref[config_idx] if hasattr(E_ref, '__getitem__') else E_ref
                    E_ref_config = ensure_scalar_energy(E_ref_raw)
                
                F_ref_config = None
                if F_ref is not None:
                    F_ref_raw = F_ref[config_idx] if F_ref.ndim > 1 else F_ref
                    F_ref_config = jnp.array(F_ref_raw)
                
                training_data.append({
                    "R": jnp.array(R[config_idx]),
                    "Z": jnp.array(Z[config_idx]),
                    "E_ref": E_ref_config,
                    "F_ref": F_ref_config,
                })
        else:
            E_ref_scalar = ensure_scalar_energy(E_ref)
            F_ref_arr = jnp.array(F_ref) if F_ref is not None else None
            
            training_data.append({
                "R": jnp.array(R),
                "Z": jnp.array(Z),
                "E_ref": E_ref_scalar,
                "F_ref": F_ref_arr,
            })
    
    # Define loss function based on mode
    if optimize_mode == "ml_only":
        # Create a factory that only computes ML (for speed, MM is precomputed)
        compute_ml_only = create_hybrid_fitting_factory(
            base_calculator_factory,
            model,
            model_params,
            atc_epsilons_jax,
            atc_rmins_jax,
            atc_qs_jax,
            at_codes_jax,
            pair_idx_atom_atom_jax,
            cutoff_params,
            optimize_mode="ml_only",  # Only compute ML
            args=args_for_factory,
        )
        
        def loss_fn(p):
            """Loss function for ML-only optimization (uses precomputed MM)."""
            total_energy_error = jnp.array(0.0)
            total_force_error = jnp.array(0.0)
            n_configs = 0
            
            for idx, data in enumerate(training_data):
                try:
                    # Compute ML contributions (fast, depends on ML params)
                    E_ml, F_ml = compute_ml_only(
                        data["R"],
                        data["Z"],
                        {"ml_params": p["ml_params"]}
                    )
                    
                    # Get precomputed MM contributions (structure matches training_data)
                    mm_data = precomputed_mm[idx]
                    E_mm = mm_data["energies"]
                    F_mm = mm_data["forces"]
                    
                    # Ensure E_ml and E_mm are scalars
                    E_ml = jnp.asarray(E_ml)
                    if E_ml.shape != ():
                        E_ml = jnp.sum(E_ml) if E_ml.size > 0 else jnp.array(0.0)
                    
                    if E_mm.shape != ():
                        E_mm = jnp.sum(E_mm) if E_mm.size > 0 else jnp.array(0.0)
                    
                    # Combine ML + MM
                    E_pred = E_ml + E_mm
                    F_pred = F_ml + F_mm
                    
                    if data["E_ref"] is not None:
                        E_ref = jnp.asarray(data["E_ref"])
                        if E_ref.shape != ():
                            E_ref = jnp.sum(E_ref) if E_ref.size > 0 else jnp.array(0.0)
                        energy_error = (E_pred - E_ref) ** 2
                        total_energy_error = total_energy_error + energy_error
                    
                    if data["F_ref"] is not None:
                        force_error = jnp.mean((F_pred - data["F_ref"]) ** 2)
                        total_force_error = total_force_error + force_error
                    
                    n_configs += 1
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Error in loss computation: {e}")
                    continue
            
            if n_configs == 0:
                return jnp.array(jnp.inf)
            
            avg_energy_error = total_energy_error / n_configs
            avg_force_error = total_force_error / n_configs
            loss = energy_weight * avg_energy_error + force_weight * avg_force_error
            
            if hasattr(loss, 'shape') and loss.shape != ():
                loss = jnp.sum(loss)
            
            return loss
    
    elif optimize_mode == "lj_only":
        # Create a factory that only computes MM (for speed, ML is precomputed)
        compute_mm_only = create_hybrid_fitting_factory(
            base_calculator_factory,
            model,
            model_params,
            atc_epsilons_jax,
            atc_rmins_jax,
            atc_qs_jax,
            at_codes_jax,
            pair_idx_atom_atom_jax,
            cutoff_params,
            optimize_mode="lj_only",  # Only compute MM
            args=args_for_factory,
        )
        
        def loss_fn(p):
            """Loss function for LJ-only optimization (uses precomputed ML)."""
            total_energy_error = jnp.array(0.0)
            total_force_error = jnp.array(0.0)
            n_configs = 0
            
            for idx, data in enumerate(training_data):
                try:
                    # Compute MM contributions (fast, depends on LJ params)
                    E_mm, F_mm = compute_mm_only(
                        data["R"],
                        data["Z"],
                        {"ep_scale": p["ep_scale"], "sig_scale": p["sig_scale"]}
                    )
                    
                    # Get precomputed ML contributions (structure matches training_data)
                    ml_data = precomputed_ml[idx]
                    E_ml = ml_data["energies"]
                    F_ml = ml_data["forces"]
                    
                    # Ensure E_ml is scalar
                    if E_ml.shape != ():
                        E_ml = jnp.sum(E_ml) if E_ml.size > 0 else jnp.array(0.0)
                    
                    # Combine ML + MM
                    E_pred = E_ml + E_mm
                    F_pred = F_ml + F_mm
                    
                    E_pred = jnp.asarray(E_pred)
                    if E_pred.shape != ():
                        E_pred = jnp.sum(E_pred) if E_pred.size > 0 else jnp.array(0.0)
                    
                    if data["E_ref"] is not None:
                        E_ref = jnp.asarray(data["E_ref"])
                        if E_ref.shape != ():
                            E_ref = jnp.sum(E_ref) if E_ref.size > 0 else jnp.array(0.0)
                        energy_error = (E_pred - E_ref) ** 2
                        total_energy_error = total_energy_error + energy_error
                    
                    if data["F_ref"] is not None:
                        force_error = jnp.mean((F_pred - data["F_ref"]) ** 2)
                        total_force_error = total_force_error + force_error
                    
                    n_configs += 1
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Error in loss computation: {e}")
                    continue
            
            if n_configs == 0:
                return jnp.array(jnp.inf)
            
            avg_energy_error = total_energy_error / n_configs
            avg_force_error = total_force_error / n_configs
            loss = energy_weight * avg_energy_error + force_weight * avg_force_error
            
            if hasattr(loss, 'shape') and loss.shape != ():
                loss = jnp.sum(loss)
            
            return loss
    
    elif optimize_mode == "cutoff_only":
        # For cutoff optimization, we compute both ML and MM, then apply switching
        # No precomputation needed since cutoffs affect both contributions
        def loss_fn(p):
            """Loss function for cutoff-only optimization."""
            # Validate and clip parameters before computing loss
            ml_cutoff = jnp.clip(
                jnp.where(jnp.isfinite(p["ml_cutoff"]), p["ml_cutoff"], 2.0),
                0.1, 10.0
            )
            mm_switch_on = jnp.clip(
                jnp.where(jnp.isfinite(p["mm_switch_on"]), p["mm_switch_on"], 5.0),
                0.1, 20.0
            )
            mm_cutoff = jnp.clip(
                jnp.where(jnp.isfinite(p["mm_cutoff"]), p["mm_cutoff"], 1.0),
                0.1, 10.0
            )
            # Ensure mm_switch_on > ml_cutoff
            mm_switch_on = jnp.maximum(mm_switch_on, ml_cutoff + 0.1)
            
            total_energy_error = jnp.array(0.0)
            total_force_error = jnp.array(0.0)
            n_configs = 0
            
            for data in training_data:
                try:
                    # Build params dict with cutoff parameters
                    params_dict_cutoff = {
                        "ml_cutoff": ml_cutoff,
                        "mm_switch_on": mm_switch_on,
                        "mm_cutoff": mm_cutoff
                    }
                    # Include optimized LJ parameters if provided (from previous optimization)
                    # This allows cutoff optimization to use optimized LJ parameters for better starting loss
                    if optimized_ep_scale is not None:
                        params_dict_cutoff["ep_scale"] = optimized_ep_scale
                    if optimized_sig_scale is not None:
                        params_dict_cutoff["sig_scale"] = optimized_sig_scale
                    
                    E_pred, F_pred = compute_energy_forces(
                        data["R"],
                        data["Z"],
                        params_dict_cutoff
                    )
                    
                    # Check for NaN/Inf in predictions
                    E_pred = jnp.asarray(E_pred)
                    if E_pred.shape != ():
                        E_pred = jnp.sum(E_pred) if E_pred.size > 0 else jnp.array(0.0)
                    
                    # Replace NaN/Inf with large penalty
                    E_pred = jnp.where(jnp.isfinite(E_pred), E_pred, 1e6)
                    F_pred = jnp.where(jnp.isfinite(F_pred), F_pred, 0.0)
                    
                    if data["E_ref"] is not None:
                        E_ref = jnp.asarray(data["E_ref"])
                        if E_ref.shape != ():
                            E_ref = jnp.sum(E_ref) if E_ref.size > 0 else jnp.array(0.0)
                        E_ref = jnp.where(jnp.isfinite(E_ref), E_ref, 0.0)
                        energy_error = (E_pred - E_ref) ** 2
                        energy_error = jnp.where(jnp.isfinite(energy_error), energy_error, 1e6)
                        total_energy_error = total_energy_error + energy_error
                    
                    if data["F_ref"] is not None:
                        F_ref = jnp.where(jnp.isfinite(data["F_ref"]), data["F_ref"], 0.0)
                        force_error = jnp.mean((F_pred - F_ref) ** 2)
                        force_error = jnp.where(jnp.isfinite(force_error), force_error, 1e6)
                        total_force_error = total_force_error + force_error
                    
                    n_configs += 1
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Error in loss computation: {e}")
                    continue
            
            if n_configs == 0:
                return jnp.array(1e6)
            
            avg_energy_error = total_energy_error / n_configs
            avg_force_error = total_force_error / n_configs
            loss = energy_weight * avg_energy_error + force_weight * avg_force_error
            
            # Ensure loss is finite
            loss = jnp.where(jnp.isfinite(loss), loss, jnp.array(1e6))
            
            if hasattr(loss, 'shape') and loss.shape != ():
                loss = jnp.sum(loss)
            
            return loss
    
    else:  # optimize_mode == "both"
        def loss_fn(p):
            """Loss function for combined ML+LJ optimization."""
            total_energy_error = jnp.array(0.0)
            total_force_error = jnp.array(0.0)
            n_configs = 0
            
            for data in training_data:
                try:
                    E_pred, F_pred = compute_energy_forces(
                        data["R"],
                        data["Z"],
                        {
                            "ml_params": p["ml_params"],
                            "ep_scale": p["ep_scale"],
                            "sig_scale": p["sig_scale"]
                        }
                    )
                    
                    E_pred = jnp.asarray(E_pred)
                    if E_pred.shape != ():
                        E_pred = jnp.sum(E_pred) if E_pred.size > 0 else jnp.array(0.0)
                    
                    if data["E_ref"] is not None:
                        E_ref = jnp.asarray(data["E_ref"])
                        if E_ref.shape != ():
                            E_ref = jnp.sum(E_ref) if E_ref.size > 0 else jnp.array(0.0)
                        energy_error = (E_pred - E_ref) ** 2
                        total_energy_error = total_energy_error + energy_error
                    
                    if data["F_ref"] is not None:
                        force_error = jnp.mean((F_pred - data["F_ref"]) ** 2)
                        total_force_error = total_force_error + force_error
                    
                    n_configs += 1
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Error in loss computation: {e}")
                    continue
            
            if n_configs == 0:
                return jnp.array(jnp.inf)
            
            avg_energy_error = total_energy_error / n_configs
            avg_force_error = total_force_error / n_configs
            loss = energy_weight * avg_energy_error + force_weight * avg_force_error
            
            if hasattr(loss, 'shape') and loss.shape != ():
                loss = jnp.sum(loss)
            
            return loss
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    
    # Optimization loop
    loss_history = []
    best_loss = jnp.inf
    best_params = params
    
    if verbose:
        print(f"\nStarting JAX optimization (mode: {optimize_mode})...")
    
    for iteration in range(n_iterations):
        def scalar_loss_fn(p):
            loss_val = loss_fn(p)
            if hasattr(loss_val, 'shape') and loss_val.shape != ():
                loss_val = jnp.sum(loss_val)
            return loss_val
        
        loss, grads = jax.value_and_grad(scalar_loss_fn)(params)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Clip parameters to reasonable bounds
        if optimize_mode in ["lj_only", "both"]:
            params["ep_scale"] = jnp.clip(params["ep_scale"], 0.1, 10.0)
            params["sig_scale"] = jnp.clip(params["sig_scale"], 0.1, 10.0)
        
        if optimize_mode == "cutoff_only":
            # Clip cutoffs to reasonable bounds (positive, with ordering constraints)
            # Also ensure they're finite
            params["ml_cutoff"] = jnp.clip(
                jnp.where(jnp.isfinite(params["ml_cutoff"]), params["ml_cutoff"], 2.0),
                0.1, 10.0
            )
            params["mm_switch_on"] = jnp.clip(
                jnp.where(jnp.isfinite(params["mm_switch_on"]), params["mm_switch_on"], 5.0),
                0.1, 20.0
            )
            params["mm_cutoff"] = jnp.clip(
                jnp.where(jnp.isfinite(params["mm_cutoff"]), params["mm_cutoff"], 1.0),
                0.1, 10.0
            )
            # Ensure mm_switch_on > ml_cutoff (for valid switching)
            params["mm_switch_on"] = jnp.maximum(params["mm_switch_on"], params["ml_cutoff"] + 0.1)
        
        if hasattr(loss, 'item'):
            loss_val = float(loss.item())
        elif hasattr(loss, '__array__'):
            loss_val = float(jnp.asarray(loss).item())
        else:
            loss_val = float(loss)
        
        loss_history.append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = jax.tree_util.tree_map(lambda x: x, params)  # Deep copy
        
        if verbose and (iteration % 10 == 0 or iteration == n_iterations - 1):
            print(f"  Iteration {iteration:4d}: Loss = {loss_val:.6f}")
            if optimize_mode in ["lj_only", "both"]:
                print(f"    ep_scale: {params['ep_scale']}")
                print(f"    sig_scale: {params['sig_scale']}")
            if optimize_mode == "cutoff_only":
                # Check gradients to diagnose if they're zero
                grad_ml_cutoff = grads.get("ml_cutoff", None)
                grad_mm_switch_on = grads.get("mm_switch_on", None)
                grad_mm_cutoff = grads.get("mm_cutoff", None)
                print(f"    ml_cutoff: {params['ml_cutoff']:.4f} (grad: {float(grad_ml_cutoff) if grad_ml_cutoff is not None else 'N/A':.6e})")
                print(f"    mm_switch_on: {params['mm_switch_on']:.4f} (grad: {float(grad_mm_switch_on) if grad_mm_switch_on is not None else 'N/A':.6e})")
                print(f"    mm_cutoff: {params['mm_cutoff']:.4f} (grad: {float(grad_mm_cutoff) if grad_mm_cutoff is not None else 'N/A':.6e})")
    
    if verbose:
        print(f"\n✓ Optimization complete!")
        print(f"  Final loss: {best_loss:.6f}")
        if optimize_mode in ["lj_only", "both"]:
            print(f"  Optimized ep_scale: {best_params['ep_scale']}")
            print(f"  Optimized sig_scale: {best_params['sig_scale']}")
        if optimize_mode == "cutoff_only":
            print(f"  Optimized ml_cutoff: {best_params['ml_cutoff']:.4f}")
            print(f"  Optimized mm_switch_on: {best_params['mm_switch_on']:.4f}")
            print(f"  Optimized mm_cutoff: {best_params['mm_cutoff']:.4f}")
    
    result_dict = {
        "loss_history": loss_history,
    }
    
    if optimize_mode in ["ml_only", "both"]:
        result_dict["ml_params"] = best_params["ml_params"]
    
    if optimize_mode in ["lj_only", "both"]:
        # Return optimized parameters (these are for the reduced set of used atom types)
        result_dict["ep_scale"] = best_params["ep_scale"]
        result_dict["sig_scale"] = best_params["sig_scale"]
        # Note: To use these with setup_calculator, you need to expand them to the full
        # parameter set using expand_scaling_parameters_to_full_set()
    
    if optimize_mode == "cutoff_only":
        result_dict["ml_cutoff"] = best_params["ml_cutoff"]
        result_dict["mm_switch_on"] = best_params["mm_switch_on"]
        result_dict["mm_cutoff"] = best_params["mm_cutoff"]
    
    return result_dict


def fit_lj_parameters_to_training_data_jax(
    train_batches: List[Dict],
    base_calculator_factory,
    atc_epsilons: np.ndarray,
    atc_rmins: np.ndarray,
    atc_qs: np.ndarray,
    at_codes: np.ndarray,
    pair_idx_atom_atom: np.ndarray,
    cutoff_params=None,
    args=None,
    initial_ep_scale: Optional[np.ndarray] = None,
    initial_sig_scale: Optional[np.ndarray] = None,
    n_samples: Optional[int] = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    learning_rate: float = 0.01,
    n_iterations: int = 100,
    verbose: bool = True,
    n_monomers: Optional[int] = None,
    skip_ml_dimers: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[float]]:
    """
    Fit LJ parameters (ep_scale, sig_scale) to training data using JAX optimization.
    
    This is a convenience wrapper for fit_hybrid_potential_to_training_data_jax with mode="lj_only".
    
    Returns:
        optimized_ep_scale: Optimized epsilon scaling factors (JAX array)
        optimized_sig_scale: Optimized sigma scaling factors (JAX array)
        loss_history: History of loss values during optimization
    """
    # Get model and params from calculator factory (needed for unified function)
    # For LJ-only mode, we can use dummy values
    dummy_model = None
    dummy_params = {}
    
    result = fit_hybrid_potential_to_training_data_jax(
        train_batches=train_batches,
        base_calculator_factory=base_calculator_factory,
        model=dummy_model,
        model_params=dummy_params,
        atc_epsilons=atc_epsilons,
        atc_rmins=atc_rmins,
        atc_qs=atc_qs,
        at_codes=at_codes,
        pair_idx_atom_atom=pair_idx_atom_atom,
        cutoff_params=cutoff_params,
        args=args,
        optimize_mode="lj_only",
        initial_ep_scale=initial_ep_scale,
        initial_sig_scale=initial_sig_scale,
        n_samples=n_samples,
        energy_weight=energy_weight,
        force_weight=force_weight,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        verbose=verbose,
        n_monomers=n_monomers,
        skip_ml_dimers=skip_ml_dimers,
    )
    
    return result["ep_scale"], result["sig_scale"], result["loss_history"]


def fit_hybrid_parameters_iteratively(
    train_batches: List[Dict],
    base_calculator_factory,
    model,
    model_params,
    atc_epsilons: np.ndarray,
    atc_rmins: np.ndarray,
    atc_qs: np.ndarray,
    at_codes: np.ndarray,
    pair_idx_atom_atom: np.ndarray,
    cutoff_params=None,
    args=None,
    initial_ep_scale: Optional[np.ndarray] = None,
    initial_sig_scale: Optional[np.ndarray] = None,
    initial_ml_cutoff: Optional[float] = None,
    initial_mm_switch_on: Optional[float] = None,
    initial_mm_cutoff: Optional[float] = None,
    n_iterations: int = 3,
    n_samples: Optional[int] = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    lj_learning_rate: float = 0.01,
    cutoff_learning_rate: float = 0.01,
    lj_n_iterations: int = 100,
    cutoff_n_iterations: int = 100,
    convergence_threshold: float = 1e-3,
    verbose: bool = True,
    n_monomers: Optional[int] = None,
    skip_ml_dimers: bool = False,
) -> Dict[str, Any]:
    """
    Iteratively optimize LJ and cutoff parameters by alternating between the two modes.
    
    This function alternates between:
    1. Optimizing LJ parameters (using current cutoff parameters)
    2. Optimizing cutoff parameters (using current LJ parameters)
    
    This iterative approach can lead to better overall optimization since the parameters
    are coupled - optimal cutoffs depend on LJ parameters and vice versa.
    
    Args:
        train_batches: List of training batches
        base_calculator_factory: Base calculator factory
        model: ML model instance
        model_params: ML model parameters
        atc_epsilons: Base epsilon values for each atom type
        atc_rmins: Base rmin values for each atom type
        atc_qs: Charges for each atom type
        at_codes: Atom type codes
        pair_idx_atom_atom: Pair indices for atom-atom interactions
        cutoff_params: Initial cutoff parameters
        args: Arguments object
        initial_ep_scale: Initial epsilon scaling factors (default: ones)
        initial_sig_scale: Initial sigma scaling factors (default: ones)
        initial_ml_cutoff: Initial ML cutoff (default: from cutoff_params)
        initial_mm_switch_on: Initial MM switch-on (default: from cutoff_params)
        initial_mm_cutoff: Initial MM cutoff (default: from cutoff_params)
        n_iterations: Number of alternating iterations (default: 3)
        n_samples: Number of training samples to use
        energy_weight: Weight for energy loss term
        force_weight: Weight for force loss term
        lj_learning_rate: Learning rate for LJ optimization
        cutoff_learning_rate: Learning rate for cutoff optimization
        lj_n_iterations: Number of iterations per LJ optimization step
        cutoff_n_iterations: Number of iterations per cutoff optimization step
        convergence_threshold: Relative loss improvement threshold for early stopping
        verbose: Print progress
        n_monomers: Number of monomers
        skip_ml_dimers: Whether to skip ML dimers
    
    Returns:
        Dictionary with:
            - "ep_scale": Final optimized epsilon scaling factors
            - "sig_scale": Final optimized sigma scaling factors
            - "ml_cutoff": Final optimized ML cutoff
            - "mm_switch_on": Final optimized MM switch-on
            - "mm_cutoff": Final optimized MM cutoff
            - "loss_history": List of loss values at each iteration
            - "lj_loss_history": List of LJ optimization loss histories
            - "cutoff_loss_history": List of cutoff optimization loss histories
    """
    from mmml.pycharmmInterface.mmml_calculator import CutoffParameters
    
    # Initialize parameters
    current_ep_scale = initial_ep_scale
    current_sig_scale = initial_sig_scale
    current_cutoff_params = cutoff_params
    
    if current_cutoff_params is None:
        current_cutoff_params = CutoffParameters(
            ml_cutoff=initial_ml_cutoff if initial_ml_cutoff is not None else 2.0,
            mm_switch_on=initial_mm_switch_on if initial_mm_switch_on is not None else 5.0,
            mm_cutoff=initial_mm_cutoff if initial_mm_cutoff is not None else 1.0,
        )
    
    # Track history
    loss_history = []
    lj_loss_history = []
    cutoff_loss_history = []
    previous_loss = None
    
    if verbose:
        print("=" * 80)
        print("ITERATIVE OPTIMIZATION: Alternating LJ and Cutoff Optimization")
        print("=" * 80)
        print(f"Number of iterations: {n_iterations}")
        print(f"Convergence threshold: {convergence_threshold}")
        print(f"Initial cutoff parameters: {current_cutoff_params}")
        print("=" * 80)
    
    for iteration in range(n_iterations):
        if verbose:
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'='*80}")
        
        # Step 1: Optimize LJ parameters using current cutoff parameters
        if verbose:
            print(f"\n[Iteration {iteration + 1}] Step 1: Optimizing LJ parameters...")
            print(f"  Using cutoff parameters: {current_cutoff_params}")
        
        result_lj = fit_hybrid_potential_to_training_data_jax(
            train_batches=train_batches,
            base_calculator_factory=base_calculator_factory,
            model=model,
            model_params=model_params,
            atc_epsilons=atc_epsilons,
            atc_rmins=atc_rmins,
            atc_qs=atc_qs,
            at_codes=at_codes,
            pair_idx_atom_atom=pair_idx_atom_atom,
            cutoff_params=current_cutoff_params,
            args=args,
            optimize_mode="lj_only",
            initial_ep_scale=current_ep_scale,
            initial_sig_scale=current_sig_scale,
            n_samples=n_samples,
            energy_weight=energy_weight,
            force_weight=force_weight,
            learning_rate=lj_learning_rate,
            n_iterations=lj_n_iterations,
            verbose=verbose,
            n_monomers=n_monomers,
            skip_ml_dimers=skip_ml_dimers,
        )
        
        current_ep_scale = result_lj["ep_scale"]
        current_sig_scale = result_lj["sig_scale"]
        lj_final_loss = result_lj["loss_history"][-1] if result_lj["loss_history"] else None
        lj_loss_history.append(result_lj["loss_history"])
        
        if verbose:
            print(f"\n[Iteration {iteration + 1}] LJ optimization complete")
            print(f"  Final loss: {lj_final_loss:.6f}")
            print(f"  ep_scale: {current_ep_scale}")
            print(f"  sig_scale: {current_sig_scale}")
        
        # Step 2: Optimize cutoff parameters using current LJ parameters
        if verbose:
            print(f"\n[Iteration {iteration + 1}] Step 2: Optimizing cutoff parameters...")
            print(f"  Using optimized LJ parameters from Step 1")
        
        result_cutoff = fit_hybrid_potential_to_training_data_jax(
            train_batches=train_batches,
            base_calculator_factory=base_calculator_factory,
            model=model,
            model_params=model_params,
            atc_epsilons=atc_epsilons,
            atc_rmins=atc_rmins,
            atc_qs=atc_qs,
            at_codes=at_codes,
            pair_idx_atom_atom=pair_idx_atom_atom,
            cutoff_params=current_cutoff_params,
            args=args,
            optimize_mode="cutoff_only",
            initial_ep_scale=current_ep_scale,
            initial_sig_scale=current_sig_scale,
            initial_ml_cutoff=current_cutoff_params.ml_cutoff,
            initial_mm_switch_on=current_cutoff_params.mm_switch_on,
            initial_mm_cutoff=current_cutoff_params.mm_cutoff,
            n_samples=n_samples,
            energy_weight=energy_weight,
            force_weight=force_weight,
            learning_rate=cutoff_learning_rate,
            n_iterations=cutoff_n_iterations,
            verbose=verbose,
            n_monomers=n_monomers,
            skip_ml_dimers=skip_ml_dimers,
        )
        
        current_cutoff_params = CutoffParameters(
            ml_cutoff=result_cutoff["ml_cutoff"],
            mm_switch_on=result_cutoff["mm_switch_on"],
            mm_cutoff=result_cutoff["mm_cutoff"],
        )
        cutoff_final_loss = result_cutoff["loss_history"][-1] if result_cutoff["loss_history"] else None
        cutoff_loss_history.append(result_cutoff["loss_history"])
        
        if verbose:
            print(f"\n[Iteration {iteration + 1}] Cutoff optimization complete")
            print(f"  Final loss: {cutoff_final_loss:.6f}")
            print(f"  Cutoff parameters: {current_cutoff_params}")
        
        # Track overall loss (use cutoff loss as it includes both LJ and cutoff effects)
        current_loss = cutoff_final_loss if cutoff_final_loss is not None else lj_final_loss
        if current_loss is not None:
            loss_history.append(current_loss)
            
            # Check convergence
            if previous_loss is not None:
                loss_improvement = abs(previous_loss - current_loss) / abs(previous_loss)
                if verbose:
                    print(f"\n[Iteration {iteration + 1}] Loss improvement: {loss_improvement:.6f}")
                
                if loss_improvement < convergence_threshold:
                    if verbose:
                        print(f"\n✓ Convergence reached! Loss improvement ({loss_improvement:.6f}) < threshold ({convergence_threshold})")
                    break
            
            previous_loss = current_loss
    
    if verbose:
        print(f"\n{'='*80}")
        print("ITERATIVE OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Final parameters:")
        print(f"  ep_scale: {current_ep_scale}")
        print(f"  sig_scale: {current_sig_scale}")
        print(f"  Cutoff parameters: {current_cutoff_params}")
        print(f"  Final loss: {loss_history[-1] if loss_history else 'N/A':.6f}")
        print(f"{'='*80}")
    
    return {
        "ep_scale": current_ep_scale,
        "sig_scale": current_sig_scale,
        "ml_cutoff": current_cutoff_params.ml_cutoff,
        "mm_switch_on": current_cutoff_params.mm_switch_on,
        "mm_cutoff": current_cutoff_params.mm_cutoff,
        "loss_history": loss_history,
        "lj_loss_history": lj_loss_history,
        "cutoff_loss_history": cutoff_loss_history,
    }
    train_batches: List[Dict],
    base_calculator_factory,
    model,
    model_params,
    atc_epsilons: np.ndarray,
    atc_rmins: np.ndarray,
    atc_qs: np.ndarray,
    at_codes: np.ndarray,
    pair_idx_atom_atom: np.ndarray,
    cutoff_params=None,
    args=None,
    initial_ep_scale: Optional[np.ndarray] = None,
    initial_sig_scale: Optional[np.ndarray] = None,
    initial_ml_cutoff: Optional[float] = None,
    initial_mm_switch_on: Optional[float] = None,
    initial_mm_cutoff: Optional[float] = None,
    n_iterations: int = 3,
    n_samples: Optional[int] = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    lj_learning_rate: float = 0.01,
    cutoff_learning_rate: float = 0.01,
    lj_n_iterations: int = 100,
    cutoff_n_iterations: int = 100,
    convergence_threshold: float = 1e-3,
    verbose: bool = True,
    n_monomers: Optional[int] = None,
    skip_ml_dimers: bool = False,
) -> Dict[str, Any]:
    """
    Iteratively optimize LJ and cutoff parameters by alternating between the two modes.
    
    This function alternates between:
    1. Optimizing LJ parameters (using current cutoff parameters)
    2. Optimizing cutoff parameters (using current LJ parameters)
    
    This iterative approach can lead to better overall optimization since the parameters
    are coupled - optimal cutoffs depend on LJ parameters and vice versa.
    
    Args:
        train_batches: List of training batches
        base_calculator_factory: Base calculator factory
        model: ML model instance
        model_params: ML model parameters
        atc_epsilons: Base epsilon values for each atom type
        atc_rmins: Base rmin values for each atom type
        atc_qs: Charges for each atom type
        at_codes: Atom type codes
        pair_idx_atom_atom: Pair indices for atom-atom interactions
        cutoff_params: Initial cutoff parameters
        args: Arguments object
        initial_ep_scale: Initial epsilon scaling factors (default: ones)
        initial_sig_scale: Initial sigma scaling factors (default: ones)
        initial_ml_cutoff: Initial ML cutoff (default: from cutoff_params)
        initial_mm_switch_on: Initial MM switch-on (default: from cutoff_params)
        initial_mm_cutoff: Initial MM cutoff (default: from cutoff_params)
        n_iterations: Number of alternating iterations (default: 3)
        n_samples: Number of training samples to use
        energy_weight: Weight for energy loss term
        force_weight: Weight for force loss term
        lj_learning_rate: Learning rate for LJ optimization
        cutoff_learning_rate: Learning rate for cutoff optimization
        lj_n_iterations: Number of iterations per LJ optimization step
        cutoff_n_iterations: Number of iterations per cutoff optimization step
        convergence_threshold: Relative loss improvement threshold for early stopping
        verbose: Print progress
        n_monomers: Number of monomers
        skip_ml_dimers: Whether to skip ML dimers
    
    Returns:
        Dictionary with:
            - "ep_scale": Final optimized epsilon scaling factors
            - "sig_scale": Final optimized sigma scaling factors
            - "ml_cutoff": Final optimized ML cutoff
            - "mm_switch_on": Final optimized MM switch-on
            - "mm_cutoff": Final optimized MM cutoff
            - "loss_history": List of loss values at each iteration
            - "lj_loss_history": List of LJ optimization loss histories
            - "cutoff_loss_history": List of cutoff optimization loss histories
    """
    # Initialize parameters
    current_ep_scale = initial_ep_scale
    current_sig_scale = initial_sig_scale
    current_cutoff_params = cutoff_params
    
    if current_cutoff_params is None:
        current_cutoff_params = CutoffParameters(
            ml_cutoff=initial_ml_cutoff if initial_ml_cutoff is not None else 2.0,
            mm_switch_on=initial_mm_switch_on if initial_mm_switch_on is not None else 5.0,
            mm_cutoff=initial_mm_cutoff if initial_mm_cutoff is not None else 1.0,
        )
    
    # Track history
    loss_history = []
    lj_loss_history = []
    cutoff_loss_history = []
    previous_loss = None
    
    if verbose:
        print("=" * 80)
        print("ITERATIVE OPTIMIZATION: Alternating LJ and Cutoff Optimization")
        print("=" * 80)
        print(f"Number of iterations: {n_iterations}")
        print(f"Convergence threshold: {convergence_threshold}")
        print(f"Initial cutoff parameters: {current_cutoff_params}")
        print("=" * 80)
    
    for iteration in range(n_iterations):
        if verbose:
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'='*80}")
        
        # Step 1: Optimize LJ parameters using current cutoff parameters
        if verbose:
            print(f"\n[Iteration {iteration + 1}] Step 1: Optimizing LJ parameters...")
            print(f"  Using cutoff parameters: {current_cutoff_params}")
        
        result_lj = fit_hybrid_potential_to_training_data_jax(
            train_batches=train_batches,
            base_calculator_factory=base_calculator_factory,
            model=model,
            model_params=model_params,
            atc_epsilons=atc_epsilons,
            atc_rmins=atc_rmins,
            atc_qs=atc_qs,
            at_codes=at_codes,
            pair_idx_atom_atom=pair_idx_atom_atom,
            cutoff_params=current_cutoff_params,
            args=args,
            optimize_mode="lj_only",
            initial_ep_scale=current_ep_scale,
            initial_sig_scale=current_sig_scale,
            n_samples=n_samples,
            energy_weight=energy_weight,
            force_weight=force_weight,
            learning_rate=lj_learning_rate,
            n_iterations=lj_n_iterations,
            verbose=verbose,
            n_monomers=n_monomers,
            skip_ml_dimers=skip_ml_dimers,
        )
        
        current_ep_scale = result_lj["ep_scale"]
        current_sig_scale = result_lj["sig_scale"]
        lj_final_loss = result_lj["loss_history"][-1] if result_lj["loss_history"] else None
        lj_loss_history.append(result_lj["loss_history"])
        
        if verbose:
            print(f"\n[Iteration {iteration + 1}] LJ optimization complete")
            print(f"  Final loss: {lj_final_loss:.6f}")
            print(f"  ep_scale: {current_ep_scale}")
            print(f"  sig_scale: {current_sig_scale}")
        
        # Step 2: Optimize cutoff parameters using current LJ parameters
        if verbose:
            print(f"\n[Iteration {iteration + 1}] Step 2: Optimizing cutoff parameters...")
            print(f"  Using optimized LJ parameters from Step 1")
        
        result_cutoff = fit_hybrid_potential_to_training_data_jax(
            train_batches=train_batches,
            base_calculator_factory=base_calculator_factory,
            model=model,
            model_params=model_params,
            atc_epsilons=atc_epsilons,
            atc_rmins=atc_rmins,
            atc_qs=atc_qs,
            at_codes=at_codes,
            pair_idx_atom_atom=pair_idx_atom_atom,
            cutoff_params=current_cutoff_params,
            args=args,
            optimize_mode="cutoff_only",
            initial_ep_scale=current_ep_scale,
            initial_sig_scale=current_sig_scale,
            initial_ml_cutoff=current_cutoff_params.ml_cutoff,
            initial_mm_switch_on=current_cutoff_params.mm_switch_on,
            initial_mm_cutoff=current_cutoff_params.mm_cutoff,
            n_samples=n_samples,
            energy_weight=energy_weight,
            force_weight=force_weight,
            learning_rate=cutoff_learning_rate,
            n_iterations=cutoff_n_iterations,
            verbose=verbose,
            n_monomers=n_monomers,
            skip_ml_dimers=skip_ml_dimers,
        )
        
        current_cutoff_params = CutoffParameters(
            ml_cutoff=result_cutoff["ml_cutoff"],
            mm_switch_on=result_cutoff["mm_switch_on"],
            mm_cutoff=result_cutoff["mm_cutoff"],
        )
        cutoff_final_loss = result_cutoff["loss_history"][-1] if result_cutoff["loss_history"] else None
        cutoff_loss_history.append(result_cutoff["loss_history"])
        
        if verbose:
            print(f"\n[Iteration {iteration + 1}] Cutoff optimization complete")
            print(f"  Final loss: {cutoff_final_loss:.6f}")
            print(f"  Cutoff parameters: {current_cutoff_params}")
        
        # Track overall loss (use cutoff loss as it includes both LJ and cutoff effects)
        current_loss = cutoff_final_loss if cutoff_final_loss is not None else lj_final_loss
        if current_loss is not None:
            loss_history.append(current_loss)
            
            # Check convergence
            if previous_loss is not None:
                loss_improvement = abs(previous_loss - current_loss) / abs(previous_loss)
                if verbose:
                    print(f"\n[Iteration {iteration + 1}] Loss improvement: {loss_improvement:.6f}")
                
                if loss_improvement < convergence_threshold:
                    if verbose:
                        print(f"\n✓ Convergence reached! Loss improvement ({loss_improvement:.6f}) < threshold ({convergence_threshold})")
                    break
            
            previous_loss = current_loss
    
    if verbose:
        print(f"\n{'='*80}")
        print("ITERATIVE OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Final parameters:")
        print(f"  ep_scale: {current_ep_scale}")
        print(f"  sig_scale: {current_sig_scale}")
        print(f"  Cutoff parameters: {current_cutoff_params}")
        print(f"  Final loss: {loss_history[-1] if loss_history else 'N/A':.6f}")
        print(f"{'='*80}")
    
    return {
        "ep_scale": current_ep_scale,
        "sig_scale": current_sig_scale,
        "ml_cutoff": current_cutoff_params.ml_cutoff,
        "mm_switch_on": current_cutoff_params.mm_switch_on,
        "mm_cutoff": current_cutoff_params.mm_cutoff,
        "loss_history": loss_history,
        "lj_loss_history": lj_loss_history,
        "cutoff_loss_history": cutoff_loss_history,
    }


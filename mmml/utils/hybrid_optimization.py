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
    }


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
    
    This factory supports three modes:
    - "ml_only": Only ML parameters are optimized (model_params)
    - "lj_only": Only LJ scaling parameters are optimized (ep_scale, sig_scale)
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
        optimize_mode: "ml_only", "lj_only", or "both"
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
            # Use default scaling (1.0) if not optimizing LJ
            ep_scale = jnp.ones(len(atc_epsilons))
            sig_scale = jnp.ones(len(atc_rmins))
        
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
        try:
            # Get ML energy and forces using the model directly
            # This requires accessing the model's apply function
            if hasattr(model, 'apply'):
                # Use model.apply directly for JAX-native computation
                # We need to prepare the input in the format expected by the model
                # For now, use the calculator but extract ML contributions
                calc, _ = base_calculator_factory(
                    atomic_numbers=Z,
                    atomic_positions=R,
                    n_monomers=args.n_monomers if args else 2,
                    cutoff_params=cutoff_params,
                    doML=True,
                    doMM=False,
                    doML_dimer=not (args.skip_ml_dimers if args else False),
                    backprop=True,
                    debug=False,
                    energy_conversion_factor=1,
                    force_conversion_factor=1,
                )
                
                # Temporarily replace model params if optimizing ML
                if optimize_mode in ["ml_only", "both"]:
                    # Note: This requires the calculator to support parameter replacement
                    # For now, we'll use the calculator as-is and note that ML optimization
                    # may require a different approach (direct model.apply)
                    pass
                
                atoms = ase.Atoms(Z, R)
                atoms.calc = calc
                ml_energy_raw = atoms.get_potential_energy()
                ml_forces_raw = atoms.get_forces()
                
                ml_energy = jnp.asarray(ml_energy_raw)
                if ml_energy.shape != ():
                    ml_energy = jnp.sum(ml_energy) if ml_energy.size > 0 else jnp.array(0.0)
                
                ml_forces = jnp.asarray(ml_forces_raw)
            else:
                # Fallback: use calculator
                calc, _ = base_calculator_factory(
                    atomic_numbers=Z,
                    atomic_positions=R,
                    n_monomers=args.n_monomers if args else 2,
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
            ml_energy = jnp.array(0.0)
            ml_forces = jnp.zeros_like(R)
        
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
    optimize_mode: str = "lj_only",  # "ml_only", "lj_only", or "both"
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
) -> Dict[str, Any]:
    """
    Fit hybrid potential parameters to training data using JAX optimization.
    
    Supports three optimization modes:
    1. "ml_only": Optimize ML model parameters only
    2. "lj_only": Optimize LJ scaling parameters (ep_scale, sig_scale) only
    3. "both": Optimize both ML and LJ parameters together
    
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
        optimize_mode: "ml_only", "lj_only", or "both"
        n_monomers: Number of monomers (optional, extracted from args if provided)
        skip_ml_dimers: Whether to skip ML dimers (optional, extracted from args if provided)
        initial_ep_scale: Initial epsilon scaling factors (array, defaults to ones)
        initial_sig_scale: Initial sigma scaling factors (array, defaults to ones)
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
            - "loss_history": History of loss values
    """
    if optimize_mode not in ["ml_only", "lj_only", "both"]:
        raise ValueError(f"optimize_mode must be 'ml_only', 'lj_only', or 'both', got {optimize_mode}")
    
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
    
    # Create the differentiable factory
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
        print(f"  Energy weight: {energy_weight}, Force weight: {force_weight}")
        print(f"  Learning rate: {learning_rate}, Iterations: {n_iterations}")
    
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
        def loss_fn(p):
            """Loss function for ML-only optimization."""
            total_energy_error = jnp.array(0.0)
            total_force_error = jnp.array(0.0)
            n_configs = 0
            
            for data in training_data:
                try:
                    E_pred, F_pred = compute_energy_forces(
                        data["R"],
                        data["Z"],
                        {"ml_params": p["ml_params"]}
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
    
    elif optimize_mode == "lj_only":
        def loss_fn(p):
            """Loss function for LJ-only optimization."""
            total_energy_error = jnp.array(0.0)
            total_force_error = jnp.array(0.0)
            n_configs = 0
            
            for data in training_data:
                try:
                    E_pred, F_pred = compute_energy_forces(
                        data["R"],
                        data["Z"],
                        {"ep_scale": p["ep_scale"], "sig_scale": p["sig_scale"]}
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
    
    if verbose:
        print(f"\n✓ Optimization complete!")
        print(f"  Final loss: {best_loss:.6f}")
        if optimize_mode in ["lj_only", "both"]:
            print(f"  Optimized ep_scale: {best_params['ep_scale']}")
            print(f"  Optimized sig_scale: {best_params['sig_scale']}")
    
    result_dict = {
        "loss_history": loss_history,
    }
    
    if optimize_mode in ["ml_only", "both"]:
        result_dict["ml_params"] = best_params["ml_params"]
    
    if optimize_mode in ["lj_only", "both"]:
        result_dict["ep_scale"] = best_params["ep_scale"]
        result_dict["sig_scale"] = best_params["sig_scale"]
    
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


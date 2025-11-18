"""
Simulation Initialization Utilities

This module provides functions for initializing simulations from data batches,
including atom reordering to match PyCHARMM's internal ordering.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import ase


def reorder_atoms_to_match_pycharmm(
    R: np.ndarray,
    Z: np.ndarray,
    pycharmm_atypes: np.ndarray,
    pycharmm_resids: np.ndarray,
    ATOMS_PER_MONOMER: int,
    N_MONOMERS: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reorder atoms from valid_data batch to match PyCHARMM's atom ordering.
    
    This function tries different atom orderings and selects the one that
    minimizes CHARMM internal energy (INTE term).
    
    Args:
        R: Positions from valid_data batch (n_atoms, 3)
        Z: Atomic numbers from valid_data batch (n_atoms,)
        pycharmm_atypes: Atom types from PyCHARMM PSF
        pycharmm_resids: Residue IDs from PyCHARMM PSF
        ATOMS_PER_MONOMER: Number of atoms per monomer
        N_MONOMERS: Number of monomers
    
    Returns:
        R_reordered: Reordered positions matching PyCHARMM ordering
        Z_reordered: Reordered atomic numbers matching PyCHARMM ordering
        reorder_indices: Indices used for reordering
    """
    from mmml.pycharmmInterface.import_pycharmm import coor
    import pycharmm.energy as energy
    
    n_atoms = len(R)
    
    print("  Reordering atoms to match PyCHARMM ordering...")
    print(f"  Original R shape: {R.shape}, Z shape: {Z.shape}")
    
    # Start with identity mapping
    base_indices = np.arange(n_atoms)
    
    # Generate candidate reorderings to try
    # Start with the identity (no reordering)
    candidate_orderings = [base_indices.copy()]
    
    # Add common swap patterns (based on user's example)
    # Swap indices 0 ↔ 3
    if n_atoms > 3:
        swap_1 = base_indices.copy()
        swap_1[0] = base_indices[3]
        swap_1[3] = base_indices[0]
        candidate_orderings.append(swap_1)
    
    # Swap indices 10 ↔ 13
    if n_atoms > 13:
        swap_2 = base_indices.copy()
        swap_2[10] = base_indices[13]
        swap_2[13] = base_indices[10]
        candidate_orderings.append(swap_2)
    
    # Combined swap: 0↔3 and 10↔13
    if n_atoms > 13:
        swap_combined = base_indices.copy()
        swap_combined[0] = base_indices[3]
        swap_combined[3] = base_indices[0]
        swap_combined[10] = base_indices[13]
        swap_combined[13] = base_indices[10]
        candidate_orderings.append(swap_combined)
    
    # Try additional swaps within each monomer if needed
    # For each monomer, try swapping first and last atoms
    for monomer_idx in range(N_MONOMERS):
        start_idx = monomer_idx * ATOMS_PER_MONOMER
        end_idx = (monomer_idx + 1) * ATOMS_PER_MONOMER
        if end_idx <= n_atoms:
            swap_monomer = base_indices.copy()
            if start_idx < n_atoms and end_idx - 1 < n_atoms:
                swap_monomer[start_idx] = base_indices[end_idx - 1]
                swap_monomer[end_idx - 1] = base_indices[start_idx]
                candidate_orderings.append(swap_monomer)
    
    print(f"  Trying {len(candidate_orderings)} different atom orderings...")
    
    # Evaluate each ordering by computing CHARMM internal energy
    best_energy = float('inf')
    best_indices = base_indices
    best_R = R
    best_Z = Z
    
    for i, reorder_indices in enumerate(candidate_orderings):
        try:
            # Apply reordering
            R_test = R[reorder_indices]
            Z_test = Z[reorder_indices]
            
            # Validate reordered arrays
            if R_test.shape != R.shape or Z_test.shape != Z.shape:
                print(f"    Ordering {i+1} failed: shape mismatch after reordering")
                continue
            
            # Check for NaN/Inf in positions
            if np.any(~np.isfinite(R_test)):
                print(f"    Ordering {i+1} failed: NaN/Inf in positions")
                continue
            
            # Set positions in PyCHARMM
            xyz = pd.DataFrame(R_test, columns=["x", "y", "z"])
            coor.set_positions(xyz)
            
            # Compute energy with error handling
            try:
                energy.get_energy()
                inte_energy = energy.get_term_by_name("INTE")
                
                # Check if energy is valid
                if not np.isfinite(inte_energy):
                    print(f"    Ordering {i+1} failed: invalid energy (NaN/Inf)")
                    continue
                
                print(f"    Ordering {i+1}/{len(candidate_orderings)}: INTE = {inte_energy:.6f} kcal/mol")
                
                # Keep track of best (lowest energy) ordering
                if inte_energy < best_energy:
                    best_energy = inte_energy
                    best_indices = reorder_indices
                    best_R = R_test
                    best_Z = Z_test
                    
            except Exception as e:
                print(f"    Ordering {i+1} energy calculation failed: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"    Ordering {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Validate that we found a valid ordering
    if best_energy == float('inf'):
        raise RuntimeError(
            "Failed to find valid atom ordering. All orderings produced invalid energies. "
            "This may indicate:\n"
            "1. PyCHARMM is not properly initialized\n"
            "2. Atom positions are invalid (NaN/Inf)\n"
            "3. Atom types/charges mismatch between batch and PyCHARMM\n"
            "4. PyCHARMM energy calculation is failing"
        )
    
    print(f"  Best ordering found: INTE = {best_energy:.6f} kcal/mol")
    print(f"  Reorder indices: {best_indices}")
    
    # Validate final arrays
    if np.any(~np.isfinite(best_R)):
        raise RuntimeError("Final reordered positions contain NaN/Inf values")
    
    # Set final positions in PyCHARMM
    try:
        xyz = pd.DataFrame(best_R, columns=["x", "y", "z"])
        coor.set_positions(xyz)
        print("  Final positions set in PyCHARMM")
    except Exception as e:
        print(f"  Warning: Could not set final positions in PyCHARMM: {e}")
        raise
    
    return best_R, best_Z, best_indices


def initialize_simulation_from_batch(
    batch: dict,
    calculator_factory,
    cutoff_params,
    args,
    pycharmm_atypes: Optional[np.ndarray] = None,
    pycharmm_resids: Optional[np.ndarray] = None,
    ATOMS_PER_MONOMER: Optional[int] = None,
    N_MONOMERS: Optional[int] = None,
) -> Tuple[ase.Atoms, any]:
    """
    Initialize a simulation from a valid_data batch.
    
    Args:
        batch: Dictionary containing 'R' and 'Z' keys (from valid_batches)
        calculator_factory: Calculator factory function (from setup_calculator)
        cutoff_params: Cutoff parameters
        args: Arguments object (for calculator factory calls)
        pycharmm_atypes: Atom types from PyCHARMM PSF (optional, for reordering)
        pycharmm_resids: Residue IDs from PyCHARMM PSF (optional, for reordering)
        ATOMS_PER_MONOMER: Number of atoms per monomer (optional, for reordering)
        N_MONOMERS: Number of monomers (optional, for reordering)
    
    Returns:
        atoms: ASE Atoms object initialized from the batch
        hybrid_calc: Hybrid calculator for the system
    """
    from mmml.pycharmmInterface.import_pycharmm import coor
    
    # Get positions and atomic numbers from batch
    R = batch["R"]
    Z = batch["Z"]
    
    # Extract the first configuration from the batch
    # Note: batches may contain multiple configurations
    if R.ndim == 3:
        # Batch shape: (batch_size, n_atoms, 3)
        R = R[0]
        Z = Z[0]
    elif R.ndim == 2:
        # Already flattened: (n_atoms, 3)
        pass
    else:
        raise ValueError(f"Unexpected R shape: {R.shape}")
    
    # Determine expected number of atoms
    if ATOMS_PER_MONOMER is not None and N_MONOMERS is not None:
        n_atoms_expected = ATOMS_PER_MONOMER * N_MONOMERS
    else:
        n_atoms_expected = len(R)
    
    # Ensure we have the right number of atoms
    if len(R) != n_atoms_expected:
        print(f"Warning: Expected {n_atoms_expected} atoms, got {len(R)}")
        R = R[:n_atoms_expected]
        Z = Z[:n_atoms_expected]
    
    print(f"Initializing simulation from batch")
    print(f"  Positions shape: {R.shape}")
    print(f"  Atomic numbers shape: {Z.shape}")
    print(f"  Number of atoms: {len(R)}")
    
    # Reorder atoms to match PyCHARMM ordering if PyCHARMM is initialized
    if (args.include_mm if hasattr(args, 'include_mm') else False) and \
       pycharmm_atypes is not None and \
       ATOMS_PER_MONOMER is not None and \
       N_MONOMERS is not None:
        R, Z, reorder_indices = reorder_atoms_to_match_pycharmm(
            R, Z, pycharmm_atypes, pycharmm_resids,
            ATOMS_PER_MONOMER, N_MONOMERS
        )
        print(f"  Atoms reordered to match PyCHARMM ordering")
    else:
        print(f"  No reordering applied (MM disabled or PyCHARMM not initialized)")
    
    # Create ASE Atoms object
    atoms = ase.Atoms(Z, R)
    
    # Sync positions with PyCHARMM if MM is enabled
    # This ensures PyCHARMM coordinates match the batch positions
    if hasattr(args, 'include_mm') and args.include_mm:
        try:
            xyz = pd.DataFrame(R, columns=["x", "y", "z"])
            coor.set_positions(xyz)
            print("  Synced positions with PyCHARMM")
        except Exception as e:
            print(f"  Warning: Could not sync positions with PyCHARMM: {e}")
    
    # Create hybrid calculator (following run_sim.py)
    # Note: MM contributions require PyCHARMM to be initialized first
    hybrid_calc, _ = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=args.n_monomers if hasattr(args, 'n_monomers') else N_MONOMERS,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=args.include_mm if hasattr(args, 'include_mm') else False,
        doML_dimer=not (args.skip_ml_dimers if hasattr(args, 'skip_ml_dimers') else False),
        backprop=True,
        debug=args.debug if hasattr(args, 'debug') else False,
        energy_conversion_factor=1,
        force_conversion_factor=1,
    )
    
    atoms.calc = hybrid_calc
    
    # Get initial energy and forces
    try:
        hybrid_energy = float(atoms.get_potential_energy())
        hybrid_forces = np.asarray(atoms.get_forces())
        print(f"Initial energy: {hybrid_energy:.6f} eV")
        print(f"Initial forces shape: {hybrid_forces.shape}")
        print(f"Max force: {np.abs(hybrid_forces).max():.6f} eV/Å")
    except Exception as e:
        print(f"Warning: Could not compute initial energy/forces: {e}")
        print("This may be due to PyCHARMM not being properly initialized")
        print("or atom ordering mismatch. Check the reordering function.")
        raise
    
    return atoms, hybrid_calc


def initialize_multiple_simulations(
    valid_batches: list,
    calculator_factory,
    cutoff_params,
    args,
    n_simulations: int = 5,
    pycharmm_atypes: Optional[np.ndarray] = None,
    pycharmm_resids: Optional[np.ndarray] = None,
    ATOMS_PER_MONOMER: Optional[int] = None,
    N_MONOMERS: Optional[int] = None,
) -> list:
    """
    Initialize multiple simulations from different valid_data batches.
    
    Args:
        valid_batches: List of batch dictionaries (from prepare_batches_jit)
        calculator_factory: Calculator factory function (from setup_calculator)
        cutoff_params: Cutoff parameters
        args: Arguments object (for calculator factory calls)
        n_simulations: Number of simulations to initialize (default: 5)
        pycharmm_atypes: Atom types from PyCHARMM PSF (optional, for reordering)
        pycharmm_resids: Residue IDs from PyCHARMM PSF (optional, for reordering)
        ATOMS_PER_MONOMER: Number of atoms per monomer (optional, for reordering)
        N_MONOMERS: Number of monomers (optional, for reordering)
    
    Returns:
        List of (atoms, hybrid_calc) tuples
    """
    simulations = []
    n_batches = len(valid_batches)
    
    for i in range(min(n_simulations, n_batches)):
        try:
            atoms, calc = initialize_simulation_from_batch(
                batch=valid_batches[i],
                calculator_factory=calculator_factory,
                cutoff_params=cutoff_params,
                args=args,
                pycharmm_atypes=pycharmm_atypes,
                pycharmm_resids=pycharmm_resids,
                ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
                N_MONOMERS=N_MONOMERS,
            )
            simulations.append((atoms, calc))
            print(f"Successfully initialized simulation {i+1}/{n_simulations}")
        except Exception as e:
            print(f"Warning: Failed to initialize simulation from batch {i}: {e}")
            continue
    
    return simulations


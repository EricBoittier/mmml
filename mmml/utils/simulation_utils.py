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
    
    # IMPORTANT: Set initial coordinates from batch data first
    # PyCHARMM may have undefined coordinates (9999.0) that need to be set before computing energies
    try:
        xyz_initial = pd.DataFrame(R, columns=["x", "y", "z"])
        coor.set_positions(xyz_initial)
        print(f"  Set initial coordinates from batch data")
    except Exception as e:
        print(f"  Warning: Could not set initial coordinates: {e}")
        # Continue anyway - will try to set for each ordering
    
    # Determine which energy terms are available and active
    # Try to use the most specific term available (IMPR > BOND > ANGLE > DIHE)
    # Note: INTE is not always available, so we don't use it as a fallback
    available_energy_terms = []
    primary_term = None
    
    # Verify coordinates were set successfully before trying to get energy terms
    coordinates_set = False
    try:
        # Check if coordinates are valid (not undefined 9999.0 values)
        current_positions = coor.get_positions()
        if current_positions is not None and len(current_positions) > 0:
            # Check if any coordinates are still undefined (9999.0)
            if not np.any(np.abs(current_positions.values.flatten()) > 1000):
                coordinates_set = True
    except Exception:
        pass
    
    if not coordinates_set:
        print(f"  Warning: Coordinates may still be undefined, will set for each ordering")
    
    try:
        # Try to get energy terms with error handling to prevent crashes
        # First ensure coordinates are set (energy.get_energy() may fail if coords are undefined)
        if coordinates_set:
            try:
                energy.get_energy()  # Update energy terms
            except Exception as e:
                print(f"  Warning: energy.get_energy() failed: {e}")
                raise
        else:
            # Skip getting energy terms now - will do it after setting coordinates for each ordering
            raise RuntimeError("Coordinates not set, will determine energy terms per ordering")
        
        try:
            term_names = energy.get_term_names()
            term_statuses = energy.get_term_statuses()
        except Exception as e:
            print(f"  Warning: Could not get term names/statuses: {e}")
            raise
        
        # Check for available terms in order of preference (IMPR first, then BOND, ANGLE, DIHE)
        preferred_terms = ["IMPR", "BOND", "ANGLE", "DIHE"]
        for term_name in preferred_terms:
            try:
                if term_name in term_names:
                    idx = term_names.index(term_name)
                    if term_statuses[idx]:  # Term is active
                        available_energy_terms.append(term_name)
            except Exception:
                continue
        
        if not available_energy_terms:
            # Fallback: use first active term (but not INTE, as it's not always available)
            for name, status in zip(term_names, term_statuses):
                if status and name != "INTE":  # Skip INTE
                    available_energy_terms.append(name)
                    break
        
        if available_energy_terms:
            primary_term = available_energy_terms[0]
            print(f"  Using energy term '{primary_term}' for reordering evaluation")
            if len(available_energy_terms) > 1:
                print(f"  (Also available: {', '.join(available_energy_terms[1:])})")
        else:
            raise RuntimeError("No active energy terms found in PyCHARMM (excluding INTE)")
            
    except Exception as e:
        print(f"  Warning: Could not determine available energy terms: {e}")
        print(f"  Will try to use IMPR, BOND, ANGLE, or DIHE during evaluation")
        # Don't set a default - let the evaluation loop try to find available terms
        available_energy_terms = ["IMPR", "BOND", "ANGLE", "DIHE"]
        primary_term = "IMPR"  # Default for display, but may not be used
    
    # Evaluate each ordering by computing CHARMM energy using available terms
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
            
            # Set positions in PyCHARMM with error handling
            # This is critical - PyCHARMM may have undefined coordinates (9999.0) that need to be set
            try:
                xyz = pd.DataFrame(R_test, columns=["x", "y", "z"])
                # Check if coordinates are valid (not undefined/placeholder values)
                if np.any(np.abs(R_test) > 1000):  # Likely undefined coordinates (9999.0)
                    print(f"    Ordering {i+1}: Setting coordinates from batch data (PyCHARMM had undefined coords)")
                coor.set_positions(xyz)
            except Exception as e:
                print(f"    Ordering {i+1} failed: Could not set positions in PyCHARMM: {e}")
                continue
            
            # Compute energy with error handling to prevent crashes
            try:
                # Try to compute energy - wrap in try-except to catch Fortran crashes
                try:
                    energy.get_energy()
                except Exception as e:
                    print(f"    Ordering {i+1} failed: energy.get_energy() crashed: {e}")
                    continue
                
                # Try to get energy from available terms, with fallbacks
                ordering_energy = None
                energy_str = ""
                
                # Try preferred terms first (IMPR, BOND, ANGLE, DIHE)
                for term_name in available_energy_terms:
                    try:
                        term_value = energy.get_term_by_name(term_name)
                        if np.isfinite(term_value):
                            if ordering_energy is None:
                                ordering_energy = term_value
                            else:
                                # Sum multiple terms if available
                                ordering_energy += term_value
                            energy_str += f"{term_name}={term_value:.6f} "
                    except Exception:
                        # Term not available, skip it
                        continue
                
                # If we still don't have energy, try to find any active term
                if ordering_energy is None:
                    try:
                        term_names = energy.get_term_names()
                        term_statuses = energy.get_term_statuses()
                        for name, status in zip(term_names, term_statuses):
                            if status and name != "INTE":  # Skip INTE
                                try:
                                    term_value = energy.get_term_by_name(name)
                                    if np.isfinite(term_value):
                                        ordering_energy = term_value
                                        energy_str = f"{name}={term_value:.6f}"
                                        break
                                except Exception:
                                    continue
                    except Exception:
                        pass
                
                # Check if energy is valid
                if ordering_energy is None or not np.isfinite(ordering_energy):
                    print(f"    Ordering {i+1} failed: invalid energy (NaN/Inf or not available)")
                    continue
                
                print(f"    Ordering {i+1}/{len(candidate_orderings)}: {energy_str.strip()} kcal/mol")
                
                # Keep track of best (lowest energy) ordering
                if ordering_energy < best_energy:
                    best_energy = ordering_energy
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
    
    if primary_term:
        print(f"  Best ordering found: Energy = {best_energy:.6f} kcal/mol (using {primary_term})")
    else:
        print(f"  Best ordering found: Energy = {best_energy:.6f} kcal/mol")
    print(f"  Reorder indices: {best_indices}")
    
    # Validate final arrays
    if np.any(~np.isfinite(best_R)):
        raise RuntimeError("Final reordered positions contain NaN/Inf values")
    
    # Set final positions in PyCHARMM
    # This ensures PyCHARMM has valid coordinates (not the undefined 9999.0 values)
    try:
        xyz = pd.DataFrame(best_R, columns=["x", "y", "z"])
        coor.set_positions(xyz)
        print("  Final positions set in PyCHARMM")
    except (AttributeError, RuntimeError, OSError, SystemError) as e:
        print(f"  ERROR: Could not set final positions in PyCHARMM (may have crashed): {e}")
        raise RuntimeError(f"Failed to set final positions. PyCHARMM may have crashed: {e}") from e
    except Exception as e:
        print(f"  Warning: Could not set final positions in PyCHARMM: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Test if PyCHARMM is responsive before doing anything with it
    pycharmm_available = False
    if hasattr(args, 'include_mm') and args.include_mm:
        try:
            from mmml.pycharmmInterface.import_pycharmm import psf, coor
            # Try a simple test call to see if PyCHARMM is alive
            try:
                _ = psf.get_atype()
                pycharmm_available = True
                print(f"  PyCHARMM is responsive")
            except (AttributeError, RuntimeError, OSError, SystemError) as e:
                print(f"  WARNING: PyCHARMM test call failed: {e}")
                print(f"  PyCHARMM may have crashed or is not properly initialized")
                print(f"  Disabling MM contributions to prevent further crashes")
                # Disable MM to prevent crashes
                if hasattr(args, 'include_mm'):
                    args.include_mm = False
                pycharmm_available = False
        except ImportError:
            print(f"  PyCHARMM not available (ImportError)")
            pycharmm_available = False
        except Exception as e:
            print(f"  WARNING: Unexpected error testing PyCHARMM: {e}")
            pycharmm_available = False
    
    # Try to get PyCHARMM data if not provided but PyCHARMM is available
    # Wrap in comprehensive error handling to prevent crashes
    if pycharmm_available and (pycharmm_atypes is None or pycharmm_resids is None):
        try:
            from mmml.pycharmmInterface.import_pycharmm import psf
            # Test if PyCHARMM is responsive before calling it
            try:
                # Try a simple call first to see if PyCHARMM is alive
                test_atypes = psf.get_atype()
                if pycharmm_atypes is None:
                    pycharmm_atypes = np.array(test_atypes)
                if pycharmm_resids is None:
                    pycharmm_resids = np.array(psf.get_resid())
                print(f"  Retrieved PyCHARMM atom types and residue IDs from PSF")
            except (AttributeError, RuntimeError, OSError, SystemError) as e:
                print(f"  Warning: PyCHARMM call failed (may have crashed): {e}")
                print(f"  PyCHARMM may not be properly initialized or may have segfaulted")
                # Set to None to disable reordering
                pycharmm_atypes = None
                pycharmm_resids = None
        except ImportError:
            print(f"  PyCHARMM not available (ImportError)")
            pycharmm_atypes = None
            pycharmm_resids = None
        except Exception as e:
            print(f"  Could not retrieve PyCHARMM data: {e}")
            import traceback
            traceback.print_exc()
            # Set to None to disable reordering
            pycharmm_atypes = None
            pycharmm_resids = None
    
    # Try to get ATOMS_PER_MONOMER and N_MONOMERS from args if not provided
    if ATOMS_PER_MONOMER is None and hasattr(args, 'n_atoms_monomer'):
        ATOMS_PER_MONOMER = args.n_atoms_monomer
    if N_MONOMERS is None and hasattr(args, 'n_monomers'):
        N_MONOMERS = args.n_monomers
    
    # Reorder atoms to match PyCHARMM ordering if PyCHARMM is initialized
    # Check if MM is enabled (either via args or if PyCHARMM data is available)
    mm_enabled = (args.include_mm if hasattr(args, 'include_mm') else False) or \
                 (pycharmm_atypes is not None)
    
    if mm_enabled and \
       pycharmm_atypes is not None and \
       ATOMS_PER_MONOMER is not None and \
       N_MONOMERS is not None:
        print(f"  Attempting to reorder atoms to match PyCHARMM ordering...")
        print(f"    ATOMS_PER_MONOMER: {ATOMS_PER_MONOMER}, N_MONOMERS: {N_MONOMERS}")
        print(f"    PyCHARMM atom types shape: {pycharmm_atypes.shape if pycharmm_atypes is not None else None}")
        try:
            R, Z, reorder_indices = reorder_atoms_to_match_pycharmm(
                R, Z, pycharmm_atypes, pycharmm_resids,
                ATOMS_PER_MONOMER, N_MONOMERS
            )
            print(f"  ✓ Atoms reordered to match PyCHARMM ordering")
        except KeyboardInterrupt:
            # Re-raise keyboard interrupts
            raise
        except Exception as e:
            print(f"  Warning: Reordering failed: {e}")
            print(f"  Continuing without reordering (this may cause issues)")
            import traceback
            traceback.print_exc()
            # Don't re-raise - allow continuation without reordering
    else:
        missing = []
        if not mm_enabled:
            missing.append("MM not enabled")
        if pycharmm_atypes is None:
            missing.append("pycharmm_atypes")
        if ATOMS_PER_MONOMER is None:
            missing.append("ATOMS_PER_MONOMER")
        if N_MONOMERS is None:
            missing.append("N_MONOMERS")
        print(f"  No reordering applied (missing: {', '.join(missing)})")
    
    # Validate atomic numbers match between batch and PyCHARMM (if available)
    # Wrap in comprehensive error handling to prevent crashes
    if hasattr(args, 'include_mm') and args.include_mm and pycharmm_atypes is not None:
        try:
            from mmml.pycharmmInterface.import_pycharmm import psf
            from mmml.pycharmmInterface.utils import get_Z_from_psf
            # Get atomic numbers from PyCHARMM PSF with error handling
            try:
                # Use get_Z_from_psf() which uses atomic masses to determine Z
                # This is more reliable than trying to parse atom type strings
                pycharmm_atomic_numbers_all = np.array(get_Z_from_psf())
                pycharmm_atomic_numbers = pycharmm_atomic_numbers_all[:len(Z)]
                
                # Fallback: Try parsing atom type strings if mass-based method fails
                if len(pycharmm_atomic_numbers) == 0 or np.all(pycharmm_atomic_numbers == 0):
                    # Try to extract element symbol from atom type string
                    # PyCHARMM atom types are IUPAC names like "C", "O", "H" (may have whitespace)
                    pycharmm_atomic_numbers = []
                    for atype in pycharmm_atypes[:len(Z)]:
                        # Strip whitespace and try to get atomic number
                        atype_clean = str(atype).strip()
                        # Try direct lookup first
                        z = ase.data.atomic_numbers.get(atype_clean, None)
                        if z is None:
                            # Try extracting first character (for cases like "C1", "O1")
                            first_char = atype_clean[0] if len(atype_clean) > 0 else ""
                            z = ase.data.atomic_numbers.get(first_char, 0)
                        pycharmm_atomic_numbers.append(z)
                    pycharmm_atomic_numbers = np.array(pycharmm_atomic_numbers)
                
                # Compare with batch atomic numbers (after reordering if applicable)
                if len(pycharmm_atomic_numbers) == len(Z):
                    matches = np.allclose(pycharmm_atomic_numbers, Z, atol=0.1)
                    if not matches:
                        print(f"  Warning: Atomic number mismatch between batch and PyCHARMM!")
                        print(f"    Batch Z: {Z}")
                        print(f"    PyCHARMM Z (from PSF): {pycharmm_atomic_numbers}")
                        print(f"    PyCHARMM atom types: {pycharmm_atypes[:len(Z)]}")
                        print(f"    This may cause force calculation issues.")
                        print(f"    Note: This warning may be safe to ignore if atom types are correct.")
                    else:
                        print(f"  ✓ Atomic numbers match between batch and PyCHARMM")
            except (AttributeError, RuntimeError, OSError, SystemError) as e:
                print(f"  Warning: Could not validate atomic numbers (PyCHARMM may have crashed): {e}")
        except ImportError:
            print(f"  Warning: PyCHARMM not available for atomic number validation")
        except Exception as e:
            print(f"  Warning: Could not validate atomic numbers: {e}")
            import traceback
            traceback.print_exc()
    
    # Create ASE Atoms object
    atoms = ase.Atoms(Z, R)
    
    # Sync positions with PyCHARMM if MM is enabled and PyCHARMM is available
    # This ensures PyCHARMM coordinates match the batch positions
    # Wrap in comprehensive error handling to prevent crashes
    if pycharmm_available and hasattr(args, 'include_mm') and args.include_mm:
        try:
            # Test if coor is available and responsive
            try:
                xyz = pd.DataFrame(R, columns=["x", "y", "z"])
                # Try to set positions - this can crash if PyCHARMM is in bad state
                coor.set_positions(xyz)
                print("  Synced positions with PyCHARMM")
            except (AttributeError, RuntimeError, OSError, SystemError) as e:
                print(f"  Warning: Could not sync positions with PyCHARMM: {e}")
                print(f"  PyCHARMM may have crashed or is in an invalid state")
                # Don't raise - continue without syncing
        except ImportError:
            print(f"  Warning: PyCHARMM not available for position sync")
        except Exception as e:
            print(f"  Warning: Unexpected error syncing positions: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise - continue without syncing
    
    # Create hybrid calculator (following run_sim.py)
    # Note: MM contributions require PyCHARMM to be initialized first
    # Wrap calculator creation in error handling
    try:
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
    except (AttributeError, RuntimeError, OSError, SystemError) as e:
        print(f"  ERROR: Calculator factory failed (PyCHARMM may have crashed): {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to create calculator. PyCHARMM may have crashed: {e}") from e
    except Exception as e:
        print(f"  ERROR: Calculator factory failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    atoms.calc = hybrid_calc
    
    # Get initial energy and forces with comprehensive error handling
    try:
        try:
            hybrid_energy = float(atoms.get_potential_energy())
        except (AttributeError, RuntimeError, OSError, SystemError) as e:
            print(f"  ERROR: get_potential_energy() failed (PyCHARMM may have crashed): {e}")
            raise RuntimeError(f"Energy calculation failed. PyCHARMM may have crashed: {e}") from e
        
        try:
            hybrid_forces = np.asarray(atoms.get_forces())
        except (AttributeError, RuntimeError, OSError, SystemError) as e:
            print(f"  ERROR: get_forces() failed (PyCHARMM may have crashed): {e}")
            raise RuntimeError(f"Force calculation failed. PyCHARMM may have crashed: {e}") from e
        
        print(f"Initial energy: {hybrid_energy:.6f} eV")
        print(f"Initial forces shape: {hybrid_forces.shape}")
        
        # Check for NaN in forces
        if np.any(~np.isfinite(hybrid_forces)):
            nan_count = np.sum(~np.isfinite(hybrid_forces))
            print(f"  WARNING: {nan_count} NaN/Inf values found in forces!")
            print(f"  This may indicate atom ordering or indexing issues.")
        else:
            print(f"Max force: {np.abs(hybrid_forces).max():.6f} eV/Å")
            
    except RuntimeError:
        # Re-raise RuntimeErrors (these are our custom errors)
        raise
    except Exception as e:
        print(f"Warning: Could not compute initial energy/forces: {e}")
        print("This may be due to PyCHARMM not being properly initialized")
        print("or atom ordering mismatch. Check the reordering function.")
        import traceback
        traceback.print_exc()
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


#!/usr/bin/env python3
"""
Debug script for hybrid calculator forces initialized from batch.
Use this to diagnose force issues after calling initialize_simulation_from_batch.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple


def debug_hybrid_calculator_forces(
    atoms,
    hybrid_calc,
    ref_forces: Optional[np.ndarray] = None,
    verbose: bool = True
) -> dict:
    """
    Comprehensive debugging of hybrid calculator forces.
    
    Args:
        atoms: ASE Atoms object with hybrid_calc attached
        hybrid_calc: Hybrid calculator instance
        ref_forces: Optional reference forces for comparison (n_atoms, 3)
        verbose: Whether to print detailed diagnostics
    
    Returns:
        Dictionary with diagnostic information
    """
    n_atoms = len(atoms)
    
    if verbose:
        print("=" * 80)
        print("HYBRID CALCULATOR FORCE DEBUGGING")
        print("=" * 80)
        print(f"Number of atoms: {n_atoms}")
    
    # Get computed forces
    try:
        computed_forces = np.asarray(atoms.get_forces())
        computed_energy = float(atoms.get_potential_energy())
    except Exception as e:
        print(f"ERROR: Could not compute forces: {e}")
        return {"error": str(e)}
    
    if verbose:
        print(f"\nComputed energy: {computed_energy:.6f} eV")
        print(f"Computed forces shape: {computed_forces.shape}")
    
    diagnostics = {
        "n_atoms": n_atoms,
        "computed_forces": computed_forces,
        "computed_energy": computed_energy,
    }
    
    # Check for NaN/Inf
    nan_mask = ~np.isfinite(computed_forces)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        print(f"\n⚠️  WARNING: {nan_count} NaN/Inf values in forces!")
        print(f"   Locations: {np.where(nan_mask)}")
        diagnostics["has_nan"] = True
        diagnostics["nan_locations"] = np.where(nan_mask)
    else:
        diagnostics["has_nan"] = False
    
    # Check zero forces
    zero_mask = np.all(np.abs(computed_forces) < 1e-10, axis=1)
    zero_count = np.sum(zero_mask)
    zero_indices = np.where(zero_mask)[0]
    
    if verbose:
        print(f"\nZero forces: {zero_count}/{n_atoms} atoms")
        if zero_count > 0:
            print(f"   Atom indices with zero forces: {zero_indices.tolist()}")
    
    diagnostics["zero_forces_count"] = zero_count
    diagnostics["zero_force_indices"] = zero_indices
    
    # Try to extract ML and MM contributions if available
    if hasattr(hybrid_calc, 'results'):
        results = hybrid_calc.results
        
        # Check for model output breakdown
        if "model_output" in results or "out" in results:
            model_out = results.get("model_output") or results.get("out")
            
            if hasattr(model_out, 'ml_2b_F'):
                ml_2b_forces = np.asarray(model_out.ml_2b_F)
                if verbose:
                    print(f"\nML dimer forces shape: {ml_2b_forces.shape}")
                    print(f"ML dimer forces zero: {np.sum(np.all(np.abs(ml_2b_forces) < 1e-10, axis=1))}/{len(ml_2b_forces)}")
                diagnostics["ml_2b_forces"] = ml_2b_forces
                diagnostics["ml_2b_zero_count"] = np.sum(np.all(np.abs(ml_2b_forces) < 1e-10, axis=1))
            
            if hasattr(model_out, 'mm_F'):
                mm_forces = np.asarray(model_out.mm_F)
                if verbose:
                    print(f"\nMM forces shape: {mm_forces.shape}")
                    print(f"MM forces zero: {np.sum(np.all(np.abs(mm_forces) < 1e-10, axis=1))}/{len(mm_forces)}")
                diagnostics["mm_forces"] = mm_forces
                diagnostics["mm_zero_count"] = np.sum(np.all(np.abs(mm_forces) < 1e-10, axis=1))
            
            if hasattr(model_out, 'internal_F'):
                internal_forces = np.asarray(model_out.internal_F)
                if verbose:
                    print(f"\nInternal (monomer) forces shape: {internal_forces.shape}")
                diagnostics["internal_forces"] = internal_forces
        
        # Check for individual force components
        for key in ["ml_forces", "mm_forces", "ml_2b_F", "internal_F", "mm_F"]:
            if key in results:
                forces = np.asarray(results[key])
                if verbose:
                    print(f"\n{key} shape: {forces.shape}")
                diagnostics[key] = forces
    
    # Compare with reference if provided
    if ref_forces is not None:
        ref_forces = np.asarray(ref_forces)
        if ref_forces.shape != computed_forces.shape:
            print(f"\n⚠️  Shape mismatch: ref={ref_forces.shape}, computed={computed_forces.shape}")
        else:
            diff = computed_forces - ref_forces
            diff_mags = np.linalg.norm(diff, axis=1)
            
            # Find atoms with zero computed but non-zero ref
            ref_zero = np.all(np.abs(ref_forces) < 1e-10, axis=1)
            problematic = zero_mask & ~ref_zero
            
            if verbose:
                print(f"\n" + "=" * 80)
                print("COMPARISON WITH REFERENCE FORCES")
                print("=" * 80)
                print(f"Mean absolute error: {diff_mags.mean():.6f} eV/Å")
                print(f"Max absolute error: {diff_mags.max():.6f} eV/Å")
                print(f"RMS error: {np.sqrt(np.mean(diff**2)):.6f} eV/Å")
                
                if np.sum(problematic) > 0:
                    print(f"\n❌ CRITICAL: {np.sum(problematic)} atoms have zero computed forces but non-zero reference!")
                    print(f"   Problematic atom indices: {np.where(problematic)[0].tolist()}")
                    print(f"\n   This suggests:")
                    print(f"   - Dimer forces may not be computed (see CRITICAL_ISSUES_SUMMARY.md)")
                    print(f"   - MM forces may not be computed for all atoms")
                    print(f"   - Atom index mapping may be incorrect")
            
            diagnostics["ref_forces"] = ref_forces
            diagnostics["force_diff"] = diff
            diagnostics["force_diff_mags"] = diff_mags
            diagnostics["problematic_atoms"] = np.where(problematic)[0] if np.sum(problematic) > 0 else []
    
    # Check atom ordering (if we can determine monomer structure)
    if hasattr(hybrid_calc, 'n_monomers') and hasattr(hybrid_calc, 'ATOMS_PER_MONOMER'):
        n_monomers = hybrid_calc.n_monomers
        atoms_per_monomer = hybrid_calc.ATOMS_PER_MONOMER
        
        if verbose:
            print(f"\n" + "=" * 80)
            print("MONOMER STRUCTURE")
            print("=" * 80)
            print(f"Number of monomers: {n_monomers}")
            print(f"Atoms per monomer: {atoms_per_monomer}")
            print(f"Expected total atoms: {n_monomers * atoms_per_monomer}")
        
        monomer_indices = []
        for i in range(n_monomers):
            start = i * atoms_per_monomer
            end = (i + 1) * atoms_per_monomer
            if end <= n_atoms:
                monomer_indices.extend(range(start, end))
        
        monomer_mask = np.zeros(n_atoms, dtype=bool)
        monomer_mask[monomer_indices] = True
        
        if verbose:
            print(f"Monomer atom indices: {monomer_indices[:20]}..." if len(monomer_indices) > 20 else f"Monomer atom indices: {monomer_indices}")
            print(f"Non-monomer atoms: {np.where(~monomer_mask)[0].tolist()}")
            
            # Check zero forces by category
            monomer_zero = np.sum(zero_mask[monomer_mask])
            nonmonomer_zero = np.sum(zero_mask[~monomer_mask])
            print(f"\nZero forces:")
            print(f"  In monomers: {monomer_zero}/{np.sum(monomer_mask)}")
            print(f"  NOT in monomers: {nonmonomer_zero}/{np.sum(~monomer_mask)}")
        
        diagnostics["n_monomers"] = n_monomers
        diagnostics["atoms_per_monomer"] = atoms_per_monomer
        diagnostics["monomer_indices"] = monomer_indices
        diagnostics["monomer_zero_count"] = np.sum(zero_mask[monomer_mask])
        diagnostics["nonmonomer_zero_count"] = np.sum(zero_mask[~monomer_mask])
    
    if verbose:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        if diagnostics.get("has_nan", False):
            print("❌ Fix NaN/Inf values in forces")
            print("   → Check atom ordering")
            print("   → Check for division by zero in force computation")
        
        if zero_count > n_atoms * 0.1:  # More than 10% zero
            print(f"⚠️  Many atoms ({zero_count}/{n_atoms}) have zero forces")
            print("   → Check if dimer forces are computed (see CRITICAL_ISSUES_SUMMARY.md)")
            print("   → Verify MM forces are computed for all atoms")
            print("   → Check atom index mapping")
        
        if ref_forces is not None and len(diagnostics.get("problematic_atoms", [])) > 0:
            print("❌ CRITICAL: Atoms with zero computed but non-zero reference forces")
            print("   → Most likely: Dimer forces set to zero (see CRITICAL_ISSUES_SUMMARY.md Issue #1)")
            print("   → Check: mmml_calculator.py:1953 - apply_dimer_switching returns forces=0")
        
        print("=" * 80)
    
    return diagnostics


def compare_ml_mm_contributions(
    atoms,
    hybrid_calc,
    verbose: bool = True
) -> dict:
    """
    Compare ML and MM force contributions separately.
    
    Args:
        atoms: ASE Atoms object
        hybrid_calc: Hybrid calculator
        verbose: Print results
    
    Returns:
        Dictionary with ML and MM force breakdowns
    """
    results = {}
    
    # Try to get ML-only forces
    try:
        # Temporarily disable MM
        original_doMM = hybrid_calc.doMM if hasattr(hybrid_calc, 'doMM') else True
        if hasattr(hybrid_calc, 'doMM'):
            hybrid_calc.doMM = False
        
        ml_forces = np.asarray(atoms.get_forces())
        ml_energy = float(atoms.get_potential_energy())
        
        results["ml_forces"] = ml_forces
        results["ml_energy"] = ml_energy
        
        # Restore MM
        if hasattr(hybrid_calc, 'doMM'):
            hybrid_calc.doMM = original_doMM
        
        if verbose:
            print(f"ML-only forces: shape={ml_forces.shape}, max={np.abs(ml_forces).max():.6f}")
            print(f"ML-only energy: {ml_energy:.6f} eV")
    except Exception as e:
        if verbose:
            print(f"Could not compute ML-only forces: {e}")
    
    # Try to get MM-only forces
    try:
        # Temporarily disable ML
        original_doML = hybrid_calc.doML if hasattr(hybrid_calc, 'doML') else True
        if hasattr(hybrid_calc, 'doML'):
            hybrid_calc.doML = False
        
        mm_forces = np.asarray(atoms.get_forces())
        mm_energy = float(atoms.get_potential_energy())
        
        results["mm_forces"] = mm_forces
        results["mm_energy"] = mm_energy
        
        # Restore ML
        if hasattr(hybrid_calc, 'doML'):
            hybrid_calc.doML = original_doML
        
        if verbose:
            print(f"MM-only forces: shape={mm_forces.shape}, max={np.abs(mm_forces).max():.6f}")
            print(f"MM-only energy: {mm_energy:.6f} eV")
    except Exception as e:
        if verbose:
            print(f"Could not compute MM-only forces: {e}")
    
    return results


if __name__ == "__main__":
    print("This module provides debugging functions for hybrid calculator forces.")
    print("\nUsage:")
    print("  from debug_hybrid_forces_from_batch import debug_hybrid_calculator_forces")
    print("  diagnostics = debug_hybrid_calculator_forces(atoms, hybrid_calc, ref_forces=ref_Fs)")


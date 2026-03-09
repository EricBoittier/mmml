#!/usr/bin/env python3
"""
ESP Grid Alignment Function

Aligns ESP grids to molecular reference frames by centering grids on atom COM.
This ensures ESP grids and atom positions are in the same coordinate frame.

Usage in notebook:
    from align_esp_grids import align_esp_grids
    train_data = align_esp_grids(train_data, verbose=True)
    valid_data = align_esp_grids(valid_data, verbose=True)
"""

import numpy as np
from typing import Dict


def align_esp_grids(data: Dict, verbose: bool = True) -> Dict:
    """
    Align ESP grids to molecular reference frames by centering grids on atom COM.
    
    This ensures ESP grids and atom positions are in the same coordinate frame,
    which is critical for accurate ESP prediction.
    
    Parameters
    ----------
    data : dict
        Data dictionary with keys: 'R', 'N', 'vdw_surface'
        - 'R': atomic positions, shape (n_samples, natoms, 3) or (natoms, 3)
        - 'N': number of atoms per sample, shape (n_samples,) or scalar
        - 'vdw_surface': ESP grid points, shape (n_samples, ngrid, 3) or (ngrid, 3)
    verbose : bool
        Whether to print alignment information
        
    Returns
    -------
    dict
        Data dictionary with aligned 'vdw_surface'
        
    Notes
    -----
    The alignment works by:
    1. Computing atom COM (center of mass) for each molecule
    2. Computing grid COM for each molecule
    3. Computing offset = grid_com - atom_com
    4. Shifting grids: vdw_surface_aligned = vdw_surface - offset
    
    After alignment, grid COM should match atom COM (within numerical precision).
    """
    # Make a copy to avoid modifying original
    data = data.copy()
    
    # Get shapes - handle both single sample and batch cases
    R = data['R']
    N = data['N']
    vdw = data['vdw_surface']
    
    # Determine if single sample or batch
    is_single_sample = R.ndim == 2
    
    if is_single_sample:
        # Single sample: add batch dimension
        R = R[None, :, :]  # (1, natoms, 3)
        N = np.array([N]) if np.isscalar(N) else N[None]  # (1,)
        if vdw.ndim == 2:
            vdw = vdw[None, :, :]  # (1, ngrid, 3)
    else:
        # Already batched
        N = np.asarray(N)
    
    # Get number of samples and max atoms
    n_samples = len(N)
    natoms_max = R.shape[1]
    
    # Create mask for real atoms (handle variable number of atoms)
    atom_mask = np.arange(natoms_max)[None, :] < N[:, None]  # (n_samples, natoms)
    
    # Compute atom COM for each molecule (handling variable number of atoms)
    atom_positions_masked = R * atom_mask[:, :, None]  # Zero out padding
    atom_com = atom_positions_masked.sum(axis=1) / N[:, None]  # (n_samples, 3)
    
    # Compute grid COM for each molecule
    grid_com = vdw.mean(axis=1)  # (n_samples, 3)
    
    # Compute offset for each molecule
    offset = grid_com - atom_com  # (n_samples, 3)
    
    # Apply alignment (broadcast over grid points)
    vdw_surface_aligned = vdw - offset[:, None, :]  # (n_samples, ngrid, 3)
    
    # Remove batch dimension if single sample
    if is_single_sample:
        vdw_surface_aligned = vdw_surface_aligned[0]
        atom_com = atom_com[0]
        grid_com = grid_com[0]
        offset = offset[0]
    
    # Update data dictionary
    data['vdw_surface'] = vdw_surface_aligned
    
    if verbose:
        print(f"✅ Aligned ESP grids to molecular reference frames")
        print(f"   Number of samples: {n_samples}")
        
        # Show first sample details
        if is_single_sample:
            grid_com_after = vdw_surface_aligned.mean(axis=0)
            atom_com_sample = atom_com
            offset_sample = offset
        else:
            grid_com_after = vdw_surface_aligned[0].mean(axis=0)
            atom_com_sample = atom_com[0]
            offset_sample = offset[0]
        
        print(f"\n   Sample 0 details:")
        print(f"     Atom COM:        {atom_com_sample}")
        print(f"     Grid COM before: {grid_com}")
        print(f"     Grid COM after:  {grid_com_after}")
        print(f"     Offset corrected: {offset_sample} Å")
        
        # Sense check: verify alignment worked
        alignment_error = np.linalg.norm(grid_com_after - atom_com_sample)
        print(f"\n   Sense check:")
        print(f"     Alignment error: {alignment_error:.6e} Å")
        
        if alignment_error > 1e-5:
            print(f"     ⚠️  WARNING: Alignment error is large!")
            print(f"        Expected: < 1e-5 Å, Got: {alignment_error:.6e} Å")
            print(f"        This suggests a potential issue with data shapes or alignment logic.")
        else:
            print(f"     ✓ Alignment verified: grids are centered on atom COM")
        
        # Check offset magnitude
        offset_magnitude = np.linalg.norm(offset_sample)
        if offset_magnitude > 1.0:
            print(f"\n     ⚠️  Large offset detected: {offset_magnitude:.3f} Å")
            print(f"        This suggests ESP grids and atoms were in different reference frames.")
        else:
            print(f"     ✓ Offset magnitude reasonable: {offset_magnitude:.3f} Å")
    
    return data


def sense_check_alignment(data: Dict, sample_idx: int = 0, verbose: bool = True) -> Dict:
    """
    Perform a comprehensive sense check on ESP grid alignment.
    
    Parameters
    ----------
    data : dict
        Data dictionary with aligned grids
    sample_idx : int
        Index of sample to check
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    dict
        Dictionary with sense check results
    """
    R = data['R']
    N = data['N']
    vdw = data['vdw_surface']
    
    # Handle single sample case
    if R.ndim == 2:
        R_sample = R
        N_sample = N if np.isscalar(N) else N[0]
        vdw_sample = vdw
    else:
        R_sample = R[sample_idx]
        N_sample = N[sample_idx] if not np.isscalar(N) else N
        vdw_sample = vdw[sample_idx]
    
    # Compute atom COM (using same logic as alignment function)
    atom_mask = np.arange(len(R_sample)) < N_sample
    atom_positions_masked = R_sample * atom_mask[:, None]
    atom_com = atom_positions_masked.sum(axis=0) / N_sample
    
    # Compute grid COM
    grid_com = vdw_sample.mean(axis=0)
    
    # Check alignment
    alignment_error = np.linalg.norm(grid_com - atom_com)
    offset = grid_com - atom_com
    
    results = {
        'atom_com': atom_com,
        'grid_com': grid_com,
        'alignment_error': alignment_error,
        'offset': offset,
        'offset_magnitude': np.linalg.norm(offset),
        'is_aligned': alignment_error < 1e-5,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Sense Check for Sample {sample_idx}")
        print(f"{'='*70}")
        print(f"\nAtom positions (first {N_sample} atoms):")
        for i in range(N_sample):
            print(f"  Atom {i}: {R_sample[i]}")
        print(f"\nAtom COM: {atom_com}")
        print(f"Grid COM:  {grid_com}")
        print(f"Offset:    {offset}")
        print(f"\nAlignment error: {alignment_error:.6e} Å")
        print(f"Offset magnitude: {results['offset_magnitude']:.3f} Å")
        
        if results['is_aligned']:
            print(f"\n✅ PASS: Grids are properly aligned")
        else:
            print(f"\n❌ FAIL: Grids are NOT properly aligned")
            print(f"   Expected alignment error < 1e-5 Å")
            print(f"   Got: {alignment_error:.6e} Å")
    
    return results


if __name__ == '__main__':
    # Test with example data
    print("ESP Grid Alignment Test")
    print("="*70)
    
    # Create dummy data
    n_samples = 2
    natoms = 3
    ngrid = 100
    
    # Atom positions centered at origin
    R = np.random.randn(n_samples, natoms, 3) * 2.0
    
    # Grid positions offset from atom COM
    grid_offset = np.array([5.0, 5.0, 5.0])
    vdw = np.random.randn(n_samples, ngrid, 3) * 3.0 + grid_offset[None, None, :]
    
    N = np.array([natoms] * n_samples)
    
    data = {
        'R': R,
        'N': N,
        'vdw_surface': vdw,
    }
    
    print("\nBefore alignment:")
    print(f"  Atom COM (sample 0): {R[0].mean(axis=0)}")
    print(f"  Grid COM (sample 0):  {vdw[0].mean(axis=0)}")
    print(f"  Offset:               {vdw[0].mean(axis=0) - R[0].mean(axis=0)}")
    
    # Align
    data_aligned = align_esp_grids(data, verbose=True)
    
    # Sense check
    sense_check_alignment(data_aligned, sample_idx=0, verbose=True)


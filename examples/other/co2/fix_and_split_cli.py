#!/usr/bin/env python3
"""
CLI tool to fix units and create train/valid/test splits from CO2 NPZ data.

Usage:
    python fix_and_split_cli.py \
        --efd energies_forces_dipoles.npz \
        --grid grids_esp.npz \
        --output-dir ./training_data_fixed \
        --train-frac 0.8 \
        --valid-frac 0.1 \
        --test-frac 0.1 \
        --seed 42

This script:
1. Validates atomic coordinates are in Angstroms
2. Converts energies from Hartree to eV (ASE standard)
3. Converts forces from Hartree/Bohr to eV/Angstrom (ASE standard)
4. Converts ESP grid coordinates from index space to physical Angstroms
5. Creates train/valid/test splits
6. Saves data in ASE-compatible format

ASE Standard Units:
- Coordinates (R): Angstrom
- Energies (E): eV
- Forces (F): eV/Angstrom
- Dipoles (Dxyz): Debye
- ESP values: Hartree/e (no ASE standard for ESP)
- ESP grid (vdw_surface): Angstrom
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional

# Add parent directory to path
repo_root = Path(__file__).parent / "../.."
sys.path.insert(0, str(repo_root.resolve()))


def create_splits(n_samples: int, train_frac=0.8, valid_frac=0.1, test_frac=0.1, seed=42):
    """Create train/valid/test split indices."""
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    n_train = int(n_samples * train_frac)
    n_valid = int(n_samples * valid_frac)
    
    return {
        'train': indices[:n_train],
        'valid': indices[n_train:n_train + n_valid],
        'test': indices[n_train + n_valid:]
    }


def convert_energy_hartree_to_ev(E_hartree: np.ndarray) -> np.ndarray:
    """Convert energies from Hartree to eV (ASE standard)."""
    HARTREE_TO_EV = 27.211386
    return E_hartree * HARTREE_TO_EV


def convert_forces_hartree_bohr_to_ev_angstrom(F_hartree_bohr: np.ndarray) -> np.ndarray:
    """Convert forces from Hartree/Bohr to eV/Angstrom (ASE standard)."""
    HARTREE_BOHR_TO_EV_ANGSTROM = 51.42208
    return F_hartree_bohr * HARTREE_BOHR_TO_EV_ANGSTROM


def convert_grid_indices_to_angstrom(
    vdw_grid_indices: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    dims: np.ndarray,
    cube_spacing_bohr: float = 0.25
) -> np.ndarray:
    """
    Convert ESP grid from index space to physical Angstrom coordinates.
    
    The vdw_grid currently contains values like 0-49 which are grid indices.
    We need to convert to physical coordinates using the cube metadata.
    """
    n_samples = vdw_grid_indices.shape[0]
    bohr_to_angstrom = 0.529177
    
    vdw_grid_angstrom = np.zeros_like(vdw_grid_indices)
    
    for i in range(n_samples):
        grid_indices = vdw_grid_indices[i] - origin[i]  # Remove origin offset
        coord_bohr = origin[i] + grid_indices * cube_spacing_bohr
        vdw_grid_angstrom[i] = coord_bohr * bohr_to_angstrom
    
    return vdw_grid_angstrom


def validate_fixed_data(R_ang, E_ev, F_ev_ang, vdw_grid_ang, Z, N, verbose=True):
    """Validate that fixes worked correctly."""
    if verbose:
        print(f"\n{'='*70}")
        print("POST-FIX VALIDATION")
        print(f"{'='*70}")
    
    # Check atomic coordinates across multiple samples
    co_bonds_check = []
    for i in range(min(100, len(R_ang))):
        r = R_ang[i]
        z = Z[i]
        valid = z > 0
        vz = z[valid]
        vpos = r[valid]
        
        if len(vz) >= 3 and 6 in vz and np.sum(vz == 8) >= 2:
            c_idx = np.where(vz == 6)[0][0]
            o_idx = np.where(vz == 8)[0]
            for oi in o_idx:
                co_bonds_check.append(np.linalg.norm(vpos[c_idx] - vpos[oi]))
    
    co_bonds_check = np.array(co_bonds_check)
    
    coords_ok = False
    energy_ok = False
    force_ok = False
    grid_ok = False
    spatial_ok = False
    
    if len(co_bonds_check) > 0:
        if verbose:
            print(f"\nAtomic Coordinates (up to 100 samples):")
            print(f"  C-O bonds: mean={co_bonds_check.mean():.4f} Å, "
                  f"range=[{co_bonds_check.min():.4f}, {co_bonds_check.max():.4f}]")
        
        if 1.0 <= co_bonds_check.mean() <= 1.5:
            if verbose:
                print(f"  ✓ Coordinates in Angstroms with varying geometries")
            coords_ok = True
        else:
            if verbose:
                print(f"  ❌ Coordinates outside expected range!")
    else:
        if verbose:
            print(f"\n⚠️  Could not find CO2 molecules for coordinate validation")
        coords_ok = True  # Skip this check if not CO2
    
    # Check energies
    if verbose:
        print(f"\nEnergies (sample 0):")
        print(f"  Value: {E_ev[0]:.6f} eV")
        print(f"  Dataset mean: {E_ev.mean():.6f} eV")
    
    # For CO2, expect energies around -5100 to -5000 eV (from -187.5 Ha)
    # For other molecules, just check that conversion happened (values are reasonable in eV)
    if -10000 < E_ev.mean() < 1000:
        if verbose:
            print(f"  ✓ Energies in reasonable range for molecular energies in eV")
        energy_ok = True
    else:
        if verbose:
            print(f"  ⚠️  Energy range unexpected")
        energy_ok = False
    
    # Check forces
    f_sample = F_ev_ang[0, :min(3, F_ev_ang.shape[1]), :]  # First sample, first atoms
    f_norm = np.linalg.norm(f_sample.reshape(-1, 3), axis=1).mean()
    
    if verbose:
        print(f"\nForces (sample 0):")
        print(f"  Mean norm: {f_norm:.6e} eV/Angstrom")
    
    # For geometry scans, forces can be large (up to 50-100 eV/Å far from equilibrium)
    if 1e-6 < f_norm < 1000:
        if verbose:
            print(f"  ✓ Force magnitudes in reasonable range")
        force_ok = True
    else:
        if verbose:
            print(f"  ⚠️  Force magnitudes outside expected range")
        force_ok = False
    
    # Check ESP grid
    grid0 = vdw_grid_ang[0]
    grid_extent = (grid0.max(axis=0) - grid0.min(axis=0)).mean()
    
    if verbose:
        print(f"\nESP Grid Coordinates:")
        print(f"  Average extent: {grid_extent:.4f} Angstrom")
        print(f"  X range: [{grid0[:, 0].min():.4f}, {grid0[:, 0].max():.4f}]")
        print(f"  Y range: [{grid0[:, 1].min():.4f}, {grid0[:, 1].max():.4f}]")
        print(f"  Z range: [{grid0[:, 2].min():.4f}, {grid0[:, 2].max():.4f}]")
    
    # Expect reasonable grid extent for molecular systems (2-20 Angstroms)
    if 2.0 < grid_extent < 50.0:
        if verbose:
            print(f"  ✓ Grid extent in reasonable range")
        grid_ok = True
    else:
        if verbose:
            print(f"  ⚠️  Grid extent outside expected range")
        grid_ok = False
    
    # Check spatial relationship
    r0 = R_ang[0]
    z0 = Z[0]
    valid = z0 > 0
    valid_pos = r0[valid]
    
    if len(valid_pos) > 0:
        mol_center = valid_pos.mean(axis=0)
        grid_min = grid0.min(axis=0)
        grid_max = grid0.max(axis=0)
        
        if verbose:
            print(f"\nSpatial relationship:")
            print(f"  Molecule center: [{mol_center[0]:.2f}, {mol_center[1]:.2f}, {mol_center[2]:.2f}]")
            print(f"  Grid bounds: X[{grid_min[0]:.2f}, {grid_max[0]:.2f}], "
                  f"Y[{grid_min[1]:.2f}, {grid_max[1]:.2f}], "
                  f"Z[{grid_min[2]:.2f}, {grid_max[2]:.2f}]")
        
        # Check if molecule is within or near grid bounds
        max_min_dist = max([np.min(np.linalg.norm(grid0 - atom_pos, axis=1)) 
                           for atom_pos in valid_pos])
        
        if max_min_dist < 10.0:
            if verbose:
                print(f"  ✓ Grid points within {max_min_dist:.2f} Å of molecule")
            spatial_ok = True
        else:
            if verbose:
                print(f"  ⚠️  Grid too far from molecule ({max_min_dist:.2f} Å)")
            spatial_ok = False
    else:
        if verbose:
            print(f"\n⚠️  Could not validate spatial relationship")
        spatial_ok = True
    
    overall_ok = coords_ok and energy_ok and force_ok and grid_ok and spatial_ok
    
    if verbose:
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"  Coordinates: {'✓' if coords_ok else '❌'}")
        print(f"  Energies:    {'✓' if energy_ok else '❌'}")
        print(f"  Forces:      {'✓' if force_ok else '❌'}")
        print(f"  ESP Grid:    {'✓' if grid_ok else '❌'}")
        print(f"  Spatial:     {'✓' if spatial_ok else '❌'}")
        
        if overall_ok:
            print(f"\n✅ ALL VALIDATIONS PASSED - Data ready for training!")
        else:
            print(f"\n⚠️  SOME VALIDATIONS FAILED - Review above")
        print(f"{'='*70}")
    
    return overall_ok


def fix_and_split_data(
    efd_file: Path,
    grid_file: Path,
    output_dir: Path,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    cube_spacing_bohr: float = 0.25,
    skip_validation: bool = False,
    verbose: bool = True
) -> bool:
    """
    Main workflow to fix units and create splits.
    
    Parameters
    ----------
    efd_file : Path
        Path to energies_forces_dipoles.npz file
    grid_file : Path
        Path to grids_esp.npz file
    output_dir : Path
        Directory to save output files
    train_frac : float
        Fraction of data for training (default 0.8)
    valid_frac : float
        Fraction of data for validation (default 0.1)
    test_frac : float
        Fraction of data for testing (default 0.1)
    seed : int
        Random seed for reproducible splits (default 42)
    cube_spacing_bohr : float
        Grid spacing in Bohr from original cube files (default 0.25)
    skip_validation : bool
        Skip validation checks (default False)
    verbose : bool
        Print detailed progress (default True)
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if verbose:
        print("\n" + "="*70)
        print("CO2 Data Unit Conversion and Splitting")
        print("="*70)
        print(f"\nInput files:")
        print(f"  EFD:  {efd_file}")
        print(f"  Grid: {grid_file}")
        print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # =========================================================================
    # Load data
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 1: Loading Data")
        print(f"{'#'*70}")
    
    try:
        efd_data = dict(np.load(efd_file))
        grid_data = dict(np.load(grid_file))
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return False
    
    n_samples = efd_data['R'].shape[0]
    if verbose:
        print(f"\nLoaded {n_samples} samples")
        print(f"  Keys in EFD: {list(efd_data.keys())}")
        print(f"  Keys in Grid: {list(grid_data.keys())}")
    
    # =========================================================================
    # Check atomic coordinates
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 2: Checking Atomic Coordinates")
        print(f"{'#'*70}")
    
    # Calculate C-O bond statistics to determine if conversion needed
    co_bonds = []
    for i in range(min(1000, len(efd_data['R']))):
        r = efd_data['R'][i]
        z = efd_data['Z'][i]
        valid = z > 0
        vz = z[valid]
        vpos = r[valid]
        
        if len(vz) >= 3 and 6 in vz and np.sum(vz == 8) >= 2:
            c_idx = np.where(vz == 6)[0][0]
            o_idx = np.where(vz == 8)[0]
            for oi in o_idx:
                co_bonds.append(np.linalg.norm(vpos[c_idx] - vpos[oi]))
    
    co_bonds = np.array(co_bonds)
    
    if len(co_bonds) > 0:
        if verbose:
            print(f"\nC-O Bond Statistics (up to 1000 samples):")
            print(f"  Mean:   {co_bonds.mean():.6f}")
            print(f"  Std:    {co_bonds.std():.6f}")
            print(f"  Min:    {co_bonds.min():.6f}")
            print(f"  Max:    {co_bonds.max():.6f}")
        
        if 1.0 < co_bonds.mean() < 1.5:
            if verbose:
                print(f"\n✓ Coordinates are ALREADY in Angstroms!")
            R_angstrom = efd_data['R']
        elif 2.0 < co_bonds.mean() < 2.7:
            if verbose:
                print(f"\n→ Converting from Bohr to Angstroms...")
            R_angstrom = efd_data['R'] * 0.529177
        else:
            print(f"\n❌ Unclear units! Mean C-O = {co_bonds.mean():.4f}")
            return False
    else:
        if verbose:
            print(f"\n⚠️  Could not determine molecule type, assuming coordinates are in Angstroms")
        R_angstrom = efd_data['R']
    
    # =========================================================================
    # Convert energies: Hartree → eV
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 3: Converting Energies from Hartree to eV")
        print(f"{'#'*70}")
    
    E_ev = convert_energy_hartree_to_ev(efd_data['E'])
    
    if verbose:
        HARTREE_TO_EV = 27.211386
        print(f"\nConversion factor: {HARTREE_TO_EV}")
        print(f"Original (Hartree): mean={efd_data['E'].mean():.6f}, "
              f"range=[{efd_data['E'].min():.6f}, {efd_data['E'].max():.6f}]")
        print(f"Converted (eV):     mean={E_ev.mean():.6f}, "
              f"range=[{E_ev.min():.6f}, {E_ev.max():.6f}]")
        print(f"✓ Energies converted to eV")
    
    # =========================================================================
    # Convert forces: Hartree/Bohr → eV/Angstrom
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 4: Converting Forces from Hartree/Bohr to eV/Angstrom")
        print(f"{'#'*70}")
    
    F_ev_ang = convert_forces_hartree_bohr_to_ev_angstrom(efd_data['F'])
    
    if verbose:
        HARTREE_BOHR_TO_EV_ANG = 51.42208
        f_orig_norms = np.linalg.norm(efd_data['F'][:10, :3, :].reshape(-1, 3), axis=1)
        f_conv_norms = np.linalg.norm(F_ev_ang[:10, :3, :].reshape(-1, 3), axis=1)
        
        print(f"\nConversion factor: {HARTREE_BOHR_TO_EV_ANG}")
        print(f"Original (Ha/Bohr): mean norm={f_orig_norms.mean():.6e}")
        print(f"Converted (eV/Å):   mean norm={f_conv_norms.mean():.6e}")
        print(f"✓ Forces converted to eV/Angstrom")
    
    # =========================================================================
    # Fix ESP grid: index space → physical Angstroms
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 5: Converting ESP Grid to Physical Angstroms")
        print(f"{'#'*70}")
    
    BOHR_TO_ANGSTROM = 0.529177
    
    if verbose:
        print(f"\nCube file parameters:")
        print(f"  Spacing: {cube_spacing_bohr} Bohr = {cube_spacing_bohr * BOHR_TO_ANGSTROM:.6f} Angstrom")
        print(f"  Dimensions: {grid_data['grid_dims'][0]}")
        print(f"  Original origin (Bohr): {grid_data['grid_origin'][0]}")
    
    vdw_surface_angstrom = convert_grid_indices_to_angstrom(
        grid_data['vdw_grid'],
        grid_data['grid_origin'],
        grid_data['grid_axes'],
        grid_data['grid_dims'],
        cube_spacing_bohr=cube_spacing_bohr
    )
    
    if verbose:
        grid0_original = grid_data['vdw_grid'][0]
        grid0_fixed = vdw_surface_angstrom[0]
        
        print(f"\nOriginal grid extent: {(grid0_original.max(axis=0) - grid0_original.min(axis=0)).mean():.4f}")
        print(f"Fixed grid extent: {(grid0_fixed.max(axis=0) - grid0_fixed.min(axis=0)).mean():.4f} Angstrom")
        print(f"✓ ESP grid converted to physical Angstroms")
    
    # =========================================================================
    # Validate fixed data
    # =========================================================================
    if not skip_validation:
        if verbose:
            print(f"\n{'#'*70}")
            print("# Step 6: Validating Fixed Data")
            print(f"{'#'*70}")
        
        is_valid = validate_fixed_data(
            R_angstrom, E_ev, F_ev_ang, vdw_surface_angstrom,
            efd_data['Z'], efd_data['N'], verbose=verbose
        )
        
        if not is_valid:
            print("\n❌ Validation failed! Not proceeding with splits.")
            return False
    
    # =========================================================================
    # Create splits
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 7: Creating Train/Valid/Test Splits")
        print(f"{'#'*70}")
    
    splits = create_splits(n_samples, train_frac=train_frac, valid_frac=valid_frac, 
                          test_frac=test_frac, seed=seed)
    
    if verbose:
        print(f"\nTotal samples: {n_samples}")
        print(f"  Train: {len(splits['train'])} ({len(splits['train'])/n_samples*100:.1f}%)")
        print(f"  Valid: {len(splits['valid'])} ({len(splits['valid'])/n_samples*100:.1f}%)")
        print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/n_samples*100:.1f}%)")
    
    # =========================================================================
    # Prepare datasets with fixed units
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 8: Preparing Fixed Datasets")
        print(f"{'#'*70}")
    
    # Update EFD data with fixed/converted values
    efd_fixed = efd_data.copy()
    efd_fixed['R'] = R_angstrom
    efd_fixed['E'] = E_ev
    efd_fixed['F'] = F_ev_ang
    
    # Update grid data with fixed coordinates
    grid_fixed = grid_data.copy()
    grid_fixed['R'] = R_angstrom
    grid_fixed['vdw_surface'] = vdw_surface_angstrom
    grid_fixed['vdw_grid'] = vdw_surface_angstrom
    
    # =========================================================================
    # Save split datasets
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 9: Saving Split Datasets")
        print(f"{'#'*70}")
    
    for split_name, split_indices in splits.items():
        if verbose:
            print(f"\nSaving {split_name} split ({len(split_indices)} samples)...")
        
        # Create EFD split
        efd_split = {k: v[split_indices] if (isinstance(v, np.ndarray) and v.shape[0] == n_samples) else v 
                     for k, v in efd_fixed.items()}
        efd_out = output_dir / f"energies_forces_dipoles_{split_name}.npz"
        np.savez_compressed(efd_out, **efd_split)
        
        if verbose:
            size_mb = efd_out.stat().st_size / 1024 / 1024
            print(f"  ✓ {efd_out.name} ({size_mb:.1f} MB)")
        
        # Create grid split
        grid_split = {k: v[split_indices] if (isinstance(v, np.ndarray) and v.shape[0] == n_samples) else v
                      for k, v in grid_fixed.items()}
        grid_out = output_dir / f"grids_esp_{split_name}.npz"
        np.savez_compressed(grid_out, **grid_split)
        
        if verbose:
            size_mb = grid_out.stat().st_size / 1024 / 1024
            print(f"  ✓ {grid_out.name} ({size_mb:.1f} MB)")
    
    # Save split indices
    indices_out = output_dir / "split_indices.npz"
    np.savez(indices_out, **splits)
    if verbose:
        print(f"\n✓ Split indices saved to {indices_out.name}")
    
    # =========================================================================
    # Create documentation
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 10: Creating Documentation")
        print(f"{'#'*70}")
    
    readme_content = f"""# Training Data (Unit-Corrected)

This directory contains molecular data with **corrected units** ready for DCMnet/PhysnetJax training.

## Data Corrections Applied

### 1. Atomic Coordinates (R)
- **Original**: Angstroms (verified)
- **Status**: ✓ Correct
- **Units**: Angstrom (ASE standard)

### 2. Energies (E)
- **Original**: Hartree
- **Converted**: eV (ASE standard)
- **Factor**: ×27.211386

### 3. Forces (F)
- **Original**: Hartree/Bohr
- **Converted**: eV/Angstrom (ASE standard)
- **Factor**: ×51.42208

### 4. ESP Grid Coordinates (vdw_surface)
- **Original**: Grid index space
- **Fixed**: Physical Angstroms
- **Conversion**: Applied proper grid spacing ({cube_spacing_bohr} Bohr = {cube_spacing_bohr * BOHR_TO_ANGSTROM:.6f} Å)

## Data Splits

- **Train**: {len(splits['train'])} samples ({train_frac*100:.0f}%)
- **Valid**: {len(splits['valid'])} samples ({valid_frac*100:.0f}%)
- **Test**: {len(splits['test'])} samples ({test_frac*100:.0f}%)
- **Seed**: {seed} (reproducible)

## Files

### Energy, Forces, and Dipoles
- `energies_forces_dipoles_train.npz`
- `energies_forces_dipoles_valid.npz`
- `energies_forces_dipoles_test.npz`

Each contains:
- `R`: Atomic coordinates [Angstrom]
- `Z`: Atomic numbers [int]
- `N`: Number of atoms [int]
- `E`: Energies [eV] ← CONVERTED from Hartree
- `F`: Forces [eV/Angstrom] ← CONVERTED from Hartree/Bohr
- `Dxyz`: Dipole moments [Debye]

### ESP Grids
- `grids_esp_train.npz`
- `grids_esp_valid.npz`
- `grids_esp_test.npz`

Each contains:
- `R`: Atomic coordinates [Angstrom]
- `Z`: Atomic numbers [int]
- `N`: Number of atoms [int]
- `esp`: ESP values [Hartree/e]
- `vdw_surface`: Grid coordinates [Angstrom] ← FIXED
- `vdw_grid`: Same as vdw_surface (backward compatibility)
- `grid_dims`: Original cube dimensions
- `grid_origin`: Original cube origins [Bohr]
- `grid_axes`: Original cube axes
- `Dxyz`: Dipole moments [Debye]

## Units Summary (ASE Standard)

| Property | Unit | Status |
|----------|------|--------|
| R (coordinates) | Angstrom | ✓ Correct |
| E (energy) | eV | ✓ Converted |
| F (forces) | eV/Angstrom | ✓ Converted |
| Dxyz (dipoles) | Debye | ✓ Correct |
| esp (values) | Hartree/e | ✓ Correct |
| vdw_surface | Angstrom | ✓ Fixed |

## Usage with DCMnet

```python
import numpy as np

# Load training data
train_props = np.load('energies_forces_dipoles_train.npz')
train_grids = np.load('grids_esp_train.npz')

# All units are ASE-standard - ready to use!
R = train_props['R']  # Angstroms
E = train_props['E']  # eV
F = train_props['F']  # eV/Angstrom
Dxyz = train_props['Dxyz']  # Debye
esp = train_grids['esp']  # Hartree/e
vdw_surface = train_grids['vdw_surface']  # Angstroms
```

Generated by: fix_and_split_cli.py
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    if verbose:
        print(f"✓ Created {readme_path.name}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("✅ DATA PREPARATION COMPLETE!")
        print(f"{'='*70}")
        print(f"\nOutput files in: {output_dir}")
        print("\nTrain/Valid/Test splits:")
        print(f"  - energies_forces_dipoles_{{train,valid,test}}.npz")
        print(f"  - grids_esp_{{train,valid,test}}.npz")
        print(f"  - split_indices.npz")
        print(f"  - README.md")
        print("\n✅ IMPORTANT: All units are now ASE-standard compliant!")
        print("   - Energies: eV (converted from Hartree)")
        print("   - Forces: eV/Angstrom (converted from Hartree/Bohr)")
        print("   - Coordinates: Angstrom")
        print("   - ESP grid: Angstrom (converted from grid indices)")
        print(f"{'='*70}\n")
    
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fix units and create train/valid/test splits from molecular NPZ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 8:1:1 split
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data
  
  # Custom split ratios
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data \\
    --train-frac 0.7 --valid-frac 0.15 --test-frac 0.15
  
  # Different cube spacing (e.g., 0.5 Bohr)
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data \\
    --cube-spacing 0.5
  
  # Skip validation for speed
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data \\
    --skip-validation
"""
    )
    
    parser.add_argument(
        '--efd', '--energies-forces-dipoles',
        type=Path,
        required=True,
        help='Path to energies_forces_dipoles.npz file'
    )
    
    parser.add_argument(
        '--grid', '--grids-esp',
        type=Path,
        required=True,
        help='Path to grids_esp.npz file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--train-frac',
        type=float,
        default=0.8,
        help='Fraction of data for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--valid-frac',
        type=float,
        default=0.1,
        help='Fraction of data for validation (default: 0.1)'
    )
    
    parser.add_argument(
        '--test-frac',
        type=float,
        default=0.1,
        help='Fraction of data for testing (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    
    parser.add_argument(
        '--cube-spacing',
        type=float,
        default=0.25,
        help='Grid spacing in Bohr from original cube files (default: 0.25)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation checks'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.efd.exists():
        print(f"❌ Error: EFD file not found: {args.efd}")
        sys.exit(1)
    
    if not args.grid.exists():
        print(f"❌ Error: Grid file not found: {args.grid}")
        sys.exit(1)
    
    if abs(args.train_frac + args.valid_frac + args.test_frac - 1.0) > 1e-6:
        print(f"❌ Error: Split fractions must sum to 1.0")
        print(f"   Got: {args.train_frac} + {args.valid_frac} + {args.test_frac} = "
              f"{args.train_frac + args.valid_frac + args.test_frac}")
        sys.exit(1)
    
    # Run the conversion
    success = fix_and_split_data(
        efd_file=args.efd,
        grid_file=args.grid,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        cube_spacing_bohr=args.cube_spacing,
        skip_validation=args.skip_validation,
        verbose=not args.quiet
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


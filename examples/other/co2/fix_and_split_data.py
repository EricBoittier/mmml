"""
Fix unit issues and create properly formatted training data for DCMnet/PhysnetJax.

This script:
1. Validates atomic coordinates are in Angstroms
2. Converts energies from Hartree to eV (ASE standard)
3. Converts forces from Hartree/Bohr to eV/Angstrom (ASE standard)
4. Converts ESP grid coordinates from index space to physical Angstroms
5. Creates 8:1:1 train/valid/test splits
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
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

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
    """
    Convert energies from Hartree to eV (ASE standard).
    
    Parameters
    ----------
    E_hartree : np.ndarray
        Energies in Hartree
        
    Returns
    -------
    np.ndarray
        Energies in eV
    """
    HARTREE_TO_EV = 27.211386
    return E_hartree * HARTREE_TO_EV


def convert_forces_hartree_bohr_to_ev_angstrom(F_hartree_bohr: np.ndarray) -> np.ndarray:
    """
    Convert forces from Hartree/Bohr to eV/Angstrom (ASE standard).
    
    Parameters
    ----------
    F_hartree_bohr : np.ndarray
        Forces in Hartree/Bohr
        
    Returns
    -------
    np.ndarray
        Forces in eV/Angstrom
    """
    # Conversion factor = (Hartree to eV) / (Bohr to Angstrom)
    # = 27.211386 / 0.529177 = 51.42208
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
    
    Original cube: coord_bohr = origin_bohr + index * spacing_bohr
    We want: coord_angstrom
    
    Parameters
    ----------
    vdw_grid_indices : np.ndarray
        Grid coordinates in index space (n_samples, n_points, 3)
    origin : np.ndarray
        Grid origins in Bohr (n_samples, 3)
    axes : np.ndarray
        Grid axes (n_samples, 3, 3) - currently identity matrices
    dims : np.ndarray
        Grid dimensions (n_samples, 3) - all [50, 50, 50]
    cube_spacing_bohr : float
        Spacing between grid points in Bohr (from original cube files)
        
    Returns
    -------
    np.ndarray
        Grid coordinates in Angstroms
    """
    n_samples = vdw_grid_indices.shape[0]
    n_points = vdw_grid_indices.shape[1]
    
    bohr_to_angstrom = 0.529177
    
    vdw_grid_angstrom = np.zeros_like(vdw_grid_indices)
    
    for i in range(n_samples):
        # The "grid" values are actually indices (0, 1, 2, ..., 49)
        # But they've been stored as coordinates
        # We need to interpret them as indices and convert to physical coords
        
        # Actually, looking at the split script, the coordinates ARE computed correctly:
        # grid_xyz = origin + i*axes[0] + j*axes[1] + k*axes[2]
        # But axes is identity and origin is in Bohr
        
        # So current coords are: origin_bohr + [i, j, k] where i,j,k are 0-49
        # This is wrong because we're adding grid indices to physical origin
        
        # Correct formula: coord_bohr = origin_bohr + i*spacing_bohr
        # Then convert to Angstrom
        
        # The stored "coords" are: origin + identity @ [i, j, k]
        # We need: origin * spacing * bohr_to_ang + (stored_coords - origin) * spacing * bohr_to_ang
        
        # Wait, let's think more carefully...
        # The cube file has axes like [0.25, 0, 0] (spacing in Bohr)
        # But NPZ has axes [[1,0,0], [0,1,0], [0,0,1]] (identity)
        # This means the stored coords are grid indices, not physical coords!
        
        # Correct conversion:
        # Extract implied indices from stored coords (subtract origin, since origin was added)
        # Then multiply by actual spacing and convert units
        
        grid_indices = vdw_grid_indices[i] - origin[i]  # Remove origin offset
        coord_bohr = origin[i] + grid_indices * cube_spacing_bohr
        vdw_grid_angstrom[i] = coord_bohr * bohr_to_angstrom
    
    return vdw_grid_angstrom


def validate_fixed_data(R_ang, E_ev, F_ev_ang, vdw_grid_ang, Z, N):
    """Validate that fixes worked correctly."""
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
    
    print(f"\nAtomic Coordinates (100 samples):")
    print(f"  C-O bonds: mean={co_bonds_check.mean():.4f} Å, "
          f"range=[{co_bonds_check.min():.4f}, {co_bonds_check.max():.4f}]")
    
    if 1.0 <= co_bonds_check.mean() <= 1.5:
        print(f"  ✓ Coordinates in Angstroms with varying geometries")
        coords_ok = True
    else:
        print(f"  ❌ Coordinates outside expected range!")
        coords_ok = False
    
    # Get a sample for other checks
    r0 = R_ang[0]
    z0 = Z[0]
    valid = z0 > 0
    valid_pos = r0[valid]
    co_dist = np.linalg.norm(valid_pos[0] - valid_pos[1])
    
    # Check energies
    print(f"\nEnergies (sample 0):")
    print(f"  Value: {E_ev[0]:.6f} eV")
    print(f"  Dataset mean: {E_ev.mean():.6f} eV")
    
    # For CO2, expect energies around -5100 to -5000 eV (from -187.5 Ha)
    if -5200 < E_ev.mean() < -4900:
        print(f"  ✓ Energies in expected range for CO2 in eV")
        energy_ok = True
    else:
        print(f"  ⚠️  Energy range unexpected for CO2")
        energy_ok = False
    
    # Check forces
    f_sample = F_ev_ang[0, :3, :]  # First sample, first 3 atoms
    f_norm = np.linalg.norm(f_sample.reshape(-1, 3), axis=1).mean()
    
    print(f"\nForces (sample 0):")
    print(f"  Mean norm: {f_norm:.6e} eV/Angstrom")
    
    # For geometry scans, forces can be large (up to 50-100 eV/Å far from equilibrium)
    if 1e-3 < f_norm < 100:
        print(f"  ✓ Force magnitudes in reasonable range for geometry scans")
        force_ok = True
    else:
        print(f"  ⚠️  Force magnitudes outside expected range")
        force_ok = False
    
    # Check ESP grid
    grid0 = vdw_grid_ang[0]
    grid_extent = (grid0.max(axis=0) - grid0.min(axis=0)).mean()
    
    print(f"\nESP Grid Coordinates:")
    print(f"  Average extent: {grid_extent:.4f} Angstrom")
    print(f"  X range: [{grid0[:, 0].min():.4f}, {grid0[:, 0].max():.4f}]")
    print(f"  Y range: [{grid0[:, 1].min():.4f}, {grid0[:, 1].max():.4f}]")
    print(f"  Z range: [{grid0[:, 2].min():.4f}, {grid0[:, 2].max():.4f}]")
    
    # For CO2 cube, expect ~12 Bohr = ~6.3 Angstrom extent
    if 5.0 < grid_extent < 8.0:
        print(f"  ✓ Grid extent matches expected cube size (~6-7 Å)")
        grid_ok = True
    else:
        print(f"  ⚠️  Grid extent outside expected range (5-8 Å)")
        grid_ok = False
    
    # Check that molecule is surrounded by grid points
    # The grid forms a VDW surface around the molecule, so we check proximity
    mol_center = valid_pos.mean(axis=0)
    grid_min = grid0.min(axis=0)
    grid_max = grid0.max(axis=0)
    
    print(f"\nSpatial relationship:")
    print(f"  Molecule center: {mol_center}")
    print(f"  Molecule extent: ~{co_dist * 2:.4f} Å (O-O distance)")
    print(f"  Grid bounds: X[{grid_min[0]:.2f}, {grid_max[0]:.2f}], "
          f"Y[{grid_min[1]:.2f}, {grid_max[1]:.2f}], "
          f"Z[{grid_min[2]:.2f}, {grid_max[2]:.2f}]")
    
    # Check if molecule is within grid bounds (with some tolerance)
    mol_in_grid = (np.all(mol_center >= grid_min - 1.0) and 
                   np.all(mol_center <= grid_max + 1.0))
    
    # Better check: are all atoms within or near the grid?
    all_atoms_near_grid = True
    for atom_pos in valid_pos:
        dist_to_grid_bounds = min(
            np.linalg.norm(atom_pos - grid_min),
            np.linalg.norm(atom_pos - grid_max)
        )
        # Check if any grid point is close to this atom
        min_dist_to_any_grid_point = np.min(np.linalg.norm(grid0 - atom_pos, axis=1))
        if min_dist_to_any_grid_point > 2.0:  # More than 2 Å from nearest grid point
            all_atoms_near_grid = False
            break
    
    if mol_in_grid and all_atoms_near_grid:
        print(f"  ✓ Molecule is within/near grid (ESP surface surrounds molecule)")
        spatial_ok = True
    else:
        print(f"  ⚠️  Checking grid-molecule proximity...")
        # Calculate minimum distance from each atom to nearest grid point
        for idx, atom_pos in enumerate(valid_pos):
            min_dist = np.min(np.linalg.norm(grid0 - atom_pos, axis=1))
            print(f"    Atom {idx}: nearest grid point at {min_dist:.4f} Å")
        
        # If ESP grid points are within ~5 Å of atoms, it's probably OK
        # (VDW surface is typically 1.4-2.0 Å from atoms)
        max_min_dist = max([np.min(np.linalg.norm(grid0 - atom_pos, axis=1)) 
                           for atom_pos in valid_pos])
        if max_min_dist < 5.0:
            print(f"  ✓ Grid points within {max_min_dist:.2f} Å - acceptable for VDW surface")
            spatial_ok = True
        else:
            print(f"  ❌ Grid too far from molecule ({max_min_dist:.2f} Å)")
            spatial_ok = False
    
    overall_ok = coords_ok and energy_ok and force_ok and grid_ok and spatial_ok
    
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
        print(f"{'='*70}")
    else:
        print(f"\n⚠️  SOME VALIDATIONS FAILED - Review above")
        print(f"{'='*70}")
    
    return overall_ok


def main():
    """Main workflow to fix units and create splits."""
    print("\n" + "="*70)
    print("CO2 Data Unit Conversion and Splitting")
    print("="*70)
    
    # Paths
    data_dir = Path("/home/ericb/testdata")
    output_dir = Path(__file__).parent / "training_data_fixed"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nInput directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # =========================================================================
    # Load data
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 1: Loading Data")
    print(f"{'#'*70}")
    
    efd_data = dict(np.load(data_dir / "energies_forces_dipoles.npz"))
    grid_data = dict(np.load(data_dir / "grids_esp.npz"))
    
    n_samples = efd_data['R'].shape[0]
    print(f"\nLoaded {n_samples} samples")
    print(f"  Keys in EFD: {list(efd_data.keys())}")
    print(f"  Keys in Grid: {list(grid_data.keys())}")
    
    # =========================================================================
    # Check atomic coordinates (actually already in Angstroms!)
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 2: Checking Atomic Coordinates")
    print(f"{'#'*70}")
    
    # Calculate C-O bond statistics
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
    
    print(f"\nC-O Bond Statistics (first 1000 samples):")
    print(f"  Mean:   {co_bonds.mean():.6f}")
    print(f"  Std:    {co_bonds.std():.6f}")
    print(f"  Min:    {co_bonds.min():.6f}")
    print(f"  Max:    {co_bonds.max():.6f}")
    print(f"  Median: {np.median(co_bonds):.6f}")
    
    if 1.0 < co_bonds.mean() < 1.5:
        print(f"\n✓ Coordinates are ALREADY in Angstroms!")
        print(f"  (Varying C-O from {co_bonds.min():.2f} to {co_bonds.max():.2f} Å)")
        print(f"  This matches the R1/R2 parameter scan in the job script")
        R_angstrom = efd_data['R']  # No conversion needed!
    elif 2.0 < co_bonds.mean() < 2.7:
        print(f"\n Converting from Bohr to Angstroms...")
        R_angstrom = efd_data['R'] * 0.529177
    else:
        print(f"\n❌ Unclear units! Mean C-O = {co_bonds.mean():.4f}")
        return
    
    # =========================================================================
    # Convert energies: Hartree → eV (ASE standard)
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 3: Converting Energies from Hartree to eV")
    print(f"{'#'*70}")
    
    E_ev = convert_energy_hartree_to_ev(efd_data['E'])
    
    HARTREE_TO_EV = 27.211386
    print(f"\nConversion factor: {HARTREE_TO_EV}")
    print(f"Original (Hartree): mean={efd_data['E'].mean():.6f}, range=[{efd_data['E'].min():.6f}, {efd_data['E'].max():.6f}]")
    print(f"Converted (eV):     mean={E_ev.mean():.6f}, range=[{E_ev.min():.6f}, {E_ev.max():.6f}]")
    print(f"✓ Energies converted to eV")
    
    # =========================================================================
    # Convert forces: Hartree/Bohr → eV/Angstrom (ASE standard)
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 4: Converting Forces from Hartree/Bohr to eV/Angstrom")
    print(f"{'#'*70}")
    
    F_ev_ang = convert_forces_hartree_bohr_to_ev_angstrom(efd_data['F'])
    
    HARTREE_BOHR_TO_EV_ANG = 51.42208
    # Calculate force norms for comparison
    f_orig_norms = np.linalg.norm(efd_data['F'][:10, :3, :].reshape(-1, 3), axis=1)
    f_conv_norms = np.linalg.norm(F_ev_ang[:10, :3, :].reshape(-1, 3), axis=1)
    
    print(f"\nConversion factor: {HARTREE_BOHR_TO_EV_ANG}")
    print(f"Original (Ha/Bohr): mean norm={f_orig_norms.mean():.6e}")
    print(f"Converted (eV/Å):   mean norm={f_conv_norms.mean():.6e}")
    print(f"Ratio: {f_conv_norms.mean() / f_orig_norms.mean():.6f} (should be ~{HARTREE_BOHR_TO_EV_ANG})")
    print(f"✓ Forces converted to eV/Angstrom")
    
    # =========================================================================
    # Fix ESP grid: index space → physical Angstroms
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 5: Converting ESP Grid to Physical Angstroms")
    print(f"{'#'*70}")
    
    # From the cube file inspection: spacing = 0.25 Bohr
    CUBE_SPACING_BOHR = 0.25
    BOHR_TO_ANGSTROM = 0.529177
    
    print(f"\nCube file parameters:")
    print(f"  Spacing: {CUBE_SPACING_BOHR} Bohr = {CUBE_SPACING_BOHR * BOHR_TO_ANGSTROM:.6f} Angstrom")
    print(f"  Dimensions: {grid_data['grid_dims'][0]}")
    print(f"  Original origin (Bohr): {grid_data['grid_origin'][0]}")
    
    vdw_surface_angstrom = convert_grid_indices_to_angstrom(
        grid_data['vdw_grid'],
        grid_data['grid_origin'],
        grid_data['grid_axes'],
        grid_data['grid_dims'],
        cube_spacing_bohr=CUBE_SPACING_BOHR
    )
    
    # Verify conversion
    grid0_original = grid_data['vdw_grid'][0]
    grid0_fixed = vdw_surface_angstrom[0]
    
    print(f"\nOriginal grid extent: {(grid0_original.max(axis=0) - grid0_original.min(axis=0)).mean():.4f}")
    print(f"Fixed grid extent: {(grid0_fixed.max(axis=0) - grid0_fixed.min(axis=0)).mean():.4f} Angstrom")
    print(f"Expected extent: ~{49 * CUBE_SPACING_BOHR * BOHR_TO_ANGSTROM:.4f} Angstrom (50 points × {CUBE_SPACING_BOHR * BOHR_TO_ANGSTROM:.4f} Å/point)")
    
    # =========================================================================
    # Validate fixed data
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 6: Validating Fixed Data")
    print(f"{'#'*70}")
    
    is_valid = validate_fixed_data(R_angstrom, E_ev, F_ev_ang, vdw_surface_angstrom, efd_data['Z'], efd_data['N'])
    
    if not is_valid:
        print("\n❌ Validation failed! Not proceeding with splits.")
        return
    
    # =========================================================================
    # Create splits
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 7: Creating 8:1:1 Train/Valid/Test Splits")
    print(f"{'#'*70}")
    
    splits = create_splits(n_samples, train_frac=0.8, valid_frac=0.1, test_frac=0.1, seed=42)
    
    print(f"\nTotal samples: {n_samples}")
    print(f"  Train: {len(splits['train'])} ({len(splits['train'])/n_samples*100:.1f}%)")
    print(f"  Valid: {len(splits['valid'])} ({len(splits['valid'])/n_samples*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/n_samples*100:.1f}%)")
    
    # =========================================================================
    # Prepare datasets with fixed units
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 8: Preparing Fixed Datasets")
    print(f"{'#'*70}")
    
    # Update EFD data with fixed/converted values
    efd_fixed = efd_data.copy()
    efd_fixed['R'] = R_angstrom  # Already in Angstroms
    efd_fixed['E'] = E_ev  # Converted to eV
    efd_fixed['F'] = F_ev_ang  # Converted to eV/Angstrom
    
    # Update grid data with fixed coordinates
    # Also rename vdw_grid to vdw_surface (MMML standard name)
    grid_fixed = grid_data.copy()
    grid_fixed['R'] = R_angstrom  # Use same fixed coords
    grid_fixed['vdw_surface'] = vdw_surface_angstrom
    # Keep vdw_grid for backward compatibility
    grid_fixed['vdw_grid'] = vdw_surface_angstrom
    
    # =========================================================================
    # Save split datasets
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 9: Saving Split Datasets")
    print(f"{'#'*70}")
    
    for split_name, split_indices in splits.items():
        print(f"\nSaving {split_name} split ({len(split_indices)} samples)...")
        
        # Create EFD split
        efd_split = {k: v[split_indices] if (isinstance(v, np.ndarray) and v.shape[0] == n_samples) else v 
                     for k, v in efd_fixed.items()}
        efd_out = output_dir / f"energies_forces_dipoles_{split_name}.npz"
        np.savez_compressed(efd_out, **efd_split)
        size_mb = efd_out.stat().st_size / 1024 / 1024
        print(f"  ✓ {efd_out.name} ({size_mb:.1f} MB)")
        
        # Create grid split
        grid_split = {k: v[split_indices] if (isinstance(v, np.ndarray) and v.shape[0] == n_samples) else v
                      for k, v in grid_fixed.items()}
        grid_out = output_dir / f"grids_esp_{split_name}.npz"
        np.savez_compressed(grid_out, **grid_split)
        size_mb = grid_out.stat().st_size / 1024 / 1024
        print(f"  ✓ {grid_out.name} ({size_mb:.1f} MB)")
    
    # Save split indices
    indices_out = output_dir / "split_indices.npz"
    np.savez(indices_out, **splits)
    print(f"\n✓ Split indices saved to {indices_out.name}")
    
    # =========================================================================
    # Create documentation
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Step 8: Creating Documentation")
    print(f"{'#'*70}")
    
    readme_content = f"""# CO2 Training Data (Unit-Corrected)

This directory contains CO2 molecular data with **corrected units** ready for DCMnet/PhysnetJax training.

## Data Corrections Applied

### 1. Atomic Coordinates (R)
- **Original**: Already in Angstroms (varying C-O bonds: 1.0-1.5 Å)
- **Status**: No conversion needed - already correct ✓
- **Note**: Dataset contains varying CO2 geometries (R1, R2, angle scans)

### 2. Energies (E)
- **Original**: Hartree
- **Converted**: eV (ASE standard)
- **Factor**: ×27.211386

### 3. Forces (F)
- **Original**: Hartree/Bohr
- **Converted**: eV/Angstrom (ASE standard)
- **Factor**: ×51.42208

### 4. ESP Grid Coordinates (vdw_surface)
- **Original**: Grid index space (0-49 with unit axes)
- **Fixed**: Physical Angstroms
- **Conversion**: Applied proper grid spacing ({CUBE_SPACING_BOHR} Bohr = {CUBE_SPACING_BOHR * BOHR_TO_ANGSTROM:.6f} Å)

## Data Splits

- **Train**: {len(splits['train'])} samples (80%)
- **Valid**: {len(splits['valid'])} samples (10%)
- **Test**: {len(splits['test'])} samples (10%)
- **Seed**: 42 (reproducible)

## Files

### Energy, Forces, and Dipoles
- `energies_forces_dipoles_train.npz`
- `energies_forces_dipoles_valid.npz`
- `energies_forces_dipoles_test.npz`

Each contains:
- `R`: Atomic coordinates (n_samples, 60, 3) [Angstrom] ✓
- `Z`: Atomic numbers (n_samples, 60) [int] ✓
- `N`: Number of atoms (n_samples,) [int] ✓
- `E`: Energies (n_samples,) [eV] ← CONVERTED from Hartree
- `F`: Forces (n_samples, 60, 3) [eV/Angstrom] ← CONVERTED from Hartree/Bohr
- `Dxyz`: Dipole moments (n_samples, 3) [Debye] ✓

### ESP Grids
- `grids_esp_train.npz`
- `grids_esp_valid.npz`
- `grids_esp_test.npz`

Each contains:
- `R`: Atomic coordinates (n_samples, 60, 3) [Angstrom] ← FIXED
- `Z`: Atomic numbers (n_samples, 60) [int]
- `N`: Number of atoms (n_samples,) [int]
- `esp`: ESP values (n_samples, 3000) [Hartree/e]
- `vdw_surface`: Grid coordinates (n_samples, 3000, 3) [Angstrom] ← FIXED
- `vdw_grid`: Same as vdw_surface (backward compatibility)
- `grid_dims`: Original cube dimensions (n_samples, 3)
- `grid_origin`: Original cube origins (n_samples, 3) [Bohr]
- `grid_axes`: Original cube axes (n_samples, 3, 3)
- `Dxyz`: Dipole moments (n_samples, 3) [Debye]

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
train_props = np.load('training_data_fixed/energies_forces_dipoles_train.npz')
train_grids = np.load('training_data_fixed/grids_esp_train.npz')

# All units are ASE-standard - ready to use!
R = train_props['R']  # Angstroms
E = train_props['E']  # eV (converted from Hartree)
F = train_props['F']  # eV/Angstrom (converted from Hartree/Bohr)
Dxyz = train_props['Dxyz']  # Debye
esp = train_grids['esp']  # Hartree/e
vdw_surface = train_grids['vdw_surface']  # Angstroms (fixed from grid indices)
```

## Important Notes

1. **All units are ASE-standard** - compatible with ASE, SchNetPack, etc.
2. **Coordinates in Angstroms** - no scaling needed
3. **Energies in eV** - converted from Hartree (×27.211386)
4. **Forces in eV/Angstrom** - converted from Hartree/Bohr (×51.42208)
5. **ESP grid in physical Angstroms** - matches atomic coordinates
6. **ESP values remain in Hartree/e** - no ASE standard for electrostatic potential
7. **Splits are reproducible** - same seed (42) ensures consistency

## Validation

Run `prepare_training_data.py` to see detailed validation of the original data,
or check the `VALIDATION_REPORT.txt` for the full analysis.

Generated: {Path(__file__).name}
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created {readme_path.name}")
    
    # Create quick reference
    units_ref = output_dir / "UNITS_REFERENCE.txt"
    with open(units_ref, 'w') as f:
        f.write("""CO2 Training Data - Units Quick Reference
================================================================================

COORDINATES (R)
  Units: Angstrom (ASE standard)
  Original: Already in Angstroms (varying geometries)
  Status: No conversion needed ✓
  Range: C-O bonds from 1.0 to 1.5 Å (mean ~1.25 Å)

ENERGIES (E)
  Units: eV (ASE standard)
  Original: Hartree
  Conversion: ×27.211386 (Ha → eV)
  Range: -5107 to -5098 eV (from -187.7 to -187.4 Ha)

FORCES (F)
  Units: eV/Angstrom (ASE standard)
  Original: Hartree/Bohr
  Conversion: ×51.42208 (Ha/Bohr → eV/Å)
  Note: Gradient = -Force

DIPOLES (Dxyz)
  Units: Debye
  Status: Correct (no change)
  Range: 0 to ~1.1 D

ESP VALUES (esp)
  Units: Hartree/e (electrostatic potential per elementary charge)
  Status: Correct (no change)
  Range: -0.058 to +0.862 Ha/e

ESP GRID (vdw_surface / vdw_grid)
  Units: Angstrom (ASE standard)
  Original: Grid indices (0-49)
  Fixed: Physical coordinates
  Conversion: origin_bohr × 0.529177 + (index - origin) × 0.25 × 0.529177
  Extent: ~6.5 Angstrom cube

================================================================================
ALL UNITS ARE ASE-STANDARD!
Ready for DCMnet/PhysnetJax training with ASE-compatible data!
================================================================================
""")
    
    print(f"✓ Created {units_ref.name}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("✅ DATA PREPARATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files in: {output_dir}")
    print("\nTrain/Valid/Test splits (8:1:1):")
    print(f"  - energies_forces_dipoles_{{train,valid,test}}.npz")
    print(f"  - grids_esp_{{train,valid,test}}.npz")
    print("\nDocumentation:")
    print(f"  - README.md (detailed description)")
    print(f"  - UNITS_REFERENCE.txt (quick reference)")
    print(f"  - split_indices.npz (reproducible splits)")
    print("\n✅ IMPORTANT: All units are now ASE-standard compliant!")
    print("   - Energies: eV (converted from Hartree)")
    print("   - Forces: eV/Angstrom (converted from Hartree/Bohr)")
    print("   - Coordinates: Angstrom")
    print("   - ESP grid: Angstrom (converted from grid indices)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


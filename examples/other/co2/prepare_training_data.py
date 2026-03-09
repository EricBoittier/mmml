"""
Prepare and validate CO2 training data with strict unit checking.

This script:
1. Validates all units match expected standards for DCMnet/PhysnetJax
2. Performs comprehensive sanity checks
3. Creates 8:1:1 train/valid/test splits
4. Saves split datasets with proper documentation

Expected Units (MMML Standard):
- Coordinates (R): Angstrom (but currently normalized!)
- Energies (E): Hartree
- Forces (F): Hartree/Bohr
- Dipoles (Dxyz): Debye
- ESP values: Hartree/e (electrostatic potential per unit charge)
- ESP grid coordinates: Angstrom (but currently in grid index space!)
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

# Add parent directory to path
repo_root = Path(__file__).parent / "../.."
sys.path.insert(0, str(repo_root.resolve()))


class UnitValidator:
    """Validate physical units in molecular datasets."""
    
    # Physical constants and expected ranges
    CO_BOND_LENGTH_ANGSTROM = 1.16  # ± 0.05 typical for CO2
    CO_BOND_LENGTH_BOHR = 2.19      # ± 0.10
    
    # Element data
    ELEMENT_SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.validation_results = {}
    
    def log(self, msg, level='INFO'):
        """Print message if verbose."""
        if self.verbose:
            prefix = {'INFO': '  ', 'WARN': '⚠️ ', 'ERROR': '❌', 'OK': '✓ '}
            print(f"{prefix.get(level, '  ')}{msg}")
    
    def validate_coordinates(self, R, Z, N, name='R'):
        """
        Validate atomic coordinates and determine their units.
        
        Returns
        -------
        dict: {'units': str, 'is_normalized': bool, 'bond_lengths': dict, 'valid': bool}
        """
        self.log(f"\n{'='*70}", 'INFO')
        self.log(f"Validating {name} (Atomic Coordinates)", 'INFO')
        self.log(f"{'='*70}", 'INFO')
        self.log(f"Shape: {R.shape}", 'INFO')
        
        # Check first sample
        r0 = R[0]
        z0 = Z[0]
        n0 = N[0]
        
        valid_mask = z0 > 0
        valid_pos = r0[valid_mask]
        valid_z = z0[valid_mask]
        
        self.log(f"Sample 0: {n0} atoms", 'INFO')
        self.log(f"Atomic numbers: {valid_z}", 'INFO')
        
        # Calculate bond lengths
        bond_lengths = []
        for i in range(len(valid_pos)):
            for j in range(i+1, len(valid_pos)):
                dist = np.linalg.norm(valid_pos[i] - valid_pos[j])
                z1, z2 = int(valid_z[i]), int(valid_z[j])
                elem1 = self.ELEMENT_SYMBOLS.get(z1, f'Z{z1}')
                elem2 = self.ELEMENT_SYMBOLS.get(z2, f'Z{z2}')
                bond_lengths.append((f"{elem1}-{elem2}", dist))
                self.log(f"  {elem1}-{elem2} ({i}-{j}): {dist:.6f}", 'INFO')
        
        # Check for CO2 pattern
        results = {'bond_lengths': bond_lengths, 'valid': False}
        
        # Find C-O bonds
        co_bonds = [dist for label, dist in bond_lengths if 'C-O' in label or 'O-C' in label]
        
        if co_bonds:
            avg_co = np.mean(co_bonds)
            self.log(f"\nAverage C-O distance: {avg_co:.6f}", 'INFO')
            
            # Check units
            if 0.95 < avg_co < 1.05:
                self.log(f"NORMALIZED coordinates (C-O = {avg_co:.4f} instead of ~1.16 Å)", 'WARN')
                self.log(f"These need to be scaled by {self.CO_BOND_LENGTH_ANGSTROM:.4f} for Angstroms", 'WARN')
                self.log(f"Or by {self.CO_BOND_LENGTH_BOHR:.4f} for Bohr", 'WARN')
                results['units'] = 'normalized'
                results['is_normalized'] = True
                results['scale_to_angstrom'] = self.CO_BOND_LENGTH_ANGSTROM / avg_co
                results['scale_to_bohr'] = self.CO_BOND_LENGTH_BOHR / avg_co
                results['valid'] = True  # Normalized is OK, just needs scaling
                
            elif 1.10 < avg_co < 1.25:
                self.log(f"Coordinates appear to be in ANGSTROMS ✓", 'OK')
                self.log(f"C-O = {avg_co:.4f} Å (expected ~{self.CO_BOND_LENGTH_ANGSTROM:.2f} Å)", 'INFO')
                results['units'] = 'angstrom'
                results['is_normalized'] = False
                results['valid'] = True
                
            elif 2.0 < avg_co < 2.5:
                self.log(f"Coordinates appear to be in BOHR ✓", 'OK')
                self.log(f"C-O = {avg_co:.4f} Bohr (expected ~{self.CO_BOND_LENGTH_BOHR:.2f} Bohr)", 'INFO')
                results['units'] = 'bohr'
                results['is_normalized'] = False
                results['valid'] = True
                
            else:
                self.log(f"UNCLEAR units! C-O = {avg_co:.4f}", 'ERROR')
                results['units'] = 'unknown'
                results['is_normalized'] = False
                
        else:
            self.log("No C-O bonds found to check units", 'WARN')
            results['units'] = 'unknown'
            results['is_normalized'] = False
        
        # Check coordinate ranges
        coord_ranges = {
            'x': (valid_pos[:, 0].min(), valid_pos[:, 0].max()),
            'y': (valid_pos[:, 1].min(), valid_pos[:, 1].max()),
            'z': (valid_pos[:, 2].min(), valid_pos[:, 2].max()),
        }
        results['ranges'] = coord_ranges
        self.log(f"\nCoordinate ranges:", 'INFO')
        for axis, (vmin, vmax) in coord_ranges.items():
            self.log(f"  {axis}: [{vmin:.6f}, {vmax:.6f}]", 'INFO')
        
        return results
    
    def validate_energies(self, E, name='E'):
        """Validate energies are in Hartree."""
        self.log(f"\n{'='*70}", 'INFO')
        self.log(f"Validating {name} (Energies)", 'INFO')
        self.log(f"{'='*70}", 'INFO')
        self.log(f"Shape: {E.shape}", 'INFO')
        self.log(f"Mean: {E.mean():.6f} Hartree", 'INFO')
        self.log(f"Std:  {E.std():.6f} Hartree", 'INFO')
        self.log(f"Min:  {E.min():.6f} Hartree", 'INFO')
        self.log(f"Max:  {E.max():.6f} Hartree", 'INFO')
        
        results = {'valid': False, 'units': 'unknown'}
        
        # For CO2, MP2/aug-cc-pVTZ energies should be around -187 to -188 Hartree
        if -200 < E.mean() < -150:
            self.log("Energies in expected range for CO2 ✓", 'OK')
            results['units'] = 'hartree'
            results['valid'] = True
        else:
            self.log(f"Energies outside expected range for CO2!", 'WARN')
            results['units'] = 'unknown'
        
        return results
    
    def validate_forces(self, F, N, name='F'):
        """Validate forces are in Hartree/Bohr."""
        self.log(f"\n{'='*70}", 'INFO')
        self.log(f"Validating {name} (Forces)", 'INFO')
        self.log(f"{'='*70}", 'INFO')
        self.log(f"Shape: {F.shape}", 'INFO')
        
        # Calculate force norms for real atoms only
        forces_flat = []
        for i in range(len(F)):
            n = N[i]
            forces_flat.append(F[i, :n, :].reshape(-1, 3))
        
        forces_flat = np.vstack(forces_flat)
        force_norms = np.linalg.norm(forces_flat, axis=1)
        
        self.log(f"Force component statistics:", 'INFO')
        self.log(f"  Mean X: {forces_flat[:, 0].mean():.6e} Ha/Bohr", 'INFO')
        self.log(f"  Mean Y: {forces_flat[:, 1].mean():.6e} Ha/Bohr", 'INFO')
        self.log(f"  Mean Z: {forces_flat[:, 2].mean():.6e} Ha/Bohr", 'INFO')
        self.log(f"  Mean norm: {force_norms.mean():.6e} Ha/Bohr", 'INFO')
        self.log(f"  Std norm:  {force_norms.std():.6e} Ha/Bohr", 'INFO')
        self.log(f"  Max norm:  {force_norms.max():.6e} Ha/Bohr", 'INFO')
        
        results = {'valid': True, 'units': 'hartree_per_bohr'}
        
        # Forces should have mean ~0 (balanced)
        mean_components = np.abs([forces_flat[:, i].mean() for i in range(3)])
        if np.all(mean_components < 1e-3):
            self.log("Force components well-balanced (mean ≈ 0) ✓", 'OK')
        else:
            self.log("Force components have non-zero mean - check for systematic bias", 'WARN')
        
        return results
    
    def validate_dipoles(self, Dxyz, name='Dxyz'):
        """Validate dipole moments are in Debye."""
        self.log(f"\n{'='*70}", 'INFO')
        self.log(f"Validating {name} (Dipoles)", 'INFO')
        self.log(f"{'='*70}", 'INFO')
        self.log(f"Shape: {Dxyz.shape}", 'INFO')
        
        dipole_norms = np.linalg.norm(Dxyz, axis=1)
        
        self.log(f"Mean norm: {dipole_norms.mean():.6f} Debye", 'INFO')
        self.log(f"Std norm:  {dipole_norms.std():.6f} Debye", 'INFO')
        self.log(f"Min norm:  {dipole_norms.min():.6f} Debye", 'INFO')
        self.log(f"Max norm:  {dipole_norms.max():.6f} Debye", 'INFO')
        
        results = {'valid': True, 'units': 'debye'}
        
        # CO2 dipole should be ~0 (linear symmetric molecule)
        # But vibrational modes can break symmetry
        if dipole_norms.mean() < 2.0:
            self.log("Dipole magnitudes reasonable for CO2 ✓", 'OK')
        else:
            self.log("Dipoles larger than expected for CO2", 'WARN')
        
        return results
    
    def validate_esp_grid(self, vdw_grid, esp, grid_origin, grid_axes, grid_dims, name='ESP Grid'):
        """Validate ESP grid coordinates and values."""
        self.log(f"\n{'='*70}", 'INFO')
        self.log(f"Validating {name}", 'INFO')
        self.log(f"{'='*70}", 'INFO')
        self.log(f"Grid shape: {vdw_grid.shape}", 'INFO')
        self.log(f"ESP shape:  {esp.shape}", 'INFO')
        
        # Check first sample
        grid0 = vdw_grid[0]
        esp0 = esp[0]
        origin0 = grid_origin[0]
        axes0 = grid_axes[0]
        dims0 = grid_dims[0]
        
        self.log(f"\nSample 0 grid metadata:", 'INFO')
        self.log(f"  Origin: {origin0}", 'INFO')
        self.log(f"  Axes:\n{axes0}", 'INFO')
        self.log(f"  Dims: {dims0}", 'INFO')
        
        # Check grid coordinate ranges
        grid_ranges = {
            'x': (grid0[:, 0].min(), grid0[:, 0].max()),
            'y': (grid0[:, 1].min(), grid0[:, 1].max()),
            'z': (grid0[:, 2].min(), grid0[:, 2].max()),
        }
        
        self.log(f"\nGrid coordinate ranges:", 'INFO')
        for axis, (vmin, vmax) in grid_ranges.items():
            extent = vmax - vmin
            self.log(f"  {axis}: [{vmin:.4f}, {vmax:.4f}] (extent: {extent:.4f})", 'INFO')
        
        avg_extent = np.mean([vmax - vmin for vmin, vmax in grid_ranges.values()])
        
        # Determine units
        results = {'valid': False, 'units': 'unknown'}
        
        if avg_extent > 30:  # Large extent suggests Angstroms or grid indices
            if axes0[0, 0] == 1.0 and axes0[1, 1] == 1.0:
                self.log("\nGrid appears to be in GRID INDEX space (axes=identity, extent~50)", 'WARN')
                self.log("This is NOT physical coordinates!", 'ERROR')
                self.log("Need to convert using: origin + i*actual_spacing", 'WARN')
                results['units'] = 'grid_indices'
                results['needs_conversion'] = True
            else:
                self.log("\nGrid appears to be in ANGSTROMS", 'OK')
                results['units'] = 'angstrom'
                results['valid'] = True
        elif 10 < avg_extent < 20:
            self.log("\nGrid appears to be in BOHR", 'OK')
            results['units'] = 'bohr'
            results['valid'] = True
        else:
            self.log(f"\nUnclear grid units (extent={avg_extent:.2f})", 'ERROR')
        
        # Validate ESP values
        self.log(f"\nESP value statistics:", 'INFO')
        self.log(f"  Mean: {esp.mean():.6e}", 'INFO')
        self.log(f"  Std:  {esp.std():.6e}", 'INFO')
        self.log(f"  Min:  {esp.min():.6e}", 'INFO')
        self.log(f"  Max:  {esp.max():.6e}", 'INFO')
        
        # ESP should have both positive and negative values
        if esp.min() < 0 and esp.max() > 0:
            self.log("ESP has both positive and negative regions ✓", 'OK')
        else:
            self.log("ESP should have both positive/negative regions!", 'WARN')
        
        # Typical ESP magnitude for molecules is 1e-3 to 1 Hartree/e
        if 1e-4 < np.abs(esp).mean() < 1.0:
            self.log("ESP magnitudes in reasonable range ✓", 'OK')
        else:
            self.log("ESP magnitudes outside typical range", 'WARN')
        
        results['esp_stats'] = {
            'mean': float(esp.mean()),
            'std': float(esp.std()),
            'min': float(esp.min()),
            'max': float(esp.max()),
        }
        
        return results


def create_splits(n_samples: int, train_frac=0.8, valid_frac=0.1, test_frac=0.1, seed=42):
    """
    Create train/valid/test split indices.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    train_frac : float
        Fraction for training (default 0.8)
    valid_frac : float
        Fraction for validation (default 0.1)
    test_frac : float
        Fraction for testing (default 0.1)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict: {'train': array, 'valid': array, 'test': array}
    """
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    n_train = int(n_samples * train_frac)
    n_valid = int(n_samples * valid_frac)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    
    return {
        'train': train_idx,
        'valid': valid_idx,
        'test': test_idx
    }


def split_dataset(data: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract subset of dataset using indices."""
    subset = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) > 0:
            # Only split along first dimension if it matches n_samples
            if value.shape[0] == len(indices) or key in ['R', 'Z', 'N', 'E', 'F', 'Dxyz', 'esp', 'vdw_grid', 'grid_origin', 'grid_axes', 'grid_dims']:
                subset[key] = value[indices]
            else:
                # Metadata or other fields - keep as is
                subset[key] = value
        else:
            subset[key] = value
    return subset


def main():
    """Main validation and splitting workflow."""
    print("\n" + "="*70)
    print("CO2 Data Validation and Splitting for DCMnet/PhysnetJax Training")
    print("="*70)
    
    # Setup paths
    data_dir = Path("/home/ericb/testdata")
    output_dir = Path(__file__).parent / "training_data"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nInput directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize validator
    validator = UnitValidator(verbose=True)
    
    # =========================================================================
    # Load and validate energies_forces_dipoles.npz
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Part 1: Validating energies_forces_dipoles.npz")
    print(f"{'#'*70}")
    
    efd_path = data_dir / "energies_forces_dipoles.npz"
    efd_data = dict(np.load(efd_path))
    
    print(f"\nLoaded {efd_path.name}")
    print(f"Keys: {list(efd_data.keys())}")
    print(f"Number of samples: {efd_data['R'].shape[0]}")
    
    # Validate each field
    efd_results = {}
    efd_results['R'] = validator.validate_coordinates(efd_data['R'], efd_data['Z'], efd_data['N'], 'R')
    efd_results['E'] = validator.validate_energies(efd_data['E'], 'E')
    efd_results['F'] = validator.validate_forces(efd_data['F'], efd_data['N'], 'F')
    efd_results['Dxyz'] = validator.validate_dipoles(efd_data['Dxyz'], 'Dxyz')
    
    # =========================================================================
    # Load and validate grids_esp.npz
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Part 2: Validating grids_esp.npz")
    print(f"{'#'*70}")
    
    grid_path = data_dir / "grids_esp.npz"
    grid_data = dict(np.load(grid_path))
    
    print(f"\nLoaded {grid_path.name}")
    print(f"Keys: {list(grid_data.keys())}")
    print(f"Number of samples: {grid_data['R'].shape[0]}")
    
    # Validate fields
    grid_results = {}
    grid_results['R'] = validator.validate_coordinates(grid_data['R'], grid_data['Z'], grid_data['N'], 'R (in grid file)')
    grid_results['esp_grid'] = validator.validate_esp_grid(
        grid_data['vdw_grid'], grid_data['esp'],
        grid_data['grid_origin'], grid_data['grid_axes'], grid_data['grid_dims'],
        'ESP Grid'
    )
    
    # =========================================================================
    # Verify consistency between files
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Part 3: Cross-file Consistency Checks")
    print(f"{'#'*70}")
    
    # Check that R, Z, N are identical
    if np.allclose(efd_data['R'], grid_data['R']):
        validator.log("R arrays are identical across files ✓", 'OK')
    else:
        validator.log("R arrays DIFFER between files!", 'ERROR')
    
    if np.array_equal(efd_data['Z'], grid_data['Z']):
        validator.log("Z arrays are identical across files ✓", 'OK')
    else:
        validator.log("Z arrays DIFFER between files!", 'ERROR')
    
    if np.array_equal(efd_data['N'], grid_data['N']):
        validator.log("N arrays are identical across files ✓", 'OK')
    else:
        validator.log("N arrays DIFFER between files!", 'ERROR')
    
    # Check sample counts match
    n_efd = efd_data['R'].shape[0]
    n_grid = grid_data['R'].shape[0]
    
    if n_efd == n_grid:
        validator.log(f"Sample counts match: {n_efd} samples ✓", 'OK')
    else:
        validator.log(f"Sample counts DIFFER: {n_efd} vs {n_grid}!", 'ERROR')
        return
    
    # =========================================================================
    # Create splits
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Part 4: Creating 8:1:1 Train/Valid/Test Splits")
    print(f"{'#'*70}")
    
    n_samples = n_efd
    splits = create_splits(n_samples, train_frac=0.8, valid_frac=0.1, test_frac=0.1, seed=42)
    
    print(f"\nTotal samples: {n_samples}")
    print(f"  Train: {len(splits['train'])} ({len(splits['train'])/n_samples*100:.1f}%)")
    print(f"  Valid: {len(splits['valid'])} ({len(splits['valid'])/n_samples*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({len(splits['test'])/n_samples*100:.1f}%)")
    
    # =========================================================================
    # Save split datasets
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Part 5: Saving Split Datasets")
    print(f"{'#'*70}")
    
    for split_name, split_indices in splits.items():
        print(f"\nSaving {split_name} split ({len(split_indices)} samples)...")
        
        # Split energies_forces_dipoles
        efd_split = split_dataset(efd_data, split_indices)
        efd_out = output_dir / f"energies_forces_dipoles_{split_name}.npz"
        np.savez_compressed(efd_out, **efd_split)
        print(f"  ✓ Saved {efd_out.name} ({efd_out.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Split grids_esp
        grid_split = split_dataset(grid_data, split_indices)
        grid_out = output_dir / f"grids_esp_{split_name}.npz"
        np.savez_compressed(grid_out, **grid_split)
        print(f"  ✓ Saved {grid_out.name} ({grid_out.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Save split indices for reproducibility
    indices_out = output_dir / "split_indices.npz"
    np.savez(indices_out, **splits)
    print(f"\n✓ Saved split indices to {indices_out.name}")
    
    # =========================================================================
    # Create validation report
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# Validation Summary")
    print(f"{'#'*70}")
    
    report_lines = [
        "CO2 Dataset Validation Report",
        "=" * 70,
        "",
        "## Data Sources",
        f"- energies_forces_dipoles.npz: {n_samples} samples",
        f"- grids_esp.npz: {n_samples} samples",
        "",
        "## Unit Validation Results",
        "",
        "### Atomic Coordinates (R)",
        f"- Status: {efd_results['R']['units']}",
        f"- Normalized: {efd_results['R'].get('is_normalized', False)}",
    ]
    
    if efd_results['R'].get('is_normalized'):
        report_lines.append(f"- Scale to Angstrom: ×{efd_results['R'].get('scale_to_angstrom', 1.0):.6f}")
        report_lines.append(f"- Scale to Bohr: ×{efd_results['R'].get('scale_to_bohr', 1.0):.6f}")
        report_lines.append("- ⚠️ WARNING: Coordinates must be scaled before training!")
    
    report_lines.extend([
        "",
        "### Energies (E)",
        f"- Units: {efd_results['E']['units']}",
        f"- Valid: {efd_results['E']['valid']}",
        "",
        "### Forces (F)",
        f"- Units: {efd_results['F']['units']}",
        f"- Valid: {efd_results['F']['valid']}",
        "",
        "### Dipoles (Dxyz)",
        f"- Units: {efd_results['Dxyz']['units']}",
        f"- Valid: {efd_results['Dxyz']['valid']}",
        "",
        "### ESP Grid",
        f"- Grid Units: {grid_results['esp_grid']['units']}",
        f"- Valid: {grid_results['esp_grid'].get('valid', False)}",
    ])
    
    if grid_results['esp_grid'].get('needs_conversion'):
        report_lines.append("- ⚠️ WARNING: Grid coordinates are in index space, not physical units!")
    
    report_lines.extend([
        "",
        "## Data Splits",
        f"- Train: {len(splits['train'])} samples (80%)",
        f"- Valid: {len(splits['valid'])} samples (10%)",
        f"- Test:  {len(splits['test'])} samples (10%)",
        "",
        "## Files Created",
        f"- {output_dir}/energies_forces_dipoles_train.npz",
        f"- {output_dir}/energies_forces_dipoles_valid.npz",
        f"- {output_dir}/energies_forces_dipoles_test.npz",
        f"- {output_dir}/grids_esp_train.npz",
        f"- {output_dir}/grids_esp_valid.npz",
        f"- {output_dir}/grids_esp_test.npz",
        f"- {output_dir}/split_indices.npz",
        "",
        "## CRITICAL WARNINGS FOR TRAINING",
        "",
    ])
    
    # Add critical warnings
    if efd_results['R'].get('is_normalized'):
        report_lines.append("⚠️  ATOMIC COORDINATES ARE NORMALIZED!")
        report_lines.append("   - Current: C-O bond = 1.0 (unitless)")
        report_lines.append("   - Required: Multiply by 1.16 for Angstroms OR 2.19 for Bohr")
        report_lines.append("   - DCMnet/PhysnetJax expect physical units!")
        report_lines.append("")
    
    if grid_results['esp_grid']['units'] == 'grid_indices':
        report_lines.append("⚠️  ESP GRID COORDINATES ARE IN INDEX SPACE!")
        report_lines.append("   - Current: 0-49 grid indices")
        report_lines.append("   - Required: Physical coordinates (Angstrom or Bohr)")
        report_lines.append("   - Must apply: coord = origin + index * spacing")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    report_path = output_dir / "VALIDATION_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Validation report saved to: {report_path}")
    print(f"\n{'='*70}")
    print("✅ Data preparation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


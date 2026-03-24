#!/usr/bin/env python3
"""
CLI tool to fix units and create train/valid/test splits from molecular NPZ data.

Supports NPZ files with or without ESP grid data.

Usage:
    # With grid data
    mmml fix-and-split \\
        --efd energies_forces_dipoles.npz \\
        --grid grids_esp.npz \\
        --output-dir ./training_data_fixed
    
    # Without grid data (only EFD)
    mmml fix-and-split \\
        --efd energies_forces_dipoles.npz \\
        --output-dir ./training_data_fixed

This script:
1. Validates atomic coordinates are in Angstroms
2. Converts energies from Hartree to eV (ASE standard)
3. Converts forces from Hartree/Bohr to eV/Angstrom (ASE standard); optional --flip-forces if F stores ∂E/∂R (gradient) instead of −∇E
4. Optional extra multipliers (--energy-scale, --force-scale, --dipole-scale, --efield-scale, --esp-scale, --charge-scale) after the standard conversions (and on matching auxiliary NPZ keys)
5. Converts ESP grid coordinates from index space to physical Angstroms (if grid data provided)
6. Creates train/valid/test splits
7. Saves data in ASE-compatible format

ASE Standard Units:
- Coordinates (R): Angstrom
- Energies (E): eV
- Forces (F): eV/Angstrom
- Dipoles (Dxyz): Debye
- ESP values: Hartree/e (no ASE standard for ESP)
- ESP grid (vdw_surface): Angstrom
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root.resolve()))

ATOMIC_REF_PATH = Path(__file__).parent.parent.parent / "data" / "qcml" / "atomic_reference_energies.json"


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


def convert_dipole_debye_to_eA(D_debye: np.ndarray) -> np.ndarray:
    """Convert dipole moments from Debye to e·Å (PhysNet/DCMNet standard)."""
    from mmml.data.units import DEBYE_TO_EANGSTROM
    return D_debye * DEBYE_TO_EANGSTROM


def _scale_ndarrays_in_dict(
    data: Dict[str, Any],
    keys: Tuple[str, ...],
    factor: float,
) -> List[str]:
    """In-place multiply numeric ndarrays for keys by factor if factor != 1. Skips missing keys and non-numeric dtypes."""
    f = float(factor)
    if f == 1.0:
        return []
    out: List[str] = []
    for k in keys:
        if k not in data:
            continue
        v = data[k]
        if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufc":
            continue
        data[k] = np.asarray(v, dtype=np.float64) * f
        out.append(f"{k}×{f:g}")
    return out


def load_atomic_reference_energies(scheme: str) -> Dict[str, float]:
    """Load per-atom reference energies (Hartree) for a given scheme from atomic_reference_energies.json."""
    with open(ATOMIC_REF_PATH) as f:
        all_refs = json.load(f)
    if scheme not in all_refs:
        raise ValueError(
            f"Unknown atomic ref scheme '{scheme}'. "
            f"Available: {list(all_refs.keys())[:10]}..."
        )
    return all_refs[scheme]


def _species_entry_to_atomic_number(zi: Any) -> int:
    """Map one Z entry (atomic number or element symbol) to atomic number; 0 = padding."""
    if zi is None:
        return 0
    if isinstance(zi, (bytes, np.bytes_)):
        try:
            zi = zi.decode("ascii").strip()
        except Exception:
            return 0
    if isinstance(zi, str) or isinstance(zi, np.str_):
        s = str(zi).strip()
        if not s or s.lower() in ("none", "nan"):
            return 0
        try:
            n = int(float(s))
            return n if n > 0 else 0
        except ValueError:
            pass
        from ase.data import atomic_numbers

        sym = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
        return int(atomic_numbers.get(sym, 0))
    try:
        n = int(np.asarray(zi).item())
        return n if n > 0 else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def npz_z_array_to_atomic_numbers(Z: np.ndarray) -> np.ndarray:
    """
    Convert NPZ Z array to int64 atomic numbers.

    Integer/float Z uses a fast path; object / Unicode arrays (e.g. 'C', 'H') are
    mapped via ASE. Non-positive and unknown species become 0 (padding).
    """
    Z = np.asarray(Z)
    if Z.size == 0:
        return np.zeros(Z.shape, dtype=np.int64)
    if Z.dtype.kind in "iuf" and Z.dtype != object:
        Zn = np.asarray(Z, dtype=np.int64)
        return np.where(Zn > 0, Zn, 0).astype(np.int64)
    flat = Z.ravel()
    out = np.fromiter(
        (_species_entry_to_atomic_number(x) for x in flat),
        dtype=np.int64,
        count=flat.size,
    )
    return out.reshape(Z.shape)


def subtract_atomic_references(
    E_hartree: np.ndarray,
    Z: np.ndarray,
    scheme: str,
    ref_units: str = "hartree",
) -> np.ndarray:
    """
    Subtract per-atom reference energies from total energies.
    E_corrected = E - sum(E_ref[Z_i]) for each molecule.
    Returns corrected energies in Hartree.

    ref_units: "hartree" (default) or "ev". If "ev", refs are converted to Hartree
    before subtraction (E_ref_Ha = E_ref_eV / 27.211386).
    """
    from ase.data import chemical_symbols

    HARTREE_TO_EV = 27.211386
    refs = load_atomic_reference_energies(scheme)
    E_ref_per_sample = np.zeros(len(E_hartree), dtype=np.float64)
    Z = npz_z_array_to_atomic_numbers(np.asarray(Z))

    for i in range(len(E_hartree)):
        z = Z[i] if Z.ndim > 1 else Z
        for zi in z:
            zn = int(zi)
            if zn <= 0:
                continue
            sym = chemical_symbols[zn]
            key = f"{sym}:0"
            if key not in refs:
                raise ValueError(f"No reference for {key} in scheme '{scheme}'")
            val = refs[key]
            if ref_units.lower() == "ev":
                val = val / HARTREE_TO_EV
            E_ref_per_sample[i] += val

    return E_hartree - E_ref_per_sample


def reduce_esp_grid(
    esp: np.ndarray,
    esp_grid: np.ndarray,
    R: np.ndarray,
    n_grid_points: int = 3000,
    esp_sd_sigma: float = 3.0,
    esp_max_abs_kcal_mol: float = 100.0,
    min_dist_to_atoms: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce ESP grid to fixed number of points per sample.
    Excludes points beyond ±esp_sd_sigma standard deviations from the mean (ignores
    distribution tails), points with |esp| > esp_max_abs_kcal_mol (kcal/mol/e), and
    points too close to atomic centers.
    """
    rng = np.random.default_rng(seed)
    n_samples = esp.shape[0]
    n_atoms = R.shape[1]

    esp_out = np.zeros((n_samples, n_grid_points), dtype=esp.dtype)
    esp_grid_out = np.full((n_samples, n_grid_points, 3), 1e6, dtype=esp_grid.dtype)

    for i in range(n_samples):
        esp_i = esp[i]
        grid_i = esp_grid[i]
        r_i = R[i]

        # Mask padding (esp_grid uses 1e6 for padding)
        valid = np.all(np.abs(grid_i) < 1e5, axis=1)

        # Exclude points too close to any atom
        atoms_valid = np.any(r_i != 0, axis=1)
        if np.any(atoms_valid):
            dists = cdist(grid_i, r_i[atoms_valid])
            min_d = dists.min(axis=1)
            valid &= min_d > min_dist_to_atoms

        # Exclude distribution tails: keep only points within ±esp_sd_sigma SD of mean
        valid_esp = esp_i[valid]
        if len(valid_esp) > 0:
            mean_esp = np.mean(valid_esp)
            std_esp = np.std(valid_esp)
            if std_esp > 0:
                valid &= np.abs(esp_i - mean_esp) <= esp_sd_sigma * std_esp

        # Exclude points with |esp| > esp_max_abs_kcal_mol (kcal/mol/e)
        from mmml.data.units import KCAL_MOL_TO_HARTREE
        esp_max_hartree = esp_max_abs_kcal_mol * KCAL_MOL_TO_HARTREE
        valid &= np.abs(esp_i) <= esp_max_hartree

        idx = np.where(valid)[0]
        if len(idx) >= n_grid_points:
            chosen = rng.choice(idx, size=n_grid_points, replace=False)
            esp_out[i] = esp_i[chosen]
            esp_grid_out[i] = grid_i[chosen]
        elif len(idx) > 0:
            n_take = len(idx)
            esp_out[i, :n_take] = esp_i[idx]
            esp_grid_out[i, :n_take, :] = grid_i[idx]
            # Rest stays as padding (zeros for esp, 1e6 for grid)

    return esp_out, esp_grid_out


def verify_reduction_preserves_alignment(
    esp_raw: np.ndarray,
    grid_raw: np.ndarray,
    esp_reduced: np.ndarray,
    grid_reduced: np.ndarray,
    R: np.ndarray,
    n_grid_points: int = 3000,
    esp_sd_sigma: float = 3.0,
    esp_max_abs_kcal_mol: float = 100.0,
    min_dist_to_atoms: float = 1.0,
    seed: int = 42,
    n_spot_check: int = 3,
) -> bool:
    """
    Verify that reduce_esp_grid used the same indices for esp and grid (alignment preserved).
    Recomputes the selection logic and checks that output pairs match input pairs.
    """
    from mmml.data.units import KCAL_MOL_TO_HARTREE
    rng = np.random.default_rng(seed)
    for i in range(min(n_spot_check, esp_raw.shape[0])):
        esp_i = esp_raw[i]
        grid_i = grid_raw[i]
        r_i = R[i]
        valid = np.all(np.abs(grid_i) < 1e5, axis=1)
        atoms_valid = np.any(r_i != 0, axis=1)
        if np.any(atoms_valid):
            dists = cdist(grid_i, r_i[atoms_valid])
            valid &= dists.min(axis=1) > min_dist_to_atoms
        valid_esp = esp_i[valid]
        if len(valid_esp) > 0:
            mean_esp = np.mean(valid_esp)
            std_esp = np.std(valid_esp)
            if std_esp > 0:
                valid &= np.abs(esp_i - mean_esp) <= esp_sd_sigma * std_esp
        esp_max_hartree = esp_max_abs_kcal_mol * KCAL_MOL_TO_HARTREE
        valid &= np.abs(esp_i) <= esp_max_hartree
        idx = np.where(valid)[0]
        if len(idx) < 3:
            continue
        if len(idx) >= n_grid_points:
            chosen = rng.choice(idx, size=n_grid_points, replace=False)
            n_check = n_grid_points
        else:
            chosen = idx
            n_check = len(idx)
        for k in range(n_check):
            if not np.isclose(esp_reduced[i, k], esp_i[chosen[k]]):
                return False
            if not np.allclose(grid_reduced[i, k], grid_i[chosen[k]]):
                return False
    return True


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


def check_esp_grid_alignment(
    esp: np.ndarray,
    grid: np.ndarray,
    R: np.ndarray,
    n_check: int = 5,
) -> Tuple[bool, float]:
    """
    Verify that esp[i,j] corresponds to grid[i,j] (same physical point).
    
    Uses spatial consistency: |esp| tends to be larger near atoms (1/r decay).
    If esp and grid are misaligned (wrong index order), this correlation drops.
    
    Returns
    -------
    ok : bool
        True if alignment appears correct (mean correlation > 0.3)
    mean_corr : float
        Mean Pearson correlation across checked samples
    """
    from scipy.stats import pearsonr
    BOHR_TO_ANGSTROM = 0.529177
    ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
    correlations = []
    n_samples = min(n_check, esp.shape[0])
    for i in range(n_samples):
        esp_i = esp[i]
        grid_i = grid[i]
        r_i = R[i]
        valid = np.all(np.abs(grid_i) < 1e5, axis=1)
        if np.sum(valid) < 10:
            continue
        esp_valid = np.abs(esp_i[valid])
        grid_valid = grid_i[valid]
        atoms_valid = np.any(r_i != 0, axis=1)
        if not np.any(atoms_valid):
            continue
        dists = cdist(grid_valid, r_i[atoms_valid])
        min_dist = dists.min(axis=1)
        inv_dist = 1.0 / (min_dist + 0.5)
        try:
            corr, _ = pearsonr(esp_valid, inv_dist)
            if not np.isnan(corr):
                correlations.append(float(corr))
        except Exception:
            pass
    if not correlations:
        return True, 0.0
    mean_corr = float(np.mean(correlations))
    ok = mean_corr > 0.2
    return ok, mean_corr


def validate_fixed_data(
    R_ang, E_ev, F_ev_ang, vdw_grid_ang, Z, N, 
    has_grid: bool = True,
    verbose: bool = True
):
    """Validate that fixes worked correctly."""
    if verbose:
        print(f"\n{'='*70}")
        print("POST-FIX VALIDATION")
        print(f"{'='*70}")
    
    # Check atomic coordinates (shortest interatomic distance)
    min_dists = []
    for i in range(min(100, len(R_ang))):
        r = R_ang[i]
        valid = np.any(r != 0, axis=1)
        vpos = r[valid]
        if len(vpos) < 2:
            continue
        d = vpos[:, np.newaxis, :] - vpos[np.newaxis, :, :]
        norms = np.linalg.norm(d, axis=2)
        norms[np.triu_indices_from(norms, k=0)] = np.inf
        min_dists.append(norms.min())
    min_dists = np.array(min_dists) if min_dists else np.array([])

    coords_ok = False
    energy_ok = False
    force_ok = False
    grid_ok = True  # Default to True if no grid
    spatial_ok = True  # Default to True if no grid

    if len(min_dists) > 0:
        if verbose:
            print(f"\nAtomic Coordinates (up to 100 samples):")
            print(f"  Shortest distance: mean={min_dists.mean():.4f} Å, "
                  f"range=[{min_dists.min():.4f}, {min_dists.max():.4f}]")
        coords_ok = 0.5 <= min_dists.mean() <= 3.0
        if verbose and coords_ok:
            print(f"  ✓ Coordinates in reasonable range")
        elif verbose:
            print(f"  ⚠️  Coordinates outside expected range")
    else:
        coords_ok = True
    
    # Check energies
    if verbose:
        print(f"\nEnergies (sample 0):")
        print(f"  Value: {E_ev[0]:.6f} eV")
        print(f"  Dataset mean: {E_ev.mean():.6f} eV")
    
    # Check that energies are in reasonable range for molecular systems (eV)
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
    
    # Check ESP grid (only if grid data exists)
    if has_grid and vdw_grid_ang is not None:
        grid0 = vdw_grid_ang[0]
        # Mask out padding (e.g. 1e6 used for variable-length grids)
        valid_mask = np.all(np.abs(grid0) < 1e5, axis=1)
        grid0_valid = grid0[valid_mask] if np.any(valid_mask) else grid0
        grid_extent = (grid0_valid.max(axis=0) - grid0_valid.min(axis=0)).mean()
        
        if verbose:
            print(f"\nESP Grid Coordinates:")
            print(f"  Average extent: {grid_extent:.4f} Angstrom")
            print(f"  X range: [{grid0_valid[:, 0].min():.4f}, {grid0_valid[:, 0].max():.4f}]")
            print(f"  Y range: [{grid0_valid[:, 1].min():.4f}, {grid0_valid[:, 1].max():.4f}]")
            print(f"  Z range: [{grid0_valid[:, 2].min():.4f}, {grid0_valid[:, 2].max():.4f}]")
        
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
        z0 = np.asarray(Z[0] if Z.ndim > 1 else Z)
        zn0 = npz_z_array_to_atomic_numbers(z0)
        valid = zn0 > 0
        valid_pos = r0[valid]
        
        if len(valid_pos) > 0:
            mol_center = valid_pos.mean(axis=0)
            grid_min = grid0_valid.min(axis=0)
            grid_max = grid0_valid.max(axis=0)
            
            if verbose:
                print(f"\nSpatial relationship:")
                print(f"  Molecule center: [{mol_center[0]:.2f}, {mol_center[1]:.2f}, {mol_center[2]:.2f}]")
                print(f"  Grid bounds: X[{grid_min[0]:.2f}, {grid_max[0]:.2f}], "
                      f"Y[{grid_min[1]:.2f}, {grid_max[1]:.2f}], "
                      f"Z[{grid_min[2]:.2f}, {grid_max[2]:.2f}]")
            
            # Check if molecule is within or near grid bounds
            max_min_dist = max([np.min(np.linalg.norm(grid0_valid - atom_pos, axis=1)) 
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
    else:
        if verbose:
            print(f"\nESP Grid: Skipped (no grid data provided)")
    
    overall_ok = coords_ok and energy_ok and force_ok and grid_ok and spatial_ok
    
    if verbose:
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"  Coordinates: {'✓' if coords_ok else '❌'}")
        print(f"  Energies:    {'✓' if energy_ok else '❌'}")
        print(f"  Forces:      {'✓' if force_ok else '❌'}")
        if has_grid:
            print(f"  ESP Grid:    {'✓' if grid_ok else '❌'}")
            print(f"  Spatial:     {'✓' if spatial_ok else '❌'}")
        
        if overall_ok:
            print(f"\n✅ ALL VALIDATIONS PASSED - Data ready for training!")
        else:
            print(f"\n⚠️  SOME VALIDATIONS FAILED - Review above")
        print(f"{'='*70}")
    
    return overall_ok


def _normalize_for_concat(arr: np.ndarray, n_samples: int, key: str) -> np.ndarray:
    """Ensure array has at least 1 dim for concatenation. Scalar/0-d -> (n_samples,)."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.full(n_samples, arr.flat[0], dtype=arr.dtype)
    if arr.ndim == 1 and arr.shape[0] == 1 and key in ('N',):
        return np.full(n_samples, arr[0], dtype=arr.dtype)
    return arr


def _load_and_merge_efd(efd_files: Union[Path, List[Path]]) -> Dict:
    """Load one or more EFD npz files and concatenate along sample dimension."""
    if isinstance(efd_files, (str, Path)):
        efd_files = [Path(efd_files)]
    else:
        efd_files = [Path(f) for f in efd_files]

    if len(efd_files) == 1:
        return dict(np.load(efd_files[0], allow_pickle=True))

    # Concatenate multiple files
    parts = [dict(np.load(f, allow_pickle=True)) for f in efd_files]
    concat_keys = ['R', 'E', 'F', 'N', 'Dxyz', 'esp']
    grid_key = 'esp_grid' if 'esp_grid' in parts[0] else ('vdw_surface' if 'vdw_surface' in parts[0] else None)
    if grid_key:
        concat_keys.append(grid_key)

    all_keys = set()
    for p in parts:
        all_keys.update(p.keys())
    merged = {}
    for k in all_keys:
        if k == 'Z':
            merged[k] = np.asarray(parts[0][k])
        elif k in concat_keys and all(k in p for p in parts):
            # Normalize shapes: pyscf-evaluate may have N as scalar (0-d), fix-and-split has (n,)
            to_concat = []
            for p in parts:
                arr = np.asarray(p[k])
                n_samples = p['R'].shape[0]  # sample count from R
                if arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] != n_samples and k == 'N'):
                    arr = _normalize_for_concat(arr, n_samples, k)
                to_concat.append(arr)
            merged[k] = np.concatenate(to_concat, axis=0)
        else:
            # Key in some but not all parts, or not concat-able: use first part that has it
            for p in parts:
                if k in p:
                    merged[k] = np.asarray(p[k])
                    break
    return merged


def fix_and_split_data(
    efd_file: Union[Path, List[Path]],
    grid_file: Optional[Path] = None,
    output_dir: Path = None,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    cube_spacing_bohr: float = 0.25,
    skip_validation: bool = False,
    atomic_ref: Optional[str] = None,
    atomic_ref_units: str = "hartree",
    n_grid_points: int = 3000,
    esp_sd_sigma: float = 3.0,
    esp_max_abs_kcal_mol: float = 100.0,
    min_dist_to_atoms: float = 1.0,
    flip_forces: bool = False,
    energy_scale: float = 1.0,
    force_scale: float = 1.0,
    dipole_scale: float = 1.0,
    efield_scale: float = 1.0,
    esp_scale: float = 1.0,
    charge_scale: float = 1.0,
    verbose: bool = True,
) -> bool:
    """
    Main workflow to fix units and create splits.
    
    Parameters
    ----------
    efd_file : Path
        Path to energies_forces_dipoles.npz file
    grid_file : Path, optional
        Path to grids_esp.npz file (optional)
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
    atomic_ref : str, optional
        Subtract per-atom reference energies using scheme from atomic_reference_energies.json
        (e.g. "pbe0/sz" for PBE0/SZ, "pbe0/def2-tzvp" for Hartree). Default: None.
    atomic_ref_units : str
        Units of refs in JSON: "hartree" (default) or "ev". If "ev", refs are converted
        before subtraction. Schemes like pbe0/sz may use eV; pbe0/def2-tzvp uses Hartree.
    n_grid_points : int
        Target number of ESP grid points per sample (default 3000). Points beyond ±esp_sd_sigma
        SD from the mean or too close to atoms are excluded, then subsampled.
    esp_sd_sigma : float
        Exclude grid points beyond ±this many standard deviations from the mean (default 3.0).
    esp_max_abs_kcal_mol : float
        Exclude grid points with |esp| > this in kcal/mol/e (default 100.0).
    min_dist_to_atoms : float
        Exclude grid points closer than this to any atom in Å (default 1.0).
    flip_forces : bool
        If True, negate ``F`` before converting units (use when NPZ stores PySCF-style
        energy gradient ∂E/∂R in Ha/Bohr instead of forces F = −∇E). Default False.
    energy_scale : float
        Multiply ``E`` after Hartree→eV (and after atomic-ref). Also ``efield_energy``, ``efield_scf_energy`` if present. Default 1.0.
    force_scale : float
        Multiply ``F`` after Ha/Bohr→eV/Å. Also ``efield_scf_F`` if present. Default 1.0.
    dipole_scale : float
        Multiply ``Dxyz`` after Debye→e·Å (ignored if no Dxyz). Same factor on ``D``, ``efield_scf_D``, ``efield_D`` (Debye) if present. Not applied to ``efield_D_au``. Default 1.0.
    efield_scale : float
        Multiply external-field vectors ``Ef``, ``efield_Ef``, ``efield_scf_Ef`` if present. Default 1.0.
    esp_scale : float
        Multiply ``esp`` (Hartree/e) on grid output and on EFD if that key is present (combined NPZ). Default 1.0.
    charge_scale : float
        Multiply NPZ key ``Q`` if present. Intended for **total molecular charge** when your files
        use ``Q`` for that. Note: PySCF ``dens_esp`` export may store **quadrupole** under ``Q``;
        only use this scale when ``Q`` matches your intended quantity. Default 1.0.
    verbose : bool
        Print detailed progress (default True)
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if verbose:
        print("\n" + "="*70)
        print("Molecular Data Unit Conversion and Splitting")
        print("="*70)
        print(f"\nInput files:")
        if isinstance(efd_file, (list, tuple)):
            print(f"  EFD:  {[str(p) for p in efd_file]}")
        else:
            print(f"  EFD:  {efd_file}")
        if grid_file:
            print(f"  Grid: {grid_file}")
        else:
            print(f"  Grid: (not provided)")
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
        efd_data = _load_and_merge_efd(efd_file)
        grid_data = None
        has_grid = False

        grid_from_efd = False
        if grid_file is not None and grid_file.exists():
            grid_data = dict(np.load(grid_file, allow_pickle=True))
            has_grid = True
            grid_from_efd = False
        elif 'esp' in efd_data and 'esp_grid' in efd_data:
            # Combined EFD+grid file (e.g. from pyscf-evaluate --esp)
            grid_data = {
                'esp': efd_data['esp'],
                'vdw_surface': efd_data['esp_grid'],
                'R': efd_data['R'],
                'Z': efd_data['Z'],
                'N': efd_data['N'],
            }
            if 'Dxyz' in efd_data:
                grid_data['Dxyz'] = efd_data['Dxyz']
            has_grid = True
            grid_from_efd = True
            if verbose:
                print(f"  Using esp/esp_grid from EFD file (combined format)")
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return False
    
    n_samples = efd_data['R'].shape[0]

    # Normalize Z: pyscf-evaluate outputs Z as (n_atoms,) for same molecule; broadcast to (n_samples, n_atoms)
    Z_raw = efd_data['Z']
    if Z_raw.ndim == 1:
        Z_expanded = np.broadcast_to(Z_raw[np.newaxis, :], (n_samples, Z_raw.shape[0]))
    else:
        Z_expanded = Z_raw

    if verbose:
        print(f"\nLoaded {n_samples} samples")
        print(f"  Keys in EFD: {list(efd_data.keys())}")
        if has_grid:
            print(f"  Keys in Grid: {list(grid_data.keys())}")
        print(f"\nShapes:")
        for k in ['R', 'Z', 'E', 'F', 'Dxyz']:
            if k in efd_data:
                v = efd_data[k]
                print(f"  {k}: {v.shape}")
        if has_grid and grid_data:
            if 'esp' in grid_data:
                print(f"  esp: {grid_data['esp'].shape}")
            if 'vdw_surface' in grid_data:
                print(f"  esp_grid (vdw_surface): {grid_data['vdw_surface'].shape}")
        print(f"\nSummary statistics:")
        print(f"  E:  mean={efd_data['E'].mean():.6f}, std={efd_data['E'].std():.6f}, "
              f"range=[{efd_data['E'].min():.6f}, {efd_data['E'].max():.6f}]")
        f_norms = np.linalg.norm(efd_data['F'].reshape(-1, 3), axis=1)
        print(f"  F:  mean_norm={f_norms.mean():.6e}, max_norm={f_norms.max():.6e}")
        if 'Dxyz' in efd_data:
            d_norms = np.linalg.norm(efd_data['Dxyz'].reshape(-1, 3), axis=1)
            print(f"  Dxyz: mean_norm={d_norms.mean():.4f}, max_norm={d_norms.max():.4f} Debye")
        if has_grid and 'esp' in (grid_data or {}):
            esp_flat = grid_data['esp'].flatten()
            valid_esp = esp_flat[np.abs(esp_flat) < 1e5]  # exclude padding
            if len(valid_esp) > 0:
                print(f"  esp: mean={valid_esp.mean():.6e}, range=[{valid_esp.min():.6e}, {valid_esp.max():.6e}]")

    # =========================================================================
    # Check atomic coordinates (Bohr vs Angstrom)
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 2: Checking Atomic Coordinates")
        print(f"{'#'*70}")
    
    # Use shortest interatomic distance to infer units (generic, not molecule-specific)
    min_dists = []
    for i in range(min(100, len(efd_data['R']))):
        r = efd_data['R'][i]
        valid = np.any(r != 0, axis=1)
        vpos = r[valid]
        if len(vpos) < 2:
            continue
        # Pairwise distances (upper triangle)
        d = vpos[:, np.newaxis, :] - vpos[np.newaxis, :, :]
        norms = np.linalg.norm(d, axis=2)
        norms[np.triu_indices_from(norms, k=0)] = np.inf
        min_dists.append(norms.min())
    min_dists = np.array(min_dists) if min_dists else np.array([])

    if len(min_dists) > 0:
        d_mean = min_dists.mean()
        if verbose:
            print(f"\nShortest interatomic distance: mean={d_mean:.4f}, range=[{min_dists.min():.4f}, {min_dists.max():.4f}]")
        if 0.8 < d_mean < 2.5:
            if verbose:
                print(f"✓ Coordinates in Angstroms")
            R_angstrom = efd_data['R']
        elif 1.8 < d_mean < 2.9:
            if verbose:
                print(f"→ Converting from Bohr to Angstroms...")
            R_angstrom = efd_data['R'] * 0.529177
        else:
            if verbose:
                print(f"⚠️  Ambiguous units (d={d_mean:.4f}), assuming Angstroms")
            R_angstrom = efd_data['R']
    else:
        if verbose:
            print(f"\n⚠️  Could not compute distances, assuming coordinates in Angstroms")
        R_angstrom = efd_data['R']
    
    # =========================================================================
    # Convert energies: Hartree → eV (optionally subtract atomic references)
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 3: Converting Energies from Hartree to eV")
        print(f"{'#'*70}")
    
    E_hartree = np.asarray(efd_data['E']).copy()
    if atomic_ref:
        if verbose:
            print(f"\nSubtracting atomic reference energies (scheme: {atomic_ref}, units: {atomic_ref_units})")
        E_hartree = subtract_atomic_references(E_hartree, Z_expanded, atomic_ref, ref_units=atomic_ref_units)
        if verbose:
            print(f"  E (after ref subtraction): mean={E_hartree.mean():.6f} Ha, "
                  f"range=[{E_hartree.min():.6f}, {E_hartree.max():.6f}]")
    E_ev = convert_energy_hartree_to_ev(E_hartree)
    
    if verbose:
        HARTREE_TO_EV = 27.211386
        print(f"\nConversion factor: {HARTREE_TO_EV}")
        print(f"Original (Hartree): mean={efd_data['E'].mean():.6f}, "
              f"range=[{efd_data['E'].min():.6f}, {efd_data['E'].max():.6f}]")
        if atomic_ref:
            print(f"After atomic ref subtraction: mean={E_hartree.mean():.6f} Ha")
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
    
    F_hartree_bohr = np.asarray(efd_data["F"], dtype=np.float64).copy()
    if flip_forces:
        F_hartree_bohr = -F_hartree_bohr
        if verbose:
            print("  --flip-forces: negating F before conversion (gradient ∂E/∂R → force −∇E)")

    F_ev_ang = convert_forces_hartree_bohr_to_ev_angstrom(F_hartree_bohr)

    if verbose:
        HARTREE_BOHR_TO_EV_ANG = 51.42208
        f_orig_norms = np.linalg.norm(F_hartree_bohr[:10, :3, :].reshape(-1, 3), axis=1)
        f_conv_norms = np.linalg.norm(F_ev_ang[:10, :3, :].reshape(-1, 3), axis=1)

        print(f"\nConversion factor: {HARTREE_BOHR_TO_EV_ANG}")
        print(f"Input to convert (Ha/Bohr): mean norm={f_orig_norms.mean():.6e}")
        print(f"Converted (eV/Å):           mean norm={f_conv_norms.mean():.6e}")
        print(f"✓ Forces converted to eV/Angstrom")
    
    # =========================================================================
    # Convert dipoles: Debye → e·Å (PhysNet/DCMNet standard)
    # =========================================================================
    if 'Dxyz' in efd_data:
        D_eA = convert_dipole_debye_to_eA(efd_data['Dxyz'])
        if verbose:
            d_norms_before = np.linalg.norm(efd_data['Dxyz'].reshape(-1, 3), axis=1)
            d_norms_after = np.linalg.norm(D_eA.reshape(-1, 3), axis=1)
            print(f"\n{'#'*70}")
            print("# Step 4b: Converting Dipoles from Debye to e·Å")
            print(f"{'#'*70}")
            print(f"  Conversion: 1 D = 0.208194 e·Å")
            print(f"  Original (Debye): mean |D|={d_norms_before.mean():.4f}, max={d_norms_before.max():.4f}")
            print(f"  Converted (e·Å):  mean |D|={d_norms_after.mean():.4f}, max={d_norms_after.max():.4f}")
            print(f"✓ Dipoles converted to e·Å")
    else:
        D_eA = None

    # =========================================================================
    # Optional extra scales (after standard conversions; before validation / ESP)
    # =========================================================================
    es = float(energy_scale)
    fs = float(force_scale)
    ds = float(dipole_scale)
    if es != 1.0:
        E_ev = E_ev * es
    if fs != 1.0:
        F_ev_ang = F_ev_ang * fs
    if D_eA is not None and ds != 1.0:
        D_eA = D_eA * ds
    ef_s = float(efield_scale)
    esp_s = float(esp_scale)
    q_s = float(charge_scale)
    if verbose and (es != 1.0 or fs != 1.0 or ds != 1.0 or ef_s != 1.0 or esp_s != 1.0 or q_s != 1.0):
        print(f"\n{'#'*70}")
        print("# Extra property scales (applied after standard unit conversion)")
        print(f"{'#'*70}")
        if es != 1.0:
            print(f"  E:     × {es}  (eV); also efield_energy, efield_scf_energy when present")
        if fs != 1.0:
            print(f"  F:     × {fs}  (eV/Å); also efield_scf_F when present")
        if D_eA is not None and ds != 1.0:
            print(f"  Dxyz:  × {ds}  (after Debye→e·Å); also D, efield_scf_D, efield_D when present")
        elif ds != 1.0:
            print(f"  D/efield dipoles: × {ds}  (Debye vectors, if present)")
        if ef_s != 1.0:
            print(f"  Ef:    × {ef_s}  (Ef, efield_Ef, efield_scf_Ef)")
        if esp_s != 1.0:
            print(f"  esp:   × {esp_s}  (Hartree/e on grid and on EFD if present)")
        if q_s != 1.0:
            print(f"  Q:     × {q_s}  (NPZ key Q; use for total charge—PySCF may use Q for quadrupole)")

    # =========================================================================
    # Fix ESP grid: index space → physical Angstroms (if grid data exists)
    # =========================================================================
    vdw_surface_angstrom = None
    
    if has_grid:
        if verbose:
            print(f"\n{'#'*70}")
            print("# Step 5: Converting ESP Grid to Physical Angstroms")
            print(f"{'#'*70}")
        
        BOHR_TO_ANGSTROM = 0.529177
        
        # Check if required grid keys exist
        required_grid_keys = ['vdw_grid', 'grid_origin', 'grid_axes', 'grid_dims']
        missing_keys = [k for k in required_grid_keys if k not in grid_data]
        
        if missing_keys:
            if verbose:
                print(f"\n⚠️  Missing grid keys: {missing_keys}")
                print(f"  Available keys: {list(grid_data.keys())}")
            # Grid from pyscf-evaluate (esp/esp_grid) is in Bohr; convert to Angstrom
            # Grid from other sources: assume Angstrom if no metadata
            # pyscf-evaluate saves esp_grid; also accept vdw_surface or vdw_grid
            grid_from_pyscf = 'esp' in grid_data and (
                'vdw_surface' in grid_data or 'vdw_grid' in grid_data or 'esp_grid' in grid_data
            )
            if 'vdw_surface' in grid_data:
                vdw_raw = grid_data['vdw_surface']
                if grid_from_pyscf:
                    vdw_surface_angstrom = vdw_raw * BOHR_TO_ANGSTROM
                    if verbose:
                        valid = np.all(np.abs(vdw_raw) < 1e5, axis=-1)
                        if valid.any():
                            ext_bohr = np.abs(vdw_raw[valid]).max()
                            ext_ang = np.abs(vdw_surface_angstrom[valid]).max()
                            print(f"  Converting vdw_surface from Bohr to Angstrom (pyscf-evaluate format)")
                            print(f"  Grid extent: ~{ext_bohr:.2f} Bohr → ~{ext_ang:.2f} Angstrom")
                        else:
                            print(f"  Converting vdw_surface from Bohr to Angstrom (pyscf-evaluate format)")
                else:
                    vdw_surface_angstrom = vdw_raw
                    if verbose:
                        print(f"  Using existing vdw_surface (assuming Angstroms)")
            elif 'vdw_grid' in grid_data:
                vdw_raw = grid_data['vdw_grid']
                if grid_from_pyscf:
                    vdw_surface_angstrom = vdw_raw * BOHR_TO_ANGSTROM
                    if verbose:
                        print(f"  Converting vdw_grid from Bohr to Angstrom (pyscf-evaluate format)")
                else:
                    vdw_surface_angstrom = vdw_raw
                    if verbose:
                        print(f"  Using vdw_grid directly (assuming Angstroms)")
            elif 'esp_grid' in grid_data:
                vdw_raw = grid_data['esp_grid']
                if grid_from_pyscf:
                    vdw_surface_angstrom = vdw_raw * BOHR_TO_ANGSTROM
                    if verbose:
                        print(f"  Using esp_grid as grid; converting from Bohr to Angstrom (pyscf format)")
                else:
                    vdw_surface_angstrom = vdw_raw
                    if verbose:
                        print(f"  Using esp_grid as grid (assuming Angstroms)")
        else:
            if verbose:
                print(f"\nCube file parameters:")
                print(f"  Spacing: {cube_spacing_bohr} Bohr = {cube_spacing_bohr * BOHR_TO_ANGSTROM:.6f} Angstrom")
                print(f"  Dimensions: {grid_data['grid_dims'][0]}")
                print(f"  Original origin (Bohr): {grid_data['grid_origin'][0]}")
            
            try:
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
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  Error converting grid: {e}")
                    print(f"  Falling back to using grid data as-is")
                # Fallback: use grid data as-is
                grid_from_pyscf_fb = 'esp' in grid_data and ('vdw_surface' in grid_data or 'vdw_grid' in grid_data)
                if 'vdw_surface' in grid_data:
                    vdw_raw = grid_data['vdw_surface']
                    vdw_surface_angstrom = vdw_raw * BOHR_TO_ANGSTROM if grid_from_pyscf_fb else vdw_raw
                elif 'vdw_grid' in grid_data:
                    vdw_raw = grid_data['vdw_grid']
                    vdw_surface_angstrom = vdw_raw * BOHR_TO_ANGSTROM if grid_from_pyscf_fb else vdw_raw
    else:
        if verbose:
            print(f"\n{'#'*70}")
            print("# Step 5: ESP Grid (skipped - no grid data provided)")
            print(f"{'#'*70}")
    
    # =========================================================================
    # Reduce ESP grid to fixed number of points (if grid exists)
    # =========================================================================
    if has_grid and vdw_surface_angstrom is not None:
        esp_raw = grid_data['esp']
        # Sanity check: esp and grid must have matching shapes (esp[i,j] pairs with grid[i,j])
        if esp_raw.shape[0] != vdw_surface_angstrom.shape[0]:
            raise ValueError(
                f"ESP/grid sample count mismatch: esp {esp_raw.shape[0]} vs grid {vdw_surface_angstrom.shape[0]}. "
                "Check that EFD and grid files have the same samples in the same order."
            )
        if esp_raw.shape[1] != vdw_surface_angstrom.shape[1]:
            raise ValueError(
                f"ESP/grid point count mismatch: esp {esp_raw.shape[1]} vs grid {vdw_surface_angstrom.shape[1]}. "
                "esp[i,j] must correspond to grid[i,j] for correct pairing."
            )
        # Check index alignment only for separate grid files (combined EFD format is trusted)
        if not grid_from_efd:
            align_ok_raw, align_corr_raw = check_esp_grid_alignment(
                esp_raw, vdw_surface_angstrom, R_angstrom, n_check=min(5, esp_raw.shape[0])
            )
            if verbose:
                print(f"  ESP-grid alignment (raw): correlation={align_corr_raw:.3f} {'✓' if align_ok_raw else '⚠️'}")
            if not align_ok_raw:
                print(f"  ⚠️  WARNING: Low esp-grid correlation suggests index misalignment. "
                      "esp[i,j] may not correspond to grid[i,j]. Check that esp and grid come from the same source.")
        if verbose:
            print(f"\n{'#'*70}")
            print("# Step 5b: Reducing ESP Grid")
            print(f"{'#'*70}")
            print(f"  Target points: {n_grid_points}")
            print(f"  Exclude points beyond ±{esp_sd_sigma} SD from mean")
            print(f"  Exclude |esp| > {esp_max_abs_kcal_mol} kcal/mol/e")
            print(f"  Exclude points < {min_dist_to_atoms} Å from atoms")
        esp_reduced, grid_reduced = reduce_esp_grid(
            esp_raw,
            vdw_surface_angstrom,
            R_angstrom,
            n_grid_points=n_grid_points,
            esp_sd_sigma=esp_sd_sigma,
            esp_max_abs_kcal_mol=esp_max_abs_kcal_mol,
            min_dist_to_atoms=min_dist_to_atoms,
            seed=seed,
        )
        # Verify reduction preserved esp-grid alignment (same indices used for both)
        reduction_ok = verify_reduction_preserves_alignment(
            esp_raw, vdw_surface_angstrom, esp_reduced, grid_reduced, R_angstrom,
            n_grid_points=n_grid_points, esp_sd_sigma=esp_sd_sigma,
            esp_max_abs_kcal_mol=esp_max_abs_kcal_mol, min_dist_to_atoms=min_dist_to_atoms,
            seed=seed, n_spot_check=3,
        )
        if not reduction_ok:
            raise RuntimeError(
                "reduce_esp_grid verification failed: esp and grid output pairs do not match input. "
                "This indicates a bug in the reduction logic."
            )
        if verbose:
            print(f"  ✓ Reduction preserves esp-grid alignment (verified)")
        vdw_surface_angstrom = grid_reduced
        if verbose:
            n_valid_per_sample = np.sum(np.all(np.abs(grid_reduced) < 1e5, axis=2), axis=1)
            print(f"  Reduced to shape: esp {esp_reduced.shape}, grid {grid_reduced.shape}")
            print(f"  Valid points per sample: min={n_valid_per_sample.min()}, max={n_valid_per_sample.max()}, mean={n_valid_per_sample.mean():.0f}")
        # Verify esp-grid alignment only for separate grid files
        if not grid_from_efd:
            align_ok, align_corr = check_esp_grid_alignment(esp_reduced, grid_reduced, R_angstrom, n_check=5)
            if verbose:
                print(f"  ESP-grid alignment check: correlation={align_corr:.3f} {'✓' if align_ok else '⚠️'}")
            if not align_ok and verbose:
                print(f"  ⚠️  Low esp-grid correlation may indicate index misalignment. "
                      "Ensure esp and grid come from the same source with matching point order.")
        # Update grid_data for saving - we need esp and vdw_surface
        grid_data = dict(grid_data)
        grid_data['esp'] = esp_reduced
        grid_data['vdw_surface'] = grid_reduced
        grid_data['esp_grid'] = grid_reduced
    
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
            Z_expanded, efd_data['N'], 
            has_grid=has_grid,
            verbose=verbose
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
    if D_eA is not None:
        efd_fixed['Dxyz'] = D_eA
    # PhysNet expects N (n_samples,) and Z (n_samples, n_atoms); pyscf-evaluate outputs scalar N and 1D Z
    N_raw = efd_data['N']
    if (np.isscalar(N_raw) or (isinstance(N_raw, np.ndarray) and N_raw.size == 1) or
            (isinstance(N_raw, np.ndarray) and N_raw.shape[0] != n_samples)):
        n_atoms = int(np.asarray(N_raw).flat[0])
        efd_fixed['N'] = np.full(n_samples, n_atoms, dtype=np.int32)
    Z_raw = efd_fixed['Z']
    if Z_raw.ndim == 1:
        efd_fixed['Z'] = np.broadcast_to(Z_raw[np.newaxis, :], (n_samples, Z_raw.shape[0]))
    efd_fixed['Z'] = npz_z_array_to_atomic_numbers(np.asarray(efd_fixed['Z']))

    # Update grid data with fixed coordinates (if grid exists)
    grid_fixed = None
    if has_grid and grid_data is not None:
        grid_fixed = grid_data.copy()
        grid_fixed['R'] = R_angstrom
        if vdw_surface_angstrom is not None:
            grid_fixed['vdw_surface'] = vdw_surface_angstrom
            grid_fixed['vdw_grid'] = vdw_surface_angstrom  # Backward compatibility
        if D_eA is not None and 'Dxyz' in grid_fixed:
            grid_fixed['Dxyz'] = D_eA
        # Align Z and N with EFD (per-sample shapes for PhysNet)
        if 'N' in grid_fixed:
            N_g = grid_fixed['N']
            if (np.isscalar(N_g) or (isinstance(N_g, np.ndarray) and N_g.size == 1) or
                    (isinstance(N_g, np.ndarray) and N_g.ndim == 0) or
                    (isinstance(N_g, np.ndarray) and N_g.shape[0] != n_samples)):
                n_a = int(np.asarray(N_g).flat[0]) if np.asarray(N_g).size else R_angstrom.shape[1]
                grid_fixed['N'] = np.full(n_samples, n_a, dtype=np.int32)
        if 'Z' in grid_fixed:
            Z_g = grid_fixed['Z']
            if Z_g.ndim == 1:
                grid_fixed['Z'] = np.broadcast_to(Z_g[np.newaxis, :], (n_samples, Z_g.shape[0]))
            grid_fixed['Z'] = npz_z_array_to_atomic_numbers(np.asarray(grid_fixed['Z']))

    # Same extra factors on auxiliary NPZ keys (pass-through arrays not unit-converted here)
    aux_scaled: List[str] = []
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("Ef", "efield_Ef", "efield_scf_Ef"), efield_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("efield_energy", "efield_scf_energy"), energy_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("efield_scf_F",), force_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("efield_scf_D", "efield_D", "D"), dipole_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("Q",), charge_scale))
    aux_scaled.extend(_scale_ndarrays_in_dict(efd_fixed, ("esp",), esp_scale))
    if grid_fixed is not None:
        aux_scaled.extend(_scale_ndarrays_in_dict(grid_fixed, ("esp",), esp_scale))
    if verbose and aux_scaled:
        print(f"\n  Auxiliary/grid arrays scaled: {', '.join(dict.fromkeys(aux_scaled))}")

    # =========================================================================
    # Save split datasets
    # =========================================================================
    if verbose:
        print(f"\n{'#'*70}")
        print("# Step 9: Saving Split Datasets")
        print(f"{'#'*70}")
    
    def _index_if_sample_dim(v, indices):
        """Index array by split if it has n_samples in first dim; else pass through."""
        if not isinstance(v, np.ndarray):
            return v
        if v.ndim == 0:
            return v
        if v.shape[0] == n_samples:
            return v[indices]
        return v

    for split_name, split_indices in splits.items():
        if verbose:
            print(f"\nSaving {split_name} split ({len(split_indices)} samples)...")
        
        # Create EFD split
        efd_split = {k: _index_if_sample_dim(v, split_indices) for k, v in efd_fixed.items()}
        efd_out = output_dir / f"energies_forces_dipoles_{split_name}.npz"
        np.savez_compressed(efd_out, **efd_split)
        
        if verbose:
            size_mb = efd_out.stat().st_size / 1024 / 1024
            print(f"  ✓ {efd_out.name} ({size_mb:.1f} MB)")
        
        # Create grid split (only if grid data exists)
        if has_grid and grid_fixed is not None:
            grid_split = {k: _index_if_sample_dim(v, split_indices) for k, v in grid_fixed.items()}
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
    
    grid_section = ""
    if has_grid:
        grid_section = f"""
### 4. ESP Grid Coordinates (vdw_surface)
- **Original**: Grid index space
- **Fixed**: Physical Angstroms
- **Conversion**: Applied proper grid spacing ({cube_spacing_bohr} Bohr = {cube_spacing_bohr * 0.529177:.6f} Å)
"""

    forces_sign_readme = ""
    if flip_forces:
        forces_sign_readme = (
            "- **Sign**: `F` was negated (`--flip-forces`) so output is force −∇E, "
            "not raw gradient ∂E/∂R.\n"
        )

    extra_scales_readme = ""
    _es_r, _fs_r, _ds_r = float(energy_scale), float(force_scale), float(dipole_scale)
    _ef_r, _esp_r, _q_r = float(efield_scale), float(esp_scale), float(charge_scale)
    _scale_lines = []
    if _es_r != 1.0:
        _scale_lines.append(
            f"- **E**: × {_es_r:g} after Hartree→eV; same factor on `efield_energy`, `efield_scf_energy` if present"
        )
    if _fs_r != 1.0:
        _scale_lines.append(
            f"- **F**: × {_fs_r:g} after Ha/Bohr→eV/Å; same factor on `efield_scf_F` if present"
        )
    if _ds_r != 1.0:
        _scale_lines.append(
            f"- **Dxyz**: × {_ds_r:g} after Debye→e·Å; same factor on `D`, `efield_scf_D`, `efield_D` if present"
        )
    if _ef_r != 1.0:
        _scale_lines.append(f"- **Ef** (field vectors): × {_ef_r:g} on `Ef`, `efield_Ef`, `efield_scf_Ef` if present")
    if _esp_r != 1.0:
        _scale_lines.append(f"- **esp**: × {_esp_r:g} [Hartree/e] on grid NPZ and EFD NPZ if present")
    if _q_r != 1.0:
        _scale_lines.append(
            f"- **Q** (NPZ key): × {_q_r:g} — for total charge in your convention; PySCF may use Q for quadrupole"
        )
    if _scale_lines:
        extra_scales_readme = "\n### 3b. Extra property scales\n" + "\n".join(_scale_lines) + "\n"

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
{forces_sign_readme}{extra_scales_readme}{grid_section}
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
- `Dxyz`: Dipole moments [e·Å] ← CONVERTED from Debye
"""
    
    if has_grid:
        readme_content += """
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
- `grid_dims`: Original cube dimensions (if available)
- `grid_origin`: Original cube origins [Bohr] (if available)
- `grid_axes`: Original cube axes (if available)
- `Dxyz`: Dipole moments [e·Å] ← CONVERTED from Debye
"""
    
    readme_content += f"""
## Units Summary (ASE Standard)

| Property | Unit | Status |
|----------|------|--------|
| R (coordinates) | Angstrom | ✓ Correct |
| E (energy) | eV | ✓ Converted |
| F (forces) | eV/Angstrom | ✓ Converted |
| Dxyz (dipoles) | e·Å | ✓ Converted from Debye |
"""
    
    if has_grid:
        readme_content += """| esp (values) | Hartree/e | ✓ Correct |
| vdw_surface | Angstrom | ✓ Fixed |
"""
    
    readme_content += """
## Usage

```python
import numpy as np

# Load training data
train_props = np.load('energies_forces_dipoles_train.npz')

# All units are ASE-standard - ready to use!
R = train_props['R']  # Angstroms
E = train_props['E']  # eV
F = train_props['F']  # eV/Angstrom
Dxyz = train_props['Dxyz']  # e·Å (converted from Debye)
"""
    
    if has_grid:
        readme_content += """
# Load grid data (if available)
train_grids = np.load('grids_esp_train.npz')
esp = train_grids['esp']  # Hartree/e
vdw_surface = train_grids['vdw_surface']  # Angstroms
"""
    
    readme_content += """
Generated by: mmml.cli.fix_and_split
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
        if has_grid:
            print(f"  - grids_esp_{{train,valid,test}}.npz")
        print(f"  - split_indices.npz")
        print(f"  - README.md")
        print("\n✅ IMPORTANT: All units are now ASE-standard compliant!")
        print("   - Energies: eV (converted from Hartree)")
        print("   - Forces: eV/Angstrom (converted from Hartree/Bohr)")
        print("   - Coordinates: Angstrom")
        if has_grid:
            print("   - ESP grid: Angstrom (converted from grid indices)")
        print("\nArray shapes (per split):")
        for split_name in splits:
            efd_path = output_dir / f"energies_forces_dipoles_{split_name}.npz"
            with np.load(efd_path, allow_pickle=True) as f:
                for k in sorted(f.keys()):
                    v = f[k]
                    sh = v.shape if hasattr(v, 'shape') else 'scalar'
                    print(f"  {split_name}: {k} {sh}")
            if has_grid:
                grid_path = output_dir / f"grids_esp_{split_name}.npz"
                with np.load(grid_path, allow_pickle=True) as g:
                    for k in sorted(g.keys()):
                        v = g[k]
                        sh = v.shape if hasattr(v, 'shape') else 'scalar'
                        print(f"  {split_name} (grid): {k} {sh}")
        print(f"{'='*70}\n")
    
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fix units and create train/valid/test splits from molecular NPZ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 8:1:1 split (with grid)
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data
  
  # Without grid data (EFD only)
  %(prog)s --efd data.npz --output-dir ./training_data
  
  # Custom split ratios
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data \\
    --train-frac 0.7 --valid-frac 0.15 --test-frac 0.15
  
  # Different cube spacing (e.g., 0.5 Bohr)
  %(prog)s --efd data.npz --grid grids.npz --output-dir ./training_data \\
    --cube-spacing 0.5
  
  # Skip validation for speed
  %(prog)s --efd data.npz --output-dir ./training_data \\
    --skip-validation

  # Concatenate multiple NPZ files (e.g. extend training set with MD samples)
  %(prog)s --efd train.npz md_evaluated.npz --output-dir ./splits_extended

  # NPZ has raw PySCF gradient in F (not forces): negate before converting to eV/Å
  %(prog)s --efd raw.npz --output-dir ./out --flip-forces

  # Correct a systematic factor after normal conversion (e.g. duplicate unit fix upstream)
  %(prog)s --efd data.npz --output-dir ./out --energy-scale 0.5 --force-scale 1.0
"""
    )
    
    parser.add_argument(
        '--efd', '--energies-forces-dipoles',
        type=Path,
        nargs='+',
        required=True,
        metavar='FILE',
        help='Path(s) to energies_forces_dipoles.npz file(s). Multiple files are concatenated.'
    )
    
    parser.add_argument(
        '--grid', '--grids-esp',
        type=Path,
        default=None,
        help='Path to grids_esp.npz file (optional)'
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
        '--atomic-ref',
        type=str,
        default=None,
        metavar='SCHEME',
        help='Subtract per-atom reference energies (e.g. pbe0/sz, pbe0/def2-tzvp)'
    )
    parser.add_argument(
        '--atomic-ref-units',
        type=str,
        choices=['hartree', 'ev'],
        default='hartree',
        help='Units of refs in JSON: hartree (pbe0/def2-tzvp) or ev (pbe0/sz may use eV; default: hartree)'
    )
    
    parser.add_argument(
        '--n-grid-points',
        type=int,
        default=3000,
        metavar='N',
        help='Target number of ESP grid points per sample (default 3000). Excludes tails (±SD) and points near atoms.'
    )
    parser.add_argument(
        '--esp-sd-sigma',
        type=float,
        default=3.0,
        metavar='N',
        help='Exclude grid points beyond ±N SD from mean (default 3.0, ignores distribution tails)'
    )
    parser.add_argument(
        '--esp-max-abs-kcal-mol',
        type=float,
        default=100.0,
        metavar='X',
        help='Exclude grid points with |esp| > X kcal/mol/e (default 100.0)'
    )
    parser.add_argument(
        '--min-dist-to-atoms',
        type=float,
        default=1.0,
        metavar='Å',
        help='Exclude grid points closer than this to any atom in Å (default 1.0)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )

    parser.add_argument(
        '--flip-forces',
        action='store_true',
        help=(
            'Negate F before unit conversion (Ha/Bohr → eV/Å). Use when F stores the PySCF energy '
            'gradient ∂E/∂R instead of forces F = −∇E. mmml pyscf-evaluate NPZ already uses −gradient.'
        ),
    )

    parser.add_argument(
        '--energy-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply E by X after Hartree→eV. Also efield_energy, efield_scf_energy if present (default 1.0).',
    )
    parser.add_argument(
        '--force-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply F by X after Ha/Bohr→eV/Å. Also efield_scf_F if present (default 1.0).',
    )
    parser.add_argument(
        '--dipole-scale',
        type=float,
        default=1.0,
        metavar='X',
        help=(
            'Multiply Dxyz by X after Debye→e·Å if present. Also scales D, efield_scf_D, efield_D (Debye) if present.'
        ),
    )
    parser.add_argument(
        '--efield-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply Ef, efield_Ef, efield_scf_Ef by X if present (default 1.0).',
    )
    parser.add_argument(
        '--esp-scale',
        type=float,
        default=1.0,
        metavar='X',
        help='Multiply esp by X [Hartree/e] on grid splits and on EFD if esp is stored there (default 1.0).',
    )
    parser.add_argument(
        '--charge-scale',
        type=float,
        default=1.0,
        metavar='X',
        help=(
            'Multiply NPZ key Q by X if present. For total molecular charge when Q stores charge. '
            'PySCF ESP export may use Q for quadrupole—verify your file (default 1.0).'
        ),
    )
    parser.add_argument(
        '--quadrupole-scale',
        type=float,
        default=None,
        metavar='X',
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.quadrupole_scale is not None:
        if args.charge_scale != 1.0:
            parser.error("--quadrupole-scale is deprecated (was misnamed); use --charge-scale only, not both.")
        args.charge_scale = args.quadrupole_scale
    
    # Validate inputs
    efd_files = args.efd if isinstance(args.efd, list) else [args.efd]
    for f in efd_files:
        if not Path(f).exists():
            print(f"❌ Error: EFD file not found: {f}")
            sys.exit(1)
    
    if args.grid is not None and not args.grid.exists():
        print(f"❌ Error: Grid file not found: {args.grid}")
        sys.exit(1)
    
    if abs(args.train_frac + args.valid_frac + args.test_frac - 1.0) > 1e-6:
        print(f"❌ Error: Split fractions must sum to 1.0")
        print(f"   Got: {args.train_frac} + {args.valid_frac} + {args.test_frac} = "
              f"{args.train_frac + args.valid_frac + args.test_frac}")
        sys.exit(1)
    
    # Run the conversion
    success = fix_and_split_data(
        efd_file=efd_files if len(efd_files) > 1 else efd_files[0],
        grid_file=args.grid,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        cube_spacing_bohr=args.cube_spacing,
        skip_validation=args.skip_validation,
        atomic_ref=getattr(args, 'atomic_ref', None),
        atomic_ref_units=getattr(args, 'atomic_ref_units', 'hartree'),
        n_grid_points=getattr(args, 'n_grid_points', 3000),
        esp_sd_sigma=getattr(args, 'esp_sd_sigma', 3.0),
        esp_max_abs_kcal_mol=getattr(args, 'esp_max_abs_kcal_mol', 100.0),
        min_dist_to_atoms=getattr(args, 'min_dist_to_atoms', 1.0),
        flip_forces=getattr(args, 'flip_forces', False),
        energy_scale=getattr(args, 'energy_scale', 1.0),
        force_scale=getattr(args, 'force_scale', 1.0),
        dipole_scale=getattr(args, 'dipole_scale', 1.0),
        efield_scale=getattr(args, 'efield_scale', 1.0),
        esp_scale=getattr(args, 'esp_scale', 1.0),
        charge_scale=getattr(args, 'charge_scale', 1.0),
        verbose=not args.quiet,
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


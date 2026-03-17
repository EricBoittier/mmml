#!/usr/bin/env python
"""
Verify esp-grid alignment at the data generation level.

Recomputes ESP at grid points using PySCF and compares to stored values.
If aligned: high correlation. If misaligned (bug in pyscf-evaluate): low correlation.

Usage:
    mmml verify-esp-alignment -i 07_evaluated.npz
    mmml verify-esp-alignment -i 07_evaluated.npz --sample 0 --n-points 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def compute_esp_at_grid_pyscf(R_angstrom: np.ndarray, Z: np.ndarray, grid_angstrom: np.ndarray,
                              basis: str = "def2-SVP", xc: str = "PBE0") -> np.ndarray:
    """
    Compute DFT ESP at grid points using PySCF (CPU).
    Returns ESP in Hartree/e, same convention as pyscf-evaluate.
    """
    import pyscf.gto
    from pyscf.dft import rks
    from ase.data import chemical_symbols

    atom = [(chemical_symbols[z], tuple(row)) for z, row in zip(Z, R_angstrom)]
    mol = pyscf.gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = rks.RKS(mol, xc=xc).run()

    grid_bohr = grid_angstrom * 0.529177
    dm = mf.make_rdm1()
    coords = mol.atom_coords(unit="Bohr")
    charges = mol.atom_charges()

    esp = np.zeros(len(grid_angstrom), dtype=np.float64)
    for i, r in enumerate(grid_bohr):
        dr = coords - r[None, :]
        dist = np.linalg.norm(dr, axis=1) + 1e-12
        v_nuc = np.sum(charges / dist)
        with mol.with_rinv_origin(r):
            v_mat = mol.intor("int1e_rinv")
        v_elec = np.einsum("ij,ij", dm, v_mat)
        esp[i] = v_nuc - v_elec
    return esp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify esp-grid alignment in evaluated NPZ (data generation check).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input NPZ (e.g. 07_evaluated.npz)")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to check (default 0)")
    parser.add_argument("--n-points", type=int, default=200,
                        help="Number of grid points to verify (default 200, for speed)")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Basis (default def2-SVP)")
    parser.add_argument("--xc", type=str, default="PBE0", help="XC functional (default PBE0)")
    parser.add_argument("--grid-in-angstrom", action="store_true",
                        help="Grid coords already in Angstrom (e.g. from fix-and-split). Default: Bohr (pyscf-evaluate)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1

    data = dict(np.load(args.input, allow_pickle=True))
    if "esp" not in data or "esp_grid" not in data:
        print("Error: NPZ must contain 'esp' and 'esp_grid'", file=sys.stderr)
        return 1

    R = data["R"]
    Z = data["Z"]
    esp = data["esp"]
    esp_grid = data["esp_grid"]

    if R.ndim == 2:
        R = R[np.newaxis, ...]
    if Z.ndim == 2:
        Z = Z[0]

    i = args.sample
    if i >= R.shape[0]:
        print(f"Error: sample {i} out of range (max {R.shape[0]-1})", file=sys.stderr)
        return 1

    R_i = R[i]
    esp_i = esp[i]
    grid_i = esp_grid[i]

    # Mask padding
    valid = np.all(np.abs(grid_i) < 1e5, axis=1)
    n_valid = np.sum(valid)
    if n_valid < 10:
        print("Error: Too few valid grid points", file=sys.stderr)
        return 1

    # Grid from pyscf-evaluate is in Bohr; convert to Angstrom for compute_esp
    valid_idx = np.where(valid)[0]
    n_check = min(args.n_points, len(valid_idx))
    rng = np.random.default_rng(42)
    idx = rng.choice(valid_idx, size=n_check, replace=False)

    grid_subset = grid_i[idx]
    BOHR_TO_ANGSTROM = 0.529177
    grid_subset_angstrom = grid_subset if args.grid_in_angstrom else grid_subset * BOHR_TO_ANGSTROM

    esp_stored = esp_i[idx]
    print(f"Verifying sample {i}: {n_check} points")
    print(f"  Recomputing ESP with PySCF ({args.basis}/{args.xc})...")

    try:
        esp_recomputed = compute_esp_at_grid_pyscf(R_i, Z, grid_subset_angstrom, basis=args.basis, xc=args.xc)
    except Exception as e:
        print(f"Error: PySCF failed: {e}", file=sys.stderr)
        return 1

    corr = np.corrcoef(esp_stored, esp_recomputed)[0, 1]
    rmse = np.sqrt(np.mean((esp_stored - esp_recomputed) ** 2))

    # Isolate V_nuc (no SCF): V_nuc = sum Z_a / |r - R_a|. If aligned, stored_esp = V_nuc - V_elec.
    BOHR = 0.529177
    coords_bohr = grid_subset_angstrom * BOHR
    R_bohr = R_i * BOHR
    atoms_valid = np.any(R_i != 0, axis=1)
    charges = np.asarray(Z)
    if charges.ndim == 0 or charges.size == 1:
        charges = np.full(np.sum(atoms_valid), int(np.asarray(Z).flat[0]))
    else:
        charges = np.asarray(Z)[atoms_valid]
    R_valid = R_bohr[atoms_valid]
    v_nuc_at_grid = np.zeros(len(grid_subset_angstrom))
    for j, r in enumerate(coords_bohr):
        dist = np.linalg.norm(R_valid - r[None, :], axis=1) + 1e-12
        v_nuc_at_grid[j] = np.sum(charges / dist)
    corr_nuc = np.corrcoef(esp_stored, v_nuc_at_grid)[0, 1]
    # Implied V_elec from stored: stored_esp = V_nuc - V_elec => V_elec = V_nuc - stored_esp
    v_elec_recomputed = v_nuc_at_grid - esp_recomputed
    v_elec_implied = v_nuc_at_grid - esp_stored
    corr_elec = np.corrcoef(v_elec_implied, v_elec_recomputed)[0, 1]

    print(f"  Pearson correlation (full ESP): {corr:.4f}")
    print(f"  Pearson correlation (V_nuc only): {corr_nuc:.4f}")
    print(f"  Pearson correlation (V_elec implied vs recomputed): {corr_elec:.4f}")
    print(f"  RMSE (Hartree/e):   {rmse:.6e}")
    if corr > 0.99:
        print("  ✓ Alignment OK: esp and grid are correctly paired (data generation is fine)")
    elif corr > 0.9:
        print("  ⚠ Correlation good but not perfect (basis/xc mismatch or numerical noise)")
    else:
        print("  ✗ MISALIGNMENT: Low correlation suggests esp[i] and grid[i] are not the same point.")
        print("    Bug likely in pyscf-evaluate (gpu4pyscf get_j_int3c2e_pass1 output order).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

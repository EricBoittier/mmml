#!/usr/bin/env python
"""
Minimal reproducer for gpu4pyscf ESP methods.

Tests:
1. get_j_int3c2e_pass1 (fakemol aux): BUG - esp/grid misalignment
2. int1e_grids (direct grid): expected correct alignment

Reference: CPU int1e_rinv gives correct esp[i] <-> grid[i] pairing.

Run: python tests/reproducers/gpu4pyscf_int3c2e_esp_ordering_bug.py

Requires: pyscf, gpu4pyscf, cupy (and CUDA-capable GPU)
"""

import numpy as np
from pyscf import gto
from pyscf.dft import rks
from gpu4pyscf.df import int3c2e
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.lib.cupy_helper import dist_matrix
import cupy


def esp_cpu_reference(mol, dm, grid_bohr):
    """ESP at grid points via int1e_rinv (correct alignment)."""
    coords = mol.atom_coords(unit="Bohr")
    charges = mol.atom_charges()
    dm_np = np.asarray(dm) if hasattr(dm, "get") else dm
    esp = np.zeros(len(grid_bohr), dtype=np.float64)
    for i, r in enumerate(grid_bohr):
        dr = coords - r[None, :]
        dist = np.linalg.norm(dr, axis=1) + 1e-12
        v_nuc = np.sum(charges / dist)
        with mol.with_rinv_origin(r):
            v_mat = mol.intor("int1e_rinv")
        v_elec = np.einsum("ij,ij", dm_np, v_mat)
        esp[i] = v_nuc - v_elec
    return esp


def esp_gpu_int3c2e(mol, dm, grid_bohr):
    """ESP at grid points via get_j_int3c2e_pass1 (buggy ordering)."""
    fakemol = gto.fakemol_for_charges(grid_bohr)
    coords_bohr = fakemol.atom_coords(unit="B")
    charges = mol.atom_charges()
    charges_cp = cupy.asarray(charges)
    mol_coords_bohr = cupy.asarray(mol.atom_coords(unit="B"))
    r = dist_matrix(mol_coords_bohr, coords_bohr)
    rinv = 1.0 / r
    intopt = int3c2e.VHFOpt(mol, fakemol, "int2e")
    intopt.build(1e-14, diag_block_with_triu=False, aosym=True, group_size=256)
    v_grids_e = 2.0 * int3c2e.get_j_int3c2e_pass1(intopt, dm, sort_j=True)
    v_grids_n = cupy.dot(charges_cp, rinv)
    return (v_grids_n - v_grids_e).get()


def esp_gpu_int1e_grids(mol, dm, grid_bohr):
    """ESP at grid points via int1e_grids (direct grid, correct ordering)."""
    v_elec = int1e_grids(mol, grid_bohr, dm=dm)
    v_elec = v_elec.get() if hasattr(v_elec, "get") else np.asarray(v_elec)
    charges = mol.atom_charges()
    atom_coords = mol.atom_coords(unit="Bohr")
    v_nuc = np.array(
        [
            np.sum(charges / (np.linalg.norm(atom_coords - r, axis=1) + 1e-12))
            for r in grid_bohr
        ],
        dtype=np.float64,
    )
    return v_nuc - v_elec


def main():
    # Water molecule
    mol = gto.M(
        atom=[
            ["O", 0.0, 0.0, 0.0],
            ["H", 0.0, -0.757, 0.587],
            ["H", 0.0, 0.757, 0.587],
        ],
        basis="def2-SVP",
        unit="Angstrom",
        verbose=0,
    )

    # Run SCF (use CPU RKS for simplicity; dm is numpy)
    mf = rks.RKS(mol, xc="PBE0").run()
    dm = mf.make_rdm1()

    # Grid points in Bohr (a few points around the molecule)
    grid_bohr = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Reference: CPU int1e_rinv (correct)
    esp_cpu = esp_cpu_reference(mol, dm, grid_bohr)

    # GPU: get_j_int3c2e_pass1 (buggy ordering)
    dm_gpu = cupy.asarray(dm)
    esp_int3c2e = esp_gpu_int3c2e(mol, dm_gpu, grid_bohr)

    # GPU: int1e_grids (direct grid, expected correct)
    esp_int1e = esp_gpu_int1e_grids(mol, dm_gpu, grid_bohr)

    # Compare
    corr_int3c2e = np.corrcoef(esp_cpu, esp_int3c2e)[0, 1]
    corr_int1e = np.corrcoef(esp_cpu, esp_int1e)[0, 1]
    rmse_int3c2e = np.sqrt(np.mean((esp_cpu - esp_int3c2e) ** 2))
    rmse_int1e = np.sqrt(np.mean((esp_cpu - esp_int1e) ** 2))

    print("gpu4pyscf ESP methods comparison")
    print("=" * 60)
    print(f"Grid points: {len(grid_bohr)}")
    print(f"  CPU (int1e_rinv):      {esp_cpu}")
    print(f"  GPU (get_j_int3c2e):   {esp_int3c2e}  corr={corr_int3c2e:.4f} rmse={rmse_int3c2e:.2e}")
    print(f"  GPU (int1e_grids):    {esp_int1e}  corr={corr_int1e:.4f} rmse={rmse_int1e:.2e}")
    print("-" * 60)
    print(f"  int3c2e: {'✓ PASS' if corr_int3c2e > 0.99 else '✗ BUG (misaligned)'}")
    print(f"  int1e_grids: {'✓ PASS' if corr_int1e > 0.99 else '✗ FAIL'}")
    print("=" * 60)
    return 0 if corr_int1e > 0.99 else 1


if __name__ == "__main__":
    exit(main())

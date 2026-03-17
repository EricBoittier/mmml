#!/usr/bin/env python
"""
Minimal reproducer for gpu4pyscf get_j_int3c2e_pass1 ESP/grid ordering bug.

When using fakemol_for_charges(coords) as auxmol, the output of get_j_int3c2e_pass1
with sort_j=True does NOT match the grid point order. esp[i] is not the ESP at grid[i].

Reference: CPU int1e_rinv gives correct esp[i] <-> grid[i] pairing.
GPU path: get_j_int3c2e_pass1 gives misaligned output.

Run: python tests/reproducers/gpu4pyscf_int3c2e_esp_ordering_bug.py

Requires: pyscf, gpu4pyscf, cupy (and CUDA-capable GPU)

For GitHub issue, paste this script and the output showing low correlation.
"""

import numpy as np
from pyscf import gto
from pyscf.dft import rks
from gpu4pyscf.df import int3c2e
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
    esp_gpu = esp_gpu_int3c2e(mol, dm_gpu, grid_bohr)

    # Compare
    corr = np.corrcoef(esp_cpu, esp_gpu)[0, 1]
    rmse = np.sqrt(np.mean((esp_cpu - esp_gpu) ** 2))

    print("gpu4pyscf int3c2e ESP ordering bug reproducer")
    print("=" * 60)
    print(f"Grid points: {len(grid_bohr)}")
    print(f"  CPU (int1e_rinv):     {esp_cpu}")
    print(f"  GPU (get_j_int3c2e):  {esp_gpu}")
    print(f"  Pearson correlation:  {corr:.6f}")
    print(f"  RMSE (Hartree/e):     {rmse:.6e}")
    if corr > 0.99:
        print("  ✓ PASS: esp and grid align correctly")
    else:
        print("  ✗ BUG: esp[i] does not correspond to grid[i]")
        print("  Expected: correlation ~1.0 when alignment is correct")
    print("=" * 60)
    return 0 if corr > 0.99 else 1


if __name__ == "__main__":
    exit(main())

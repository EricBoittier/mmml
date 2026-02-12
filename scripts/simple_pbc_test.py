#!/usr/bin/env python
"""
Simple PBC test: load init-packmol.pdb, set up calculator with cell, run energy invariance check.

Usage:
  python scripts/simple_pbc_test.py
  python scripts/simple_pbc_test.py --checkpoint /path/to/ckpt --pdb pdb/init-packmol.pdb

Run make_box first to create pdb/init-packmol.pdb.
"""
from pathlib import Path
import argparse
import numpy as np

# Optional: enable JAX 64-bit for consistency
# import os
# os.environ["JAX_ENABLE_X64"] = "True"

DEFAULT_CHECKPOINT = Path("/pchem-data/meuwly/boittier/home/ckpts/run-983a1749-5cc7-4f50-856c-458e47744d71")

def main():
    parser = argparse.ArgumentParser(description="Simple PBC test")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pdb", type=Path, default=Path("pdb/init-packmol.pdb"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--L", type=float, default=28.0)
    args = parser.parse_args()

    config = {"RES": "ACO", "N": args.n, "L": args.L}
    checkpoint = args.checkpoint
    pdb_path = args.pdb

    if not pdb_path.exists():
        print(f"PDB not found: {pdb_path}")
        print("Run make_box first: python -m mmml.cli.make_box --res ACO --n 10 --side_length 28")
        return 1

    # Load PSF into CHARMM (required for MM; make_box does this too)
    from mmml.pycharmmInterface.setupBox import setup_box_generic
    setup_box_generic(str(pdb_path), side_length=config["L"], tag="aco")

    import ase.io
    from mmml.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.pycharmmInterface.cutoffs import CutoffParameters

    atoms = ase.io.read(str(pdb_path))
    Z = atoms.get_atomic_numbers()
    R = atoms.get_positions()
    n_atoms = len(Z)
    n_monomers = args.n
    n_atoms_monomer = n_atoms // n_monomers

    assert n_atoms == n_monomers * n_atoms_monomer, f"Atom count mismatch: {n_atoms} != {n_monomers} * {n_atoms_monomer}"

    factory = setup_calculator(
        ATOMS_PER_MONOMER=n_atoms_monomer,
        N_MONOMERS=n_monomers,
        doML=True,
        doMM=True,
        doML_dimer=True,
        model_restart_path=checkpoint,
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        cell=config["L"],
    )

    cutoff_params = CutoffParameters(ml_cutoff=1.0, mm_switch_on=8.0, mm_cutoff=5.0)
    calc, _ = factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
        do_pbc_map=getattr(factory, "do_pbc_map", True),
        pbc_map=getattr(factory, "pbc_map", None),
    )

    atoms.set_cell([config["L"], config["L"], config["L"]])
    atoms.set_pbc(True)
    atoms.calc = calc

    # Energy and forces
    E0 = atoms.get_potential_energy()
    F0 = atoms.get_forces()
    print(f"Energy (eV): {E0:.6f}")
    print(f"Forces shape: {F0.shape}, max |F|: {np.max(np.linalg.norm(F0, axis=1)):.6f}")

    # PBC invariance: translate monomer 0 by lattice vector
    L = config["L"]
    a = np.array([L, 0.0, 0.0])
    g0 = np.arange(n_atoms_monomer)
    R_shift = R.copy()
    R_shift[g0] += a
    atoms.set_positions(R_shift)
    E1 = atoms.get_potential_energy()
    atoms.set_positions(R)  # restore

    delta_E = float(E1 - E0)
    print(f"PBC invariance: E0={E0:.6f}, E1={E1:.6f}, delta={delta_E:.6e}")
    if abs(delta_E) < 1e-4:
        print("PASS: Energy invariant under lattice translation")
    else:
        print(f"WARN: Energy changed by {delta_E:.6e} (expected < 1e-4)")

    return 0


if __name__ == "__main__":
    exit(main())

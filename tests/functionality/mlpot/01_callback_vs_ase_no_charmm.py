#!/usr/bin/env python3
"""
Step 1: compare ASE calculator vs PyCharmm_Calculator.calculate_charmm (no MLpot registration).

Isolates the Python callback path from CHARMM's energy driver.
"""

from __future__ import annotations

import argparse
import sys

import ase
import e3x
import numpy as np

from _common import (
    add_cluster_args,
    build_ase_cluster,
    load_physnet_for_cluster,
    print_header,
    resolve_checkpoint,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol
from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc, get_pyc


def _run_calculate_charmm(pyc, positions: np.ndarray, z: np.ndarray) -> tuple[float, np.ndarray]:
    n = len(z)
    dst, src = e3x.ops.sparse_pairwise_indices(n)
    n_pairs = len(dst)

    x = positions[:, 0].astype(float)
    y = positions[:, 1].astype(float)
    zc = positions[:, 2].astype(float)
    dx = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    dz = np.zeros(n, dtype=float)
    idxp = np.arange(n, dtype=int)

    energy_kcal = pyc.calculate_charmm(
        Natom=n,
        Ntrans=0,
        Natim=n,
        idxp=idxp,
        x=x,
        y=y,
        z=zc,
        dx=dx,
        dy=dy,
        dz=dz,
        Nmlp=n_pairs,
        Nmlmmp=0,
        idxi=dst,
        idxj=src,
        idxjp=idxp[:n_pairs],
        idxu=np.zeros(0, dtype=int),
        idxv=np.zeros(0, dtype=int),
        idxup=np.zeros(0, dtype=int),
        idxvp=np.zeros(0, dtype=int),
    )
    forces_results = np.asarray(pyc.results["forces"], dtype=float)
    forces_dx = -np.stack([dx, dy, dz], axis=-1)
    return float(energy_kcal), forces_results, forces_dx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.02,
        help="Relative tolerance energy (ASE eV vs callback kcal/mol)",
    )
    parser.add_argument(
        "--force-atol",
        type=float,
        default=5.0,
        help="Max |F_ASE - F_callback| in kcal/mol/Å",
    )
    args = parser.parse_args()

    print_header("ASE vs calculate_charmm (no CHARMM ENER)")
    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)
    print(f"Cluster: {args.residue} x {args.n_molecules} -> {n_atoms} atoms")
    print(f"Checkpoint: {ckpt}")

    params, model = load_physnet_for_cluster(ckpt, n_atoms)
    model.natoms = n_atoms
    atoms = ase.Atoms(numbers=z, positions=r)

    ase_calc = get_ase_calc(
        params,
        model,
        atoms,
        conversion={"energy": ev2kcalmol, "forces": ev2kcalmol},
    )
    atoms.calc = ase_calc
    _ = atoms.get_potential_energy()  # trigger calculate()
    # Read stored properties (kcal/mol) — same convention as get_pyc / CHARMM.
    e_ase_kcal = float(ase_calc.results["energy"])
    f_ase_kcal = np.asarray(ase_calc.results["forces"], dtype=float)
    e_model_ev = e_ase_kcal / ev2kcalmol

    pyCModel = get_pyc(params, model, atoms)
    pyc = pyCModel.get_pycharmm_calculator()
    e_cb_kcal, f_cb_kcal, f_dx_kcal = _run_calculate_charmm(pyc, r, z)

    e_scale = max(abs(e_ase_kcal), abs(e_cb_kcal), 1e-6)
    f_diff = np.abs(f_ase_kcal - f_cb_kcal).max()
    f_dx_diff = np.abs(f_ase_kcal - f_dx_kcal).max()

    print(f"\n  model energy (eV, ref):    {e_model_ev:.8f}")
    print(f"  ASE energy (kcal/mol):     {e_ase_kcal:.8f}")
    print(f"  callback energy (kcal/mol): {e_cb_kcal:.8f}")
    print(f"  |dE| (kcal/mol):           {abs(e_ase_kcal - e_cb_kcal):.8f}")
    print(f"  max |dF| results vs ASE:   {f_diff:.8f}")
    print(f"  max |dF| dx-array vs ASE:   {f_dx_diff:.8f}")

    energy_ok = abs(e_ase_kcal - e_cb_kcal) <= args.rtol * e_scale
    force_ok = f_diff <= args.force_atol
    force_dx_ok = f_dx_diff <= args.force_atol

    if energy_ok and force_ok:
        print("\nPASS: energy and forces match ASE.")
        return 0

    print("\nFAIL:")
    if energy_ok:
        print("  energy: PASS")
    else:
        print(f"  energy: FAIL (rtol={args.rtol}, scale={e_scale:.4f})")
    if force_ok:
        print("  forces (model results): PASS")
    else:
        print(f"  forces (model results): FAIL (atol={args.force_atol} kcal/mol/Å)")
    if not force_dx_ok:
        print(f"  forces (CHARMM dx scatter): FAIL (atol={args.force_atol})")
    return 1


if __name__ == "__main__":
    sys.exit(main())

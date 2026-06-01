#!/usr/bin/env python3
"""
Step 3: numeric comparison — ASE vs direct callback vs CHARMM after MLpot.

For an all-ML cluster, bonded MM terms are stripped; total ENER should be
dominated by the ML callback (often reported as USER in CHARMM).
"""

from __future__ import annotations

import argparse
import sys

import ase
import e3x
import numpy as np

from _common import (
    add_cluster_args,
    all_atom_selection,
    build_ase_cluster,
    charmm_energy_row,
    check_mlpot_symbols,
    load_physnet_for_cluster,
    print_header,
    resolve_checkpoint,
    setup_charmm_nbonds,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol
from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc, get_pyc


def _callback_energy_forces(pyc, positions: np.ndarray) -> tuple[float, np.ndarray]:
    n = positions.shape[0]
    dst, src = e3x.ops.sparse_pairwise_indices(n)
    n_pairs = len(dst)
    x = positions[:, 0].astype(float)
    y = positions[:, 1].astype(float)
    zc = positions[:, 2].astype(float)
    dx = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    dz = np.zeros(n, dtype=float)
    idxp = np.arange(n, dtype=int)
    e_kcal = pyc.calculate_charmm(
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
        idxu=[],
        idxv=[],
        idxup=[],
        idxvp=[],
    )
    return float(e_kcal), -np.stack([dx, dy, dz], axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    parser.add_argument("--rtol", type=float, default=0.05, help="Relative E tolerance")
    parser.add_argument(
        "--force-atol",
        type=float,
        default=10.0,
        help="Max force diff kcal/mol/Å (callback vs ASE)",
    )
    args = parser.parse_args()

    print_header("Energy comparison: ASE / callback / CHARMM+MLpot")
    if check_mlpot_symbols():
        print("FAIL: MLpot symbols missing on libcharmm")
        return 1

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.energy as energy

    setup_charmm_nbonds()

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
    e_ase_kcal = float(atoms.get_potential_energy()) * ev2kcalmol
    f_ase_kcal = atoms.get_forces() * ev2kcalmol

    pyCModel = get_pyc(params, model, atoms)
    pyc = pyCModel.get_pycharmm_calculator()
    e_cb_kcal, f_cb_kcal = _callback_energy_forces(pyc, r)

    ml_sel = all_atom_selection()
    mlpot = pycharmm.MLpot(
        ml_model=pyCModel,
        ml_Z=list(z),
        ml_selection=ml_sel,
        ml_charge=0,
        ml_fq=True,
    )
    energy.show()
    terms = charmm_energy_row()
    mlpot.unset_mlpot()

    e_charmm_total = terms.get("ENER", float("nan"))
    e_user = terms.get("USER", float("nan"))
    # MM bonded terms should be ~0 after MLpot PSF edits
    bonded = sum(terms.get(k, 0.0) for k in ("BOND", "ANGL", "UREY", "DIHE", "IMPR"))

    print(f"\n  ASE E (kcal/mol):        {e_ase_kcal:.8f}")
    print(f"  callback E (kcal/mol):   {e_cb_kcal:.8f}")
    print(f"  CHARMM ENER (kcal/mol):  {e_charmm_total:.8f}")
    print(f"  CHARMM USER (kcal/mol):  {e_user:.8f}")
    print(f"  sum bonded MM terms:     {bonded:.8f}")
    print(f"  max |F_ASE - F_cb|:      {np.abs(f_ase_kcal - f_cb_kcal).max():.8f}")

    ref = max(abs(e_ase_kcal), 1e-6)
    ok_ase_cb = abs(e_ase_kcal - e_cb_kcal) <= args.rtol * ref
    ok_cb_user = (
        np.isfinite(e_user)
        and abs(e_cb_kcal - e_user) <= args.rtol * max(abs(e_cb_kcal), 1e-6)
    )
    ok_total = (
        np.isfinite(e_charmm_total)
        and abs(e_ase_kcal - e_charmm_total) <= args.rtol * ref
    )

    print("\nChecks:")
    print(f"  ASE vs callback:     {'PASS' if ok_ase_cb else 'FAIL'}")
    print(f"  callback vs USER:  {'PASS' if ok_cb_user else 'FAIL/SKIP (no USER column)'}")
    print(f"  ASE vs CHARMM ENER: {'PASS' if ok_total else 'FAIL'}")

    if ok_ase_cb and (ok_cb_user or ok_total):
        print("\nPASS: MLpot path is in the right ballpark.")
        return 0

    print(
        "\nFAIL or partial — capture output and we can narrow "
        "(units, pair list from CHARMM vs e3x, USER term name)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

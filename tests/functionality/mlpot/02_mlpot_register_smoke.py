#!/usr/bin/env python3
"""
Step 2: build CHARMM cluster, register pycharmm.MLpot on all atoms, run ENER.

Confirms MLpot hooks into libcharmm without requiring energy agreement yet.
"""

from __future__ import annotations

import argparse
import sys

import ase
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    args = parser.parse_args()

    print_header("MLpot registration smoke test")
    missing = check_mlpot_symbols()
    if missing:
        print(f"FAIL: missing libcharmm symbols: {missing}")
        return 1

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)
    print(f"Cluster: {n_atoms} atoms, checkpoint: {ckpt}")

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    import pycharmm.coor as coor
    import pycharmm.energy as energy

    setup_charmm_nbonds()

    params, model = load_physnet_for_cluster(ckpt, n_atoms)
    model.natoms = n_atoms
    atoms = ase.Atoms(numbers=z, positions=r)

    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

    pyCModel = get_pyc(params, model, atoms)
    ml_sel = all_atom_selection()

    print("Registering MLpot (all atoms ML)...")
    mlpot = pycharmm.MLpot(
        ml_model=pyCModel,
        ml_Z=list(z),
        ml_selection=ml_sel,
        ml_charge=0,
        ml_fq=True,
    )
    assert mlpot.is_set, "MLpot.is_set should be True after __init__"

    print("Running CHARMM energy.show() ...")
    try:
        energy.show()
    except Exception as exc:
        print(f"FAIL: energy.show() raised: {exc}")
        mlpot.unset_mlpot()
        return 1

    terms = charmm_energy_row()
    print("\nCHARMM energy terms (kcal/mol):")
    for key in sorted(terms):
        if abs(terms[key]) > 1e-8 or key in ("ENER", "USER", "TOTE"):
            print(f"  {key:12s} {terms[key]:14.6f}")

    pos = coor.get_positions().to_numpy(dtype=float)
    print(f"\nPositions unchanged after ENER: {np.allclose(pos, r)}")

    mlpot.unset_mlpot()
    print("\nPASS: MLpot registered, ENER completed, unset OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Step 4 (stub): SD minimization with MLpot active.

Default: print planned workflow only. Pass ``--run`` to execute a few SD steps.
ML region and fixed atoms use CHARMM ``resid`` (default: residue 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _common import (
    add_cluster_args,
    build_ase_cluster,
    print_header,
    resolve_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute minimization (default: dry-run stub message only)",
    )
    parser.add_argument("--nstep", type=int, default=10, help="SD steps when --run")
    parser.add_argument(
        "--ml-resid",
        type=int,
        default=1,
        help="Residue ID for MLpot region and cons_fix (CHARMM resid)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/functionality/mlpot/output/minimize"),
    )
    args = parser.parse_args()

    print_header("MLpot minimization stub (step 4)")
    out_dir = args.out_dir.resolve()
    pdb_path = out_dir / f"mini_resid{args.ml_resid}.pdb"
    crd_path = out_dir / f"mini_resid{args.ml_resid}.crd"

    print("Workflow (see mmml/interfaces/pycharmmInterface/mlpot/dynamics.py):")
    print(f"  1. register_mlpot on resid {args.ml_resid}")
    print(f"  2. cons_fix.setup(resid {args.ml_resid})")
    print("  3. minimize.run_sd(...); cons_fix.turn_off(); minimize.run_sd(...)")
    print(f"  4. write CRD/PDB -> {crd_path}")

    if not args.run:
        print("\nSTUB: pass --run to execute a short SD test.")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    out_dir.mkdir(parents=True, exist_ok=True)

    import ase
    from mmml.interfaces.pycharmmInterface.mlpot import (
        MinimizeWithMlpotConfig,
        load_physnet_mlpot_bundle,
        minimize_with_mlpot,
        register_mlpot,
        select_by_resid,
        setup_default_nbonds,
    )

    setup_default_nbonds()
    ml_sel = select_by_resid(args.ml_resid)
    ml_idx = np.array(ml_sel.get_atom_indexes(), dtype=int)
    if ml_idx.size == 0:
        print(f"FAIL: no atoms in resid {args.ml_resid}")
        return 1

    ml_z = z[ml_idx]
    ml_r = r[ml_idx]
    n_ml = len(ml_idx)
    print(f"MLpot on resid {args.ml_resid}: {n_ml} atoms (indices {ml_idx.min()}..{ml_idx.max()})")

    atoms_ml = ase.Atoms(numbers=ml_z, positions=ml_r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_ml, atoms_ml)
    ctx = register_mlpot(pyCModel, ml_z, ml_sel)
    try:
        ran = minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                fixed_ml_selection=select_by_resid(args.ml_resid),
                nstep=args.nstep,
                nprint=max(1, args.nstep // 2),
                pdb_path=pdb_path,
                crd_path=crd_path,
                skip_if_crd_exists=False,
            )
        )
    finally:
        ctx.unset()

    print(f"\nMinimization ran={ran}; files under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Step 4 (stub): SD minimization with MLpot on the full system.

- MLpot: all atoms (full PhysNet cluster).
- cons_fix: only ``resid`` 1 (default) to test fixed-atom constraints.

Pass ``--run`` to execute a short SD test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--nstep", type=int, default=10, help="SD steps per SD pass")
    parser.add_argument(
        "--fix-resid",
        type=int,
        default=1,
        help="Residue ID held fixed with cons_fix during the first SD pass",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/functionality/mlpot/output/minimize"),
    )
    args = parser.parse_args()

    print_header("MLpot minimization stub (step 4)")
    out_dir = args.out_dir.resolve()
    pdb_path = out_dir / "mini_full_mlpot.pdb"
    crd_path = out_dir / "mini_full_mlpot.crd"

    print("Workflow:")
    print("  1. register_mlpot on ALL atoms")
    print(f"  2. cons_fix.setup(resid {args.fix_resid})  # constraint test only")
    print("  3. minimize.run_sd; cons_fix.turn_off(); minimize.run_sd")
    print(f"  4. write CRD/PDB -> {crd_path}")

    if not args.run:
        print("\nSTUB: pass --run to execute.")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)
    out_dir.mkdir(parents=True, exist_ok=True)

    import ase
    from mmml.interfaces.pycharmmInterface.mlpot import (
        MinimizeWithMlpotConfig,
        load_physnet_mlpot_bundle,
        minimize_with_mlpot,
        register_mlpot,
        select_all_atoms,
        select_by_resid,
        setup_default_nbonds,
    )

    setup_default_nbonds()
    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)

    fix_sel = select_by_resid(args.fix_resid)
    if len(fix_sel.get_atom_indexes()) == 0:
        print(f"FAIL: no atoms in fix-resid {args.fix_resid}")
        return 1

    print(f"MLpot: all {n_atoms} atoms | cons_fix: resid {args.fix_resid}")
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        ran = minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                fixed_ml_selection=fix_sel,
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

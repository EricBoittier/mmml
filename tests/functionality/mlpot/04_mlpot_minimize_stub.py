#!/usr/bin/env python3
"""
Step 4 (stub): SD minimization with MLpot active.

Default: print planned workflow only. Pass ``--run`` to execute a few SD steps
on the ACO dimer cluster (all atoms ML, no fixed segment).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    add_cluster_args,
    build_ase_cluster,
    load_physnet_for_cluster,
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
        "--out-dir",
        type=Path,
        default=Path("tests/functionality/mlpot/output/minimize"),
    )
    args = parser.parse_args()

    print_header("MLpot minimization stub (step 4)")
    out_dir = args.out_dir.resolve()
    pdb_path = out_dir / "mini_aco.pdb"
    crd_path = out_dir / "mini_aco.crd"

    print("Workflow (see mmml/interfaces/pycharmmInterface/mlpot/dynamics.py):")
    print("  1. register_mlpot(...)")
    print("  2. optional cons_fix.setup(selection)  # e.g. seg_id='AMM1'")
    print("  3. minimize.run_sd(...)")
    print("  4. cons_fix.turn_off(); minimize.run_sd(...)  # second pass in examples")
    print(f"  5. write CRD/PDB -> {crd_path}")
    print("  Prefer reloading CRD (not PDB) to preserve ML nb exclusions.")

    if not args.run:
        print("\nSTUB: pass --run to execute a short SD test.")
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
        setup_default_nbonds,
    )

    setup_default_nbonds()
    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        ran = minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                fixed_ml_selection=None,
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

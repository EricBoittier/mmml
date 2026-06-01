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

import numpy as np

from _common import (
    add_charmm_output_args,
    add_cluster_args,
    add_dcd_save_args,
    apply_charmm_output_from_args,
    build_acetone_dimer_cluster,
    build_ase_cluster,
    print_cluster_geometry_summary,
    print_header,
    resolve_checkpoint,
    resolve_dcd_nsavc,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    add_charmm_output_args(parser)
    add_dcd_save_args(parser)
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
    parser.add_argument(
        "--save",
        action="store_true",
        help="After minimization, write PDB/CRD/PSF, energy JSON, XYZ, and minimization DCD",
    )
    parser.add_argument(
        "--no-save-vmd-topology",
        action="store_true",
        help="Do not write cluster_for_vmd.psf/pdb before MLpot strips PSF bonds",
    )
    args = parser.parse_args()

    print_header("MLpot minimization stub (step 4)")
    out_dir = args.out_dir.resolve()
    pdb_path = out_dir / "mini_full_mlpot.pdb"
    crd_path = out_dir / "mini_full_mlpot.crd"
    psf_path = out_dir / "mini_full_mlpot.psf"
    energy_json_path = out_dir / "mini_full_mlpot_energy.json"
    xyz_path = out_dir / "mini_full_mlpot.xyz"
    dcd_path = out_dir / "mini_full_mlpot.dcd"
    print("Workflow:")
    print("  1. register_mlpot on ALL atoms")
    print(f"  2. cons_fix.setup(resid {args.fix_resid})  # constraint test only")
    print("  3. minimize.run_sd; cons_fix.turn_off(); minimize.run_sd")
    if args.save:
        print(f"  4. --save -> {out_dir}/mini_full_mlpot.*")
    else:
        print("  4. (optional) pass --save to write minimized structures and energy JSON")

    if not args.run:
        print("\nSTUB: pass --run to execute.")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    if args.residue.upper() == "ACO" and args.n_molecules == 2:
        z, r = build_acetone_dimer_cluster(spacing=args.spacing)
    else:
        z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)
    print_cluster_geometry_summary(r, args.n_molecules)
    out_dir.mkdir(parents=True, exist_ok=True)

    import ase
    from mmml.interfaces.pycharmmInterface.mlpot import (
        MinimizeWithMlpotConfig,
        get_charmm_positions_array,
        load_physnet_mlpot_bundle,
        minimize_with_mlpot,
        register_mlpot,
        save_cluster_topology_for_vmd,
        select_all_atoms,
        select_by_resid,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    nprint = apply_charmm_output_from_args(args)
    dcd_nsavc = resolve_dcd_nsavc(
        dcd_nsavc=args.dcd_nsavc,
        nstep=args.nstep,
    )
    print(
        f"CHARMM output: PRNLev={0 if args.quiet else args.prnlev} "
        f"WRNLev={0 if args.quiet else args.warnlev} nprint={nprint}"
    )
    setup_default_nbonds()
    sync_charmm_positions(r)
    if not args.no_save_vmd_topology:
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem="cluster_for_vmd", title="pre-MLpot cluster"
        )
        print(
            "VMD topology (full PSF bonds, before MLpot): "
            f"{vmd_files['psf'].name} + {vmd_files['pdb'].name}"
        )
        print(f"  vmd {vmd_files['psf']} {vmd_files['pdb']}")
    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)

    fix_sel = select_by_resid(args.fix_resid)
    if len(fix_sel.get_atom_indexes()) == 0:
        print(f"FAIL: no atoms in fix-resid {args.fix_resid}")
        return 1

    if args.save:
        print(f"Minimization DCD nsavc={dcd_nsavc} (frame every {dcd_nsavc} SD steps)")
    print(f"MLpot: all {n_atoms} atoms | cons_fix: resid {args.fix_resid}")
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    sync_charmm_positions(r)
    pos_chk = get_charmm_positions_array()
    if np.allclose(pos_chk, 0.0):
        print("WARN: CHARMM coordinates are zero after MLpot; re-syncing from cluster build")
        sync_charmm_positions(r)
    try:
        ran = minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                fixed_ml_selection=fix_sel,
                nstep=args.nstep,
                nprint=nprint,
                verbose=not args.quiet,
                reference_positions=r,
                pyCModel=pyCModel,
                save=args.save,
                pdb_path=pdb_path if args.save else None,
                crd_path=crd_path if args.save else None,
                psf_path=psf_path if args.save else None,
                energy_json_path=energy_json_path if args.save else None,
                xyz_path=xyz_path if args.save else None,
                dcd_path=dcd_path if args.save else None,
                dcd_nsavc=dcd_nsavc if args.save else 0,
                skip_if_crd_exists=False,
            )
        )
    finally:
        ctx.unset()

    if args.save:
        print(f"\nMinimization ran={ran}; saved under {out_dir} (incl. {dcd_path.name})")
    else:
        print(f"\nMinimization ran={ran} (no files written; use --save)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

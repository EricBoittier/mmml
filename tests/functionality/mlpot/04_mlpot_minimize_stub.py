#!/usr/bin/env python3
"""
Step 4: SD minimization with MLpot on the full cluster.

- MLpot: all atoms (PhysNet on the full system).
- SD pass 1: free minimization (all atoms); pass 2: ``cons_fix`` on ``--fix-resids`` monomers.

Pass ``--run`` to execute. Use ``--n-molecules`` for larger acetone clusters (ACO × N).
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
    add_flat_bottom_args,
    apply_flat_bottom_from_args,
    add_monomer_constraint_args,
    apply_charmm_output_from_args,
    build_cluster_from_args,
    format_resid_constraint_message,
    print_cluster_geometry_summary,
    print_header,
    print_vmd_load_help,
    resolve_checkpoint,
    resolve_dcd_nsavc,
    resolve_fix_resids,
    validate_resids_for_cluster,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    add_charmm_output_args(parser)
    add_dcd_save_args(parser)
    add_flat_bottom_args(parser)
    add_monomer_constraint_args(parser, for_dynamics=False)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute minimization (default: dry-run stub message only)",
    )
    parser.add_argument("--nstep", type=int, default=10, help="SD steps per SD pass")
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

    fix_resids = resolve_fix_resids(args)

    print_header("MLpot minimization (step 4)")
    out_dir = args.out_dir.resolve()
    tag = f"{args.residue.lower()}_{args.n_molecules}mer"
    pdb_path = out_dir / f"mini_full_mlpot_{tag}.pdb"
    crd_path = out_dir / f"mini_full_mlpot_{tag}.crd"
    psf_path = out_dir / f"mini_full_mlpot_{tag}.psf"
    energy_json_path = out_dir / f"mini_full_mlpot_{tag}_energy.json"
    xyz_path = out_dir / f"mini_full_mlpot_{tag}.xyz"
    dcd_path = out_dir / f"mini_full_mlpot_{tag}.dcd"

    print("Workflow:")
    print(f"  Cluster: {args.residue} × {args.n_molecules} monomers (spacing {args.spacing} Å)")
    print("  1. register_mlpot on ALL atoms")
    print("  2. SD pass 1: free minimization (all atoms)")
    print(f"  3. SD pass 2: {format_resid_constraint_message(fix_resids, context='cons_fix')}")
    if args.save:
        print(f"  4. --save -> {out_dir}/mini_full_mlpot_{tag}.*")

    if not args.run:
        print("\nExamples:")
        print("  --n-molecules 4 --fix-resids 1,3   # tetramer; free mini then fix 1,3")
        print("  --n-molecules 3 --no-fix           # trimer; free minimization only")
        print("\nSTUB: pass --run to execute.")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r, n_atoms = build_cluster_from_args(args)
    validate_resids_for_cluster(fix_resids, args.n_molecules)
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
        select_by_resids,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    nprint = apply_charmm_output_from_args(args)
    dcd_nsavc = resolve_dcd_nsavc(dcd_nsavc=args.dcd_nsavc, nstep=args.nstep)
    print(
        f"CHARMM output: PRNLev={0 if args.quiet else args.prnlev} "
        f"WRNLev={0 if args.quiet else args.warnlev} nprint={nprint}"
    )
    setup_default_nbonds()
    sync_charmm_positions(r)
    apply_flat_bottom_from_args(args)
    vmd_topo_psf = out_dir / f"cluster_for_vmd_{tag}.psf"
    if not args.no_save_vmd_topology:
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem=f"cluster_for_vmd_{tag}", title="pre-MLpot cluster"
        )
        vmd_topo_psf = vmd_files["psf"]
        print(
            "VMD topology (full PSF bonds, before MLpot): "
            f"{vmd_files['psf'].name} + {vmd_files['pdb'].name}"
        )

    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)

    fix_sel = select_by_resids(fix_resids) if fix_resids else None
    if fix_sel is not None and len(fix_sel.get_atom_indexes()) == 0:
        print(f"FAIL: no atoms for --fix-resids {fix_resids}")
        return 1

    if args.save:
        print(f"Minimization DCD nsavc={dcd_nsavc} (frame every {dcd_nsavc} SD steps)")
    print(
        f"MLpot: all {n_atoms} atoms | "
        f"{format_resid_constraint_message(fix_resids, context='cons_fix')}"
    )
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
        print_vmd_load_help(
            out_dir=out_dir,
            tag=tag,
            topology_psf=vmd_topo_psf,
            trajectory=dcd_path,
            n_atoms=n_atoms,
            bondless_psf=psf_path,
        )
    else:
        print(f"\nMinimization ran={ran} (no files written; use --save)")
        if vmd_topo_psf.is_file():
            print_vmd_load_help(
                out_dir=out_dir,
                tag=tag,
                topology_psf=vmd_topo_psf,
                trajectory=None,
                n_atoms=n_atoms,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

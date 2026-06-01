#!/usr/bin/env python3
"""
Step 5: short CHARMM dynamics with MLpot.

Supports multiple monomers (``--n-molecules``) and optional frozen monomers
(``--constrain-resids``) during the run.

Default: dry-run only. ``--run`` executes a minimal NVE segment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    add_charmm_output_args,
    add_cluster_args,
    add_dcd_save_args,
    add_monomer_constraint_args,
    apply_charmm_output_from_args,
    build_cluster_from_args,
    format_resid_constraint_message,
    print_cluster_geometry_summary,
    print_header,
    print_vmd_load_help,
    resolve_checkpoint,
    resolve_constrain_resids,
    resolve_dcd_nsavc,
    setup_cons_fix_for_resids,
    turn_off_cons_fix,
    validate_resids_for_cluster,
)

NVE_TIMESTEP_PS = 0.00025


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    add_charmm_output_args(parser)
    add_dcd_save_args(parser)
    add_monomer_constraint_args(parser, for_dynamics=True)
    parser.add_argument("--run", action="store_true", help="Run a short NVE segment")
    parser.add_argument("--nstep", type=int, default=20, help="Dynamics steps when --run")
    parser.add_argument(
        "--temp",
        type=float,
        default=300.0,
        help="Initial Maxwell-Boltzmann temperature (K) when starting fresh NVE",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/functionality/mlpot/output/dynamics"),
    )
    parser.add_argument(
        "--no-save-vmd-topology",
        action="store_true",
        help="Do not write cluster_for_vmd.psf/pdb before MLpot strips PSF bonds",
    )
    args = parser.parse_args()

    constrain_resids = resolve_constrain_resids(args)

    print_header("MLpot dynamics (step 5)")
    out_dir = args.out_dir.resolve()
    tag = f"{args.residue.lower()}_{args.n_molecules}mer"
    res_path = out_dir / f"nve_{tag}.res"
    dcd_path = out_dir / f"nve_{tag}.dcd"

    print(f"Cluster: {args.residue} × {args.n_molecules} monomers (spacing {args.spacing} Å)")
    print(format_resid_constraint_message(constrain_resids, context="Dynamics"))

    if not args.run:
        print("\nExamples:")
        print("  --n-molecules 4 --run --nstep 50")
        print("  --n-molecules 3 --constrain-resids 1,2 --run  # freeze monomers 1–2")
        print("\nDry-run only. Example:")
        print("  python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --nstep 20")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r, n_atoms = build_cluster_from_args(args)
    validate_resids_for_cluster(constrain_resids, args.n_molecules)
    print_cluster_geometry_summary(r, args.n_molecules)
    out_dir.mkdir(parents=True, exist_ok=True)

    import ase
    import pycharmm.energy as energy
    from mmml.interfaces.pycharmmInterface.mlpot import (
        CharmmTrajectoryFiles,
        build_nve_dynamics,
        load_physnet_mlpot_bundle,
        register_mlpot,
        run_dynamics_with_io,
        save_cluster_topology_for_vmd,
        select_all_atoms,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    nprint = apply_charmm_output_from_args(args)
    dcd_nsavc = resolve_dcd_nsavc(
        dcd_nsavc=args.dcd_nsavc,
        dcd_interval_ps=args.dcd_interval_ps,
        timestep_ps=NVE_TIMESTEP_PS,
        nstep=args.nstep,
    )
    print(
        f"CHARMM output: PRNLev={0 if args.quiet else args.prnlev} "
        f"WRNLev={0 if args.quiet else args.warnlev} nprint={nprint}"
    )
    setup_default_nbonds()
    sync_charmm_positions(r)
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

    print(
        f"MLpot: all {n_atoms} atoms | NVE {args.nstep} steps @ {NVE_TIMESTEP_PS} ps | "
        f"{format_resid_constraint_message(constrain_resids, context='cons_fix')}"
    )
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        if constrain_resids:
            setup_cons_fix_for_resids(constrain_resids)
        if not args.quiet:
            print("CHARMM energy before dynamics:")
            energy.show()
        io = CharmmTrajectoryFiles(
            restart_write=res_path,
            trajectory=dcd_path,
        )
        kw = build_nve_dynamics(
            timestep_ps=NVE_TIMESTEP_PS,
            duration_ps=args.nstep * NVE_TIMESTEP_PS,
            save_interval_ps=NVE_TIMESTEP_PS * dcd_nsavc,
            restart=False,
            temp=args.temp,
        )
        kw["new"] = True
        kw["start"] = True
        kw["nstep"] = args.nstep
        kw["nprint"] = nprint
        kw["iprfrq"] = nprint
        kw["nsavc"] = dcd_nsavc
        print(f"DCD nsavc={dcd_nsavc} (frame every {dcd_nsavc * NVE_TIMESTEP_PS:.6f} ps)")
        run_dynamics_with_io(kw, io)
        print("CHARMM energy after dynamics:")
        energy.show()
    finally:
        if constrain_resids:
            turn_off_cons_fix()
        ctx.unset()

    missing = [p for p in (res_path, dcd_path) if not p.is_file()]
    if missing:
        print(f"FAIL: expected outputs missing: {missing}")
        return 1

    print(f"\nNVE OK; restart={res_path.name} trajectory={dcd_path.name} ({out_dir})")
    print_vmd_load_help(
        out_dir=out_dir,
        tag=tag,
        topology_psf=vmd_topo_psf,
        trajectory=dcd_path,
        n_atoms=n_atoms,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

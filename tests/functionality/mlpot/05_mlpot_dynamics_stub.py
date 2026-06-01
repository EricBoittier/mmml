#!/usr/bin/env python3
"""
Step 5: CHARMM dynamics with MLpot after two-stage minimization.

Workflow with MLpot registered:
  1. SD pass 1 — free minimization (all atoms)
  2. SD pass 2 — constrained minimization (``--fix-resids``)
  3. NVE dynamics (optional ``--constrain-resids`` during MD)

Use ``--no-pre-minimize`` to skip steps 1–2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    add_charmm_output_args,
    add_cluster_args,
    add_dcd_save_args,
    add_dynamics_stability_args,
    add_flat_bottom_args,
    add_monomer_constraint_args,
    apply_flat_bottom_from_args,
    apply_charmm_output_from_args,
    build_cluster_from_args,
    format_resid_constraint_message,
    print_cluster_geometry_summary,
    print_header,
    print_vmd_load_help,
    resolve_checkpoint,
    resolve_constrain_resids,
    resolve_dcd_nsavc,
    resolve_dynamics_print_kwargs,
    resolve_echeck_from_args,
    resolve_fix_resids,
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
    add_dynamics_stability_args(parser)
    add_flat_bottom_args(parser)
    add_monomer_constraint_args(parser, for_dynamics=True)
    parser.add_argument("--run", action="store_true", help="Run minimization + NVE")
    parser.add_argument("--nstep", type=int, default=20, help="NVE dynamics steps when --run")
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

    fix_resids = resolve_fix_resids(args)
    dynamics_constrain = resolve_constrain_resids(args)

    print_header("MLpot dynamics (step 5)")
    out_dir = args.out_dir.resolve()
    tag = f"{args.residue.lower()}_{args.n_molecules}mer"
    res_path = out_dir / f"nve_{tag}.res"
    dcd_path = out_dir / f"nve_{tag}.dcd"

    print(f"Cluster: {args.residue} × {args.n_molecules} monomers (spacing {args.spacing} Å)")
    if not args.no_pre_minimize:
        print("  1. SD pass 1: free minimization")
        print(f"  2. SD pass 2: {format_resid_constraint_message(fix_resids, context='cons_fix')}")
        print(
            f"  3. NVE: {format_resid_constraint_message(dynamics_constrain, context='cons_fix during MD')}"
        )
    else:
        print("  (pre-minimize skipped)")
        print(
            f"  NVE: {format_resid_constraint_message(dynamics_constrain, context='cons_fix during MD')}"
        )

    if not args.run:
        print("\nExamples:")
        print("  --run --n-molecules 4 --fix-resids 1,3 --mini-nstep 30 --nstep 50")
        print("  --run --fix-resids 1 --constrain-resids 1  # same monomer fixed in mini + MD")
        print("  --run --no-pre-minimize --nstep 20")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r, n_atoms = build_cluster_from_args(args)
    validate_resids_for_cluster(fix_resids, args.n_molecules)
    validate_resids_for_cluster(dynamics_constrain, args.n_molecules)
    print_cluster_geometry_summary(r, args.n_molecules)
    out_dir.mkdir(parents=True, exist_ok=True)

    import ase
    import pycharmm.energy as energy
    from mmml.interfaces.pycharmmInterface.mlpot import (
        CharmmTrajectoryFiles,
        MinimizeWithMlpotConfig,
        build_nve_dynamics,
        get_charmm_positions_array,
        load_physnet_mlpot_bundle,
        minimize_with_mlpot,
        register_mlpot,
        run_dynamics_with_io,
        save_cluster_topology_for_vmd,
        select_all_atoms,
        select_by_resids,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    mini_nprint = apply_charmm_output_from_args(args)
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=args.nstep)
    echeck = resolve_echeck_from_args(args)
    dcd_nsavc = resolve_dcd_nsavc(
        dcd_nsavc=args.dcd_nsavc,
        dcd_interval_ps=args.dcd_interval_ps,
        timestep_ps=NVE_TIMESTEP_PS,
        nstep=args.nstep,
    )
    print(
        f"CHARMM output: PRNLev={0 if args.quiet else args.prnlev} "
        f"WRNLev={0 if args.quiet else args.warnlev} | "
        f"SD nprint={mini_nprint} | "
        f"dyn nprint={dyn_print['nprint']} iprfrq={dyn_print['iprfrq']} | "
        f"echeck={echeck}"
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

    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        if not args.no_pre_minimize:
            fix_sel = select_by_resids(fix_resids) if fix_resids else None
            print(
                f"\nPre-dynamics minimization ({args.mini_nstep} SD steps per pass) | "
                f"MLpot on all {n_atoms} atoms"
            )
            minimize_with_mlpot(
                MinimizeWithMlpotConfig(
                    fixed_ml_selection=fix_sel,
                    nstep=args.mini_nstep,
                    nprint=mini_nprint,
                    verbose=not args.quiet,
                    reference_positions=r,
                    pyCModel=pyCModel,
                    save=False,
                    show_energy=True,
                    skip_if_crd_exists=False,
                )
            )
            sync_charmm_positions(get_charmm_positions_array())

        if dynamics_constrain:
            setup_cons_fix_for_resids(dynamics_constrain)

        print(
            f"\nNVE: {args.nstep} steps @ {NVE_TIMESTEP_PS} ps | "
            f"{format_resid_constraint_message(dynamics_constrain, context='cons_fix')}"
        )
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
            nprint=dyn_print["nprint"],
            iprfrq=dyn_print["iprfrq"],
            isvfrq=dyn_print["isvfrq"],
            echeck=echeck,
        )
        kw["new"] = True
        kw["start"] = True
        kw["nstep"] = args.nstep
        kw["nsavc"] = dcd_nsavc
        print(
            f"DCD nsavc={dcd_nsavc} ({dcd_nsavc * NVE_TIMESTEP_PS:.6f} ps/frame) | "
            f"dyn print every {dyn_print['nprint']} steps | "
            f"echeck={echeck} kcal/mol (stop if |dE| exceeded)"
        )
        run_dynamics_with_io(kw, io)
        print("CHARMM energy after dynamics:")
        energy.show()
    finally:
        if dynamics_constrain:
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

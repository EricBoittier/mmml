#!/usr/bin/env python3
"""
Step 5: short CHARMM dynamics with MLpot (acetone dimer, 20 atoms).

Default: dry-run only. ``--run`` executes a minimal NVE segment (few steps).
Full heat / equil / production chains use ``mlpot.dynamics`` builders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    add_charmm_output_args,
    add_cluster_args,
    apply_charmm_output_from_args,
    build_acetone_dimer_cluster,
    build_ase_cluster,
    print_cluster_geometry_summary,
    print_header,
    resolve_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    add_charmm_output_args(parser)
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
    args = parser.parse_args()

    print_header("MLpot dynamics (step 5)")
    out_dir = args.out_dir.resolve()
    res_path = out_dir / "nve_stub.res"
    dcd_path = out_dir / "nve_stub.dcd"

    print("Stages in mmml/interfaces/pycharmmInterface/mlpot/dynamics.py:")
    print("  build_heat_dynamics              -> NVT heating")
    print("  build_nve_dynamics               -> NVE")
    print("  build_cpt_equilibration_dynamics -> NPT equil")
    print("  build_cpt_production_dynamics    -> NPT production")
    print("  production_restart_chain         -> chained dyna.{i}.res/dcd")

    if not args.run:
        print("\nDry-run only. Example:")
        print("  python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --nstep 20")
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
    import pycharmm.energy as energy
    from mmml.interfaces.pycharmmInterface.mlpot import (
        CharmmTrajectoryFiles,
        assign_boltzmann_velocities,
        build_nve_dynamics,
        load_physnet_mlpot_bundle,
        register_mlpot,
        run_dynamics_with_io,
        select_all_atoms,
        setup_default_nbonds,
        sync_charmm_positions,
    )

    nprint = apply_charmm_output_from_args(args)
    print(
        f"CHARMM output: PRNLev={0 if args.quiet else args.prnlev} "
        f"WRNLev={0 if args.quiet else args.warnlev} nprint={nprint}"
    )
    setup_default_nbonds()
    sync_charmm_positions(r)
    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)

    print(f"MLpot: all {n_atoms} atoms | NVE {args.nstep} steps @ 0.25 fs")
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        if not args.quiet:
            print("CHARMM energy before dynamics:")
            energy.show()
        assign_boltzmann_velocities(args.temp)
        io = CharmmTrajectoryFiles(
            restart_write=res_path,
            trajectory=dcd_path,
        )
        kw = build_nve_dynamics(
            timestep_ps=0.00025,
            duration_ps=args.nstep * 0.00025,
            save_interval_ps=0.00025,
            restart=False,
        )
        kw["new"] = True
        kw["start"] = True
        kw["nstep"] = args.nstep
        kw["nprint"] = nprint
        kw["iprfrq"] = nprint
        kw["nsavc"] = max(1, args.nstep // 2)
        run_dynamics_with_io(kw, io)
        print("CHARMM energy after dynamics:")
        energy.show()
    finally:
        ctx.unset()

    missing = [p for p in (res_path, dcd_path) if not p.is_file()]
    if missing:
        print(f"FAIL: expected outputs missing: {missing}")
        return 1

    print(f"\nNVE OK; restart={res_path.name} trajectory={dcd_path.name} ({out_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Step 5 (stub): short CHARMM dynamics with MLpot.

Default: dry-run only. ``--run`` executes a minimal NVE segment (few steps).
Full heat / equil / production chains use ``mlpot.dynamics`` builders.
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
    parser.add_argument("--run", action="store_true", help="Run a short NVE segment")
    parser.add_argument("--nstep", type=int, default=20, help="Dynamics steps when --run")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/functionality/mlpot/output/dynamics"),
    )
    args = parser.parse_args()

    print_header("MLpot dynamics stub (step 5)")
    out_dir = args.out_dir.resolve()

    print("Stages from example mini-MD scripts (mmml/.../mlpot/dynamics.py):")
    print("  build_heat_dynamics      -> NVT heating")
    print("  build_nve_dynamics       -> NVE")
    print("  build_cpt_equilibration_dynamics -> NPT equil")
    print("  build_cpt_production_dynamics    -> NPT production")
    print("  production_restart_chain -> plan dyna.{i}.res/dcd files")
    print("Use run_dynamics_with_io(kwargs, CharmmTrajectoryFiles(...)).")

    if not args.run:
        print("\nSTUB: pass --run for a short NVE test.")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = build_ase_cluster(args.residue, args.n_molecules, args.spacing)
    n_atoms = len(z)
    out_dir.mkdir(parents=True, exist_ok=True)

    import ase
    from mmml.interfaces.pycharmmInterface.mlpot import (
        CharmmTrajectoryFiles,
        build_nve_dynamics,
        load_physnet_mlpot_bundle,
        register_mlpot,
        select_all_atoms,
        run_dynamics_with_io,
        setup_default_nbonds,
    )

    setup_default_nbonds()
    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(ckpt, n_atoms, atoms)
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    try:
        io = CharmmTrajectoryFiles(
            restart_write=out_dir / "nve_stub.res",
            trajectory=out_dir / "nve_stub.dcd",
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
        kw["nsavc"] = max(1, args.nstep // 2)
        run_dynamics_with_io(kw, io)
    finally:
        ctx.unset()

    print(f"\nShort NVE finished; outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

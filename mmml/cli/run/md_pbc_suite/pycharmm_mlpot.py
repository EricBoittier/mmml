#!/usr/bin/env python3
"""CHARMM MLpot backend for ``mmml md-system --backend pycharmm`` (vacuum, non-PBC)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    add_charmm_output_args,
    add_cluster_args,
    add_dcd_save_args,
    add_dynamics_stability_args,
    add_flat_bottom_args,
    add_monomer_constraint_args,
)
from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import run_workflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CHARMM MLpot workflows: two-pass SD minimization and vacuum NVE/NVT dynamics. "
            "Invoked via ``mmml md-system --backend pycharmm``."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["full", "minimize", "dynamics"],
        default="full",
        help="full = pre-minimize + MD; minimize = SD only; dynamics = MD only",
    )
    parser.add_argument(
        "--ensemble",
        choices=["nve", "nvt"],
        default="nve",
        help="MD ensemble after minimization (nvt = CHARMM heating/Hoover-style)",
    )
    add_cluster_args(parser)
    add_charmm_output_args(parser)
    add_dcd_save_args(parser)
    add_dynamics_stability_args(parser)
    add_flat_bottom_args(parser)
    add_monomer_constraint_args(parser, for_dynamics=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pycharmm_mlpot"),
        help="Directory for PSF/DCD/restart outputs",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=0,
        help="Dynamics integration steps (0 = derive from --ps and --dt-fs)",
    )
    parser.add_argument("--ps", type=float, default=1.0, help="Dynamics length in ps")
    parser.add_argument("--dt-fs", type=float, default=0.25, help="Timestep in fs")
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Target / initial temperature in K",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write minimization artifacts when --phase minimize",
    )
    parser.add_argument(
        "--no-save-vmd-topology",
        action="store_true",
        help="Skip cluster_for_vmd PSF/PDB before MLpot strips bonds",
    )
    parser.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="ANG",
        help="MMFP droff (Å); also enables Packmol when --composition is set (legacy)",
    )
    parser.add_argument(
        "--packmol-sphere",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pack --composition inside a sphere with Packmol",
    )
    parser.add_argument(
        "--packmol-radius",
        type=float,
        default=None,
        metavar="ANG",
        help="Packmol sphere radius (overrides --flat-bottom-radius for packing only)",
    )
    parser.add_argument(
        "--packmol-center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help="Packmol sphere center in Å (default: 0 0 0)",
    )
    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol distance tolerance in Å (default: 2.0)",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for Packmol placement")
    parser.add_argument(
        "--no-scale-mini-nstep",
        action="store_true",
        help="Do not auto-increase --mini-nstep for large clusters",
    )
    parser.add_argument(
        "--no-scale-echeck",
        action="store_true",
        help="Keep --echeck exactly as given for large clusters",
    )
    parser.add_argument(
        "--allow-high-grms",
        action="store_true",
        help="Run dynamics even if post-min GRMS is high (not recommended)",
    )
    parser.add_argument(
        "--max-grms-before-dyn",
        type=float,
        default=50.0,
        help="Abort dynamics if post-min GRMS exceeds this (kcal/mol/Å)",
    )
    parser.add_argument(
        "--charmm-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CGENFF SD/ABNR before registering MLpot (default: on; use --no-charmm-pre-minimize to skip)",
    )
    parser.add_argument(
        "--charmm-sd-steps",
        type=int,
        default=50,
        help="CHARMM SD steps before MLpot (default: 50)",
    )
    parser.add_argument(
        "--charmm-abnr-steps",
        type=int,
        default=100,
        help="CHARMM ABNR steps before MLpot (default: 100; 0 to skip)",
    )
    parser.add_argument("--charmm-tolenr", type=float, default=1e-3)
    parser.add_argument("--charmm-tolgrd", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Alias for run_workflow helpers that read ``temp``.
    args.temp = args.temperature
    try:
        return run_workflow(
            args,
            phase=args.phase,
            ensemble=args.ensemble,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"pycharmm_mlpot: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

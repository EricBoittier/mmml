#!/usr/bin/env python3
"""Standalone restrained two-residue PyCHARMM sampling utility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pycharmm
import pycharmm.write as write

from mmml.cli.run.pycharmm_runner import (
    NBONDS_SCRIPT,
    add_two_residue_sampling_args,
    run_two_residue_harmonic_sampling,
)
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    coor,
    pycharmm_quiet,
    pycharmm_soft,
    reset_block,
)
from mmml.interfaces.pycharmmInterface.setupBox import setup_box_generic


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Set up a two-residue CHARMM system and run restrained sampling.",
    )
    parser.add_argument(
        "--pdbfile",
        type=Path,
        required=True,
        help="Path to a PDB file containing the two-residue CHARMM system.",
    )
    parser.add_argument(
        "--cell",
        type=float,
        default=1000.0,
        help="Cubic cell side length in Angstrom for periodic boundary conditions (default: 1000).",
    )
    parser.add_argument(
        "--skip-setup-energy-show",
        action="store_true",
        help="Skip energy.show() in setup_box for faster startup.",
    )
    parser.add_argument(
        "--pycharmm-minimize-steps",
        type=int,
        default=1000,
        metavar="N",
        help="Fallback ABNR steps when --two-residue-sampling-steps is not set (default: 1000).",
    )
    parser.add_argument(
        "--output-pdb",
        type=Path,
        default=Path("pdb/two-residue-sampled.pdb"),
        help="Path for sampled coordinates as PDB (default: pdb/two-residue-sampled.pdb).",
    )
    add_two_residue_sampling_args(parser, include_toggle=False)
    parser.set_defaults(two_residue_sampling=True)
    return parser.parse_args()


def run(args: argparse.Namespace):
    """Set up nbonds/block state, run restrained sampling, and write coordinates."""
    atoms = setup_box_generic(
        str(args.pdbfile),
        side_length=args.cell,
        skip_energy_show=getattr(args, "skip_setup_energy_show", False),
    )

    reset_block()
    pycharmm_soft()
    pycharmm.lingo.charmm_script(NBONDS_SCRIPT)
    atoms = run_two_residue_harmonic_sampling(args, atoms=atoms)

    atoms.set_positions(coor.get_positions())
    args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
    write.coor_pdb(str(args.output_pdb))
    pycharmm_quiet()
    return atoms


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    if not args.pdbfile.exists():
        print(f"Error: PDB file not found: {args.pdbfile}", file=sys.stderr)
        return 1
    run(args)
    print(f"Two-residue restrained sampling complete: {args.output_pdb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

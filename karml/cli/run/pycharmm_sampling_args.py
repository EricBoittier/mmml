"""Argument helpers for PyCHARMM sampling commands."""

from __future__ import annotations

import argparse


def add_two_residue_sampling_args(
    parser: argparse.ArgumentParser,
    *,
    include_toggle: bool = True,
) -> None:
    """Add CLI options for restrained two-residue PyCHARMM sampling."""
    if include_toggle:
        parser.add_argument(
            "--two-residue-sampling",
            dest="two_residue_sampling",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Run restrained two-residue PyCHARMM sampling after nbonds/block setup (default: True).",
        )
    parser.add_argument(
        "--two-residue-restraint-force",
        type=float,
        default=1.0,
        metavar="K",
        help="CHARMM harmonic restraint force constant for two-residue sampling (default: 1.0).",
    )
    parser.add_argument(
        "--two-residue-restraint-r0",
        type=float,
        default=2.5,
        metavar="ANGSTROM",
        help="CHARMM harmonic restraint target distance r0 for two-residue sampling (default: 2.5 Angstrom).",
    )
    parser.add_argument(
        "--two-residue-sampling-steps",
        type=int,
        default=None,
        metavar="N",
        help="ABNR steps for restrained two-residue sampling (default: --pycharmm-minimize-steps).",
    )
    parser.add_argument(
        "--two-residue-restraint-resid1",
        default=1,
        help="First CHARMM residue id for two-residue sampling (default: 1).",
    )
    parser.add_argument(
        "--two-residue-restraint-resid2",
        default=2,
        help="Second CHARMM residue id for two-residue sampling (default: 2).",
    )

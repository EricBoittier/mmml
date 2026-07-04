#!/usr/bin/env python3
"""Convert CHARMM restart (.res) files to ASE trajectory (.traj) with velocities."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.cli.run.md_handoff import res_to_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CHARMM restart (.res) to ASE trajectory (.traj) with velocities.",
    )
    parser.add_argument(
        "res",
        type=Path,
        help="Input .res file, or directory containing numbered heat.NNNN.res files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output ASE trajectory (.traj)",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Handoff NPZ with atomic_numbers (default: sibling handoff/state.npz)",
    )
    parser.add_argument(
        "--numbered-stem",
        default=None,
        help="When input is a directory, restart stem to collect (default: heat)",
    )
    parser.add_argument(
        "--velocity-units",
        choices=("auto", "akma", "ase"),
        default="auto",
        help="Restart velocity interpretation (default: auto)",
    )
    args = parser.parse_args()

    out = res_to_trajectory(
        args.res,
        args.output,
        npz_path=args.npz,
        velocity_units=args.velocity_units,
        numbered_stem=args.numbered_stem,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

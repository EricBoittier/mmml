"""
Sets up a residue for an MD simulation.

Note: The final energy.show() call can segfault in CHARMM's bond routines when run
under SLURM, on some cluster nodes, or with certain MPI/threading. Set
SKIP_CHARMM_ENERGY_SHOW=1 or use --skip-energy-show to skip it; residue and
coordinates are already written before that call.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str)
    parser.add_argument(
        "--skip-energy-show",
        action="store_true",
        help="Skip the final CHARMM energy.show() (avoids segfault on some clusters/SLURM).",
    )
    return parser.parse_args()


def main_loop(args):
    from mmml.pycharmmInterface import setupRes
    from mmml.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
        energy,
    )
    atoms = setupRes.main(args.res)
    atoms = setupRes.generate_coordinates()
    _ = setupRes.coor.get_positions()
    atoms.set_positions(_)
    reset_block()
    reset_block_no_internal()
    reset_block()
    atoms = setupRes.generate_coordinates()
    _ = setupRes.coor.get_positions()
    atoms.set_positions(_)
    reset_block()
    reset_block_no_internal()
    reset_block()
    skip_energy = getattr(args, "skip_energy_show", False) or os.environ.get("SKIP_CHARMM_ENERGY_SHOW")
    if not skip_energy:
        energy.show()
    else:
        print("Skipping energy.show() (--skip-energy-show or SKIP_CHARMM_ENERGY_SHOW).")

def main():
    args = parse_args()
    print(args)
    main_loop(args)

if __name__ == "__main__":
    main()
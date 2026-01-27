"""
Sets up a residue for an MD simulation.

Note: The final energy.show() call can segfault in CHARMM's bond routines (e.g.
__eintern_fast_MOD_ebondfs) when run under SLURM, on some cluster nodes, or with
certain MPI/threading. Residue and coordinates are already written before that call.
To avoid the segfault: use --skip-energy-show, or set SKIP_CHARMM_ENERGY_SHOW=1
(or "yes"/"true"). When SLURM_JOB_ID is set, energy.show() is skipped by default
unless RUN_CHARMM_ENERGY_SHOW=1 is set.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

import argparse


def _should_skip_energy_show(args) -> bool:
    """True if CHARMM energy.show() should be skipped to avoid segfault."""
    if getattr(args, "skip_energy_show", False):
        return True
    from mmml.pycharmmInterface.import_pycharmm import should_skip_charmm_energy_show
    return should_skip_charmm_energy_show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str)
    parser.add_argument(
        "--skip-energy-show",
        dest="skip_energy_show",
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
    if not _should_skip_energy_show(args):
        energy.show()
    else:
        print("Skipping energy.show() (--skip-energy-show, SKIP_CHARMM_ENERGY_SHOW, or SLURM).")

def main():
    args = parse_args()
    print(args)
    main_loop(args)

if __name__ == "__main__":
    main()
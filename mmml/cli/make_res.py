"""
Sets up a residue for an MD simulation.
"""

import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str)
    return parser.parse_args()

def main_loop(args):
    from mmml.pycharmmInterface import setupRes
    from mmml.pycharmmInterface.import_pycharmm import (
        reset_block, reset_block_no_internal, energy
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
    energy.show()

def main():
    args = parse_args()
    print(args)
    main_loop(args)

if __name__ == "__main__":
    main()
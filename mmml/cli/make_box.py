"""
Sets up a box for an MD simulation.
"""

"""
Sets up a residue for an MD simulation.
"""

import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import os

import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--res", type=str)
    parser.add_argument("--side_length", type=float, default=300)
    parser.add_argument("--pdb", type=str, default=None)
    parser.add_argument("--solvent", type=str, default=None)
    parser.add_argument("--density", type=float, default=None)

    return parser.parse_args()

def main_loop(args):
    from mmml.pycharmmInterface import setupBox

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)} {type(getattr(args, arg))}")

    cwd = Path(os.getcwd())

    if args.pdb is not None:
        mol = ase.io.read(args.pdb)
        print(mol)
        print(mol.get_chemical_symbols())

    else:
        mol = setupBox.read_initial_pdb(cwd)
        print(mol)
        print(mol.get_chemical_symbols())

        if args.solvent is None:
            n_molecules = args.n 
            setupBox.run_packmol(n_molecules, args.side_length)
        else:
            n_molecules =  setupBox.determine_n_molecules_from_density(args.density, mol, 
            args.side_length, args.solvent)
            setupBox.run_packmol_solvation(n_molecules, args.side_length, args.solvent)
    pdb_path = args.pdb if args.pdb is not None else "pdb/init-packmol.pdb"
    setupBox.setup_box_generic(pdb_path, side_length=args.side_length, tag=str(args.res).lower())
    
    from mmml.pycharmmInterface.import_pycharmm import reset_block
    reset_block()
    from mmml.pycharmmInterface.import_pycharmm import reset_block, pycharmm, reset_block_no_internal
    reset_block()
    reset_block_no_internal()
    reset_block()
    nbonds = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################

! Non-bonding parameters
nbonds atom cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
vswitch NBXMOD 3 -
inbfrq -1 imgfrq -1
"""
    pycharmm.lingo.charmm_script(nbonds)
    pycharmm.energy.show()
    setupBox.minimize_box()


def main():
    args = parse_args()
    print(args)
    main_loop(args)

if __name__ == "__main__":
    main()
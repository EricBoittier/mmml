"""
Sets up a box for an MD simulation.
"""

"""
Sets up a residue for an MD simulation.
"""

from pathlib import Path
import os


import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--res", type=str)
    parser.add_argument(
        "--side_length",
        "--box-size",
        dest="side_length",
        type=float,
        default=300,
        help="Cubic box side length in Å. '--box-size' is an alias, matching the "
        "naming used elsewhere in the CLI suite.",
    )
    parser.add_argument("--pdb", type=str, default=None)
    parser.add_argument(
        "--solvent",
        type=str,
        default=None,
        help=(
            "CGenFF solvent residue name (any RESI in top_all36_cgenff.rtf), "
            "e.g. TIP3, MEOH, ACO, OCOH. Aliases: water→TIP3, octanol→OCOH."
        ),
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help=(
            "Solvent (or neat liquid) density in kg/m³. Built-in for TIP3/water "
            "(1000) and OCOH/octanol (824); required for other solvents when "
            "sizing N from density."
        ),
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)

def main_loop(args):
    from mmml.interfaces.pycharmmInterface import setupBox
    from mmml.interfaces.pycharmmInterface.utils import set_up_directories

    set_up_directories()  # ensure pdb/, psf/, xyz/, res/, dcd/ exist

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)} {type(getattr(args, arg))}")

    cwd = Path(os.getcwd())

    if args.pdb is not None:
        import ase.io
        mol = ase.io.read(args.pdb)
        print(mol)
        print(mol.get_chemical_symbols())
        pdb_path = args.pdb
    else:
        mol = setupBox.read_initial_pdb(cwd)
        print(mol)
        print(mol.get_chemical_symbols())

        if args.solvent is None:
            n_molecules = args.n
            setupBox.run_packmol(n_molecules, args.side_length)
            pdb_path = "pdb/init-packmol.pdb"
        else:
            from mmml.interfaces.pycharmmInterface.cgenff_residues import (
                require_cgenff_residue_name,
            )

            solvent = require_cgenff_residue_name(args.solvent)
            if args.density is not None:
                n_molecules = setupBox.determine_n_molecules_from_density(
                    args.density, mol, args.side_length, solvent
                )
            else:
                n_molecules = args.n
            setupBox.run_packmol_solvation(
                n_molecules,
                args.side_length,
                solvent,
                solute_mol=mol,
                density_kg_m3=args.density,
            )
            pdb_path = f"pdb/init-{solvent.lower()}box.pdb"
    setupBox.setup_box_generic(pdb_path, side_length=args.side_length, tag=str(args.res).lower())
    
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
    )
    reset_block()
    reset_block_no_internal()
    reset_block()
    setupBox.minimize_box()
    


def main():
    args = parse_args()
    print(args)
    main_loop(args)

if __name__ == "__main__":
    main()
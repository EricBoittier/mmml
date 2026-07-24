"""
Sets up a box for an MD simulation.

Packs a solute (``pdb/initial.pdb``, or ``--pdb`` staged there) into a cubic
cell — neat liquid or with ``--solvent`` — then builds a CHARMM PSF and
minimizes contacts.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pack a solute into a periodic box (vacuum copies or explicit solvent). "
            "Stages --pdb to pdb/initial.pdb when given, then runs Packmol + CHARMM."
        ),
    )
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--res",
        type=str,
        default="box",
        help="Tag for output files (psf/system-<res>.psf, pdb/init-<res>.pdb).",
    )
    parser.add_argument(
        "--side_length",
        "--box-size",
        dest="side_length",
        type=float,
        default=300,
        help="Cubic box side length in Å. '--box-size' is an alias, matching the "
        "naming used elsewhere in the CLI suite.",
    )
    parser.add_argument(
        "--pdb",
        type=str,
        default=None,
        help=(
            "Solute PDB (CGenFF residue/atom names). Copied to pdb/initial.pdb "
            "before Packmol. When omitted, pdb/initial.pdb must already exist "
            "(e.g. from mmml make-res)."
        ),
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default=None,
        help=(
            "CGenFF solvent residue name (any RESI in "
            "top_all36_cgenff.rtf), e.g. TIP3, MEOH, ACO, OCOH, ACN, DMSO. "
            "Aliases: water→TIP3, octanol→OCOH."
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


def _stage_solute_pdb(pdb: str | Path) -> Path:
    """Copy *pdb* to ``pdb/initial.pdb`` (Packmol input), preserving atom names."""
    src = Path(pdb).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Solute PDB not found: {src}")
    Path("pdb").mkdir(parents=True, exist_ok=True)
    dst = Path("pdb/initial.pdb").resolve()
    if src != dst:
        shutil.copy2(src, dst)
    return dst


def main_loop(args):
    from mmml.interfaces.pycharmmInterface import setupBox
    from mmml.interfaces.pycharmmInterface.utils import set_up_directories

    set_up_directories()  # ensure pdb/, psf/, xyz/, res/, dcd/ exist

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)} {type(getattr(args, arg))}")

    cwd = Path(os.getcwd())

    if args.pdb is not None:
        _stage_solute_pdb(args.pdb)

    mol = setupBox.read_initial_pdb(cwd)
    print(mol)
    print(mol.get_chemical_symbols())

    if args.solvent is None:
        if args.density is not None:
            n_molecules = setupBox.determine_n_molecules_from_density(
                args.density, mol, args.side_length, solvent=None
            )
        else:
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

    tag = str(args.res or "box").lower()
    setupBox.setup_box_generic(pdb_path, side_length=args.side_length, tag=tag)

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

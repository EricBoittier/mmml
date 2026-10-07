"""
Sets up a residue for an MD simulation.

Note: The final energy.show() call can segfault in CHARMM's bond routines (e.g.
__eintern_fast_MOD_ebondfs) when run under SLURM, on some cluster nodes, or with
certain MPI/threading. Residue and coordinates are already written before that call.
To avoid the segfault: use --skip-energy-show, or set SKIP_CHARMM_ENERGY_SHOW=1
(or "yes"/"true"). When SLURM_JOB_ID is set, energy.show() is skipped by default
unless RUN_CHARMM_ENERGY_SHOW=1 is set.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a CGENFF residue (PDB, PSF, topology) via PyCHARMM.",
        epilog=(
            "Examples:\n"
            "  mmml make-res --list-residues\n"
            "  mmml make-res --list-residues --no-pager | grep -i acetone\n"
            "  mmml make-res --res ACO"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--res",
        type=str,
        help="CGENFF residue name (RESI in top_all36_cgenff.rtf), e.g. ACO, CYBZ, TIP3.",
    )
    parser.add_argument(
        "--list-residues",
        action="store_true",
        help="List valid CGENFF residue names and descriptions (opens less on a TTY).",
    )
    parser.add_argument(
        "--no-pager",
        action="store_true",
        help="With --list-residues, print the table to stdout instead of piping to less.",
    )
    parser.add_argument(
        "--skip-energy-show",
        dest="skip_energy_show",
        action="store_true",
        help="Skip the final CHARMM energy.show() (avoids segfault on some clusters/SLURM).",
    )
    return parser


def parse_args():
    return build_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "list_residues", False):
        return
    if not getattr(args, "res", None):
        build_parser().error("--res is required unless --list-residues is set")


def main_loop(args):
    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.utils import set_up_directories

    set_up_directories()  # ensure pdb/, psf/, xyz/, res/, dcd/ exist before CHARMM
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        reset_block,
        reset_block_no_internal,
    )

    skip_energy_show = getattr(args, "skip_energy_show", False)
    atoms = setupRes.main(args.res, skip_energy_show=skip_energy_show, max_attempts=2)
    reset_block()
    reset_block_no_internal()
    reset_block()

    # ensure xyz files exist for downstream use
    import ase.io
    resid = args.res.upper()
    ase.io.write("xyz/initial.xyz", atoms)
    ase.io.write(f"xyz/{resid.lower()}.xyz", atoms)

    return atoms


def main():
    args = parse_args()
    validate_args(args)
    print(args)
    atoms = main_loop(args)
    return atoms


if __name__ == "__main__":
    main()

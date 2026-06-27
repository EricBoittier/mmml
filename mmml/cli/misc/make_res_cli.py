#!/usr/bin/env python
"""
CLI for make_res: generate residue structure (PDB, PSF, topology) via PyCHARMM/CGENFF.

Usage:
    mmml make-res --list-residues
    mmml make-res --res CYBZ
    mmml make-res --res CYBZ --skip-energy-show

Requires: CHARMM, PyCHARMM (charmm-interface) for --res; --list-residues needs only the RTF.
"""

import sys
import time


def main() -> int:
    """Run make-res CLI."""
    from mmml.cli.make.make_res import parse_args, validate_args

    args = parse_args()
    validate_args(args)

    if args.list_residues:
        from mmml.interfaces.pycharmmInterface.cgenff_residues import show_cgenff_residue_list

        show_cgenff_residue_list(pager=not args.no_pager)
        return 0

    t0 = time.perf_counter()
    try:
        from mmml.cli.make.make_res import main_loop
    except ModuleNotFoundError as e:
        if "pycharmm" in str(e).lower() or "charmm" in str(e).lower():
            print("Error: make-res requires PyCHARMM/CHARMM.", file=sys.stderr)
            print("See setup docs for CHARMM installation.", file=sys.stderr)
            return 1
        raise

    atoms = main_loop(args)
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(atoms)} atoms")
    print("Output: pdb/initial.pdb, psf/initial.psf, xyz/initial.xyz, CHARMM topology files")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
CLI for make_res: generate residue structure (PDB, PSF, topology) via PyCHARMM/CGENFF.

Usage:
    mmml make-res --res CYBZ
    mmml make-res --res CYBZ --skip-energy-show

Requires: CHARMM, PyCHARMM (charmm-interface)
"""

import sys
import time


def main() -> int:
    """Run make-res CLI."""
    t0 = time.perf_counter()
    try:
        from mmml.cli.make.make_res import parse_args, main_loop
    except ModuleNotFoundError as e:
        if "pycharmm" in str(e).lower() or "charmm" in str(e).lower():
            print("Error: make-res requires PyCHARMM/CHARMM.", file=sys.stderr)
            print("See setup docs for CHARMM installation.", file=sys.stderr)
            return 1
        raise

    args = parse_args()
    atoms = main_loop(args)
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(atoms)} atoms")
    print("Output: pdb/initial.pdb, psf/initial.psf, xyz/initial.xyz, CHARMM topology files")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

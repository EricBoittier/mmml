#!/usr/bin/env python
"""
CLI for make_box: pack molecules into a periodic box (vacuum or solvated).

Usage:
    mmml make-box --res CYBZ --n 50 --side_length 25.0
    mmml make-box --res CYBZ --n 50 --side_length 25.0 --solvent TIP3 --density 1.0

Requires: CHARMM, PyCHARMM, PackMol (charmm-interface)
"""

import sys
import time


def main() -> int:
    """Run make-box CLI."""
    t0 = time.perf_counter()
    try:
        from mmml.cli.make.make_box import parse_args, main_loop
    except ModuleNotFoundError as e:
        if "pycharmm" in str(e).lower() or "charmm" in str(e).lower():
            print("Error: make-box requires PyCHARMM/CHARMM.", file=sys.stderr)
            print("See setup docs for CHARMM installation.", file=sys.stderr)
            return 1
        raise

    args = parse_args()
    main_loop(args)
    elapsed = time.perf_counter() - t0
    print("Output: pdb/init-packmol.pdb (or pdb/init-TIP3box.pdb if solvated)")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

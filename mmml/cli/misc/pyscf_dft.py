#!/usr/bin/env python
"""
CLI for GPU-accelerated DFT calculations via PySCF/gpu4pyscf.

Usage:
    mmml pyscf-dft --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy
    mmml pyscf-dft --mol water.xyz --energy --gradient --output results.hdf5
    mmml pyscf-dft --mol water.xyz --energy --hessian --harmonic --thermo

Requires: gpu4pyscf, pyscf (GPU/quantum environment)
"""

import sys
import argparse
from pathlib import Path


def main() -> int:
    """Run pyscf-dft CLI."""
    from mmml.interfaces.pyscf4gpuInterface.calcs import (
        parse_args,
        process_calcs,
        compute_dft,
        save_output,
    )

    args = parse_args()
    calcs, extra = process_calcs(args)

    if not calcs:
        print("Error: At least one calculation flag is required.", file=sys.stderr)
        print("Use --energy, --gradient, --hessian, --optimize, etc.", file=sys.stderr)
        return 1

    output = compute_dft(args, calcs, extra)
    save_output(args.output, output, args.save_option)
    print(f"Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

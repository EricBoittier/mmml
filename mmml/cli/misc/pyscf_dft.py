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
import time
import argparse
from pathlib import Path


def main() -> int:
    """Run pyscf-dft CLI."""
    t0 = time.perf_counter()
    try:
        from mmml.interfaces.pyscf4gpuInterface.calcs import (
            parse_args,
            process_calcs,
            compute_dft,
            save_pyscf_results,
        )
    except ModuleNotFoundError as e:
        if "cupy" in str(e).lower() or "gpu4pyscf" in str(e).lower():
            print("Error: pyscf-dft requires cupy and gpu4pyscf.", file=sys.stderr)
            print("Install with: uv sync --extra quantum-gpu", file=sys.stderr)
            print("Or: uv pip install cupy-cuda13x gpu4pyscf-cuda13x", file=sys.stderr)
            return 1
        raise

    args = parse_args()
    calcs, extra = process_calcs(args)

    if not calcs:
        print("Error: At least one calculation flag is required.", file=sys.stderr)
        print("Use --energy, --gradient, --hessian, --optimize, etc.", file=sys.stderr)
        return 1

    output = compute_dft(args, calcs, extra)
    save_pyscf_results(args.output, output)
    elapsed = time.perf_counter() - t0
    base = Path(args.output).with_suffix("").name
    print(f"Results saved to {base}.npz (ML keys) and {base}.h5 (all arrays)")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

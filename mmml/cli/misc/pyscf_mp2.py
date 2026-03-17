#!/usr/bin/env python
"""
CLI for GPU-accelerated MP2 (post-HF) calculations via PySCF/gpu4pyscf.

MP2 is not DFT; use pyscf-dft for DFT and pyscf-mp2 for MP2.

Usage:
    mmml pyscf-mp2 --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy
    mmml pyscf-mp2 --mol water.xyz --energy --gradient --output results

Requires: gpu4pyscf, pyscf (GPU/quantum environment)
"""

import sys
import time
import argparse
from pathlib import Path


def main() -> int:
    """Run pyscf-mp2 CLI."""
    t0 = time.perf_counter()
    try:
        from mmml.interfaces.pyscf4gpuInterface.calcs import (
            compute_mp2,
            save_pyscf_results,
        )
    except ModuleNotFoundError as e:
        if "cupy" in str(e).lower() or "gpu4pyscf" in str(e).lower():
            print("Error: pyscf-mp2 requires cupy and gpu4pyscf.", file=sys.stderr)
            print("Install with: uv sync --extra quantum-gpu", file=sys.stderr)
            print("Or: uv pip install cupy-cuda13x gpu4pyscf-cuda13x", file=sys.stderr)
            return 1
        raise

    parser = argparse.ArgumentParser(description="GPU-accelerated MP2 calculations")
    parser.add_argument("--mol", type=str, required=True, help="Molecule (xyz string or file)")
    parser.add_argument("--output", type=str, default="output", help="Output base path (.npz and .h5)")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Basis set")
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--energy", action="store_true", help="Compute MP2 energy")
    parser.add_argument("--gradient", action="store_true", help="Compute MP2 gradient")
    parser.add_argument("--log_file", type=str, default="pyscf.log")
    args = parser.parse_args()

    if not args.energy and not args.gradient:
        print("Error: At least one of --energy or --gradient is required.", file=sys.stderr)
        return 1

    # Resolve mol: if it looks like a file path, read it
    mol_str = args.mol
    if Path(args.mol).exists():
        mol_str = Path(args.mol).read_text()

    output = compute_mp2(
        mol_str=mol_str,
        basis=args.basis,
        spin=args.spin,
        charge=args.charge,
        energy=args.energy,
        gradient=args.gradient,
        log_file=args.log_file,
    )
    save_pyscf_results(args.output, output)
    elapsed = time.perf_counter() - t0
    base = Path(args.output).with_suffix("").name
    print(f"Results saved to {base}.npz (ML keys) and {base}.h5 (all arrays)")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
CLI to evaluate sampled geometries with pyscf-dft (energy, forces, dipoles, ESP).

Runs all geometries in one process (same GPU context) for speed.
Input: NPZ with R (n_samples, n_atoms, 3), Z, N (e.g. from normal-mode-sample)
Output: NPZ with R, Z, N, E, F, Dxyz, esp, esp_grid (if --esp)

Usage:
    mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz
    mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz --esp
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    """Run pyscf-evaluate CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate geometries with pyscf-dft (energy, forces, dipoles, ESP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input NPZ with R, Z, N (e.g. from normal-mode-sample)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("evaluated.npz"),
        help="Output NPZ path (default: evaluated.npz)",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="def2-SVP",
        help="Basis set (default: def2-SVP)",
    )
    parser.add_argument(
        "--xc",
        type=str,
        default="PBE0",
        help="XC functional (default: PBE0)",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2*spin (0=singlet, 1=doublet, default: 0)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total charge (default: 0)",
    )
    parser.add_argument(
        "--no-energy",
        action="store_true",
        help="Skip energy (not recommended)",
    )
    parser.add_argument(
        "--no-gradient",
        action="store_true",
        help="Skip forces/gradients",
    )
    parser.add_argument(
        "--no-dipole",
        action="store_true",
        help="Skip dipole moments",
    )
    parser.add_argument(
        "--esp",
        action="store_true",
        help="Compute ESP on density-selected grid (slower)",
    )
    parser.add_argument(
        "--esp-cpu-fallback",
        action="store_true",
        help="Use CPU path for ESP (slower; default: GPU int1e_grids)",
    )

    args = parser.parse_args()
    t0 = time.perf_counter()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        from mmml.interfaces.pyscf4gpuInterface.calcs import compute_dft_batch
    except ModuleNotFoundError as e:
        if "cupy" in str(e).lower() or "gpu4pyscf" in str(e).lower():
            print("Error: pyscf-evaluate requires cupy and gpu4pyscf.", file=sys.stderr)
            print("Install with: uv sync --extra quantum-gpu", file=sys.stderr)
            return 1
        raise

    data = np.load(args.input, allow_pickle=True)
    R = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    if R.ndim == 2:
        R = R[np.newaxis, ...]
    if Z.ndim == 2:
        Z = Z[0]

    n_samples = R.shape[0]
    print(f"Evaluating {n_samples} geometries with pyscf-dft (GPU)...")

    result = compute_dft_batch(
        R,
        Z,
        basis=args.basis,
        xc=args.xc,
        spin=args.spin,
        charge=args.charge,
        energy=not args.no_energy,
        gradient=not args.no_gradient,
        dipole=not args.no_dipole,
        dens_esp=args.esp,
        esp_cpu_fallback=args.esp_cpu_fallback,
        verbose=0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)

    elapsed = time.perf_counter() - t0
    print(f"Saved to {args.output}")
    print(f"  R: {result['R'].shape}, E: {result.get('E', np.array([])).shape}, F: {result.get('F', np.array([])).shape}")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

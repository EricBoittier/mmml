#!/usr/bin/env python
"""
CLI to evaluate sampled geometries with pyscf (DFT or MP2).

Runs all geometries in one process (same GPU context).

Input: NPZ with R (n_samples, n_atoms, 3), Z
Output: NPZ with energies, forces, dipoles, etc.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _parse_efield_vector(s: str) -> np.ndarray:
    p = [float(x.strip()) for x in s.replace(" ", "").split(",")]
    if len(p) != 3:
        raise ValueError(f"Expected Ex,Ey,Ez comma-separated, got {s!r}")
    return np.array(p, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("out.npz"))

    parser.add_argument("--method", choices=["dft", "mp2"], default="dft")

    parser.add_argument("--basis", default="def2-SVP")
    parser.add_argument("--xc", default="PBE0")

    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--charge", type=int, default=0)

    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--no-gradient", action="store_true")
    parser.add_argument("--no-dipole", action="store_true")

    parser.add_argument("--esp", action="store_true")
    parser.add_argument("--esp-cpu-fallback", action="store_true")

    parser.add_argument("--EF", dest="efield_enabled", action="store_true")
    parser.add_argument("--efield", type=str, default=None)
    parser.add_argument("--efield-sigma", type=float, default=0.01)

    parser.add_argument("--add-random-noise", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--no-efield-include-nuclear-energy",
        dest="efield_include_nuclear_energy",
        action="store_false",
    )
    parser.set_defaults(efield_include_nuclear_energy=True)

    args = parser.parse_args()
    t0 = time.perf_counter()

    # -------------------------
    # imports (method switch)
    # -------------------------
    try:
        if args.method == "mp2":
            from mmml.interfaces.pyscf4gpuInterface.calcs_mp2 import compute_mp2_batch as compute_fn
        else:
            from mmml.interfaces.pyscf4gpuInterface.calcs import compute_dft_batch as compute_fn

    except ModuleNotFoundError as e:
        if "cupy" in str(e).lower() or "gpu4pyscf" in str(e).lower():
            print("Error: requires cupy + gpu4pyscf", file=sys.stderr)
            return 1
        raise

    # -------------------------
    # load data
    # -------------------------
    data = np.load(args.input, allow_pickle=True)
    R = np.asarray(data["R"], dtype=np.float64)
    Z = np.asarray(data["Z"])

    if R.ndim == 2:
        R = R[np.newaxis, ...]
    if Z.ndim == 2:
        Z = Z[0]

    n_samples = R.shape[0]
    rng = np.random.default_rng(args.seed)

    # -------------------------
    # noise
    # -------------------------
    if args.add_random_noise is not None and args.add_random_noise > 0:
        R = R + rng.normal(0.0, args.add_random_noise, size=R.shape)

    # -------------------------
    # electric field
    # -------------------------
    efield_pass = None
    if args.efield_enabled:
        if args.efield:
            efield_pass = _parse_efield_vector(args.efield)
        else:
            efield_pass = rng.normal(0.0, args.efield_sigma, size=(n_samples, 3))

    # -------------------------
    # logging
    # -------------------------
    print(f"Evaluating {n_samples} geometries with pyscf-{args.method.upper()} (GPU)...")

    if args.efield_enabled:
        print(f"  E-field enabled")

    if args.add_random_noise:
        print(f"  Noise sigma: {args.add_random_noise} Å")

    # -------------------------
    # compute
    # -------------------------
    result = compute_fn(
        R,
        Z,
        basis=args.basis,
        spin=args.spin,
        charge=args.charge,
        energy=not args.no_energy,
        gradient=not args.no_gradient,
        dipole=not args.no_dipole,
        verbose=0,
        efield=efield_pass,
        efield_include_nuclear_energy=args.efield_include_nuclear_energy,
    )

    # -------------------------
    # save
    # -------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)

    # -------------------------
    # summary
    # -------------------------
    print(f"Saved to {args.output}")

    print(
        f"  R: {result['R'].shape}, "
        f"E: {result.get('E', np.array([])).shape}, "
        f"F: {result.get('F', np.array([])).shape}"
    )

    if "Ef" in result:
        print(f"  Ef: {result['Ef'].shape}")

    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.2f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())

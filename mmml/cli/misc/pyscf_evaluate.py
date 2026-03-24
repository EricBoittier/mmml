#!/usr/bin/env python
"""
CLI to evaluate sampled geometries with pyscf-dft (energy, forces, dipoles, ESP).

Runs all geometries in one process (same GPU context) for speed.
Input: NPZ with R (n_samples, n_atoms, 3), Z, N (e.g. from normal-mode-sample)
Output: NPZ with R, Z, N, E, F, Dxyz, esp, esp_grid (if --esp), Ef (if --EF)

Usage:
    mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz
    mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz --esp
    mmml pyscf-evaluate -i traj.npz -o out.npz --EF
    mmml pyscf-evaluate -i traj.npz -o out.npz --EF --efield 0,0,0.01
    mmml pyscf-evaluate -i traj.npz -o out.npz --EF --no-efield-include-nuclear-energy
    mmml pyscf-evaluate -i traj.npz -o out.npz --add-random-noise 0.1
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
    parser.add_argument(
        "--EF",
        dest="efield_enabled",
        action="store_true",
        help=(
            "Include uniform electric field in the Hamiltonian (atomic units). "
            "Without --efield, draw a random (Ex,Ey,Ez) per geometry (see --efield-sigma). "
            "Giving --efield alone also enables the field (same vector for all frames)."
        ),
    )
    parser.add_argument(
        "--efield",
        type=str,
        default=None,
        metavar="Ex,Ey,Ez",
        help="Fixed field in a.u., same for all geometries (enables E-field even without --EF)",
    )
    parser.add_argument(
        "--efield-sigma",
        type=float,
        default=0.01,
        help="Std dev (a.u.) per component for random fields when --EF is set without --efield (default: 0.01)",
    )
    parser.add_argument(
        "--add-random-noise",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Gaussian noise std dev in Angstrom added to all R components before evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --add-random-noise and random --EF draws",
    )
    parser.add_argument(
        "--no-efield-include-nuclear-energy",
        dest="efield_include_nuclear_energy",
        action="store_false",
        help="With --EF/--efield: use mf.kernel energy only (omit nuclear-field term after SCF).",
    )
    parser.set_defaults(efield_include_nuclear_energy=True)

    args = parser.parse_args()
    t0 = time.perf_counter()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.efield:
        args.efield_enabled = True

    try:
        from mmml.interfaces.pyscf4gpuInterface.calcs import compute_dft_batch
    except ModuleNotFoundError as e:
        if "cupy" in str(e).lower() or "gpu4pyscf" in str(e).lower():
            print("Error: pyscf-evaluate requires cupy and gpu4pyscf.", file=sys.stderr)
            print("Install with: uv sync --extra quantum-gpu", file=sys.stderr)
            return 1
        raise

    data = np.load(args.input, allow_pickle=True)
    R = np.asarray(data["R"], dtype=np.float64)
    Z = np.asarray(data["Z"])
    if R.ndim == 2:
        R = R[np.newaxis, ...]
    if Z.ndim == 2:
        Z = Z[0]

    n_samples = R.shape[0]
    rng = np.random.default_rng(args.seed)

    if args.add_random_noise is not None:
        if args.add_random_noise < 0:
            print("Error: --add-random-noise must be >= 0", file=sys.stderr)
            return 1
        if args.add_random_noise > 0:
            R = R + rng.normal(0.0, args.add_random_noise, size=R.shape)

    efield_pass = None
    if args.efield_enabled:
        if args.efield:
            try:
                efield_pass = _parse_efield_vector(args.efield)
            except ValueError as err:
                print(f"Error: {err}", file=sys.stderr)
                return 1
        else:
            efield_pass = rng.normal(0.0, args.efield_sigma, size=(n_samples, 3))

    print(f"Evaluating {n_samples} geometries with pyscf-dft (GPU)...")
    if args.efield_enabled:
        print(
            f"  E-field: {'fixed ' + str(efield_pass) if args.efield else f'random per frame (sigma={args.efield_sigma} a.u.)'}"
        )
        if not args.efield_include_nuclear_energy:
            print("  E-field energy: mf.kernel only (--no-efield-include-nuclear-energy)")
    if args.add_random_noise and args.add_random_noise > 0:
        print(f"  Position noise: Gaussian sigma={args.add_random_noise} Angstrom on R")

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
        efield=efield_pass,
        efield_include_nuclear_energy=args.efield_include_nuclear_energy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)

    elapsed = time.perf_counter() - t0
    print(f"Saved to {args.output}")
    print(f"  R: {result['R'].shape}, E: {result.get('E', np.array([])).shape}, F: {result.get('F', np.array([])).shape}")
    if "Ef" in result:
        print(f"  Ef: {result['Ef'].shape}")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

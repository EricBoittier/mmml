#!/usr/bin/env python
"""
CLI for normal mode sampling from pyscf-dft harmonic output.

Samples geometries along vibrational modes for downstream QM/ML.
Input: .h5 from mmml pyscf-dft --harmonic
Output: NPZ with R (n_samples, n_atoms, 3), Z, N

Usage:
    mmml normal-mode-sample -i out/04_results.h5 -o out/06_sampled.npz --amplitude 0.1
    mmml normal-mode-sample -i out/04_results.h5 -o out/06_sampled.npz --amplitude 0.1 --include-equilibrium
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


BOHR_TO_ANG = 0.529177


def _load_h5(path: Path) -> dict:
    """Load R, Z, harmonic/norm_mode, harmonic/freq_wavenumber from h5."""
    import h5py

    data = {}
    with h5py.File(path, "r") as f:
        data["R"] = np.asarray(f["R"])
        data["Z"] = np.asarray(f["Z"])
        if "harmonic" not in f:
            raise ValueError(f"No 'harmonic' group in {path}. Run pyscf-dft with --harmonic --hessian.")
        grp = f["harmonic"]
        data["norm_mode"] = np.asarray(grp["norm_mode"])
        data["freq_wavenumber"] = np.asarray(grp["freq_wavenumber"])

    return data


def sample_normal_modes(
    R_eq: np.ndarray,
    Z: np.ndarray,
    norm_mode: np.ndarray,
    freq_wavenumber: np.ndarray,
    *,
    amplitudes: list[float],
    freq_min: float = 50.0,
    include_equilibrium: bool = False,
    samples_per_mode: int = 2,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate displaced geometries along vibrational modes.

    Parameters
    ----------
    R_eq : (n_atoms, 3)
        Equilibrium geometry in Angstrom.
    Z : (n_atoms,)
        Atomic numbers.
    norm_mode : (n_modes, n_atoms, 3)
        Normal mode vectors in Bohr (PySCF convention).
    freq_wavenumber : (n_modes,)
        Frequencies in cm^-1 (may be complex for imaginary).
    amplitudes : list of float
        Displacement amplitudes in Angstrom.
    freq_min : float
        Minimum frequency (cm^-1) to include; skip translational/rotational.
    include_equilibrium : bool
        Include R_eq as first sample.
    samples_per_mode : int
        2 for +/- amplitude; 1 for + only.
    max_samples : int, optional
        Maximum number of structures to generate (None = no limit).

    Returns
    -------
    R_samples : (n_samples, n_atoms, 3)
    Z : (n_atoms,)
    """
    freq_real = np.real(freq_wavenumber)
    mask = freq_real > freq_min
    mode_indices = np.where(mask)[0]

    if len(mode_indices) == 0:
        raise ValueError(
            f"No vibrational modes with freq > {freq_min} cm^-1. "
            "Try lowering --freq-min or check harmonic output."
        )

    mode_ang = norm_mode * BOHR_TO_ANG  # (n_modes, n_atoms, 3) in Angstrom

    samples = []
    if include_equilibrium:
        samples.append(R_eq.copy())

    n_modes = len(mode_indices)

    # 1. Single-mode displacements
    for k in mode_indices:
        if max_samples is not None and len(samples) >= max_samples:
            break
        mode_k = mode_ang[k]  # (n_atoms, 3)
        for amp in amplitudes:
            if max_samples is not None and len(samples) >= max_samples:
                break
            if samples_per_mode >= 2:
                samples.append(R_eq + amp * mode_k)
                if max_samples is not None and len(samples) >= max_samples:
                    break
                samples.append(R_eq - amp * mode_k)
            else:
                samples.append(R_eq + amp * mode_k)

    # 2. Two-mode combination displacements (when max_samples exceeds single-mode count)
    if max_samples is not None and len(samples) < max_samples and n_modes >= 2:
        scale = 1.0 / np.sqrt(2)  # keep similar displacement magnitude
        for i in range(n_modes):
            if max_samples is not None and len(samples) >= max_samples:
                break
            mode_i = mode_ang[mode_indices[i]]
            for j in range(i + 1, n_modes):
                if max_samples is not None and len(samples) >= max_samples:
                    break
                mode_j = mode_ang[mode_indices[j]]
                for amp in amplitudes:
                    if max_samples is not None and len(samples) >= max_samples:
                        break
                    disp = scale * amp * (mode_i + mode_j)
                    samples.append(R_eq + disp)
                    if max_samples is not None and len(samples) >= max_samples:
                        break
                    disp = scale * amp * (mode_i - mode_j)
                    samples.append(R_eq + disp)

    R_samples = np.stack(samples, axis=0)
    return R_samples, Z


def main() -> int:
    """Run normal-mode-sample CLI."""
    parser = argparse.ArgumentParser(
        description="Sample geometries along vibrational modes from pyscf-dft harmonic output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to pyscf-dft output .h5 (must contain harmonic group)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("sampled.npz"),
        help="Output NPZ path (default: sampled.npz)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Displacement amplitude in Angstrom (default: 0.1)",
    )
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=None,
        help="List of amplitudes (overrides --amplitude)",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=50.0,
        help="Minimum frequency (cm^-1) to include (default: 50)",
    )
    parser.add_argument(
        "--include-equilibrium",
        action="store_true",
        help="Add equilibrium geometry as first sample",
    )
    parser.add_argument(
        "--samples-per-mode",
        type=int,
        choices=[1, 2],
        default=2,
        help="2 for +/- amplitude (default), 1 for + only",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of structures to generate (default: no limit)",
    )

    args = parser.parse_args()
    t0 = time.perf_counter()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        data = _load_h5(args.input)
    except Exception as e:
        print(f"Error loading {args.input}: {e}", file=sys.stderr)
        return 1

    R_eq = data["R"]
    if R_eq.ndim == 3:
        R_eq = R_eq[0]
    Z = data["Z"]
    if Z.ndim == 2:
        Z = Z[0]
    norm_mode = data["norm_mode"]
    freq_wavenumber = data["freq_wavenumber"]

    amplitudes = args.amplitudes if args.amplitudes is not None else [args.amplitude]

    R_samples, Z_out = sample_normal_modes(
        R_eq,
        Z,
        norm_mode,
        freq_wavenumber,
        amplitudes=amplitudes,
        freq_min=args.freq_min,
        include_equilibrium=args.include_equilibrium,
        samples_per_mode=args.samples_per_mode,
        max_samples=args.max_samples,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        R=R_samples,
        Z=Z_out,
        N=np.array(len(Z_out)),
    )

    elapsed = time.perf_counter() - t0
    print(f"Saved {R_samples.shape[0]} geometries to {args.output}")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

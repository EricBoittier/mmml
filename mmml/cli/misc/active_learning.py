#!/usr/bin/env python
"""
CLI to extract frames from MD trajectories for active learning.

Filters frames by temperature (e.g. T < 300 K) and saves to NPZ format
compatible with mmml pyscf-evaluate for extending the training set.

Usage:
    mmml active-learning -i out/physnet_md/physnet_ase.traj -o md_sampled.npz
    mmml active-learning -i traj1.traj traj2.traj -o md_sampled.npz --max-temp 300
    mmml active-learning -i "out/*.traj" -o md_sampled.npz --stride 5
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

K_B = 8.617333e-5  # eV/K


def _get_temperature(atoms) -> float | None:
    """Compute instantaneous temperature from velocities (eV, Å, amu). Returns None if no velocities."""
    v = atoms.get_velocities()
    if v is None:
        return None
    m = atoms.get_masses()
    if m is None or len(m) == 0:
        return None
    # E_kin = 0.5 * sum(m * v^2), T = 2 * E_kin / (3 * N * k_B)
    e_kin = 0.5 * np.sum(m[:, None] * v**2)
    n_atoms = len(atoms)
    T = 2 * e_kin / (3 * n_atoms * K_B)
    return float(T)


def _load_trajectory_frames(path: Path):
    """Yield (atoms, temperature) from .traj or .xyz. T is None if unknown."""
    from ase.io import read

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {path}")
    if path.is_dir():
        raise ValueError(
            f"Expected a trajectory file, got directory: {path}\n"
            f"Use a .traj or .xyz file, e.g. programmatic/out/physnet_md/physnet_ase.traj"
        )

    if path.suffix.lower() == ".traj":
        from ase.io.trajectory import Trajectory

        traj = Trajectory(str(path))
        for atoms in traj:
            T = _get_temperature(atoms)
            yield atoms, T
    else:
        # .xyz or other multi-frame format
        frames = read(str(path), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        for atoms in frames:
            T = _get_temperature(atoms)
            yield atoms, T


def _collect_frames(
    paths: list[Path],
    max_temp: float | None,
    stride: int,
    max_frames: int | None,
    verbose: bool,
) -> tuple[list, list, np.ndarray]:
    """Collect R, Z, N from trajectories. Returns (R_list, Z, N_array)."""
    R_list = []
    Z_ref = None
    n_per_frame = []

    total_seen = 0
    total_kept = 0

    for path in paths:
        path = Path(path)
        for i, (atoms, T) in enumerate(_load_trajectory_frames(path)):
            total_seen += 1
            if stride > 1 and (i % stride) != 0:
                continue
            if max_temp is not None and T is not None and T >= max_temp:
                continue
            if max_frames is not None and total_kept >= max_frames:
                break

            R = np.asarray(atoms.get_positions(), dtype=np.float64)
            Z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
            n_atoms = len(atoms)

            if Z_ref is None:
                Z_ref = Z
            elif not np.array_equal(Z_ref, Z):
                if verbose:
                    print(
                        f"  Warning: Skipping frame (Z mismatch) in {path} frame {i}",
                        file=sys.stderr,
                    )
                continue

            R_list.append(R)
            n_per_frame.append(n_atoms)
            total_kept += 1

        if max_frames is not None and total_kept >= max_frames:
            break

    if not R_list:
        Z_out = np.array([], dtype=np.int32) if Z_ref is None else Z_ref
        return np.array([]), Z_out, np.array([], dtype=np.int32)

    R_arr = np.stack(R_list, axis=0)
    N_arr = np.array(n_per_frame, dtype=np.int32)
    return R_arr, Z_ref, N_arr


def main() -> int:
    """Run active-learning CLI."""
    parser = argparse.ArgumentParser(
        description="Extract frames from MD trajectories for active learning (pyscf-evaluate input).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        metavar="TRAJ",
        help="Trajectory file(s) (.traj, .xyz). Globs supported, e.g. 'out/*.traj'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("md_sampled.npz"),
        help="Output NPZ path (default: md_sampled.npz)",
    )
    parser.add_argument(
        "--max-temp",
        type=float,
        default=300.0,
        metavar="K",
        help="Keep only frames with T < max-temp K (default: 300). Ignored if trajectories have no velocities.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Maximum frames to extract (default: no limit)",
    )
    parser.add_argument(
        "--no-temp-filter",
        action="store_true",
        help="Do not filter by temperature (keep all frames)",
    )

    args = parser.parse_args()

    # Expand globs
    paths = []
    for p in args.input:
        expanded = glob.glob(p)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)

    paths = [Path(p).resolve() for p in paths]
    if not paths:
        print("Error: No trajectory files found.", file=sys.stderr)
        return 1

    for p in paths:
        if not p.exists():
            print(f"Error: File not found: {p}", file=sys.stderr)
            return 1

    max_temp = None if args.no_temp_filter else args.max_temp
    if max_temp is not None:
        print(f"Filtering frames with T < {max_temp} K")

    R_arr, Z, N = _collect_frames(
        paths,
        max_temp=max_temp,
        stride=args.stride,
        max_frames=args.max_frames,
        verbose=True,
    )

    if len(R_arr) == 0:
        print("Error: No frames collected. Check --max-temp, --stride, or trajectory format.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, R=R_arr, Z=Z, N=N)

    print(f"Saved {len(R_arr)} frames to {args.output}")
    print(f"  R: {R_arr.shape}, Z: {Z.shape}, N: {N.shape}")
    print("Next: mmml pyscf-evaluate -i", args.output, "-o md_evaluated.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())

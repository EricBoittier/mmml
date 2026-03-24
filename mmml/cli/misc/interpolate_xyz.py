#!/usr/bin/env python
"""
CLI: interpolate between two XYZ geometries in internal (Z-matrix) coordinates.

Uses the first structure's Z-matrix topology; the second XYZ must match atom
order and count. Writes a compressed NPZ with R, Z, N per frame (same layout as
interpolate_xyzs_to_npz).

Usage:
    mmml interpolate-xyz start.xyz end.xyz -o path.npz --steps 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate between two XYZ files via Z-matrix coordinates and "
            "save frames to NPZ (R, Z, N)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "xyz1",
        type=Path,
        help="First XYZ file (defines Z-matrix connectivity)",
    )
    parser.add_argument(
        "xyz2",
        type=Path,
        help="Second XYZ file (same atoms and ordering as the first)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("interpolated.npz"),
        help="Output NPZ path (default: interpolated.npz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        metavar="N",
        help="Number of interpolation segments (N+1 frames written; default: 1000)",
    )

    args = parser.parse_args()
    t0 = time.perf_counter()

    if args.steps < 1:
        print("Error: --steps must be >= 1", file=sys.stderr)
        return 1

    for label, p in ("xyz1", args.xyz1), ("xyz2", args.xyz2):
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            return 1

    try:
        from mmml.interfaces.chemcoordInterface.interface import (
            interpolate_xyzs_to_npz,
        )
    except ImportError as e:
        print(f"Error: could not import chemcoord interface: {e}", file=sys.stderr)
        return 1

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        interpolate_xyzs_to_npz(
            str(args.xyz1.resolve()),
            str(args.xyz2.resolve()),
            steps=args.steps,
            out_fn=str(args.output.resolve()),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    n_frames = args.steps + 1
    print(f"Wrote {n_frames} frames to {args.output}")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

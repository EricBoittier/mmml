#!/usr/bin/env python3
"""Estimate MMFP droff from a centered cluster: R = 1 + sqrt(x_max^2 + y_max^2 + z_max^2).

Reads CHARMM CRD or PDB (Å). Translates to COM at origin, then reports axis maxima and R.
Use after minimization to set --flat-bottom-radius / --fb-rad for dynamics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _read_coords(path: Path) -> list[list[float]]:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if path.suffix.lower() == ".pdb":
        coords: list[list[float]] = []
        for line in text:
            if line.startswith(("ATOM", "HETATM")):
                coords.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
        return coords
    # CHARMM extended CRD: skip header, read x y z from fixed columns
    coords = []
    for line in text[2:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        coords.append([x, y, z])
    return coords


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=Path, help="CHARMM CRD or PDB")
    p.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Add this Å to sqrt(x^2+y^2+z^2) before droff (default: 1.0)",
    )
    args = p.parse_args(argv)
    coords = _read_coords(args.structure.expanduser().resolve())
    if not coords:
        print(f"No coordinates in {args.structure}", file=sys.stderr)
        return 1
    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    xs = [abs(c[0] - cx) for c in coords]
    ys = [abs(c[1] - cy) for c in coords]
    zs = [abs(c[2] - cz) for c in coords]
    xmax, ymax, zmax = max(xs), max(ys), max(zs)
    core = math.sqrt(xmax * xmax + ymax * ymax + zmax * zmax)
    droff = args.margin + core
    print(f"atoms={n}  COM=({cx:.3f}, {cy:.3f}, {cz:.3f}) Å")
    print(f"x_max={xmax:.3f}  y_max={ymax:.3f}  z_max={zmax:.3f} Å")
    print(f"sqrt(x^2+y^2+z^2)={core:.3f} Å")
    print(f"droff R = {args.margin} + ... = {droff:.3f} Å")
    print(f"Suggested: --flat-bottom-radius {droff:.1f} --packmol-radius {0.65 * core:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

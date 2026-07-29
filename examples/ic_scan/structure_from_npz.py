#!/usr/bin/env python3
"""Write an XYZ structure from an ACEM (or other) EF NPZ frame.

Default: lowest-energy frame. Use this so ic-scan atom order matches training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True, help="NPZ with R, E, and Z")
    p.add_argument("--out", type=Path, required=True, help="Output XYZ path")
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Frame index (default: argmin E)",
    )
    args = p.parse_args(argv)

    data = np.load(args.npz)
    if "R" not in data or "E" not in data:
        raise SystemExit(f"{args.npz} must contain R and E")
    R = np.asarray(data["R"], dtype=float)
    E = np.asarray(data["E"], dtype=float).reshape(-1)
    if "Z" not in data:
        raise SystemExit(f"{args.npz} must contain Z (atomic numbers)")
    Z = np.asarray(data["Z"])
    z0 = Z[0] if Z.ndim == 2 else Z
    idx = int(np.argmin(E)) if args.index is None else int(args.index)
    symbols = [chemical_symbols[int(z)] for z in np.asarray(z0).reshape(-1)]
    atoms = Atoms(symbols=symbols, positions=R[idx])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, atoms)
    print(
        f"wrote {args.out}  frame={idx}  E={float(E[idx]):.6f}  "
        f"Z={list(map(int, z0))}  symbols={symbols}"
    )
    print(
        "If Z order is not CGenFF ACEM (C,C,N,H,H,O,H,H,H), update "
        "dihedral atom indices in the ic-scan YAML."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

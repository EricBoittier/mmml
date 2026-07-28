#!/usr/bin/env python3
"""Export NEB / umbrella / DMC endpoint XYZ files from the bundled NPZ.

Atom order matches the dataset and :mod:`mmml.neb` defaults:
``Cl, N, C, H×3(N), H×3(C)`` (indices 0–8).

Writes under ``examples/m/neb/`` by default:

| File | Basin |
|------|-------|
| ``reag_0_opt.xyz`` | Reactant-like (short Cl–C, longer C–N) |
| ``prod_0_opt.xyz`` | Product-like (long Cl–C, short C–N) |
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from _geometry import (  # noqa: E402
    DEFAULT_NPZ,
    find_frame_near_rc,
    load_dimer_frame,
)
DEFAULT_OUT = EXAMPLE_DIR / "neb"

# (label, target r_ClC Å, target r_CN Å)
_DEFAULT_BASINS: tuple[tuple[str, float, float], ...] = (
    ("reag_0_opt", 1.85, 2.80),
    ("prod_0_opt", 3.80, 1.57),
)

_ELEMENT = ("Cl", "N", "C", "H", "H", "H", "H", "H", "H")


def _write_xyz(path: Path, z: np.ndarray, r: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(z)), comment]
    for sym, xyz in zip(_ELEMENT, r, strict=True):
        x, y, zc = (float(v) for v in xyz)
        lines.append(f"{sym:2s}  {x:12.6f}  {y:12.6f}  {zc:12.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_basins(
    *,
    npz_path: Path,
    out_dir: Path,
    basins: tuple[tuple[str, float, float], ...] = _DEFAULT_BASINS,
) -> list[Path]:
    written: list[Path] = []
    for stem, rcl, rcn in basins:
        idx, xi, r_clc, r_cn = find_frame_near_rc(rcl, rcn, npz_path)
        _z, r = load_dimer_frame(npz_path, index=idx)
        out = out_dir / f"{stem}.xyz"
        comment = (
            f"NPZ index {idx}; xi={xi:.4f}; r(Cl-C)={r_clc:.4f}; r(C-N)={r_cn:.4f} Å"
        )
        _write_xyz(out, _z, r, comment)
        written.append(out)
        print(f"Wrote {out}  ({comment})")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ,
        help=f"Filtered dataset (default: {DEFAULT_NPZ.name})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Directory for endpoint XYZ (default: {DEFAULT_OUT.relative_to(EXAMPLE_DIR.parent.parent)})",
    )
    args = parser.parse_args()
    export_basins(npz_path=args.npz, out_dir=args.output_dir)


if __name__ == "__main__":
    main()

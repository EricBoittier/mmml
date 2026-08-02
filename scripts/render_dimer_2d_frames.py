#!/usr/bin/env python3
"""Render selected frames of the 2D dimer grid with POV-Ray, for verification.

The surface plots assert that a given (R, theta) cell holds a particular dimer
geometry. That assertion is worth checking rather than trusting: this session
found `--evaluate-npz` silently ignoring term flags, a campaign silently running
one-molecule systems, and ASE silently reading CHARMM's OG as oganesson. A
picture of the actual frame, pulled from the same NPZ the evaluator was given,
is the cheapest way to confirm the geometry is what the axes claim.

Frames are selected by index so the caller can point at exactly the cells marked
on the surface. Each render is accompanied by the frame's measured R, theta and
closest intermolecular contact, recomputed here from the coordinates rather than
copied from the grid metadata -- so a mismatch between the file and its own
labels would show up.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

MASS = {1: 1.008, 8: 15.999}
SYMBOL = {1: "H", 8: "O"}


def com(xyz: np.ndarray, z: np.ndarray) -> np.ndarray:
    m = np.array([MASS[int(v)] for v in z])
    return (xyz * m[:, None]).sum(0) / m.sum()


def frame_metrics(xyz: np.ndarray, z: np.ndarray) -> dict[str, float]:
    """R and closest contact, measured from the coordinates themselves."""
    a, b = xyz[:3], xyz[3:]
    za, zb = z[:3], z[3:]
    r = float(np.linalg.norm(com(b, zb) - com(a, za)))
    contact = float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min())
    # theta from the orientation of monomer B's bisector relative to A's.
    def bisector(m: np.ndarray) -> np.ndarray:
        v = (m[1] - m[0]) + (m[2] - m[0])
        return v / np.linalg.norm(v)

    ca, cb = bisector(a), bisector(b)
    theta = float(np.degrees(np.arccos(np.clip(ca @ cb, -1.0, 1.0))))
    return {"R_A": r, "min_contact_A": contact, "bisector_angle_deg": theta}


def write_pdb(path: Path, xyz: np.ndarray, z: np.ndarray) -> None:
    """PDB with a real element column, so ASE cannot misread the elements."""
    lines = []
    for i, (p, zi) in enumerate(zip(xyz, z), start=1):
        sym = SYMBOL[int(zi)]
        res = 1 if i <= 3 else 2
        lines.append(
            f"ATOM  {i:5d} {sym:<3s} HOH  {res:4d}    "
            f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {sym:>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--rotation", default="-80x, 0y, 0z")
    ap.add_argument("--box-side", type=float, default=16.0,
                    help="fixed cubic cell (A) so every frame renders at the SAME "
                         "scale; without it POV-Ray auto-fits per frame and the "
                         "geometries cannot be compared by eye")
    a = ap.parse_args(argv)

    g = np.load(a.grid)
    R, Z = np.asarray(g["R"], dtype=np.float64), np.asarray(g["Z"])
    a.outdir.mkdir(parents=True, exist_ok=True)

    render = Path(__file__).with_name("render_liquid_box_povray.py")
    for idx in a.frames:
        xyz, z = R[idx], Z[idx]
        m = frame_metrics(xyz, z)
        # Centre the dimer in a fixed cell: identical camera framing for every
        # frame, and the drawn cell edges give a visual scale bar.
        xyz = xyz - com(xyz, z) + np.full(3, a.box_side / 2.0)
        pdb = a.outdir / f"frame_{idx:04d}.pdb"
        png = a.outdir / f"frame_{idx:04d}.png"
        write_pdb(pdb, xyz, z)
        cmd = [
            sys.executable, str(render), str(pdb), "-o", str(png),
            "--width", str(a.width), "--rotation", a.rotation,
            "--box-side", str(a.box_side),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        ok = png.is_file()
        print(
            f"  frame {idx:4d}  R={m['R_A']:5.2f} A  contact={m['min_contact_A']:5.2f} A"
            f"  bisector={m['bisector_angle_deg']:6.1f} deg  "
            f"{'rendered' if ok else 'RENDER FAILED'}"
        )
        if not ok:
            tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-3:]
            for line in tail:
                print(f"      {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

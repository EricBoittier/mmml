#!/usr/bin/env python3
"""Convert an ASE-readable trajectory to a CHARMM/NAMD DCD for VMD.

Example
-------
uv run python scripts/ase_traj_to_charmm_dcd.py \
  --traj artifacts/trialanine_phi_psi_pes/phi_psi_pes.traj \
  --out artifacts/trialanine_phi_psi_pes/phi_psi_pes.dcd

Then load in VMD with a matching PSF:
  vmd topology.psf phi_psi_pes.dcd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

from mmml.utils.dcd_writer import save_trajectory_dcd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", required=True, type=Path, help="ASE-readable input trajectory")
    parser.add_argument("--out", required=True, type=Path, help="Output DCD path")
    parser.add_argument("--index", default=":", help="ASE frame index/slice, default ':'")
    parser.add_argument("--dt-ps", type=float, default=1.0, help="DCD timestep metadata in ps")
    parser.add_argument("--steps-per-frame", type=int, default=1, help="DCD NSAVC metadata")
    args = parser.parse_args()

    frames = read(args.traj, index=args.index)
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise ValueError(f"No frames read from {args.traj}")

    n_atoms = len(frames[0])
    for frame_index, frame in enumerate(frames):
        if len(frame) != n_atoms:
            raise ValueError(
                f"Frame {frame_index} has {len(frame)} atoms; expected {n_atoms}"
            )

    positions = np.asarray([frame.get_positions() for frame in frames], dtype=np.float32)
    boxes = []
    for frame in frames:
        cell = np.asarray(frame.get_cell().array, dtype=float)
        boxes.append(cell if np.linalg.norm(cell) > 1e-12 else np.zeros(3))
    has_box = any(np.linalg.norm(box) > 1e-12 for box in boxes)

    save_trajectory_dcd(
        args.out,
        positions,
        frames[0],
        boxes=boxes if has_box else None,
        dt_ps=args.dt_ps,
        steps_per_frame=args.steps_per_frame,
    )
    print(f"Wrote {args.out} ({len(frames)} frames, {n_atoms} atoms)")


if __name__ == "__main__":
    main()

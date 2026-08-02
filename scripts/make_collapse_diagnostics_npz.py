#!/usr/bin/env python3
"""Build the two NPZs that explain the bulk descent.

The bulk NVT run does not collapse intermolecularly -- the closest O-O contact
holds at ~2.3 A for the whole trajectory. What comes apart is the molecules
themselves: O-H bonds spread from 0.95-0.99 A to 0.57-1.83 A while E_pot falls
without bound (-350 kcal/mol per water by the last recorded frame, still
falling).

The DES training dimers are perfectly rigid -- O-H = 0.9840 A with std exactly
0.0, HOH = 104.60 deg, in all 295 of them. So the ML term owns the internal
monomer energy (see setup_calculator's docstring) but has seen exactly one
internal geometry. Away from it the monomer surface is pure extrapolation.

Two datasets, both evaluated as single points with terms switched on and off:

  intramolecular_scan  one O-H bond of monomer A scanned through the range the
                       MD actually visited, everything else held at the training
                       geometry. A potential fit to rigid monomers has no
                       restoring wall here; a sound one has a steep minimum near
                       0.984 A. This is the mechanism test.

  collapse_frames      the recorded frames of the failing NVT trajectory. Arm
                       differences attribute the released energy to a term.
                       This is the attribution test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The single internal geometry every DES training dimer uses.
TRAIN_OH_A = 0.9840
TRAIN_HOH_DEG = 104.60


def _write(path: Path, R: np.ndarray, Z: np.ndarray, meta: dict) -> None:
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.int32)
    n_frames, n_atoms, _ = R.shape
    np.savez_compressed(
        path,
        R=R,
        Z=Z if Z.ndim == 2 else np.broadcast_to(Z.reshape(1, -1), (n_frames, n_atoms)).copy(),
        N=np.full(n_frames, n_atoms, dtype=np.int32),
        # Placeholder reference: these datasets exist to read the model's own
        # energy, not to score it against anything.
        E=np.zeros(n_frames, dtype=np.float64),
        _mmml_units=np.array(json.dumps({"E": "ev", "R": "angstrom"})),
        metadata=json.dumps(meta),
    )
    print(f"wrote {path}  frames={n_frames} atoms={n_atoms}")


def build_scan(des_npz: Path, out: Path, n_points: int) -> None:
    """One O-H of monomer A scanned; all other internal coordinates frozen."""
    data = np.load(des_npz, allow_pickle=True)
    R_all, Z_all = data["R"], data["Z"]
    water = np.array([tuple(z) == (8, 1, 1, 8, 1, 1) for z in Z_all])
    if not water.any():
        raise SystemExit("no water-dimer frame found in the reference NPZ")

    # Pick the frame whose O-O separation is closest to the liquid first peak,
    # so the scan sits in a condensed-phase-relevant environment.
    idx = np.where(water)[0]
    oo = np.array([np.linalg.norm(R_all[i][0] - R_all[i][3]) for i in idx])
    base_i = int(idx[int(np.argmin(np.abs(oo - 2.75)))])
    base = np.asarray(R_all[base_i], dtype=np.float64).copy()
    z = np.asarray(Z_all[base_i], dtype=np.int32)
    print(f"base frame {base_i}, O-O = {np.linalg.norm(base[0] - base[3]):.3f} A")

    o, h = base[0], base[1]
    unit = (h - o) / np.linalg.norm(h - o)
    # Cover the range the MD visited (0.571 .. 1.825 A) with margin.
    lengths = np.linspace(0.55, 1.95, n_points)

    frames = []
    for d in lengths:
        f = base.copy()
        f[1] = o + unit * d
        frames.append(f)

    _write(
        out,
        np.stack(frames),
        z,
        {
            "kind": "intramolecular_oh_scan",
            "base_frame": base_i,
            "scanned_atom": 1,
            "oh_lengths_A": lengths.tolist(),
            "training_oh_A": TRAIN_OH_A,
            "md_oh_range_A": [0.571, 1.825],
        },
    )


def build_frames(traj: Path, out: Path, stride: int) -> None:
    """The recorded frames of the failing NVT trajectory."""
    from ase.io import Trajectory

    t = Trajectory(str(traj))
    sel = list(range(0, len(t), stride))
    R = np.stack([np.asarray(t[i].get_positions(), dtype=np.float64) for i in sel])
    z = np.asarray(t[0].numbers, dtype=np.int32)
    cell = float(t[0].cell.lengths()[0])
    _write(
        out,
        R,
        z,
        {
            "kind": "bulk_collapse_frames",
            "trajectory": str(traj),
            "frame_indices": sel,
            "box_A": cell,
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--des-npz", type=Path, help="DES dimer NPZ, source of the scan base geometry")
    p.add_argument("--traj", type=Path, help="failing NVT trajectory (.traj)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scan-points", type=int, default=71)
    p.add_argument("--frame-stride", type=int, default=1)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.des_npz:
        build_scan(args.des_npz, args.out_dir / "intramolecular_scan.npz", args.scan_points)
    if args.traj:
        build_frames(args.traj, args.out_dir / "collapse_frames.npz", args.frame_stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

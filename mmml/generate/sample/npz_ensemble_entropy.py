#!/usr/bin/env python
"""
Compute the same metrics as compare_ensemble_entropy.py from a compact NPZ:
expects ``R`` with shape (n_frames, n_atoms, 3) and ``z`` atomic numbers (n_atoms,).

Avoids writing a multi-frame XYZ. Reference topology for the Z-matrix is the first frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms as ASEAtoms

from mmml.generate.sample.compare_ensemble_entropy import (
    DEFAULT_L_MAX,
    DEFAULT_N_MAX,
    DEFAULT_R_CUT,
    DEFAULT_SIGMA,
    DEFAULT_SPECIES,
    _make_soap,
    ase_to_chemcoord,
    com_center_positions,
    construction_table_from_zmat,
    process_ensemble_frames,
)


def npz_to_frames(path: Path, *, key_r: str, key_z: str) -> list[ASEAtoms]:
    data = np.load(path, allow_pickle=True)
    if key_r not in data or key_z not in data:
        raise KeyError(f"NPZ must contain {key_r!r} and {key_z!r}; got {list(data.keys())}")
    R = np.asarray(data[key_r], dtype=np.float64)
    z = np.asarray(data[key_z])
    if R.ndim != 3:
        raise ValueError(f"{key_r} must have shape (n_frames, n_atoms, 3), got {R.shape}")
    z = z.reshape(-1).astype(int)
    if R.shape[1] != len(z):
        raise ValueError(
            f"{key_r} has n_atoms={R.shape[1]} but len({key_z})={len(z)}"
        )
    frames: list[ASEAtoms] = []
    for i in range(R.shape[0]):
        frames.append(ASEAtoms(numbers=z, positions=R[i]))
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="NPZ with R and z")
    p.add_argument("--key-r", default="R", help="Array name for positions (default R)")
    p.add_argument("--key-z", default="z", help="Array name for atomic numbers (default z)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kabsch", action="store_true")
    p.add_argument("--hist-bins", type=int, default=32)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--r-cut", type=float, default=DEFAULT_R_CUT)
    p.add_argument("--n-max", type=int, default=DEFAULT_N_MAX)
    p.add_argument("--l-max", type=int, default=DEFAULT_L_MAX)
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write scalar metrics as JSON (excludes soap_array)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write one row CSV compatible with ensemble_entropy_per_file.csv",
    )
    args = p.parse_args()

    path = args.npz.resolve()
    frames = npz_to_frames(path, key_r=args.key_r, key_z=args.key_z)
    if not frames:
        print("No frames in NPZ.", file=sys.stderr)
        raise SystemExit(1)

    ref_atoms = frames[0]
    ref_pos = com_center_positions(ref_atoms)
    ref_cart = ase_to_chemcoord(ref_atoms)
    ref_cart.loc[:, ["x", "y", "z"]] = ref_pos
    ref_zmat = ref_cart.get_zmat()
    c_table = construction_table_from_zmat(ref_zmat)

    try:
        soap_engine = _make_soap(
            DEFAULT_SPECIES,
            args.r_cut,
            args.n_max,
            args.l_max,
            args.sigma,
        )
    except ImportError as e:
        raise SystemExit(
            "dscribe is required for SOAP metrics. Install with: pip install -e '.[quantum]'"
        ) from e

    rng = np.random.default_rng(args.seed)
    r = process_ensemble_frames(
        frames,
        label=str(path),
        c_table=c_table,
        ref_positions_com=ref_pos if args.kabsch else None,
        kabsch=args.kabsch,
        soap_engine=soap_engine,
        max_frames=args.max_frames,
        rng=rng,
        hist_bins=args.hist_bins,
    )
    soap_arr = r.pop("soap_array", None)
    row = {k: v for k, v in r.items() if k != "soap_array"}

    # Human-readable summary
    print(json.dumps(row, indent=2, default=str))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(row, indent=2, default=str) + "\n")
        print(f"Wrote {args.out_json}")

    if args.out_csv:
        fieldnames = [
            "path",
            "n_frames",
            "zmat_failures",
            "gzip_bytes_cart",
            "gzip_bytes_zmat",
            "gzip_bytes_soap",
            "bits_per_frame_cart",
            "bits_per_frame_zmat",
            "bits_per_frame_soap",
            "shannon_pca2d_soap",
            "ipr_pca2d_soap",
            "shannon_cc_dist_1d",
            "soap_dim",
        ]
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerow(row)
        print(f"Wrote {args.out_csv}")

    if soap_arr is not None:
        del soap_arr  # allow GC; large


if __name__ == "__main__":
    main()

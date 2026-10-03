#!/usr/bin/env python
"""Convert SPICE-α HDF5 files to ragged NPZ shards (one per file).

Usage:
  python scripts/spice_alpha/convert_full_ragged.py H5_DIR OUT_DIR [--only STEM ...]
Molecule ids are made unique across files as file_index * 10_000_000 + group index.
"""
import argparse
import json
from pathlib import Path

from mmml.data.spice_alpha_ragged import SUBSETS, convert_hdf5_ragged

ap = argparse.ArgumentParser()
ap.add_argument("h5_dir", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--only", nargs="*", default=None, help="file stems to convert (default: all)")
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
for i, stem in enumerate(sorted(SUBSETS)):
    if args.only and stem not in args.only:
        continue
    stats = convert_hdf5_ragged(args.h5_dir / f"{stem}.hdf5", args.out_dir / f"{stem}.npz",
                                mol_id_base=i * 10_000_000)
    print(stem, json.dumps(stats), flush=True)

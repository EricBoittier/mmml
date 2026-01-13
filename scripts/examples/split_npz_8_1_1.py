#!/usr/bin/env python3
"""Split an NPZ into 8:1:1 train/valid/test NPZs without loading everything at once.

By default writes files next to the input as:
  - train.npz
  - valid.npz
  - test.npz

Features
- Uses streaming zip writing to avoid 3x memory duplication.
- Preserves all keys; splits arrays whose first dimension equals the number of structures.
- Copies non-sample arrays (e.g., metadata, constants) to all outputs.
- Adds split metadata (source_file, split, indices) unless disabled.

Usage
  python scripts/split_npz_8_1_1.py input.npz [--out-dir DIR] [--seed 42]
  python scripts/split_npz_8_1_1.py input.npz --train T.npz --valid V.npz --test X.npz
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, Tuple
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np


def choose_count_key(npz: np.lib.npyio.NpzFile) -> Tuple[str, int]:
    """Select a key to determine number of structures and return (key, count)."""
    prefer = ["E", "R", "Z", "F", "N"]
    for k in prefer:
        if k in npz.files:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                return k, int(arr.shape[0])
    # fallback: pick first array-like with ndim>=1
    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            return k, int(arr.shape[0])
    raise ValueError("Could not determine number of structures from NPZ.")


def write_array_to_zip(zf: ZipFile, key: str, arr: np.ndarray):
    """Write a numpy array as key.npy into open ZipFile."""
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(f"{key}.npy", buf.getvalue())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Input NPZ file")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    ap.add_argument("--train", type=Path, default=None, help="Train NPZ path")
    ap.add_argument("--valid", type=Path, default=None, help="Valid NPZ path")
    ap.add_argument("--test", type=Path, default=None, help="Test NPZ path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--add-indices", action="store_true", help="Store split indices in each NPZ"
    )
    args = ap.parse_args()

    inp = args.input
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    out_dir = args.out_dir if args.out_dir is not None else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.train or (out_dir / "train.npz")
    valid_path = args.valid or (out_dir / "valid.npz")
    test_path = args.test or (out_dir / "test.npz")

    with np.load(inp, allow_pickle=True) as npz:
        count_key, n = choose_count_key(npz)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n)

        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)
        n_test = n - n_train - n_valid
        idx_train = perm[:n_train]
        idx_valid = perm[n_train : n_train + n_valid]
        idx_test = perm[n_train + n_valid :]

        with ZipFile(train_path, "w", compression=ZIP_DEFLATED) as ztr, \
             ZipFile(valid_path, "w", compression=ZIP_DEFLATED) as zv, \
             ZipFile(test_path, "w", compression=ZIP_DEFLATED) as zte:

            for key in npz.files:
                arr = npz[key]
                # Arrays with first-dim == n are split;
                # everything else is copied as-is to all.
                if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n:
                    write_array_to_zip(ztr, key, arr[idx_train])
                    write_array_to_zip(zv, key, arr[idx_valid])
                    write_array_to_zip(zte, key, arr[idx_test])
                else:
                    write_array_to_zip(ztr, key, arr)
                    write_array_to_zip(zv, key, arr)
                    write_array_to_zip(zte, key, arr)

            # Add split metadata if requested
            if args.add_indices:
                meta_train = np.array([
                    {
                        "source_file": str(inp),
                        "split": "train",
                        "indices": idx_train.astype(np.int64),
                        "count_key": count_key,
                        "n_total": n,
                    }
                ], dtype=object)
                meta_valid = np.array([
                    {
                        "source_file": str(inp),
                        "split": "valid",
                        "indices": idx_valid.astype(np.int64),
                        "count_key": count_key,
                        "n_total": n,
                    }
                ], dtype=object)
                meta_test = np.array([
                    {
                        "source_file": str(inp),
                        "split": "test",
                        "indices": idx_test.astype(np.int64),
                        "count_key": count_key,
                        "n_total": n,
                    }
                ], dtype=object)
                write_array_to_zip(ztr, "metadata", meta_train)
                write_array_to_zip(zv, "metadata", meta_valid)
                write_array_to_zip(zte, "metadata", meta_test)

    print(f"Wrote: {train_path}")
    print(f"Wrote: {valid_path}")
    print(f"Wrote: {test_path}")


if __name__ == "__main__":
    main()


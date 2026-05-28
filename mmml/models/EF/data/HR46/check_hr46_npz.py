#!/usr/bin/env python3
"""
Sanity-check HR46 NPZ files produced for EF training.

Checks required keys and prints compact statistics per file:
  electric_field, Z, R, E, F
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_KEYS = ("electric_field", "Z", "R", "E", "F")


def fmt_range(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    return f"[{arr.min():.6g}, {arr.max():.6g}]"


def check_file(path: Path) -> bool:
    ok = True
    data = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        print(f"[FAIL] {path.name}: missing keys {missing}")
        return False

    ef = np.asarray(data["electric_field"])
    z = np.asarray(data["Z"])
    r = np.asarray(data["R"])
    e = np.asarray(data["E"])
    f = np.asarray(data["F"])

    n = r.shape[0] if r.ndim >= 1 else 0
    natoms = r.shape[1] if r.ndim >= 2 else 0

    # Basic shape checks expected by the extractor.
    if ef.shape[0] != n or z.shape[0] != n or e.shape[0] != n or f.shape[0] != n:
        print(
            f"[FAIL] {path.name}: inconsistent first dimension "
            f"(ef={ef.shape}, z={z.shape}, r={r.shape}, e={e.shape}, f={f.shape})"
        )
        ok = False

    if r.shape != f.shape:
        print(f"[FAIL] {path.name}: F shape {f.shape} != R shape {r.shape}")
        ok = False

    zero_forces = float(np.max(np.abs(f))) == 0.0
    status = "OK" if ok else "FAIL"
    print(
        f"[{status}] {path.name} | n={n} natoms={natoms} "
        f"field={fmt_range(ef)} E={fmt_range(e)} "
        f"Z_unique={np.unique(z[z > 0]).tolist()} F_zero={zero_forces}"
    )
    return ok


def iter_npz_files(root: Path) -> Iterable[Path]:
    return sorted(root.glob("*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HR46 EF NPZ files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing generated NPZ files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    files = list(iter_npz_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {input_dir}")

    print(f"Checking {len(files)} files in {input_dir}")
    n_ok = 0
    for path in files:
        if check_file(path):
            n_ok += 1

    print(f"\nSummary: {n_ok}/{len(files)} files passed.")
    if n_ok != len(files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge model and reference dimer scans without duplicating grid points."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = [
    "molecule_a",
    "molecule_b",
    "backend",
    "distance_angstrom",
    "offset_angstrom",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for path in args.input:
        frame = pd.read_csv(path)
        missing = [key for key in KEYS if key not in frame]
        if missing:
            raise ValueError(f"{path}: missing merge keys {missing}")
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=KEYS, keep="last")
    merged = merged.sort_values(KEYS, kind="stable").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    counts = merged.groupby("backend", sort=True).size()
    print(f"Wrote {len(merged)} rows to {args.output}")
    for backend, count in counts.items():
        pairs = merged.loc[merged.backend == backend, ["molecule_a", "molecule_b"]]
        print(f"  {backend}: {count} points, {len(pairs.drop_duplicates())} pairs")


if __name__ == "__main__":
    main()

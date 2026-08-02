#!/usr/bin/env python3
"""Write a dimer-only NPZ slice for evaluation (N=9 frames)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "examples" / "m" / "nh3_ch3cl_filtered.npz"
DEFAULT_OUT = REPO / "artifacts" / "nh3_ch3cl" / "dimer_only.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_IN)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=9, help="Keep frames with this atom count")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    if "N" not in data.files:
        raise SystemExit(f"{args.data} has no N array")
    mask = np.asarray(data["N"]) == int(args.n)
    if not np.any(mask):
        raise SystemExit(f"No frames with N={args.n} in {args.data}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(data[k])[mask] for k in data.files})
    print(f"Wrote {out}  ({int(mask.sum())} frames with N={args.n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

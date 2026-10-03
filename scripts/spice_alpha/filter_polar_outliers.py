#!/usr/bin/env python
"""Copy an efield splits directory, dropping frames with corrupt polarizability labels.

A handful of SPICE-alpha frames carry |alpha_ij| ~ 6e11 Bohr^3 (typical max
element ~70, 99th pct ~145). One such valid frame dominates valid_loss, so
every run reports best_epoch=1 and params-best is the untrained epoch-1 model.

Usage:
  python scripts/spice_alpha/filter_polar_outliers.py SRC_SPLITS DST_SPLITS [--max-abs 1000]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def filter_npz(src: Path, dst: Path, max_abs: float) -> dict:
    data = np.load(src, allow_pickle=False)
    polar = data["polar"]
    n = polar.shape[0]
    bad = np.abs(polar).reshape(n, -1).max(axis=1) > max_abs
    bad |= ~np.isfinite(polar).reshape(n, -1).all(axis=1)
    keep = ~bad
    out = {}
    for key in data.files:
        arr = data[key]
        out[key] = arr[keep] if arr.ndim > 0 and arr.shape[0] == n else arr
    np.savez(dst, **out)
    return {"file": src.name, "n_in": int(n), "n_dropped": int(bad.sum()),
            "dropped_idx": np.flatnonzero(bad).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-abs", type=float, default=1000.0,
                    help="drop frames with any |polar_ij| above this (Bohr^3)")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    report = {"src": str(args.src), "max_abs_bohr3": args.max_abs, "files": []}
    for src in sorted(args.src.glob("*.npz")):
        info = filter_npz(src, args.dst / src.name, args.max_abs)
        report["files"].append(info)
        print(f"{src.name}: kept {info['n_in'] - info['n_dropped']}/{info['n_in']}"
              f" (dropped idx {info['dropped_idx']})")
    for extra in args.src.glob("*.json"):
        shutil.copy2(extra, args.dst / extra.name)
    (args.dst / "polar_filter_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

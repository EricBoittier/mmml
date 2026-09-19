#!/usr/bin/env python3
"""Gate an efield-train NPZ (Ef=0, polar Bohr³). No downloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mmml.data.spice_alpha import check_efield_train_npz, max_atomic_number


def _summarize(path: Path) -> None:
    raw = np.load(path, allow_pickle=True)
    n = int(np.asarray(raw["E"]).reshape(-1).shape[0])
    zmax = max_atomic_number({k: raw[k] for k in raw.files})
    polar = np.asarray(raw["polar"])
    finite = int(np.isfinite(polar).all(axis=(-2, -1)).sum())
    units = {}
    if "_mmml_units" in raw.files:
        units = json.loads(str(np.asarray(raw["_mmml_units"]).reshape(-1)[0]))
    print(
        f"{path}: n={n} pad={raw['R'].shape[1]} Zmax={zmax} "
        f"polar_finite={finite}/{n} E[0]={float(np.asarray(raw['E']).reshape(-1)[0]):.4g} "
        f"units={units.get('E')}/{units.get('polar')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = 0
    for path in args.npz:
        problems = check_efield_train_npz(path)
        if problems:
            failed = 1
            for line in problems:
                print(line, file=sys.stderr)
            continue
        _summarize(path)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())

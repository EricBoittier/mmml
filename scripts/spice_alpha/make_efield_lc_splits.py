#!/usr/bin/env python3
"""Build efield learning-curve split dirs (same valid, smaller train). No downloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmml.data.efield_lc import DEFAULT_LC_FRACS, parse_fracs, prepare_lc_grid


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "src_splits",
        type=Path,
        help="Full splits dir with energies_forces_dipoles_{train,valid}.npz",
    )
    p.add_argument("out_root", type=Path, help="Write splits_p01/, splits_p03/, … here")
    p.add_argument(
        "--fracs",
        default=",".join(str(f) for f in DEFAULT_LC_FRACS),
        help="Comma-separated train fractions (default 0.01,0.03,0.10,0.30)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rows = prepare_lc_grid(
        args.src_splits,
        args.out_root,
        parse_fracs(args.fracs),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

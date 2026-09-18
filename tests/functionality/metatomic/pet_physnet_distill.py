#!/usr/bin/env python3
"""Label a tiny acetone pool with PET-MAD (optional live teacher).

Without --checkpoint, only builds geometries (dummy-free, CI-safe).
Does not train PhysNet or run MD.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.cli.misc.pet_physnet_distill import main as distill_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/functionality/metatomic/output/pet_physnet_distill"),
    )
    parser.add_argument("--preset", default="smoke")
    args = parser.parse_args()
    argv = ["--out-dir", str(args.out_dir), "--preset", args.preset, "--seed", "0"]
    if args.checkpoint is None:
        argv.append("--geometries-only")
    else:
        argv.extend(["--checkpoint", str(args.checkpoint)])
    return distill_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

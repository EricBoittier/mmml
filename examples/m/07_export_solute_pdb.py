#!/usr/bin/env python3
"""Export one NH3–CH3Cl dimer frame from the NPZ as a CGenFF PDB for make-box."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from _geometry import DEFAULT_NPZ, write_solute_pdb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--frame", type=int, default=None, help="Absolute NPZ index (N=9).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/nh3_ch3cl/solute_amm1_ch3cl.pdb",
    )
    args = parser.parse_args()

    out = write_solute_pdb(
        args.output,
        args.data,
        index=args.frame,
        seed=int(args.seed),
    )
    atom_lines = [
        ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("ATOM")
    ]
    n_amm1 = sum("AMM1" in ln for ln in atom_lines)
    n_ch3cl = sum("CH3CL" in ln for ln in atom_lines)
    print(f"Wrote {out}")
    print(f"  AMM1 atom records: {n_amm1}  CH3CL atom records: {n_ch3cl}")
    if n_amm1 != 4 or n_ch3cl != 5:
        print("FAIL: expected 4 AMM1 + 5 CH3CL ATOM lines", file=sys.stderr)
        return 1
    print("PASS: solute PDB for make-box")
    return 0


if __name__ == "__main__":
    sys.exit(main())

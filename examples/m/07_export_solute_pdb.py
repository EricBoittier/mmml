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

from _geometry import (  # noqa: E402
    DEFAULT_NPZ,
    find_frame_near_rc,
    find_frame_near_xi,
    write_solute_pdb,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--frame", type=int, default=None, help="Absolute NPZ index (N=9).")
    parser.add_argument(
        "--xi",
        type=float,
        default=None,
        help="Seed near reaction coord xi=r(Cl-C)/r(C-N): pick the N=9 frame "
        "closest to this value (e.g. --xi 1.0 for a transition-state-like start). "
        "Overrides --frame/--seed.",
    )
    parser.add_argument(
        "--rcl",
        type=float,
        default=None,
        help="Seed near a 2D point on the (r_ClC, r_CN) plane: r(Cl-C) target in Å "
        "(use with --rcn). Picks the nearest N=9 frame; e.g. --rcl 3.8 --rcn 1.57 "
        "seeds the product basin (broken C-Cl). Overrides --xi/--frame/--seed.",
    )
    parser.add_argument(
        "--rcn",
        type=float,
        default=None,
        help="r(C-N) target in Å for 2D (r_ClC, r_CN) seeding (use with --rcl).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="Keep raw NPZ coords (default centers mass-weighted COM at origin).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/nh3_ch3cl/solute_amm1_ch3cl.pdb",
    )
    args = parser.parse_args()

    frame = args.frame
    if (args.rcl is None) != (args.rcn is None):
        parser.error("--rcl and --rcn must be given together (2D seeding)")
    if args.rcl is not None:
        frame, xi, r_clc, r_cn = find_frame_near_rc(args.rcl, args.rcn, args.data)
        print(
            f"(r_ClC, r_CN) target ({args.rcl:.2f}, {args.rcn:.2f}) -> frame {frame}: "
            f"r(Cl-C)={r_clc:.2f} Å r(C-N)={r_cn:.2f} Å (xi={xi:.3f})"
        )
    elif args.xi is not None:
        frame, xi, r_clc, r_cn = find_frame_near_xi(args.xi, args.data)
        print(
            f"xi target {args.xi:.3f} -> frame {frame}: "
            f"xi={xi:.3f} r(Cl-C)={r_clc:.2f} Å r(C-N)={r_cn:.2f} Å"
        )

    out = write_solute_pdb(
        args.output,
        args.data,
        index=frame,
        seed=int(args.seed),
        center=bool(args.center),
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

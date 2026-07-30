#!/usr/bin/env python3
"""Print monomer counts for the DCM / ACO density sweep.

Uses mmml's own sizing helper so the table can never drift from what
``md-system`` / ``liquid-box`` would actually build.

    uv run python workflows/liquid_density_sweep/scripts/print_density_table.py
    uv run python .../print_density_table.py --bash   # N_MONOMERS for common.sh
"""

from __future__ import annotations

import argparse

SOLVENTS = ("DCM", "ACO")
FRACTIONS = (0.50, 0.75, 1.00)
BOX_SIDES = (28.0, 32.0, 36.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bash",
        action="store_true",
        help="emit the N_MONOMERS associative-array body for common.sh",
    )
    args = parser.parse_args()

    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        SOLVENT_BULK_PROPS,
        n_molecules_for_target_density_in_fixed_box,
    )

    def count(solvent: str, side: float, frac: float) -> int:
        rho = float(SOLVENT_BULK_PROPS[solvent]["rho_g_cm3"]) * frac
        sized = n_molecules_for_target_density_in_fixed_box(
            composition={solvent: 1},
            box_side_A=side,
            target_density_g_cm3=rho,
        )
        return int(sized[solvent])

    if args.bash:
        for side in BOX_SIDES:
            for solvent in SOLVENTS:
                cells = " ".join(
                    f"[{solvent}_{side:g}_{frac:.2f}]={count(solvent, side, frac)}"
                    for frac in FRACTIONS
                )
                print(f"  {cells}")
        return 0

    print(f"{'L (Å)':>6} {'solvent':>8} {'ρ_bulk':>7} "
          + " ".join(f"{f'{frac:.2f}':>18}" for frac in FRACTIONS))
    for side in BOX_SIDES:
        for solvent in SOLVENTS:
            rho = float(SOLVENT_BULK_PROPS[solvent]["rho_g_cm3"])
            atoms_per = {"DCM": 5, "ACO": 10}[solvent]
            cells = []
            for frac in FRACTIONS:
                n = count(solvent, side, frac)
                cells.append(f"{n:>6} ({n * atoms_per:>5} atoms)")
            print(f"{side:>6g} {solvent:>8} {rho:>7.3f} " + " ".join(cells))
    print("\nAtom counts are the ML-region size: compare against max_Nml from "
          "`mmml doctor`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

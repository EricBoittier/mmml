"""CLI adapter for the canonical reproducible 1D dimer scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.dimer_scan import DimerScanConfig, run_dimer_scan


def _distance_grid(value: str) -> tuple[float, ...]:
    try:
        start_text, stop_text, step_text = value.split(":")
        start, stop, step = float(start_text), float(stop_text), float(step_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected START:STOP:STEP") from exc
    if step <= 0.0 or stop < start:
        raise argparse.ArgumentTypeError("distance grid requires STEP > 0 and STOP >= START")
    count = int(round((stop - start) / step))
    values = tuple(round(start + index * step, 12) for index in range(count + 1))
    if not values or abs(values[-1] - stop) > 1.0e-9:
        raise argparse.ArgumentTypeError("STEP must land exactly on STOP")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml dimer-scan",
        description="Run a reproducible rigid 1D dimer energy/force scan.",
    )
    parser.add_argument(
        "residues",
        nargs="+",
        metavar="RESIDUE",
        help="One residue for a homodimer or two for a heterodimer",
    )
    parser.add_argument(
        "--calculator",
        required=True,
        choices=("physnet", "xtb"),
        help="Explicit ASE calculator type",
    )
    parser.add_argument("--checkpoint", type=Path, help="PhysNet checkpoint path")
    parser.add_argument(
        "--distance",
        type=_distance_grid,
        default=_distance_grid("2.5:6.0:0.1"),
        metavar="START:STOP:STEP",
        help="Inclusive distance grid in angstrom (default: 2.5:6.0:0.1)",
    )
    parser.add_argument(
        "--energy-definition",
        choices=("interaction", "total"),
        default="interaction",
    )
    parser.add_argument("--charge", type=float)
    parser.add_argument("--spin", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.residues) not in (1, 2):
        build_parser().error("provide one residue for a homodimer or two for a heterodimer")
    residues = (
        (args.residues[0], args.residues[0])
        if len(args.residues) == 1
        else tuple(args.residues)
    )
    config = DimerScanConfig(
        residues=residues,
        calculator=args.calculator,
        checkpoint=args.checkpoint,
        distances_angstrom=args.distance,
        energy_definition=args.energy_definition,
        failure_policy="allow_partial" if args.allow_partial else "fail",
        charge=args.charge,
        spin=args.spin,
        seed=args.seed,
    )
    result = run_dimer_scan(config)
    paths = result.write(args.output, overwrite=args.overwrite)
    print(f"Wrote {len(result.records)} scan points to {paths['manifest']}")
    if result.has_failures and not args.allow_partial:
        print("One or more requested scan points failed; see data.csv for diagnostics.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

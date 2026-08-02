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
        "--config",
        type=Path,
        help="YAML/JSON DimerScanConfig; command-line output/failure flags remain available",
    )
    parser.add_argument(
        "residues",
        nargs="*",
        metavar="RESIDUE",
        help=(
            "One residue for a homodimer or two for a heterodimer. "
            "Accepts campaign labels (DCM, ACE, BENZ, TIP3, MEOH) or any "
            "CGenFF RESI name (e.g. ACO, CYBZ); non-campaign pairs use a "
            "generic centroid–centroid orientation."
        ),
    )
    parser.add_argument(
        "--calculator",
        required=False,
        choices=(
            "physnet",
            "spookynet",
            "mbd",
            "multipoles",
            "efield",
            "kernnn",
            "xtb",
            "dftb3-d4",
            "pyscf",
        ),
        help="Explicit ASE calculator type",
    )
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint/parameter path")
    parser.add_argument("--calculator-config", type=Path, help="Calculator model config JSON")
    parser.add_argument("--method", help="Calculator method (for example pyscf: dft or hf)")
    parser.add_argument("--basis", help="PySCF basis (default: def2-svp)")
    parser.add_argument("--xc", help="PySCF DFT functional (default: pbe0)")
    parser.add_argument(
        "--electric-field",
        nargs=3,
        type=float,
        metavar=("EX", "EY", "EZ"),
        help="EField vector in atomic units",
    )
    parser.add_argument("--slako-dir", type=Path, help="DFTB+ 3ob-3-1 directory")
    parser.add_argument("--calculator-workdir", type=Path, help="External calculator scratch directory")
    parser.add_argument("--calculator-executable", help="External calculator executable")
    parser.add_argument(
        "--multipole-force-step",
        type=float,
        default=1.0e-4,
        metavar="ANGSTROM",
        help="Central finite-difference step for learned-multipole forces",
    )
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
    parser.add_argument(
        "--spin",
        type=float,
        help="Spin multiplicity for calculators that require it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is not None:
        import json
        import os

        path = args.config.expanduser().resolve()
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            build_parser().error("--config document must contain a mapping")
        for key in ("checkpoint", "calculator_config", "interaction_policy", "slako_dir", "workdir"):
            value = data.get(key)
            if value:
                expanded = os.path.expandvars(str(value))
                if "$" in expanded:
                    build_parser().error(f"unresolved environment variable in config field {key}")
                candidate = Path(expanded).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                data[key] = str(candidate)
        config = DimerScanConfig.from_dict(data)
        if args.allow_partial and config.failure_policy != "allow_partial":
            from dataclasses import replace

            config = replace(config, failure_policy="allow_partial")
    else:
        if args.calculator is None:
            build_parser().error("--calculator is required unless --config is provided")
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
            method=args.method,
            basis=args.basis,
            xc=args.xc,
            calculator_config=args.calculator_config,
            electric_field_au=(
                tuple(args.electric_field) if args.electric_field is not None else None
            ),
            slako_dir=args.slako_dir,
            workdir=args.calculator_workdir,
            executable=args.calculator_executable,
            multipole_force_step_angstrom=args.multipole_force_step,
            seed=args.seed,
        )
    result = run_dimer_scan(config)
    paths = result.write(args.output, overwrite=args.overwrite)
    print(f"Wrote {len(result.records)} scan points to {paths['manifest']}")
    if result.has_failures and config.failure_policy != "allow_partial":
        print("One or more requested scan points failed; see data.csv for diagnostics.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

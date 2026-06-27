"""``mmml liquid-box`` — MM-only periodic liquid box build and certification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml liquid-box",
        description=(
            "Build and certify a periodic liquid box under CHARMM MM only "
            "(Packmol → MC density → SD/ABNR → optional lattice/NPT → geometry gate). "
            "Writes model.psf/crd, box.json, and REPORT.md. "
            "See docs/liquid-box-workflow.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--composition",
        type=str,
        required=True,
        help="Composition for Packmol cube build, e.g. DCM:206 or MEOH:100.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory for certified box artifacts.",
    )
    parser.add_argument(
        "--profile",
        choices=["standard", "dense", "conservative"],
        default="dense",
        help=(
            "standard: MC + CHARMM pre-minimize only; "
            "dense: liquid-prep preventive stack; "
            "conservative: dense with looser initial density."
        ),
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=4.0,
        help="Monomer template spacing for cluster build (Å).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for Packmol / MC placement.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature for mini box equilibration (K).",
    )
    parser.add_argument(
        "--dt-fs",
        type=float,
        default=0.25,
        help="Timestep for mini box equilibration (fs).",
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import add_dynamics_stability_args

    add_dynamics_stability_args(parser)
    parser.add_argument(
        "--charmm-mm-pretreat-echeck",
        type=float,
        default=None,
        metavar="KCAL",
        help=(
            "Enable ECHECK during mini box CPT equil (kcal/mol). "
            "Default: disabled (NPT prep often exceeds ML echeck floors)."
        ),
    )
    parser.add_argument(
        "--charmm-sd-steps",
        type=int,
        default=50,
        help="CHARMM SD steps during MM pre-minimize (dense profile bumps to ≥1000).",
    )
    parser.add_argument(
        "--charmm-abnr-steps",
        type=int,
        default=100,
        help="CHARMM ABNR steps during MM pre-minimize (dense profile bumps to ≥1000).",
    )
    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol distance tolerance (Å).",
    )
    parser.add_argument(
        "--reuse-packmol-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse on-disk Packmol cache when composition matches.",
    )
    parser.add_argument(
        "--rebuild-packmol",
        action="store_true",
        help="Ignore Packmol cache and rebuild placement.",
    )
    parser.add_argument(
        "--packmol-cache-dir",
        type=Path,
        default=None,
        help="Packmol cache root (default: output-dir/.packmol_cache).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import add_box_sizing_args

    add_box_sizing_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(parsed_argv)
    args.setup = "pbc_nvt"
    args.save = True
    args.tag = None

    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        maybe_rerun_mmml_under_mpirun,
        prepare_serial_charmm_mpi_env,
    )

    prepare_serial_charmm_mpi_env()
    rerun_code = maybe_rerun_mmml_under_mpirun(parsed_argv, subcommand="liquid-box")
    if rerun_code is not None:
        return int(rerun_code)

    try:
        from mmml.interfaces.pycharmmInterface.mlpot.liquid_box_build import (
            run_liquid_box_build,
        )

        result = run_liquid_box_build(args)
    except ModuleNotFoundError as exc:
        if "pycharmm" in str(exc).lower() or "charmm" in str(exc).lower():
            print(
                "Error: liquid-box requires PyCHARMM/CHARMM and Packmol.",
                file=sys.stderr,
            )
            return 1
        raise
    except Exception as exc:
        print(f"liquid-box failed: {exc}", file=sys.stderr)
        if not args.quiet:
            raise
        return 1

    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Run supplementary QC cross-checks against a reference (PySCF, ORCA QM, xTB, Molpro, ML).

Examples
--------
From YAML config:

    mmml cross-check -c cross_check.example.yaml

CLI flags (minimal smoke):

    mmml cross-check -i sampled.npz --reference-npz ref.npz \\
        --backend ml --checkpoint epoch.pkl -o validation/

    mmml cross-check -i water.xyz --reference pyscf --backend xtb --max-frames 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mmml.interfaces.qc_backends.factory import backend_from_dict
from mmml.interfaces.qc_backends.protocol import BackendSpec
from mmml.interfaces.qc_backends.runner import CrossCheckConfig, CrossCheckRunner


def _parse_backend_flags(argv: list[str]) -> tuple[list[BackendSpec], argparse.Namespace, list[str]]:
    """Parse repeated --backend/--checkpoint style flags before main argparse."""
    backends: list[BackendSpec] = []
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--backend" and i + 1 < len(argv):
            name = argv[i + 1]
            opts: dict = {"name": name}
            i += 2
            while i < len(argv) and not argv[i].startswith("--"):
                token = argv[i]
                if "=" in token:
                    key, val = token.split("=", 1)
                    opts[key.lstrip("-").replace("-", "_")] = val
                i += 1
            backends.append(backend_from_dict(opts))
            continue
        remaining.append(argv[i])
        i += 1
    return backends, argparse.Namespace(), remaining


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supplementary QC cross-check (PySCF, ORCA QM, xTB, Molpro, ML).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="YAML config file (see examples/cross_check/cross_check.example.yaml)",
    )
    parser.add_argument(
        "-i",
        "--input",
        "--structures",
        dest="structures",
        type=Path,
        help="Input NPZ or XYZ with structures to evaluate",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("cross_check_out"),
        help="Output directory (default: cross_check_out)",
    )
    parser.add_argument(
        "--reference-npz",
        type=Path,
        help="Use existing reference NPZ instead of running a reference backend",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference backend name when --reference-npz is not set (default: pyscf)",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backend_names",
        help="Backend to evaluate (repeatable): pyscf, ml, orca, xtb, molpro",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="ML checkpoint (shorthand for --backend ml)",
    )
    parser.add_argument(
        "--functional",
        "--xc",
        dest="functional",
        type=str,
        help="XC functional for pyscf/orca reference backend",
    )
    parser.add_argument(
        "--basis",
        type=str,
        help="Basis set for pyscf/orca/molpro backends",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of structures to evaluate",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (default: 1)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total charge (default: 0)",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2*spin for PySCF (default: 0)",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity for ORCA/Molpro/xTB (default: spin+1)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib comparison plots",
    )
    parser.add_argument(
        "--no-save-backend-npz",
        action="store_true",
        help="Do not write per-backend NPZ files",
    )
    parser.add_argument(
        "--orca-template",
        type=Path,
        help="Custom ORCA input template with {xyz},{method},{basis},{charge},{mult} placeholders",
    )
    parser.add_argument(
        "--molpro-template",
        type=Path,
        help="Custom Molpro input template with {geometry},{basis},{method},{charge},{mult} placeholders",
    )
    return parser


def _config_from_args(args: argparse.Namespace, cli_backends: list[BackendSpec]) -> CrossCheckConfig:
    if args.config is not None:
        return CrossCheckConfig.from_yaml(args.config)

    if args.structures is None:
        raise ValueError("Provide --config or --input/--structures.")

    backends = list(cli_backends)
    if args.checkpoint is not None:
        backends.append(
            BackendSpec(
                name="ml",
                options={"checkpoint": str(args.checkpoint)},
            )
        )
    if args.backend_names:
        for name in args.backend_names:
            opts: dict = {"name": name}
            if args.functional:
                opts["functional"] = args.functional
                opts["xc"] = args.functional
            if args.basis:
                opts["basis"] = args.basis
            if args.orca_template and name in {"orca", "orca_qm"}:
                opts["template"] = str(args.orca_template)
            if args.molpro_template and name == "molpro":
                opts["template"] = str(args.molpro_template)
            backends.append(backend_from_dict(opts))

    mult = args.multiplicity if args.multiplicity is not None else args.spin + 1
    mapping = {
        "structures": str(args.structures),
        "output": str(args.output_dir),
        "reference_npz": str(args.reference_npz) if args.reference_npz else None,
        "reference": args.reference or ("pyscf" if args.reference_npz is None else None),
        "backends": [{"name": b.name, **dict(b.options)} for b in backends],
        "max_frames": args.max_frames,
        "stride": args.stride,
        "charge": args.charge,
        "spin": args.spin,
        "multiplicity": mult,
        "no_plots": args.no_plots,
        "save_backend_npz": not args.no_save_backend_npz,
    }
    return CrossCheckConfig.from_mapping(mapping)


def main() -> int:
    cli_backends, _, remaining = _parse_backend_flags(sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(remaining)

    try:
        config = _config_from_args(args, cli_backends)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not config.backends and config.reference_npz is None:
        print(
            "Error: specify at least one --backend or --checkpoint to compare against the reference.",
            file=sys.stderr,
        )
        return 1

    if not config.structures.is_file():
        print(f"Error: structures file not found: {config.structures}", file=sys.stderr)
        return 1

    if config.reference_npz is not None and not config.reference_npz.is_file():
        print(f"Error: reference NPZ not found: {config.reference_npz}", file=sys.stderr)
        return 1

    try:
        summary = CrossCheckRunner(config).run()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for label, row in summary.get("backends", {}).items():
        if "error" in row:
            continue
        e = row["metrics"]["energy"]
        print(f"\n{label}: energy MAE={e['mae']:.6g} RMSE={e['rmse']:.6g} R²={e['r2']:.4f}")
        if "forces" in row["metrics"]:
            f = row["metrics"]["forces"]
            print(f"  forces MAE={f['mae']:.6g} RMSE={f['rmse']:.6g} R²={f['r2']:.4f}")

    if summary.get("method_warnings"):
        print("\nMethod mismatch warnings (sanity screen, not pass/fail):")
        for msg in summary["method_warnings"]:
            print(f"  - {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CLI for MBAR post-processing of lambda-dynamics runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmml.cli.run.lambda_dynamics import (
    MbarConfig,
    merge_mbar_into_summary,
    parse_couple_residue_numbers,
    print_lambda_summary,
    run_mbar_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MBAR analysis for a completed lambda-dynamics run. "
            "Reads lambda_ti_snapshots.npz from --run-dir and updates lambda_ti_summary.json."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output directory from mmml md-system --setup lambda_ti or scripts/meoh_dimer_lambda_ti.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint (default: read from summary args if present).",
    )
    parser.add_argument("--temperature-K", type=float, default=None, help="kT for reduced potentials (default: from summary or 100 K).")
    parser.add_argument(
        "--couple-residues",
        type=str,
        default=None,
        help="Override 1-based coupled residue numbers (default: read from snapshots).",
    )
    parser.add_argument("--ml-cutoff", type=float, default=1.0)
    parser.add_argument("--mm-switch-on", type=float, default=5.0)
    parser.add_argument("--mm-cutoff", type=float, default=5.0)
    parser.add_argument("--mbar-verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip updating diagnostic plots.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _temperature_from_summary(run_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    summary_path = run_dir / "lambda_ti_summary.json"
    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        args = data.get("args") or {}
        if "temperature_K" in args:
            return float(args["temperature_K"])
    return 100.0


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    temp_K = _temperature_from_summary(run_dir, args.temperature_K)

    couple_override = None
    if args.couple_residues:
        summary_path = run_dir / "lambda_ti_summary.json"
        n_mol = 2
        if summary_path.is_file():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            n_mol = int((data.get("system") or {}).get("n_molecules", n_mol))
        couple_override = [
            i + 1 for i in parse_couple_residue_numbers(args.couple_residues, n_mol)
        ]

    cfg = MbarConfig(
        run_dir=run_dir,
        checkpoint=args.checkpoint,
        temperature_K=temp_K,
        couple_residue_numbers=couple_override,
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        mbar_verbose=args.mbar_verbose,
        write_plots=not args.no_plots,
    )
    mbar_block = run_mbar_analysis(cfg)
    summary_path = merge_mbar_into_summary(run_dir, mbar_block, write_plots=not args.no_plots)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print_lambda_summary(summary)
    print(f"Updated {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

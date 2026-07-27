#!/usr/bin/env python3
"""CLI for MBAR post-processing of umbrella sampling runs.

Usage:
    mmml umbrella-mbar --run-dir out/umbrella [--checkpoint PATH] [--temperature-K 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmml.umbrella.config import UmbrellaMbarConfig
from mmml.umbrella.io import SUMMARY_JSON
from mmml.umbrella.mbar import run_umbrella_mbar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml umbrella-mbar",
        description=(
            "MBAR analysis for a completed umbrella-sample run. "
            "Reads umbrella_snapshots.npz from --run-dir and updates umbrella_summary.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output directory from mmml umbrella-sample",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint for U_ML re-evaluation (default: from snapshots/summary).",
    )
    parser.add_argument(
        "--temperature-K",
        type=float,
        default=None,
        help="kT for reduced potentials (default: from snapshots/summary).",
    )
    parser.add_argument(
        "--mbar-verbose",
        action="store_true",
        help="Verbose pymbar output",
    )
    parser.add_argument(
        "--ml-batch-size",
        type=int,
        default=32,
        help="Reserved for batched U_ML re-eval (default: 32)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = UmbrellaMbarConfig(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        temperature_K=args.temperature_K,
        mbar_verbose=args.mbar_verbose,
        ml_batch_size=args.ml_batch_size,
    )
    result = run_umbrella_mbar(cfg)
    if "error" in result:
        print(f"MBAR error: {result['error']}")
        return 1
    print("Umbrella MBAR done:")
    xi0 = result.get("xi0") or []
    pmf = result.get("pmf_rel_kcal_mol") or []
    for x, f in zip(xi0, pmf):
        print(f"  ξ₀={x:.4f} Å   PMF={f:.4f} kcal/mol")
    summary_path = Path(args.run_dir).expanduser().resolve() / SUMMARY_JSON
    print(f"  summary: {summary_path}")
    # Ensure JSON-serializable echo for scripting
    print(json.dumps({"N_k_effective": result.get("N_k_effective")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

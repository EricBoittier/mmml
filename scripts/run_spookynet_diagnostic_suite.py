#!/usr/bin/env python3
"""Run SpookyNet diagnostic suite across multiple checkpoints to generate time series comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.debug_spookynet_checkpoint import analyze_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Multi-checkpoint SpookyNet Diagnostic Runner")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint JSON files or directories (e.g. epoch 2, epoch 8, etc.)",
    )
    parser.add_argument("--pairs", nargs="+", default=["TIP3+TIP3", "DCM+DCM"], help="Dimer pairs to analyze")
    parser.add_argument("--output-dir", default="./spookynet_diagnostics_suite", help="Master output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================================")
    print(f" Multi-Checkpoint SpookyNet Suite Runner")
    print(f" Checkpoints: {len(args.checkpoints)} | Pairs: {args.pairs}")
    print(f" Output Directory: {output_dir.resolve()}")
    print(f"==================================================================")

    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        for pair in args.pairs:
            try:
                analyze_checkpoint(
                    checkpoint_path=ckpt_path,
                    pair_name=pair,
                    output_dir=output_dir / ckpt_path.name,
                )
            except Exception as exc:
                print(f"[!] Error analyzing checkpoint {ckpt_path} on pair {pair}: {exc}")

    print("\n[+] All diagnostic scans completed.")


if __name__ == "__main__":
    main()

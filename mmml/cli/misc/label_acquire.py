"""CLI: ``mmml label-acquire`` — stages of the structure-selection workflow.

``build_parser`` is argparse-only so ``mmml label-acquire --help`` and docs
generation do not import JAX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STAGES = (
    "prepare-pool",
    "fingerprint-models",
    "extract",
    "fit-pca",
    "select",
    "label",
    "train-eval",
    "report",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmml label-acquire",
        description=(
            "Compare structure-selection methods for acquiring expensive "
            "reference labels.  Teacher potentials are cheap surrogates, not "
            "ground truth.  Use the Snakemake workflow for the full DAG."
        ),
    )
    p.add_argument("--config", "-c", type=Path, required=True, help="YAML config")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Override output_root from the config",
    )
    p.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=STAGES,
        help="Pipeline stage (default: all)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from mmml.acquisition.pipeline import (
        load_config,
        run_all,
        stage_extract,
        stage_fingerprint_models,
        stage_fit_pca,
        stage_label,
        stage_prepare_pool,
        stage_report,
        stage_select,
        stage_train_eval,
        workdir,
    )

    cfg = load_config(args.config)
    out = workdir(cfg, args.output)
    dispatch = {
        "prepare-pool": stage_prepare_pool,
        "fingerprint-models": stage_fingerprint_models,
        "extract": stage_extract,
        "fit-pca": stage_fit_pca,
        "select": stage_select,
        "label": stage_label,
        "train-eval": stage_train_eval,
        "report": stage_report,
        "all": run_all,
    }
    path = dispatch[args.stage](cfg, out)
    print(path)
    return 0

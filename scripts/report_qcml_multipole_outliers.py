#!/usr/bin/env python3
"""Report QCML multipole target scales and outlier thresholds from Orbax shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmml.data.orbax_shards import partition_shards

try:
    from scripts.train_qcml_multipoles import compute_target_statistics
except ModuleNotFoundError:
    from train_qcml_multipoles import compute_target_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--validation-shards", type=int, default=2)
    parser.add_argument("--test-shards", type=int, default=2)
    parser.add_argument(
        "--target-scale-mode",
        choices=("rms", "q95", "q99"),
        default="q95",
    )
    parser.add_argument("--outlier-quantile", type=float, default=0.99)
    parser.add_argument(
        "--outlier-degree-mode",
        choices=("component", "norm"),
        default="component",
    )
    parser.add_argument("--target-scale-floor", type=float, default=1e-6)
    args = parser.parse_args()

    if (args.cache / "manifest.json").exists():
        shard_split = partition_shards(
            args.cache,
            validation_shards=args.validation_shards,
            test_shards=args.test_shards,
        )
        training_paths = shard_split["train"]
    else:
        training_paths = [args.cache]

    report = compute_target_statistics(
        training_paths,
        max_structures=args.max_structures,
        max_atoms=args.max_atoms,
        scale_mode=args.target_scale_mode,
        outlier_quantile=args.outlier_quantile,
        outlier_degree_mode=args.outlier_degree_mode,
        floor=args.target_scale_floor,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

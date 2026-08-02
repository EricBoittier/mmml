#!/usr/bin/env python3
"""Aggregate one or more evaluate_so3lr_spooky_extxyz.py --output JSONs into one table.

evaluate_so3lr_spooky_extxyz.py --output <path.json> already writes a
{dataset_name: {energy_mae, energy_rmse, forces_mae, forces_rmse,
dipole_mae, dipole_rmse, charge_mae, charge_rmse}} mapping covering every
.extxyz file it evaluated. This script merges one or more such JSONs (e.g.
one per checkpoint, or one per --extxyz subset if you ran it that way) into
a single long-format table, tagged by source file, and writes CSV + prints
a markdown table.

Usage:
    python scripts/aggregate_so3lr_eval_results.py \\
        spooky_so3lr_muon3_epoch0010.json \\
        --out eval_summary.csv

    # Compare several checkpoints (source filename becomes the "run" column)
    python scripts/aggregate_so3lr_eval_results.py \\
        eval_out/muon3_epoch0010.json eval_out/muon3_epoch0020.json \\
        --out eval_out/compare.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

METRIC_COLUMNS = [
    "energy_mae",
    "energy_rmse",
    "forces_mae",
    "forces_rmse",
    "dipole_mae",
    "dipole_rmse",
    "charge_mae",
    "charge_rmse",
]


def _run_name(path: Path) -> str:
    return path.stem


def load_rows(json_paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in json_paths:
        with path.open() as f:
            results = json.load(f)
        if not isinstance(results, dict):
            raise ValueError(f"{path} is not a {{dataset: metrics}} JSON (got {type(results)})")
        run = _run_name(path)
        for dataset, metrics in results.items():
            row = {"run": run, "dataset": dataset}
            for col in METRIC_COLUMNS:
                row[col] = metrics.get(col, float("nan"))
            rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "dataset", *METRIC_COLUMNS]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict]) -> None:
    multi_run = len({r["run"] for r in rows}) > 1
    cols = (["run"] if multi_run else []) + ["dataset", *METRIC_COLUMNS]
    widths = {c: max(len(c), *(len(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}

    def fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-|-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in sorted(rows, key=lambda r: (r["run"], r["dataset"])):
        print(" | ".join(fmt(r[c]).ljust(widths[c]) for c in cols))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_paths", nargs="+", type=Path, help="One or more --output JSONs from evaluate_so3lr_spooky_extxyz.py")
    p.add_argument("--out", type=Path, default=None, help="Write the merged table to this CSV path")
    args = p.parse_args()

    missing = [p for p in args.json_paths if not p.is_file()]
    if missing:
        print(f"Missing input JSON(s): {missing}", file=sys.stderr)
        return 1

    rows = load_rows(args.json_paths)
    print_markdown(rows)
    if args.out is not None:
        write_csv(rows, args.out)
        print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

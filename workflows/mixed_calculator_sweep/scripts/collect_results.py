#!/usr/bin/env python3
"""Aggregate per-(setting, seed) status.json files into a summary table.

Always exits 0 once the summary is written (see
workflows/unified_backend_sweep/scripts/collect_results.py for why: a
non-zero exit here would make Snakemake delete the report that's meant to
surface per-setting failures).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FIELDS = [
    "setting",
    "seed",
    "system",
    "checkpoint",
    "calculator_kwargs",
    "mm_nonbonded_kwargs",
    "completed",
    "atom_count",
    "n_steps",
    "n_frames",
    "energy_initial_ev",
    "energy_final_ev",
    "energy_mean_ev",
    "energy_std_ev",
    "energy_drift_ev",
    "energy_max_abs_deviation_ev",
    "elapsed_seconds",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("statuses", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def flatten(path: Path) -> dict[str, object]:
    status = json.loads(path.read_text(encoding="utf-8"))
    return {field: status.get(field, "") for field in FIELDS}


def main() -> None:
    args = parse_args()
    rows = sorted(
        (flatten(path) for path in args.statuses),
        key=lambda row: (str(row["setting"]), int(row["seed"])),
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Mixed-system / calculator sweep (10000-step NVE)",
        "",
        "| setting | seed | system | status | frames | E0 (eV) | Efinal (eV) | "
        "std (eV) | max|dev| (eV) | elapsed (s) |",
        "|---|---:|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    n_failed = sum(1 for row in rows if not row["completed"])
    for row in rows:
        status_icon = "✅" if row["completed"] else f"❌ {row['error']}"
        lines.append(
            f"| {row['setting']} | {row['seed']} | {row['system']} | {status_icon} | "
            f"{row['n_frames']} | {row['energy_initial_ev']} | {row['energy_final_ev']} | "
            f"{row['energy_std_ev']} | {row['energy_max_abs_deviation_ev']} | "
            f"{row['elapsed_seconds']} |"
        )
    lines.append("")
    lines.append(f"**{len(rows) - n_failed}/{len(rows)} settings completed.**")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if n_failed:
        print(f"warning: {n_failed} setting(s) failed; see {args.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()

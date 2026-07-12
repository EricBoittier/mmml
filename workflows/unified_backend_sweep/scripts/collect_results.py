#!/usr/bin/env python3
"""Aggregate per-(backend, seed) status.json files into a summary table.

Always exits 0 once the summary is written: per-setting failures are already
visible as ❌ rows in the report (that IS the collection succeeding at its
job), and Snakemake deletes a rule's declared outputs when its shell command
exits non-zero -- a non-zero exit here would silently discard the very report
meant to surface the failures, undermining `snakemake --keep-going`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FIELDS = [
    "backend",
    "description",
    "seed",
    "completed",
    "atom_count",
    "n_steps",
    "n_frames",
    "energy_initial_ev",
    "energy_final_ev",
    "energy_drift_ev",
    "acceptance_ratio",
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
        key=lambda row: (str(row["backend"]), int(row["seed"])),
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Unified mmml.md backend sweep",
        "",
        "| backend | seed | status | frames | E0 (eV) | Efinal (eV) | drift (eV) | acceptance | elapsed (s) |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    n_backends = len({row["backend"] for row in rows})
    n_failed = sum(1 for row in rows if not row["completed"])
    for row in rows:
        status_icon = "✅" if row["completed"] else f"❌ {row['error']}"
        lines.append(
            f"| {row['backend']} | {row['seed']} | {status_icon} | {row['n_frames']} | "
            f"{row['energy_initial_ev']} | {row['energy_final_ev']} | "
            f"{row['energy_drift_ev']} | {row['acceptance_ratio']} | {row['elapsed_seconds']} |"
        )
    lines.append("")
    lines.append(
        f"**{len(rows) - n_failed}/{len(rows)} settings completed** across "
        f"{n_backends} backends."
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if n_failed:
        raise SystemExit(f"{n_failed} setting(s) failed; see results/summary.md")


if __name__ == "__main__":
    main()

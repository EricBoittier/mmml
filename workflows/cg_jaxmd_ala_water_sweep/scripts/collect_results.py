#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "mode",
    "description",
    "dt_fs",
    "completed",
    "returncode",
    "elapsed_seconds",
    "final_nve_step",
    "final_total_energy_ev",
    "final_energy_drift_ev",
    "final_temperature_k",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("statuses", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def flatten(path: Path) -> dict[str, object]:
    status = json.loads(path.read_text(encoding="utf-8"))
    nve = status.get("final_nve") or {}
    return {
        "mode": status["mode"],
        "description": status["description"],
        "dt_fs": status["dt_fs"],
        "completed": status["completed"],
        "returncode": status["returncode"],
        "elapsed_seconds": status["elapsed_seconds"],
        "final_nve_step": nve.get("step", ""),
        "final_total_energy_ev": nve.get("energy", ""),
        "final_energy_drift_ev": nve.get("drift", ""),
        "final_temperature_k": nve.get("temperature", ""),
    }


def main() -> None:
    args = parse_args()
    rows = sorted((flatten(path) for path in args.statuses), key=lambda row: (row["mode"], -float(row["dt_fs"])))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# cg_jaxmd ALA + 50-water sweep",
        "",
        "| mode | dt (fs) | completed | final NVE step | drift (eV) | temperature (K) | elapsed (s) |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['dt_fs']} | {row['completed']} | "
            f"{row['final_nve_step']} | {row['final_energy_drift_ev']} | "
            f"{row['final_temperature_k']} | {row['elapsed_seconds']} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


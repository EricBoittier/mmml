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
    "seed",
    "atom_count",
    "completed",
    "returncode",
    "elapsed_seconds",
    "final_nve_step",
    "final_total_energy_ev",
    "final_energy_drift_ev",
    "nve_time_ps",
    "abs_drift_ev_per_ps",
    "abs_drift_ev_per_atom",
    "abs_drift_ev_per_atom_ps",
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
    step = nve.get("step")
    drift = nve.get("drift")
    atom_count = status.get("atom_count")
    nve_time_ps = float(step) * float(status["dt_fs"]) * 0.001 if step else None
    abs_drift = abs(float(drift)) if drift else None
    drift_per_ps = abs_drift / nve_time_ps if abs_drift is not None and nve_time_ps else None
    drift_per_atom = abs_drift / atom_count if abs_drift is not None and atom_count else None
    drift_per_atom_ps = drift_per_atom / nve_time_ps if drift_per_atom is not None and nve_time_ps else None
    return {
        "mode": status["mode"],
        "description": status["description"],
        "dt_fs": status["dt_fs"],
        "seed": status["seed"],
        "atom_count": atom_count or "",
        "completed": status["completed"],
        "returncode": status["returncode"],
        "elapsed_seconds": status["elapsed_seconds"],
        "final_nve_step": step or "",
        "final_total_energy_ev": nve.get("energy", ""),
        "final_energy_drift_ev": drift or "",
        "nve_time_ps": nve_time_ps if nve_time_ps is not None else "",
        "abs_drift_ev_per_ps": drift_per_ps if drift_per_ps is not None else "",
        "abs_drift_ev_per_atom": drift_per_atom if drift_per_atom is not None else "",
        "abs_drift_ev_per_atom_ps": drift_per_atom_ps if drift_per_atom_ps is not None else "",
        "final_temperature_k": nve.get("temperature", ""),
    }


def main() -> None:
    args = parse_args()
    rows = sorted(
        (flatten(path) for path in args.statuses),
        key=lambda row: (row["mode"], -float(row["dt_fs"]), int(row["seed"])),
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# cg_jaxmd ALA + 50-water sweep",
        "",
        "| mode | dt (fs) | seed | completed | NVE (ps) | drift (eV) | |drift|/atom/ps | temperature (K) | elapsed (s) |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['dt_fs']} | {row['seed']} | {row['completed']} | "
            f"{row['nve_time_ps']} | {row['final_energy_drift_ev']} | "
            f"{row['abs_drift_ev_per_atom_ps']} | "
            f"{row['final_temperature_k']} | {row['elapsed_seconds']} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

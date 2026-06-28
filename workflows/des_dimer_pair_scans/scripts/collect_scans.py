#!/usr/bin/env python3
"""Aggregate des_dimer_pair_scans NPZ outputs into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scan_lib import iter_pairs, load_config, output_dir, workflow_root  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=workflow_root() / "config.yaml")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    return p.parse_args()


def _grid_min(arr: np.ndarray) -> float:
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    rows: list[dict[str, str | float | int]] = []

    for pair in iter_pairs(cfg):
        npz_path = output_dir(cfg, pair) / "scan_2d.npz"
        row: dict[str, str | float | int] = {
            "pair_tag": pair.tag,
            "composition": pair.composition,
            "label": pair.label,
            "npz": str(npz_path),
            "exists": int(npz_path.is_file()),
        }
        if npz_path.is_file():
            data = np.load(npz_path, allow_pickle=False)
            for key in (
                "charmm_ENER_kcal",
                "xtb_energy_kcal",
                "orca_mp2_energy_kcal",
            ):
                if key in data:
                    row[f"min_{key}"] = _grid_min(data[key])
            if "meta_json" in data:
                try:
                    meta = json.loads(str(data["meta_json"]))
                    row["backends"] = json.dumps(meta.get("backends", {}))
                except json.JSONDecodeError:
                    row["backends"] = ""
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    done = sum(int(r.get("exists", 0)) for r in rows)
    lines = [
        "# DES dimer pair scan summary",
        "",
        f"Completed: {done}/{len(rows)} pairs",
        "",
        "| pair | composition | min CHARMM (kcal/mol) | min xTB | min ORCA MP2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_tag']} | {row['composition']} | "
            f"{row.get('min_charmm_ENER_kcal', '')} | "
            f"{row.get('min_xtb_energy_kcal', '')} | "
            f"{row.get('min_orca_mp2_energy_kcal', '')} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

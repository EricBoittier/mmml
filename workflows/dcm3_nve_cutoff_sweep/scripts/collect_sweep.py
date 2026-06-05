#!/usr/bin/env python3
"""Aggregate DCM:3 NVE cutoff sweep metrics into CSV and Markdown ranking."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cutoff_lib import (  # noqa: E402
    geometry_config,
    geometry_ids,
    load_config,
    preset_config,
    preset_ids,
    run_dir,
    workflow_root,
)

_CSV_FIELDS = [
    "preset_id",
    "geom_id",
    "mm_switch_on",
    "mm_switch_width",
    "ml_switch_width",
    "d01_A",
    "d02_A",
    "angle_02_deg",
    "n_frames",
    "duration_ps",
    "etot_std_kcal",
    "max_abs_etot_step_delta_kcal",
    "rms_etot_step_delta_kcal",
    "etot_drift_kcal",
    "etot_drift_per_ps_kcal",
    "epot_std_kcal",
    "max_abs_epot_step_delta_kcal",
    "smoothness_score",
    "temp_mean_K",
    "status",
    "notes",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _row_for_job(cfg: dict[str, Any], preset_id: str, geom_id: str) -> dict[str, Any]:
    preset = preset_config(cfg, preset_id)
    geom = geometry_config(cfg, geom_id)
    metrics_path = run_dir(cfg, preset_id, geom_id) / "nve_metrics.json"
    metrics = _read_json(metrics_path) or {}

    row: dict[str, Any] = {
        "preset_id": preset_id,
        "geom_id": geom_id,
        "mm_switch_on": preset.get("mm_switch_on", ""),
        "mm_switch_width": preset.get("mm_switch_width", ""),
        "ml_switch_width": preset.get("ml_switch_width", ""),
        "d01_A": geom.get("d01", ""),
        "d02_A": geom.get("d02", ""),
        "angle_02_deg": geom.get("angle_02_deg", ""),
        "status": metrics.get("status", "missing"),
        "notes": metrics.get("notes", ""),
    }
    for key in _CSV_FIELDS:
        if key in metrics:
            row[key] = metrics[key]
        row.setdefault(key, "")
    return row


def collect(
    *,
    csv_path: Path,
    md_path: Path,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    rows: list[dict[str, Any]] = []
    for preset_id in preset_ids(cfg):
        for geom_id in geometry_ids(cfg):
            rows.append(_row_for_job(cfg, preset_id, geom_id))

    rows.sort(
        key=lambda r: (
            float(r["smoothness_score"]) if str(r.get("smoothness_score", "")).strip() else float("inf"),
            str(r["preset_id"]),
            str(r["geom_id"]),
        )
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_report(rows, md_path, cfg)
    return rows


def _write_report(rows: list[dict[str, Any]], path: Path, cfg: dict[str, Any]) -> None:
    presets = preset_ids(cfg)
    geoms = geometry_ids(cfg)
    passed = [r for r in rows if r.get("status") == "pass"]

    lines = [
        "# DCM:3 NVE cutoff sweep report",
        "",
        f"- Composition: `{cfg.get('composition', 'DCM:3')}`",
        f"- NVE length: **{cfg['ps_nve']} ps** @ {cfg['dt_fs']} fs",
        f"- Matrix: {len(presets)} cutoff presets × {len(geoms)} COM geometries = {len(rows)} jobs",
        "",
        "Lower **smoothness_score** (= etot_std + max step Δ + 0.1×|drift|) indicates smoother NVE.",
        "",
        "## Ranked runs (smoothest first)",
        "",
        "| rank | preset | geom | mm_on | mm_w | ml_w | etot_std | max_ΔE/step | drift | score | status |",
        "|------|--------|------|-------|------|------|----------|-------------|-------|-------|--------|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            "| {i} | {preset_id} | {geom_id} | {mm_switch_on} | {mm_switch_width} | "
            "{ml_switch_width} | {etot_std_kcal} | {max_abs_etot_step_delta_kcal} | "
            "{etot_drift_kcal} | {smoothness_score} | {status} |".format(i=i, **row)
        )

    if passed:
        by_preset: dict[str, list[float]] = {}
        for row in passed:
            score = float(row["smoothness_score"])
            by_preset.setdefault(str(row["preset_id"]), []).append(score)
        mean_scores = {
            pid: sum(vals) / len(vals) for pid, vals in by_preset.items()
        }
        best_preset = min(mean_scores, key=mean_scores.get)
        lines.extend(
            [
                "",
                "## Preset mean smoothness (passed runs only)",
                "",
                "| preset | mean smoothness_score | n |",
                "|--------|----------------------|---|",
            ]
        )
        for pid in sorted(mean_scores, key=mean_scores.get):
            lines.append(
                f"| {pid} | {mean_scores[pid]:.6f} | {len(by_preset[pid])} |"
            )
        lines.extend(
            [
                "",
                f"**Suggested preset (lowest mean score):** `{best_preset}`",
            ]
        )

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=workflow_root() / "results" / "cutoff_sweep_summary.csv",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=workflow_root() / "results" / "cutoff_sweep_report.md",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
    )
    args = parser.parse_args()
    rows = collect(csv_path=args.csv, md_path=args.md, config_path=args.config)
    print(f"Wrote {args.csv} ({len(rows)} jobs)")
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()

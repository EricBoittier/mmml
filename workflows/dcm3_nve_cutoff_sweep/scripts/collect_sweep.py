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
    energy_catastrophe_score,
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


def _effective_status(row: dict[str, Any], catastrophe: float) -> str:
    status = str(row.get("status", "missing"))
    score_raw = str(row.get("smoothness_score", "")).strip()
    if not score_raw:
        return status
    score = float(score_raw)
    if score > catastrophe and status == "pass":
        return "fail_energy"
    return status


def _row_for_job(
    cfg: dict[str, Any],
    preset_id: str,
    geom_id: str,
    *,
    catastrophe: float,
) -> dict[str, Any]:
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
    row["status"] = _effective_status(row, catastrophe)
    if row["status"] == "fail_energy" and not row["notes"]:
        row["notes"] = (
            f"smoothness_score {float(row['smoothness_score']):.3g} "
            f"> catastrophe threshold {catastrophe:g}"
        )
    return row


def _preset_mean_scores(
    rows: list[dict[str, Any]],
    *,
    exclude_statuses: frozenset[str],
) -> dict[str, tuple[float, int]]:
    by_preset: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("status", "")) in exclude_statuses:
            continue
        score_raw = str(row.get("smoothness_score", "")).strip()
        if not score_raw:
            continue
        by_preset.setdefault(str(row["preset_id"]), []).append(float(score_raw))
    return {
        pid: (sum(vals) / len(vals), len(vals))
        for pid, vals in by_preset.items()
        if vals
    }


def collect(
    *,
    csv_path: Path,
    md_path: Path,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    catastrophe = energy_catastrophe_score(cfg)
    rows: list[dict[str, Any]] = []
    for preset_id in preset_ids(cfg):
        for geom_id in geometry_ids(cfg):
            rows.append(
                _row_for_job(cfg, preset_id, geom_id, catastrophe=catastrophe)
            )

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

    _write_report(rows, md_path, cfg, catastrophe=catastrophe)
    return rows


def _write_report(
    rows: list[dict[str, Any]],
    path: Path,
    cfg: dict[str, Any],
    *,
    catastrophe: float,
) -> None:
    presets = preset_ids(cfg)
    geoms = geometry_ids(cfg)
    n_catastrophe = sum(1 for r in rows if r.get("status") == "fail_energy")

    lines = [
        "# DCM:3 NVE cutoff sweep report",
        "",
        f"- Composition: `{cfg.get('composition', 'DCM:3')}`",
        f"- NVE length: **{cfg['ps_nve']} ps** @ {cfg['dt_fs']} fs",
        f"- Matrix: {len(presets)} cutoff presets × {len(geoms)} COM geometries = {len(rows)} jobs",
        f"- Energy catastrophe threshold: **{catastrophe:g}** (`smoothness_score`; → `fail_energy`)",
        "",
        "Lower **smoothness_score** (= etot_std + max step Δ + 0.1×|drift|) indicates smoother NVE.",
        "",
        "## Ranked runs (smoothest first)",
        "",
        "| rank | preset | geom | mm_on | mm_w | ml_w | etot_std | max_ΔE/step | drift | score | status |",
        "|------|--------|------|-------|------|------|----------|-------------|-------|-------|--------",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            "| {i} | {preset_id} | {geom_id} | {mm_switch_on} | {mm_switch_width} | "
            "{ml_switch_width} | {etot_std_kcal} | {max_abs_etot_step_delta_kcal} | "
            "{etot_drift_kcal} | {smoothness_score} | {status} |".format(i=i, **row)
        )

    all_means = _preset_mean_scores(rows, exclude_statuses=frozenset({"missing", "fail"}))
    sane_means = _preset_mean_scores(
        rows,
        exclude_statuses=frozenset({"missing", "fail", "fail_energy"}),
    )

    if all_means:
        lines.extend(
            [
                "",
                "## Preset mean smoothness (all completed runs)",
                "",
                "| preset | mean smoothness_score | n |",
                "|--------|----------------------|---|",
            ]
        )
        for pid in sorted(all_means, key=lambda p: all_means[p][0]):
            mean, n = all_means[pid]
            lines.append(f"| {pid} | {mean:.6f} | {n} |")

    if sane_means:
        best_preset = min(sane_means, key=lambda p: sane_means[p][0])
        lines.extend(
            [
                "",
                "## Preset mean smoothness (excluding `fail_energy`)",
                "",
                "| preset | mean smoothness_score | n |",
                "|--------|----------------------|---|",
            ]
        )
        for pid in sorted(sane_means, key=lambda p: sane_means[p][0]):
            mean, n = sane_means[pid]
            lines.append(f"| {pid} | {mean:.6f} | {n} |")
        lines.extend(
            [
                "",
                f"**Suggested preset (lowest sane mean):** `{best_preset}`",
            ]
        )
        if n_catastrophe:
            lines.append(
                f"\n_{n_catastrophe} run(s) flagged `fail_energy` (score > {catastrophe:g}); "
                "excluded from sane mean._"
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

#!/usr/bin/env python3
"""Aggregate prep_sweep run logs into a sortable CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    cell_run_tag,
    iter_matrix_cells,
    load_config,
    prep_sweep_enabled,
    repo_root,
    run_output_dir,
)

_GRMS_POST_MINI = re.compile(
    r"Post MLpot SD pass 1[\s\S]*?\|\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
)
_GRMS_PRE_DYN = re.compile(
    r"Pre-dynamics GRMS OK:\s*([0-9.]+)\s*kcal/mol/Å\s*\(limit\s*([0-9.]+)\)"
)
_GRMS_PRE_DYN_FAIL = re.compile(
    r"Pre-dynamics GRMS\s+([0-9.]+)\s*kcal/mol/Å\s*>\s*([0-9.]+)"
)
_GRMS_OVERLAP_FAIL = re.compile(
    r"post-overlap-rescue hybrid GRMS\s+([0-9.]+)\s*kcal/mol/Å\s*>\s*([0-9.]+)"
)
_ERROR = re.compile(r"pycharmm_mlpot: error:\s*(.+)", re.MULTILINE)
_DONE = re.compile(r"Campaign summary", re.MULTILINE)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _last_match(pattern: re.Pattern[str], text: str) -> str | None:
    hits = pattern.findall(text)
    if not hits:
        return None
    last = hits[-1]
    if isinstance(last, tuple):
        return "|".join(str(x) for x in last)
    return str(last)


def _campaign_exit(summary_path: Path) -> int | None:
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    runs = data.get("runs") or {}
    for job in runs.values():
        if isinstance(job, dict) and job.get("exit_code") not in (None, 0):
            return int(job["exit_code"])
    return 0


def _row_for_cell(cfg: dict[str, Any], cell: Any) -> dict[str, Any]:
    out_dir = run_output_dir(cfg, cell)
    stdout = _read_text(out_dir / "stdout.log")
    summary = out_dir / "campaign_summary.json"
    campaign = out_dir / "campaign.yaml"
    defaults: dict[str, Any] = {}
    if campaign.is_file():
        try:
            loaded = yaml.safe_load(campaign.read_text(encoding="utf-8"))
            defaults = dict((loaded or {}).get("defaults") or {})
        except yaml.YAMLError:
            defaults = {}

    status = "missing"
    if (out_dir / "done.txt").is_file():
        status = "done"
    elif stdout:
        status = "failed" if _ERROR.search(stdout) else "running_or_incomplete"

    return {
        "run_tag": cell_run_tag(cell, cfg),
        "sweep_id": cell.sweep_id or "",
        "status": status,
        "exit_code": _campaign_exit(summary),
        "post_mini_grms": _last_match(_GRMS_POST_MINI, stdout),
        "pre_dynamics_grms": _last_match(_GRMS_PRE_DYN, stdout),
        "pre_dynamics_grms_fail": _last_match(_GRMS_PRE_DYN_FAIL, stdout),
        "overlap_rescue_grms_fail": _last_match(_GRMS_OVERLAP_FAIL, stdout),
        "error": _last_match(_ERROR, stdout),
        "dt_fs": defaults.get("dt_fs", ""),
        "spacing": defaults.get("spacing", ""),
        "packmol_tolerance": defaults.get("packmol_tolerance", ""),
        "mm_switch_on": defaults.get("mm_switch_on", ""),
        "mm_switch_width": defaults.get("mm_switch_width", ""),
        "ml_switch_width": defaults.get("ml_switch_width", ""),
        "bonded_mm_mini": defaults.get("bonded_mm_mini", ""),
        "stdout_log": str(out_dir / "stdout.log"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_SCRIPTS.parent / "config.prep_sweep.yaml",
        help="Workflow config with prep_sweep.enabled",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_SCRIPTS.parent / "results" / "prep_sweep_summary.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not prep_sweep_enabled(cfg):
        raise SystemExit("prep_sweep.enabled is false — use config.prep_sweep.yaml")

    rows = [_row_for_cell(cfg, cell) for cell in iter_matrix_cells(cfg)]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run_tag"]
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    done = sum(1 for r in rows if r["status"] == "done")
    failed = sum(1 for r in rows if r["status"] == "failed")
    print(f"Wrote {len(rows)} rows -> {args.csv} (done={done}, failed={failed})")
    print(f"Repo root: {repo_root()}")


if __name__ == "__main__":
    main()

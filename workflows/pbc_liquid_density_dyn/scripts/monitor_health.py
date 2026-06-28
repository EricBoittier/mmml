#!/usr/bin/env python3
"""Hourly health monitor for pbc_liquid_density_dyn Snakemake campaigns.

Scans production, profile, and large-box matrices; detects stale failures;
optionally restarts Snakemake drivers and reruns failed cells.

Usage:
  python scripts/monitor_health.py              # report only
  python scripts/monitor_health.py --react      # report + auto-remediation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
_WORKFLOW = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    campaign_job_order,
    cell_run_tag,
    iter_matrix_cells,
    load_config,
    paths_for_run,
    repo_root,
    workflow_root,
)

# Log signatures that are fixed in current main — safe to delete stdout and rerun.
_STALE_FAILURE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "config_path",
        re.compile(r"FileNotFoundError.*config\.yaml", re.I),
    ),
    (
        "warmup_pmi",
        re.compile(
            r"refuse to run under mpirun/PMI launcher env|ERROR: warmup-mlpot-jax failed",
            re.I,
        ),
    ),
]

_ERROR_MARKERS = re.compile(
    r"Traceback|liquid-density dynamics campaign failed|Campaign summary reports failed|"
    r"ERROR:|pycharmm_mlpot: error:|Segmentation fault|CANCELLED|Killed",
    re.I,
)

_PROGRESS_MARKERS = re.compile(
    r"Campaign jobs|Packmol|warmup-mlpot-jax: done|MLpot profile:|DYNA>|"
    r"density_prep_ladder|pycharmm_init",
    re.I,
)

CAMPAIGNS: list[dict[str, Any]] = [
    {
        "name": "production",
        "config": "config.yaml",
        "driver_log": "snakemake_slurm.log",
        "max_jobs": 4,
    },
    {
        "name": "profile_gpu",
        "config": "config.profile.gpu.yaml",
        "driver_log": "snakemake_profile.log",
        "max_jobs": 2,
    },
    {
        "name": "large_boxes",
        "config": "config.large_boxes.yaml",
        "driver_log": "snakemake_large_boxes.log",
        "max_jobs": 2,
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _resolve_config(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path.resolve()
    candidate = workflow_root() / name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(name)


def _read_tail(path: Path, *, max_bytes: int = 256_000) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            raw = f.read()
    except OSError:
        return ""
    if b"\x00" in raw[:2048]:
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""
    return raw.decode("utf-8", errors="replace")


def _snakemake_driver_running(config_path: Path) -> bool:
    cfg_name = config_path.name
    try:
        out = subprocess.run(
            ["pgrep", "-af", "snakemake"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    for line in out.stdout.splitlines():
        if "snakemake" not in line.lower():
            continue
        if cfg_name in line or str(config_path) in line:
            return True
        if cfg_name == "config.yaml" and "--configfile" not in line and "config." not in line:
            if "pbc_liquid_density_dyn/Snakefile" in line or "pbc_liquid_density_dyn" in line:
                if "config.profile" not in line and "config.large" not in line:
                    return True
    return False


def _slurm_jobs_for_user() -> list[str]:
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T %M %R"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


@dataclass
class CellHealth:
    tag: str
    campaign: str
    n_monomers: int
    box_A: float
    status: str
    legs_done: str
    log_kb: int
    log_age_min: float | None
    stale_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def _inspect_cell(cfg: dict[str, Any], cell, *, campaign: str) -> CellHealth:
    paths = paths_for_run(cfg, cell)
    tag = cell_run_tag(cell, cfg)
    order = campaign_job_order(cfg)
    legs_done = sum(
        1
        for jid in order
        if (paths["out_dir"] / jid / "handoff" / "state.npz").is_file()
    )
    stdout = paths["out_dir"] / "stdout.log"
    text = _read_tail(stdout)
    log_kb = stdout.stat().st_size // 1024 if stdout.is_file() else 0
    log_age_min: float | None = None
    if stdout.is_file():
        age_s = datetime.now(timezone.utc).timestamp() - stdout.stat().st_mtime
        log_age_min = round(age_s / 60.0, 1)

    stale_reason: str | None = None
    if text and not paths["done"].is_file():
        for label, pat in _STALE_FAILURE_PATTERNS:
            if pat.search(text) and "Campaign jobs" not in text:
                stale_reason = label
                break

    errors: list[str] = []
    if text and _ERROR_MARKERS.search(text) and not paths["done"].is_file():
        for line in text.splitlines():
            if _ERROR_MARKERS.search(line):
                errors.append(line.strip()[:160])
                if len(errors) >= 3:
                    break

    if paths["done"].is_file():
        status = "done"
    elif stale_reason:
        status = "stale_fail"
    elif errors and log_age_min is not None and log_age_min > 20:
        status = "failed"
    elif text and _PROGRESS_MARKERS.search(text):
        status = "running"
    elif stdout.is_file() and log_kb > 0:
        status = "started"
    else:
        status = "pending"

    return CellHealth(
        tag=tag,
        campaign=campaign,
        n_monomers=int(cell.n_monomers),
        box_A=float(cell.box_size),
        status=status,
        legs_done=f"{legs_done}/{len(order)}",
        log_kb=log_kb,
        log_age_min=log_age_min,
        stale_reason=stale_reason,
        errors=errors,
    )


def _done_target(cfg: dict[str, Any], cell) -> Path:
    paths = paths_for_run(cfg, cell)
    return paths["done"]


def _rerun_cell(cfg_path: Path, cell, *, dry_run: bool) -> str:
    cfg = load_config(cfg_path)
    tag = cell_run_tag(cell, cfg)
    paths = paths_for_run(cfg, cell)
    stdout = paths["out_dir"] / "stdout.log"
    actions: list[str] = []
    if stdout.is_file():
        actions.append(f"remove {stdout}")
        if not dry_run:
            stdout.unlink(missing_ok=True)
    target = f"../../{paths['done'].relative_to(repo_root())}"
    cmd = f"MMML_WORKFLOW_CONFIG={cfg_path.name} bash scripts/snakemake_slurm.sh 1 {target}"
    actions.append(cmd)
    if not dry_run:
        env = os.environ.copy()
        env["MMML_WORKFLOW_CONFIG"] = str(cfg_path)
        subprocess.Popen(  # noqa: S603
            ["bash", "scripts/snakemake_slurm.sh", "1", target],
            cwd=workflow_root(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return "; ".join(actions)


def _ensure_driver(spec: dict[str, Any], cfg_path: Path, *, incomplete: int, dry_run: bool) -> list[str]:
    actions: list[str] = []
    if incomplete <= 0:
        return actions
    if _snakemake_driver_running(cfg_path):
        actions.append(f"driver already running for {cfg_path.name}")
        return actions
    log = workflow_root() / str(spec["driver_log"])
    cmd = f"nohup bash scripts/snakemake_slurm.sh {int(spec['max_jobs'])} >> {log.name} 2>&1 &"
    actions.append(f"start driver: MMML_WORKFLOW_CONFIG={cfg_path.name} {cmd}")
    if not dry_run:
        env = os.environ.copy()
        env["MMML_WORKFLOW_CONFIG"] = str(cfg_path)
        subprocess.Popen(  # noqa: S603
            ["bash", "scripts/snakemake_slurm.sh", str(spec["max_jobs"])],
            cwd=workflow_root(),
            env=env,
            stdout=open(log, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return actions


def run_monitor(*, react: bool, dry_run: bool) -> dict[str, Any]:
    dry_run = dry_run or not react
    report: dict[str, Any] = {
        "timestamp": _utc_now(),
        "hostname": os.uname().nodename,
        "slurm_queue": _slurm_jobs_for_user(),
        "campaigns": [],
    }
    all_cells: list[CellHealth] = []

    for spec in CAMPAIGNS:
        cfg_path = _resolve_config(str(spec["config"]))
        cfg = load_config(cfg_path)
        camp_report: dict[str, Any] = {
            "name": spec["name"],
            "config": str(cfg_path),
            "driver_running": _snakemake_driver_running(cfg_path),
            "cells": [],
            "actions": [],
        }
        incomplete = 0
        for cell in iter_matrix_cells(cfg):
            health = _inspect_cell(cfg, cell, campaign=str(spec["name"]))
            all_cells.append(health)
            if health.status != "done":
                incomplete += 1
            cell_actions: list[str] = []
            if react and health.stale_reason and health.status == "stale_fail":
                cell_actions.append(_rerun_cell(cfg_path, cell, dry_run=dry_run))
                health.actions.extend(cell_actions)
            camp_report["cells"].append(asdict(health))
        if react:
            camp_report["actions"] = _ensure_driver(
                spec, cfg_path, incomplete=incomplete, dry_run=dry_run
            )
        report["campaigns"].append(camp_report)

    n_done = sum(1 for c in all_cells if c.status == "done")
    n_stale = sum(1 for c in all_cells if c.status == "stale_fail")
    n_run = sum(1 for c in all_cells if c.status == "running")
    report["summary"] = {
        "total": len(all_cells),
        "done": n_done,
        "running": n_run,
        "stale_fail": n_stale,
        "failed": sum(1 for c in all_cells if c.status == "failed"),
        "pending": sum(1 for c in all_cells if c.status == "pending"),
    }

    print(f"\n=== pbc_liquid_density_dyn monitor @ {_utc_now()} ===")
    print(
        f"Cells: {n_done}/{len(all_cells)} done | {n_run} running | "
        f"{n_stale} stale (auto-rerun) | queue lines: {len(report['slurm_queue'])}"
    )
    print(f"{'campaign':<14} {'tag':<22} {'N':>4} {'L':>4} {'legs':>8} {'status':<12} note")
    for camp in report["campaigns"]:
        for row in camp["cells"]:
            note = row.get("stale_reason") or (row["errors"][0][:40] if row.get("errors") else "")
            print(
                f"{camp['name']:<14} {row['tag']:<22} {row['n_monomers']:4d} "
                f"{int(row['box_A']):4d} {row['legs_done']:>8} {row['status']:<12} {note}"
            )
        drv = "driver=UP" if camp["driver_running"] else "driver=DOWN"
        print(f"  [{camp['name']}] {drv} incomplete={sum(1 for r in camp['cells'] if r['status'] != 'done')}")
        for act in camp.get("actions") or []:
            print(f"    ACTION: {act}")

    results_dir = workflow_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "monitor_latest.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log_path = workflow_root() / "logs" / "monitor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{_utc_now()} done={n_done}/{len(all_cells)} run={n_run} "
            f"stale={n_stale} react={react}\n"
        )
    print(f"\nWrote {json_path}")
    print(f"Appended {log_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--react",
        action="store_true",
        help="Restart dead Snakemake drivers and rerun cells with stale known failures",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show react actions without executing (implies no --react side effects)",
    )
    args = parser.parse_args()
    if not os.environ.get("MMML_CKPT", "").strip():
        print("WARNING: MMML_CKPT is unset — reruns will fail at runtime", file=sys.stderr)
    run_monitor(react=bool(args.react), dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

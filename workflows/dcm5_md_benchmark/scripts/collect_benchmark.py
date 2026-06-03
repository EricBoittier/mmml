#!/usr/bin/env python3
"""Aggregate DCM:5 benchmark job outputs into CSV and Markdown report."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# Snakemake injects ``snakemake`` when run as a script rule.
try:
    snakemake  # type: ignore[name-defined]
except NameError:
    snakemake = None  # type: ignore[assignment,misc]

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_lib import job_metadata, load_config, workflow_root  # noqa: E402

_CSV_FIELDS = [
    "job_id",
    "backend",
    "setup",
    "pbc",
    "integrator",
    "ps",
    "nsteps",
    "nsteps_target",
    "status",
    "temp_mean_K",
    "temp_end_K",
    "energy_drift",
    "wall_s",
    "notes",
]

_RESTART_STEP_RE = re.compile(
    r"(?:restart_step=|NSTEP\s*=?\s*)(\d+)", re.IGNORECASE
)
_HEAT_COMPLETE_RE = re.compile(
    r"(?:HEAT|NVE|EQUI|PROD)\s+complete:", re.IGNORECASE
)
_AVER_TEMP_RE = re.compile(
    r"AVER(?:AGE)?\s+(?:TEMP|TEMPERATURE)\s*=?\s*([0-9.+-eE]+)", re.IGNORECASE
)
_ECHECK_ABORT_RE = re.compile(r"echeck|ECHECK|tolerance exceeded", re.IGNORECASE)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _ase_run_key(job: dict[str, Any]) -> str:
    setup = str(job["setup"])
    integrator = str(job.get("integrator", ""))
    if setup == "free_nve":
        return "vac_nve"
    if setup == "pbc_nve":
        return "pbc_nve"
    if setup == "free_nvt":
        return "vac_nvt_langevin" if "langevin" in integrator else "vac_nvt_nhc"
    if setup == "pbc_nvt":
        return "pbc_nvt_langevin" if "langevin" in integrator else "pbc_nvt_nhc"
    return integrator


def _parse_ase(out_dir: Path, job: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    row = dict(meta)
    row["status"] = "missing"
    row["notes"] = ""

    summary = _read_json(out_dir / "suite_summary.json")
    if not summary:
        row["notes"] = "suite_summary.json not found"
        return row

    key = _ase_run_key(job)
    run = summary.get("runs", {}).get(key)
    if not run:
        row["notes"] = f"run key {key!r} missing in suite_summary.json"
        return row

    log_samples = int(run.get("log_samples", 0))
    nsteps = int(round(meta["ps"] * 1000.0 / 0.25))  # fallback
    row["nsteps"] = nsteps
    row["temp_mean_K"] = run.get("temp_mean_K", "")
    row["temp_end_K"] = run.get("temp_end_K", "")
    row["energy_drift"] = run.get("etot_drift_eV", "")
    timings = run.get("timings_s") or {}
    row["wall_s"] = timings.get("md_integrator_loop_s", "")

    if log_samples > 0:
        row["status"] = "pass"
    else:
        row["status"] = "fail"
        row["notes"] = "log_samples=0"

    target = meta["nsteps_target"]
    if meta["integrator"] == "nve" and run.get("etot_drift_eV") is not None:
        row["notes"] = f"etot_drift_eV={run['etot_drift_eV']:.4f}"

    temp_mean = run.get("temp_mean_K")
    if temp_mean is not None and "nvt" in meta["integrator"]:
        target_t = 300.0
        if abs(float(temp_mean) - target_t) / target_t > 0.15:
            row["status"] = "warn"
            row["notes"] = (
                f"temp_mean_K={temp_mean:.1f} outside ±15% of {target_t} K"
            )

    return row


def _parse_jaxmd(out_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    row = dict(meta)
    row["status"] = "missing"
    row["notes"] = ""

    summary = _read_json(out_dir / "suite_summary_jaxmd.json")
    if not summary:
        row["notes"] = "suite_summary_jaxmd.json not found"
        return row

    status = str(summary.get("status", ""))
    nsteps_completed = int(summary.get("nsteps_completed", 0))
    nsteps_requested = int(summary.get("nsteps_requested", meta["nsteps_target"]))
    row["nsteps"] = nsteps_completed
    row["temp_mean_K"] = summary.get("temperature_K", "")

    min_ok = int(0.95 * meta["nsteps_target"])
    if status == "complete" and nsteps_completed >= min_ok:
        row["status"] = "pass"
    elif status == "complete":
        row["status"] = "warn"
        row["notes"] = f"nsteps_completed={nsteps_completed} < {min_ok}"
    else:
        row["status"] = "fail"
        row["notes"] = summary.get("error") or f"status={status!r}"

    return row


def _parse_pycharmm(out_dir: Path, log_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    row = dict(meta)
    row["status"] = "missing"
    row["notes"] = ""
    row["nsteps"] = ""

    if not log_path.is_file():
        row["notes"] = "stdout.log not found"
        return row

    text = log_path.read_text(encoding="utf-8", errors="replace")
    restart_step: int | None = None
    for match in _RESTART_STEP_RE.finditer(text):
        restart_step = int(match.group(1))

    if restart_step is not None:
        row["nsteps"] = restart_step

    min_ok = int(0.95 * meta["nsteps_target"])
    complete = bool(_HEAT_COMPLETE_RE.search(text))
    echeck = bool(_ECHECK_ABORT_RE.search(text)) and "complete" not in text.lower()

    aver = _AVER_TEMP_RE.search(text)
    if aver:
        row["temp_mean_K"] = float(aver.group(1))

    if echeck:
        row["status"] = "fail"
        row["notes"] = "possible echeck abort"
    elif complete and restart_step is not None and restart_step >= min_ok:
        row["status"] = "pass"
    elif complete:
        row["status"] = "warn"
        row["notes"] = f"restart_step={restart_step} < {min_ok}"
    elif restart_step is not None and restart_step >= min_ok:
        row["status"] = "warn"
        row["notes"] = "restart ok but stage complete line not found"
    else:
        row["status"] = "fail"
        row["notes"] = f"restart_step={restart_step}"

    return row


def _parse_job(out_dir: Path, cfg: dict[str, Any], job_id: str) -> dict[str, Any]:
    job = cfg["jobs"][job_id]
    meta = job_metadata(cfg, job_id)
    if not out_dir.is_dir() or not (out_dir / "stdout.log").is_file():
        meta["status"] = "skipped" if meta.get("optional") else "missing"
        meta["notes"] = "optional job not run" if meta.get("optional") else "no output dir"
        meta["nsteps"] = ""
        meta["temp_mean_K"] = ""
        meta["temp_end_K"] = ""
        meta["energy_drift"] = ""
        meta["wall_s"] = ""
        return meta
    backend = meta["backend"]
    log_path = out_dir / "stdout.log"

    if backend == "ase":
        return _parse_ase(out_dir, job, meta)
    if backend == "jaxmd":
        return _parse_jaxmd(out_dir, meta)
    if backend == "pycharmm":
        return _parse_pycharmm(out_dir, log_path, meta)

    meta["status"] = "fail"
    meta["notes"] = f"unknown backend {backend!r}"
    meta["nsteps"] = ""
    meta["temp_mean_K"] = ""
    meta["temp_end_K"] = ""
    meta["energy_drift"] = ""
    meta["wall_s"] = ""
    return meta


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(rows: list[dict[str, Any]], path: Path, cfg: dict[str, Any]) -> None:
    n_pass = sum(1 for r in rows if r["status"] == "pass")
    n_warn = sum(1 for r in rows if r["status"] == "warn")
    n_fail = sum(1 for r in rows if r["status"] in {"fail", "missing"})

    lines = [
        "# DCM:5 cross-backend MD benchmark report",
        "",
        f"- Composition: `{cfg['composition']}`",
        f"- Target length: **{cfg['ps']} ps** ({cfg['nsteps_target']} steps @ {cfg['dt_fs']} fs)",
        f"- Jobs: {len(rows)} (pass={n_pass}, warn={n_warn}, fail/missing={n_fail})",
        "",
        "| job | backend | setup | PBC | integrator | status | nsteps | temp_mean_K | notes |",
        "|-----|---------|-------|-----|------------|--------|--------|-------------|-------|",
    ]
    for row in rows:
        lines.append(
            "| {job_id} | {backend} | {setup} | {pbc} | {integrator} | {status} | "
            "{nsteps} | {temp_mean_K} | {notes} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect(
    *,
    results_root: Path,
    csv_path: Path,
    md_path: Path,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    rows: list[dict[str, Any]] = []
    for job_id in sorted(cfg["jobs"]):
        out_dir = results_root / job_id
        row = _parse_job(out_dir, cfg, job_id)
        for field in _CSV_FIELDS:
            row.setdefault(field, "")
        rows.append(row)

    _write_csv(rows, csv_path)
    _write_report(rows, md_path, cfg)
    return rows


def main() -> None:
    if snakemake is not None:
        csv_path = Path(snakemake.output.csv)
        md_path = Path(snakemake.output.md)
        results_root = workflow_root() / "results"
        rows = collect(
            results_root=results_root,
            csv_path=csv_path,
            md_path=md_path,
        )
        print(f"Wrote {csv_path} ({len(rows)} jobs)")
        print(f"Wrote {md_path}")
        return

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=workflow_root() / "results",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=workflow_root() / "results" / "benchmark_summary.csv",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=workflow_root() / "results" / "benchmark_report.md",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
    )
    args = parser.parse_args()
    collect(
        results_root=args.results_root,
        csv_path=args.csv,
        md_path=args.md,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

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

_ANSI = re.compile(r"\x1b\[[0-9;]*(?:[0-9;]*[A-Za-z])|\x1b\][^\x07]*(?:\x07|\x1b\\)")
_GRMS_PARTIAL = re.compile(r"GRMS≈([0-9.]+)\s*kcal/mol/Å")
_GRMS_SD_STALL = re.compile(
    r"MLpot SD pass 1 stalled.*?GRMS≈([0-9.]+)\s*kcal/mol/Å",
    re.DOTALL,
)
_GRMS_SD_PARTIAL = re.compile(
    r"MLpot SD pass 1 partial.*?GRMS≈([0-9.]+)\s*kcal/mol/Å",
    re.DOTALL,
)
_RICH_GRMS_CELL = re.compile(r"([\d.]+)\s*kcal/mol/Å")
_GRMS_HYBRID_KV = re.compile(
    r"hybrid GRMS[= ]+([0-9.]+)\s*kcal/mol/Å",
    re.IGNORECASE,
)
_GRMS_HYBRID_LABEL = re.compile(
    r"Hybrid GRMS[^\d\n]*([\d.]+)\s*kcal/mol/Å",
    re.IGNORECASE,
)
_GRMS_PRE_DYN_OK = re.compile(
    r"Pre-dynamics GRMS OK:\s*([0-9.]+)\s*kcal/mol/Å\s*\(limit\s*([0-9.]+)\)"
)
_GRMS_PRE_DYN_FAIL = re.compile(
    r"Pre-dynamics GRMS\s+([0-9.]+)\s*kcal/mol/Å\s*>\s*([0-9.]+)"
)
_GRMS_OVERLAP_FAIL = re.compile(
    r"post-overlap-rescue hybrid GRMS\s+([0-9.]+)\s*kcal/mol/Å\s*>\s*([0-9.]+)"
)
_ERROR = re.compile(r"pycharmm_mlpot: error:\s*(.+)", re.MULTILINE)
_FAIL = re.compile(
    r"(?:pycharmm_mlpot: error:|Unknown config key\(s\)|setup-compare campaign failed|Traceback \(most recent call last\))",
    re.MULTILINE,
)


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _rich_hybrid_grms_after_context(text: str, context: str) -> float | None:
    """First hybrid GRMS value cell after each Rich ``Hybrid GRMS`` header in a panel."""
    stripped = _strip_ansi(text)
    idx = stripped.rfind(context)
    if idx < 0:
        return None
    window = stripped[idx : idx + 4000]
    best: float | None = None
    for hdr in re.finditer("Hybrid GRMS", window):
        chunk = window[hdr.end() : hdr.end() + 600]
        m = _RICH_GRMS_CELL.search(chunk)
        if m:
            best = float(m.group(1))
    return best


def _hybrid_grms_after_context(text: str, context: str) -> float | None:
    rich = _rich_hybrid_grms_after_context(text, context)
    if rich is not None:
        return rich
    stripped = _strip_ansi(text)
    idx = stripped.rfind(context)
    if idx < 0:
        return None
    window = stripped[idx : idx + 2500]
    for pat in (_GRMS_HYBRID_LABEL, _GRMS_HYBRID_KV, _GRMS_PARTIAL):
        m = pat.search(window)
        if m:
            return float(m.group(1))
    return None


def _extract_post_mini_grms(stdout: str) -> tuple[float | None, float | None, float | None]:
    """Return (final post-mini hybrid GRMS, SD stall GRMS, SD partial GRMS)."""
    stall = _last_float(_GRMS_SD_STALL, stdout)
    partial = _last_float(_GRMS_SD_PARTIAL, stdout)

    final: float | None = None
    for context in ("Post MLpot mini", "Post MLpot mini GRMS"):
        val = _hybrid_grms_after_context(stdout, context)
        if val is not None:
            final = val
            break

    if final is None:
        final = _hybrid_grms_after_context(stdout, "Post MLpot SD pass 1")

    if final is None:
        final = partial if partial is not None else stall

    return final, stall, partial


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    hits = pattern.findall(_strip_ansi(text))
    if not hits:
        return None
    last = hits[-1]
    if isinstance(last, tuple):
        return float(last[0])
    return float(last)


def _extract_grms_metrics(stdout: str) -> dict[str, Any]:
    """Best-effort hybrid GRMS from stdout (Rich tables / plain text)."""
    pre_ok = _last_match(_GRMS_PRE_DYN_OK, stdout)
    pre_fail = _last_match(_GRMS_PRE_DYN_FAIL, stdout)

    post_mini, sd_stall, sd_partial = _extract_post_mini_grms(stdout)

    pre_dyn: float | None = None
    pre_limit: float | None = None
    if pre_ok:
        parts = pre_ok.split("|")
        pre_dyn = float(parts[0])
        if len(parts) > 1:
            pre_limit = float(parts[1])
    elif pre_fail:
        parts = pre_fail.split("|")
        pre_dyn = float(parts[0])
        if len(parts) > 1:
            pre_limit = float(parts[1])

    gate_grms = pre_dyn if pre_dyn is not None else post_mini
    under_50: str | bool = ""
    if gate_grms is not None:
        under_50 = gate_grms <= 50.0

    return {
        "post_mini_grms": f"{post_mini:.4f}" if post_mini is not None else "",
        "sd_stall_grms": f"{sd_stall:.4f}" if sd_stall is not None else "",
        "sd_partial_grms": f"{sd_partial:.4f}" if sd_partial is not None else "",
        "pre_dynamics_grms": f"{pre_dyn:.4f}" if pre_dyn is not None else "",
        "pre_dynamics_limit": f"{pre_limit:.1f}" if pre_limit is not None else "",
        "pre_dynamics_grms_fail": pre_fail or "",
        "gate_grms": f"{gate_grms:.4f}" if gate_grms is not None else "",
        "under_50": under_50,
    }


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
        status = "failed" if _FAIL.search(stdout) else "running_or_incomplete"

    metrics = _extract_grms_metrics(stdout)
    return {
        "run_tag": cell_run_tag(cell, cfg),
        "sweep_id": cell.sweep_id or "",
        "status": status,
        "exit_code": _campaign_exit(summary),
        **metrics,
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
    passed = sum(1 for r in rows if r.get("under_50") is True)
    print(f"Wrote {len(rows)} rows -> {args.csv} (done={done}, failed={failed}, gate_grms<=50={passed})")
    print(f"Repo root: {repo_root()}")


if __name__ == "__main__":
    main()

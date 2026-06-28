#!/usr/bin/env python3
"""Hourly health monitor for pbc_liquid_density_dyn Snakemake campaigns.

Classifies known failure modes and applies targeted mediations (tier prebuild,
fresh rerun, resume rerun, driver restart). Tracks per-cell retry budgets.

Usage:
  python scripts/monitor_health.py              # report only
  python scripts/monitor_health.py --react      # report + auto-remediation
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

_SCRIPTS = Path(__file__).resolve().parent
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

MediationAction = Literal[
    "rerun_fresh",
    "rerun_resume",
    "prebuild_tier_rerun",
    "submit_pending",
    "skip_manual",
]

_PROGRESS_MARKERS = re.compile(
    r"Campaign jobs|Packmol|warmup-mlpot-jax: done|MLpot profile:|DYNA>|"
    r"density_prep_ladder|pycharmm_init|liquid-density dynamics campaign failed",
    re.I,
)

CAMPAIGNS: list[dict[str, Any]] = [
    {
        "name": "production",
        "config": "config.yaml",
        "driver_log": "snakemake_slurm.log",
        "max_jobs": 4,
        "auto_start": False,  # paused: gpu08_local handles DCM L28/L32 on-node
    },
    {
        "name": "profile_gpu",
        "config": "config.profile.gpu.yaml",
        "driver_log": "snakemake_profile.log",
        "max_jobs": 2,
        "auto_start": False,
    },
    # large_boxes (L=36,40) paused — OOM / stuck warmup; config.large_boxes.yaml kept for later.
    {
        "name": "gpu08_local",
        "config": "config.gpu08.local.yaml",
        "driver_log": "snakemake_gpu08.log",
        "max_jobs": 2,
        "local": True,
    },
]


@dataclass(frozen=True)
class FailureRule:
    id: str
    pattern: re.Pattern[str]
    action: MediationAction
    max_retries: int = 3
    note: str = ""


# Known errors → mediation. Order matters: first match wins per scan pass.
FAILURE_RULES: tuple[FailureRule, ...] = (
    FailureRule(
        "oom_kill",
        re.compile(r"OUT_OF_MEMORY|Killed process|CUDA out of memory|slurmstepd: error: Detected \d+ oom", re.I),
        "skip_manual",
        max_retries=0,
        note="reduce N or box size (OOM)",
    ),
    FailureRule(
        "config_path",
        re.compile(r"FileNotFoundError.*config\.yaml", re.I),
        "rerun_fresh",
        max_retries=5,
        note="fixed config path resolution",
    ),
    FailureRule(
        "warmup_pmi",
        re.compile(
            r"refuse to run under mpirun/PMI launcher env|ERROR: warmup-mlpot-jax failed",
            re.I,
        ),
        "rerun_fresh",
        max_retries=5,
        note="PMI scrub + allow-under-mpirun in job_shell",
    ),
    FailureRule(
        "opencl_missing",
        re.compile(r"libOpenCL\.so\.1 not found|libOpenCL\.so not found", re.I),
        "skip_manual",
        max_retries=0,
        note="needs GPU compute node (Slurm gpu partition)",
    ),
    FailureRule(
        "charmm_tier_exceeded",
        re.compile(r"largest tier .* is insufficient|max_Npr>=\d+ pairs; largest tier", re.I),
        "skip_manual",
        max_retries=0,
        note="reduce N or bulk fraction (largest NPR tier exceeded)",
    ),
    FailureRule(
        "charmm_tier",
        re.compile(
            r"could not resolve CHARMM NPR tier|ERROR: could not resolve CHARMM NPR tier",
            re.I,
        ),
        "prebuild_tier_rerun",
        max_retries=2,
        note="prebuild tier via ensure_charmm_mlpot_limits.sh",
    ),
    FailureRule(
        "checkpoint_missing",
        re.compile(
            r"checkpoint not found|MMML_CKPT is not set|warmup-mlpot-jax: checkpoint not found",
            re.I,
        ),
        "skip_manual",
        max_retries=0,
        note="set MMML_CKPT in cron / job env",
    ),
    FailureRule(
        "echeck_abort",
        re.compile(
            r"dynamics incomplete.*echeck|ENERGY CHANGE TOLERANCE|"
            r"integrated step \d+ < \d+.*expected",
            re.I,
        ),
        "rerun_resume",
        max_retries=3,
        note="resume campaign from last handoff (--resume)",
    ),
    FailureRule(
        "campaign_leg_fail",
        re.compile(r"Campaign summary reports failed legs|liquid-density dynamics campaign failed", re.I),
        "rerun_resume",
        max_retries=3,
        note="resume incomplete legs",
    ),
    FailureRule(
        "overlap_abort",
        re.compile(
            r"intra-monomer close contact|dynamics aborted after chunk|"
            r"dynamics_overlap_action|overlap_rescue",
            re.I,
        ),
        "rerun_resume",
        max_retries=2,
        note="resume after overlap rescue ladder",
    ),
    FailureRule(
        "oom",
        re.compile(r"exit code -9|Out of memory|Killed\s*$|CANCELLED.*TIME", re.I),
        "skip_manual",
        max_retries=0,
        note="reduce concurrency or request more mem",
    ),
    FailureRule(
        "segfault",
        re.compile(r"Segmentation fault|Signal: Segmentation|exit code -?11", re.I),
        "rerun_fresh",
        max_retries=1,
        note="one fresh rerun after segfault",
    ),
)


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
    failure_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


class RetryTracker:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, int]] = {}
        if path.is_file():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _key(self, campaign: str, tag: str) -> str:
        return f"{campaign}:{tag}"

    def count(self, campaign: str, tag: str, failure_id: str) -> int:
        return int(self._data.get(self._key(campaign, tag), {}).get(failure_id, 0))

    def record(self, campaign: str, tag: str, failure_id: str) -> None:
        key = self._key(campaign, tag)
        bucket = self._data.setdefault(key, {})
        bucket[failure_id] = int(bucket.get(failure_id, 0)) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2) + "\n", encoding="utf-8")

    def can_retry(self, campaign: str, tag: str, rule: FailureRule) -> bool:
        if rule.max_retries <= 0:
            return False
        return self.count(campaign, tag, rule.id) < rule.max_retries


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


def _read_tail(path: Path, *, max_bytes: int = 384_000) -> str:
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
    if b"\x00" in raw[:4096]:
        return raw.decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _classify_failures(text: str) -> list[FailureRule]:
    hits: list[FailureRule] = []
    for rule in FAILURE_RULES:
        if rule.pattern.search(text):
            hits.append(rule)
    return hits


def _campaign_spec(name: str) -> dict[str, Any] | None:
    for spec in CAMPAIGNS:
        if spec["name"] == name:
            return spec
    return None


def _launcher_script(campaign: str) -> str:
    spec = _campaign_spec(campaign)
    if spec and spec.get("local"):
        return "snakemake_local.sh"
    return "snakemake_slurm.sh"


def _snakemake_driver_running(config_path: Path) -> bool:
    cfg_name = config_path.name
    cfg_resolved = str(config_path.resolve())
    lock = Path(f"/tmp/mmml_snakemake_locks_{os.environ.get('USER') or os.environ.get('LOGNAME') or 'unknown'}") / f"{cfg_name}.driver.lock"
    if lock.is_file():
        try:
            with lock.open("rb") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return True
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
        if cfg_resolved in line or f"--configfile {cfg_name}" in line or f"--configfile {cfg_resolved}" in line:
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


def _estimate_n_ml(cell) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import estimate_ml_atoms

    return int(estimate_ml_atoms(int(cell.n_monomers), solvent=cell.solvent))


def _prebuild_charmm_tier(cell, *, dry_run: bool) -> str:
    n_ml = _estimate_n_ml(cell)
    cmd = [
        "bash",
        str(repo_root() / "scripts" / "ensure_charmm_mlpot_limits.sh"),
        "--n-ml",
        str(n_ml),
        "--pbc",
        "--box-size",
        str(float(cell.box_size)),
    ]
    desc = " ".join(cmd)
    if not dry_run:
        subprocess.run(cmd, cwd=repo_root(), check=False)
    return f"prebuild tier: {desc}"


def _submit_snakemake_target(cfg_path: Path, cell, *, campaign: str, dry_run: bool) -> str:
    paths = paths_for_run(load_config(cfg_path), cell)
    target = f"../../{paths['done'].relative_to(repo_root())}"
    launcher = _launcher_script(campaign)
    cmd = f"MMML_WORKFLOW_CONFIG={cfg_path.name} bash scripts/{launcher} 1 {target}"
    if not dry_run:
        env = _driver_subprocess_env(cfg_path)
        if launcher == "snakemake_local.sh":
            env["MMML_LOCAL_GPU_PIN"] = "1"
        subprocess.Popen(  # noqa: S603
            ["bash", f"scripts/{launcher}", "1", target],
            cwd=workflow_root(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return cmd


def _rerun_fresh(cfg_path: Path, cell, *, campaign: str, dry_run: bool) -> str:
    paths = paths_for_run(load_config(cfg_path), cell)
    stdout = paths["out_dir"] / "stdout.log"
    actions: list[str] = []
    if stdout.is_file():
        actions.append(f"remove {stdout}")
        if not dry_run:
            stdout.unlink(missing_ok=True)
    actions.append(_submit_snakemake_target(cfg_path, cell, campaign=campaign, dry_run=dry_run))
    return "; ".join(actions)


def _rerun_resume(cfg_path: Path, cell, *, campaign: str, dry_run: bool) -> str:
    # Keep prep ladder / partial legs; md-system --resume inside run_job.py.
    cmd = _submit_snakemake_target(cfg_path, cell, campaign=campaign, dry_run=dry_run)
    return f"resume rerun: {cmd}"


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

    failure_rules = _classify_failures(text) if text else []
    failure_ids = [r.id for r in failure_rules]

    errors: list[str] = []
    for line in text.splitlines():
        if any(r.pattern.search(line) for r in failure_rules):
            errors.append(line.strip()[:160])
            if len(errors) >= 3:
                break

    if paths["done"].is_file():
        status = "done"
    elif failure_rules and any(r.action != "skip_manual" for r in failure_rules):
        status = "failed"
    elif failure_rules:
        status = "manual"
    elif text and _PROGRESS_MARKERS.search(text):
        status = "running"
    elif stdout.is_file() and log_kb > 0:
        status = "started"
    else:
        status = "pending"

    # Stuck: log exists but no leg progress for a long time.
    if (
        status in {"running", "started"}
        and legs_done == 0
        and log_age_min is not None
        and log_age_min > 180
        and "stuck_no_legs" not in failure_ids
    ):
        failure_ids.append("stuck_no_legs")
        failure_rules = list(failure_rules) + [
            FailureRule(
                "stuck_no_legs",
                re.compile(r"."),
                "rerun_resume",
                max_retries=2,
                note="no leg handoff after 3h",
            )
        ]

    return CellHealth(
        tag=tag,
        campaign=campaign,
        n_monomers=int(cell.n_monomers),
        box_A=float(cell.box_size),
        status=status,
        legs_done=f"{legs_done}/{len(order)}",
        log_kb=log_kb,
        log_age_min=log_age_min,
        failure_ids=failure_ids,
        errors=errors,
    )


def _mediate_cell(
    health: CellHealth,
    cell,
    cfg_path: Path,
    tracker: RetryTracker,
    *,
    driver_running: bool,
    dry_run: bool,
) -> list[str]:
    actions: list[str] = []
    if health.status == "done":
        return actions

    text = _read_tail(paths_for_run(load_config(cfg_path), cell)["out_dir"] / "stdout.log")
    rules = _classify_failures(text)
    if health.failure_ids and "stuck_no_legs" in health.failure_ids:
        rules.append(
            FailureRule("stuck_no_legs", re.compile(r"."), "rerun_resume", max_retries=2)
        )

    for rule in rules:
        if not tracker.can_retry(health.campaign, health.tag, rule):
            actions.append(f"skip {rule.id} (retry budget exhausted)")
            continue
        if rule.action == "skip_manual":
            actions.append(f"manual: {rule.id} — {rule.note}")
            return actions
        if rule.action == "rerun_fresh":
            if "Campaign jobs" in text:
                actions.append(f"skip fresh rerun for {rule.id} (campaign already started)")
                continue
            actions.append(f"{rule.id}: {_rerun_fresh(cfg_path, cell, campaign=health.campaign, dry_run=dry_run)}")
            if not dry_run:
                tracker.record(health.campaign, health.tag, rule.id)
            continue
        if rule.action == "rerun_resume":
            actions.append(f"{rule.id}: {_rerun_resume(cfg_path, cell, campaign=health.campaign, dry_run=dry_run)}")
            if not dry_run:
                tracker.record(health.campaign, health.tag, rule.id)
            continue
        if rule.action == "prebuild_tier_rerun":
            actions.append(f"{rule.id}: {_prebuild_charmm_tier(cell, dry_run=dry_run)}")
            actions.append(f"{rule.id}: {_rerun_fresh(cfg_path, cell, campaign=health.campaign, dry_run=dry_run)}")
            if not dry_run:
                tracker.record(health.campaign, health.tag, rule.id)

    # Pending cells with driver up but no stdout — nudge Snakemake for this target.
    # When driver is down, _ensure_driver starts the batch driver instead.
    if health.status == "pending" and health.log_kb == 0 and driver_running:
        pending_rule = FailureRule(
            "pending_submit",
            re.compile(r"^$"),
            "submit_pending",
            max_retries=4,
            note="submit cell to Slurm",
        )
        if tracker.can_retry(health.campaign, health.tag, pending_rule):
            actions.append(
                f"pending: {_submit_snakemake_target(cfg_path, cell, campaign=health.campaign, dry_run=dry_run)}"
            )
            if not dry_run:
                tracker.record(health.campaign, health.tag, pending_rule.id)

    return actions


def _driver_subprocess_env(cfg_path: Path) -> dict[str, str]:
    """Environment for Snakemake driver subprocesses (cron-safe PATH + uv)."""
    env = os.environ.copy()
    home = Path.home()
    path_parts = [env.get("PATH", "")]
    for bindir in (home / ".local" / "bin", home / ".cargo" / "bin", home / "bin"):
        if bindir.is_dir():
            path_parts.insert(0, str(bindir))
    env["PATH"] = ":".join(p for p in path_parts if p)
    env["MMML_WORKFLOW_CONFIG"] = str(cfg_path)
    env.setdefault("JAX_ENABLE_X64", "1")
    if not env.get("MMML_CKPT"):
        default_ckpt = (
            "/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/"
            "dcm1-c137fb42-1f65-4748-880b-8f8184a20f70"
        )
        if Path(default_ckpt).is_dir():
            env["MMML_CKPT"] = default_ckpt
    if not env.get("MMML_UV"):
        import shutil

        uv_bin = shutil.which("uv", path=env["PATH"])
        if uv_bin:
            env["MMML_UV"] = uv_bin
    return env


def _ensure_driver(spec: dict[str, Any], cfg_path: Path, *, incomplete: int, dry_run: bool) -> list[str]:
    actions: list[str] = []
    if incomplete <= 0:
        return actions
    if spec.get("auto_start") is False:
        actions.append(f"auto_start disabled for {cfg_path.name}")
        return actions
    if _snakemake_driver_running(cfg_path):
        actions.append(f"driver already running for {cfg_path.name}")
        return actions
    log = workflow_root() / str(spec["driver_log"])
    launcher = _launcher_script(str(spec["name"]))
    cmd = f"nohup bash scripts/{launcher} {int(spec['max_jobs'])} >> {log.name} 2>&1 &"
    actions.append(f"start driver: MMML_WORKFLOW_CONFIG={cfg_path.name} {cmd}")
    if not dry_run:
        env = _driver_subprocess_env(cfg_path)
        if launcher == "snakemake_local.sh":
            env["MMML_LOCAL_GPU_PIN"] = "1"
        subprocess.Popen(  # noqa: S603
            ["bash", f"scripts/{launcher}", str(spec["max_jobs"])],
            cwd=workflow_root(),
            env=env,
            stdout=open(log, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return actions


def run_monitor(*, react: bool, dry_run: bool) -> dict[str, Any]:
    dry_run = dry_run or not react
    tracker = RetryTracker(workflow_root() / "results" / "monitor_retries.json")
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
        driver_running = _snakemake_driver_running(cfg_path)
        camp_report: dict[str, Any] = {
            "name": spec["name"],
            "config": str(cfg_path),
            "driver_running": driver_running,
            "cells": [],
            "actions": [],
        }
        incomplete = 0
        for cell in iter_matrix_cells(cfg):
            health = _inspect_cell(cfg, cell, campaign=str(spec["name"]))
            all_cells.append(health)
            if health.status != "done":
                incomplete += 1
            if react:
                health.actions = _mediate_cell(
                    health,
                    cell,
                    cfg_path,
                    tracker,
                    driver_running=driver_running,
                    dry_run=dry_run,
                )
            camp_report["cells"].append(asdict(health))
        camp_report["driver_running"] = driver_running
        if react:
            camp_report["actions"] = _ensure_driver(
                spec, cfg_path, incomplete=incomplete, dry_run=dry_run
            )
        report["campaigns"].append(camp_report)

    n_done = sum(1 for c in all_cells if c.status == "done")
    n_fail = sum(1 for c in all_cells if c.status == "failed")
    n_run = sum(1 for c in all_cells if c.status == "running")
    n_pending = sum(1 for c in all_cells if c.status == "pending")
    report["summary"] = {
        "total": len(all_cells),
        "done": n_done,
        "running": n_run,
        "failed": n_fail,
        "manual": sum(1 for c in all_cells if c.status == "manual"),
        "pending": n_pending,
    }

    print(f"\n=== pbc_liquid_density_dyn monitor @ {_utc_now()} ===")
    print(
        f"Cells: {n_done}/{len(all_cells)} done | {n_run} running | "
        f"{n_fail} failed (mediated) | {n_pending} pending | "
        f"queue: {len(report['slurm_queue'])}"
    )
    print(f"{'campaign':<14} {'tag':<22} {'N':>4} {'L':>4} {'legs':>8} {'status':<10} failures")
    for camp in report["campaigns"]:
        for row in camp["cells"]:
            fails = ",".join(row.get("failure_ids") or []) or "-"
            print(
                f"{camp['name']:<14} {row['tag']:<22} {row['n_monomers']:4d} "
                f"{int(row['box_A']):4d} {row['legs_done']:>8} {row['status']:<10} {fails}"
            )
            for act in row.get("actions") or []:
                print(f"    -> {act}")
        drv = "driver=UP" if camp["driver_running"] else "driver=DOWN"
        print(f"  [{camp['name']}] {drv}")
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
            f"fail={n_fail} pending={n_pending} react={react}\n"
        )
    print(f"\nWrote {json_path}")
    print(f"Retry state: {tracker.path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--react",
        action="store_true",
        help="Apply mediations: driver restart, tier prebuild, reruns (fresh/resume)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show mediations without executing",
    )
    args = parser.parse_args()
    if not os.environ.get("MMML_CKPT", "").strip():
        print("WARNING: MMML_CKPT is unset — reruns will fail at runtime", file=sys.stderr)
    run_monitor(react=bool(args.react), dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

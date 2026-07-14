"""Core library for the MMML validation campaign.

Loads ``campaign.yaml`` and ``environments/*.yaml``, owns the on-disk proof
layout, renders job scripts, and resolves state *only* from proof receipts.

The central rule: a task is PASS only when every acceptance check declared in
``campaign.yaml`` appears in that task's ``proof.json`` and is true. A job that
was submitted, or that exited zero without writing the checks it promised, is
INCOMPLETE -- never PASS. Only ``exit_zero`` may be certified by the harness
itself; every other check must be asserted by a scientific driver.

Artifact layout (accumulating; runs never clobber each other):

    artifacts/validation_campaign/<run_id>/<environment>/<task/path>/
        request.json  provenance.json  status.json
        proof.json    metrics.json     stdout.log  stderr.log
        job.slurm | job.sh
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

WORKFLOW = Path(__file__).resolve().parents[1]
REPO = WORKFLOW.parents[1]
CAMPAIGN_FILE = WORKFLOW / "campaign.yaml"
ENV_DIR = WORKFLOW / "environments"

# States, most severe first: this ordering drives every rollup.
FAIL = "FAIL"
BLOCKED = "BLOCKED"
GATED = "GATED"
NEEDS_DRIVER = "NEEDS_DRIVER"
INCOMPLETE = "INCOMPLETE"
RUNNING = "RUNNING"
PASS = "PASS"

STATE_ORDER = [FAIL, BLOCKED, GATED, NEEDS_DRIVER, INCOMPLETE, RUNNING, PASS]

# The only check the harness may certify without a driver asserting it.
HARNESS_CHECKS = {"exit_zero"}


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def campaign() -> dict[str, Any]:
    return load_yaml(CAMPAIGN_FILE)


def environment(name: str) -> dict[str, Any]:
    path = ENV_DIR / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in ENV_DIR.glob("*.yaml")))
        raise SystemExit(f"unknown environment {name!r}; available: {known}")
    env = load_yaml(path)
    env.setdefault("name", name)
    return env


def new_run_id() -> str:
    return os.environ.get("MMML_VALIDATION_RUN_ID") or dt.datetime.now(dt.UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def artifact_root(cfg: dict[str, Any]) -> Path:
    return REPO / cfg.get("artifact_root", "artifacts/validation_campaign")


def task_path(task_id: str) -> str:
    return task_id.replace(".", "/")


def output_dir(cfg: dict[str, Any], run_id: str, task_id: str, env_name: str) -> Path:
    return artifact_root(cfg) / run_id / env_name / task_path(task_id)


def select_tasks(
    cfg: dict[str, Any], args: Any
) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for task_id, task in (cfg.get("tasks") or {}).items():
        env = getattr(args, "environment", None)
        if env and env not in task.get("environments", []):
            continue
        if getattr(args, "tier", None) and args.tier != task.get("tier"):
            continue
        if getattr(args, "goal", None) and args.goal != task.get("goal"):
            continue
        if getattr(args, "task", None) and args.task != task_id:
            continue
        rows.append((task_id, task))
    return sorted(rows, key=lambda kv: (kv[1].get("goal", ""), kv[0]))


def declared_state(task: dict[str, Any]) -> str:
    return task.get("state", "ready")


def is_runnable(task: dict[str, Any]) -> bool:
    """Blocked/gated/needs_driver tasks are catalogued but must not be dispatched."""
    return declared_state(task) == "ready"


def render_command(task: dict[str, Any], out: Path, *, python: str) -> str:
    return str(task["command"]).format(output_dir=shlex.quote(str(out)), python=python)


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO, text=True, capture_output=True, timeout=30
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def git_provenance() -> dict[str, Any]:
    diff = _git("diff", "HEAD") or ""
    return {
        "revision": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(diff.strip()),
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()
        if diff.strip()
        else None,
    }


def request_payload(
    task_id: str, task: dict[str, Any], env: dict[str, Any], run_id: str, command: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "task_id": task_id,
        "goal": task["goal"],
        "tier": task["tier"],
        "environment": env["name"],
        "systems": task.get("systems", []),
        "methods": task.get("methods", []),
        "backends": task.get("backends", []),
        "acceptance": task.get("acceptance", []),
        "command": command,
        "declared_state": declared_state(task),
        "blocker": task.get("blocker"),
        "requires": task.get("requires", []),
        "git": git_provenance(),
        "created_utc": utcnow(),
    }


def shell_exports(env: dict[str, Any]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in (env.get("environment") or {}).items()
    )


def runtime_provenance() -> dict[str, Any]:
    info: dict[str, Any] = {
        "generated_utc": utcnow(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": git_provenance(),
    }
    try:
        import jax  # noqa: PLC0415

        info["jax"] = {
            "version": jax.__version__,
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "default_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
        }
    except Exception as exc:
        info["jax"] = {"error": repr(exc), "x64_enabled": False}
    return info


# --------------------------------------------------------------------------
# Proof evaluation
# --------------------------------------------------------------------------


def _normalize_checks(proof: dict[str, Any]) -> dict[str, bool]:
    """Accept either {"checks": {name: bool}} or {"checks": [{name, passed}]}."""
    checks = proof.get("checks")
    if isinstance(checks, dict):
        return {str(k): bool(v) for k, v in checks.items()}
    if isinstance(checks, list):
        out = {}
        for entry in checks:
            if isinstance(entry, dict) and entry.get("name"):
                out[str(entry["name"])] = bool(entry.get("passed"))
        return out
    return {}


def find_receipts(
    cfg: dict[str, Any], task_id: str, env_name: str
) -> Path | None:
    """Newest run directory holding a status.json for this (task, environment)."""
    root = artifact_root(cfg)
    if not root.is_dir():
        return None
    candidates = [
        d
        for d in root.glob(f"*/{env_name}/{task_path(task_id)}")
        if (d / "status.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "status.json").stat().st_mtime)


def evaluate_unit(
    cfg: dict[str, Any], task_id: str, task: dict[str, Any], env_name: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": task_id,
        "goal": task["goal"],
        "tier": task["tier"],
        "environment": env_name,
        "acceptance": task.get("acceptance", []),
        "checks": {},
        "note": "",
        "receipt": None,
    }

    if declared_state(task) == "blocked":
        result["state"] = BLOCKED
        result["note"] = task.get("blocker", "declared blocked in campaign.yaml")
        return result

    out = find_receipts(cfg, task_id, env_name)
    if out is None:
        result["state"] = (
            NEEDS_DRIVER if declared_state(task) == "needs_driver" else INCOMPLETE
        )
        result["note"] = (
            "scientific driver not implemented"
            if result["state"] == NEEDS_DRIVER
            else "no receipt: never run in this environment"
        )
        return result

    result["receipt"] = str(out.relative_to(REPO))
    status = read_json(out / "status.json") or {}

    if str(status.get("state", "")).upper() == "NEEDS_DRIVER":
        result["state"] = NEEDS_DRIVER
        result["note"] = status.get("message", "driver not implemented")
        return result

    if status.get("state") in {"RUNNING", "SUBMITTED"}:
        result["state"] = RUNNING
        result["note"] = str(status.get("state")).lower()
        return result

    exit_code = status.get("exit_code")
    proof = read_json(out / "proof.json")
    result["metrics"] = read_json(out / "metrics.json") or {}

    checks = _normalize_checks(proof or {})
    result["checks"] = checks

    required = list(task.get("acceptance", []))
    missing = [c for c in required if c not in checks]
    failed = [c for c, ok in checks.items() if not ok]

    if failed:
        result["state"] = FAIL
        result["note"] = "failed checks: " + ", ".join(sorted(failed))
    elif missing:
        result["state"] = INCOMPLETE
        result["note"] = "missing proof for: " + ", ".join(missing)
    elif exit_code not in (0, None):
        result["state"] = FAIL
        result["note"] = f"nonzero exit_code={exit_code}"
    elif not required:
        result["state"] = INCOMPLETE
        result["note"] = "task declares no acceptance checks"
    else:
        result["state"] = PASS
    return result


def rollup(states: list[str]) -> str:
    for state in STATE_ORDER:
        if state in states:
            return state
    return INCOMPLETE


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    tasks = cfg.get("tasks") or {}
    units: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        for env_name in task.get("environments", []):
            units.append(evaluate_unit(cfg, task_id, task, env_name))

    task_states: dict[str, str] = {}
    for task_id, task in tasks.items():
        states = [u["state"] for u in units if u["task_id"] == task_id]
        task_states[task_id] = rollup(states) if states else INCOMPLETE

    # Gating: prerequisites that are not PASS hold a task at GATED, which is a
    # different statement from "nobody ran it".
    for task_id, task in tasks.items():
        requires = task.get("requires") or []
        if not requires or task_states[task_id] == PASS:
            continue
        unmet = [r for r in requires if task_states.get(r) != PASS]
        if unmet:
            task_states[task_id] = GATED
            detail = ", ".join(f"{r}={task_states.get(r, '?')}" for r in unmet)
            for unit in units:
                if unit["task_id"] == task_id and unit["state"] in {
                    INCOMPLETE,
                    NEEDS_DRIVER,
                }:
                    unit["state"] = GATED
                    unit["note"] = f"gated by {detail}"

    goals = cfg.get("goals") or {}
    goal_states = {
        goal: rollup([task_states[t] for t, v in tasks.items() if v["goal"] == goal])
        for goal in goals
    }

    return {
        "generated_utc": utcnow(),
        "git": git_provenance(),
        "overall": rollup(list(task_states.values())),
        "goals": {
            goal: {
                "state": goal_states[goal],
                "description": (goals[goal] or {}).get("description", ""),
            }
            for goal in goals
        },
        "tasks": {
            task_id: {
                "state": state,
                "goal": tasks[task_id]["goal"],
                "tier": tasks[task_id]["tier"],
                "blocker": tasks[task_id].get("blocker"),
            }
            for task_id, state in task_states.items()
        },
        "units": units,
    }

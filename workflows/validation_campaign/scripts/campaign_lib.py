"""Core library for the MMML validation campaign.

Loads ``campaign.yaml`` and ``environments/*.yaml``, expands task x environment
work units, owns the on-disk proof layout, and resolves task state *only* from
proof receipts. A submitted or exited job is never treated as success: the
acceptance checks declared in ``campaign.yaml`` must each appear in the task's
``proof.json`` and pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

WORKFLOW_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = WORKFLOW_DIR.parent.parent
CAMPAIGN_FILE = WORKFLOW_DIR / "campaign.yaml"
ENVIRONMENTS_DIR = WORKFLOW_DIR / "environments"

DEFAULT_RUN_ID = "current"

# Terminal states, most severe first. Ordering drives the summary rollup.
STATE_FAIL = "FAIL"
STATE_BLOCKED = "BLOCKED"
STATE_GATED = "GATED"
STATE_NEEDS_DRIVER = "NEEDS_DRIVER"
STATE_INCOMPLETE = "INCOMPLETE"
STATE_RUNNING = "RUNNING"
STATE_PASS = "PASS"

STATE_ORDER = [
    STATE_FAIL,
    STATE_BLOCKED,
    STATE_GATED,
    STATE_NEEDS_DRIVER,
    STATE_INCOMPLETE,
    STATE_RUNNING,
    STATE_PASS,
]

# Checks the harness can certify itself, without the driver asserting them.
# Everything else must be proven by the driver in proof.json.
HARNESS_CHECKS = {"exit_zero"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass
class Environment:
    name: str
    kind: str  # "slurm" | "local"
    role: str = ""
    repo_root: str = "."
    cpus: int = 1
    gpus: int = 0
    environment: dict[str, str] = field(default_factory=dict)
    # slurm-only
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int | None = None
    mem_mb_per_cpu: int | None = None
    runtime_min: int = 60
    gpu_constraint: str | None = None
    max_concurrent: int = 4
    launcher: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_slurm(self) -> bool:
        return self.kind == "slurm"


@dataclass
class Task:
    task_id: str
    goal: str
    tier: str
    environments: list[str]
    command: str
    acceptance: list[str]
    state: str | None = None  # declared: needs_driver | blocked | gated
    blocker: str | None = None
    requires: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Campaign:
    artifact_root: Path
    defaults: dict[str, Any]
    goals: dict[str, dict[str, Any]]
    tasks: dict[str, Task]


def load_environment(name: str) -> Environment:
    path = ENVIRONMENTS_DIR / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in ENVIRONMENTS_DIR.glob("*.yaml"))
        raise SystemExit(
            f"unknown environment {name!r}; available: {', '.join(available)}"
        )
    data = yaml.safe_load(path.read_text()) or {}
    known = {f for f in Environment.__dataclass_fields__ if f != "raw"}
    kwargs = {k: v for k, v in data.items() if k in known}
    kwargs.setdefault("name", name)
    return Environment(raw=data, **kwargs)


def all_environment_names() -> list[str]:
    return sorted(p.stem for p in ENVIRONMENTS_DIR.glob("*.yaml"))


def load_campaign() -> Campaign:
    data = yaml.safe_load(CAMPAIGN_FILE.read_text()) or {}
    tasks: dict[str, Task] = {}
    for task_id, spec in (data.get("tasks") or {}).items():
        known = {f for f in Task.__dataclass_fields__ if f not in {"raw", "task_id"}}
        kwargs = {k: v for k, v in spec.items() if k in known}
        tasks[task_id] = Task(task_id=task_id, raw=spec, **kwargs)
    return Campaign(
        artifact_root=REPO_ROOT / (data.get("artifact_root") or "artifacts/validation_campaign"),
        defaults=data.get("defaults") or {},
        goals=data.get("goals") or {},
        tasks=tasks,
    )


def select_tasks(
    campaign: Campaign,
    *,
    environment: str | None = None,
    tier: str | None = None,
    goal: str | None = None,
    task_ids: list[str] | None = None,
) -> list[Task]:
    out = []
    for task in campaign.tasks.values():
        if task_ids and task.task_id not in task_ids:
            continue
        if environment and environment not in task.environments:
            continue
        if tier and task.tier != tier:
            continue
        if goal and task.goal != goal:
            continue
        out.append(task)
    return sorted(out, key=lambda t: (t.goal, t.tier, t.task_id))


# --------------------------------------------------------------------------
# Artifact layout
# --------------------------------------------------------------------------


def output_dir(campaign: Campaign, run_id: str, task_id: str, environment: str) -> Path:
    """One task on one environment gets one proof directory."""
    return campaign.artifact_root / run_id / task_id / environment


def render_command(task: Task, campaign: Campaign, out_dir: Path) -> str:
    return task.command.format(
        output_dir=str(out_dir),
        repo_root=str(REPO_ROOT),
        task_id=task.task_id,
        checkpoint=str(campaign.defaults.get("checkpoint", "")),
        seed=str(campaign.defaults.get("seed", "")),
    )


def git_provenance() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    diff = _git("diff", "HEAD") or ""
    return {
        "revision": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(diff.strip()),
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff.strip() else None,
    }


def runtime_provenance() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "timestamp": utcnow(),
    }
    try:
        import jax  # noqa: PLC0415

        info["jax"] = jax.__version__
        info["jax_x64"] = bool(jax.config.read("jax_enable_x64"))
        info["jax_devices"] = [str(d) for d in jax.devices()]
        info["jax_backend"] = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        info["jax_error"] = f"{type(exc).__name__}: {exc}"

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            info["cuda_gpus"] = subprocess.run(
                [nvidia, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip().splitlines()
        except subprocess.CalledProcessError:
            pass

    info["charmm_lib_dir"] = os.environ.get("CHARMM_LIB_DIR")
    return info


def write_provenance(out_dir: Path, environment: Environment) -> dict[str, Any]:
    prov = {
        "git": git_provenance(),
        "runtime": runtime_provenance(),
        "environment": environment.name,
        "environment_kind": environment.kind,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    return prov


def write_request(
    out_dir: Path, task: Task, campaign: Campaign, environment: Environment, command: str
) -> dict[str, Any]:
    """Immutable record of what was asked for, written before the work starts."""
    request = {
        "task_id": task.task_id,
        "goal": task.goal,
        "tier": task.tier,
        "environment": environment.name,
        "command": command,
        "acceptance": task.acceptance,
        "declared_state": task.state,
        "blocker": task.blocker,
        "requires": task.requires,
        "systems": task.systems,
        "methods": task.methods,
        "backends": task.backends,
        "checkpoint": campaign.defaults.get("checkpoint"),
        "seed": campaign.defaults.get("seed"),
        "require_x64": campaign.defaults.get("require_x64", True),
        "created": utcnow(),
    }
    path = out_dir / "request.json"
    if not path.exists():
        path.write_text(json.dumps(request, indent=2) + "\n")
    return request


def write_status(out_dir: Path, **fields: Any) -> None:
    status = {"updated": utcnow(), **fields}
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# Proof evaluation
# --------------------------------------------------------------------------


def evaluate_unit(
    campaign: Campaign, task: Task, environment: str, run_id: str
) -> dict[str, Any]:
    """Resolve one (task, environment) unit's state purely from its receipts."""
    out_dir = output_dir(campaign, run_id, task.task_id, environment)
    status = read_json(out_dir / "status.json")
    proof = read_json(out_dir / "proof.json")
    metrics = read_json(out_dir / "metrics.json")

    result: dict[str, Any] = {
        "task_id": task.task_id,
        "goal": task.goal,
        "tier": task.tier,
        "environment": environment,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)) if out_dir.exists() else None,
        "acceptance": task.acceptance,
        "checks": {},
        "metrics": metrics or {},
        "blocker": task.blocker,
        "notes": [],
    }

    # Declared blockers/gates stay visible even if a stale receipt exists.
    if task.state == "blocked":
        result["state"] = STATE_BLOCKED
        result["notes"].append(task.blocker or "declared blocked in campaign.yaml")
        return result

    if not out_dir.exists() or status is None:
        result["state"] = STATE_INCOMPLETE
        result["notes"].append("no receipt: task has not run in this environment")
        if task.state == "needs_driver":
            result["state"] = STATE_NEEDS_DRIVER
            result["notes"] = ["driver not implemented (state: needs_driver)"]
        return result

    phase = status.get("phase")
    if phase in {"submitted", "running"}:
        result["state"] = STATE_RUNNING
        result["notes"].append(f"phase={phase}")
        return result

    if status.get("state") == "needs_driver":
        result["state"] = STATE_NEEDS_DRIVER
        result["notes"].append(status.get("detail") or "driver not implemented")
        return result

    exit_code = status.get("exit_code")

    # Harness-certifiable checks, derived from the runner's own observation.
    checks: dict[str, dict[str, Any]] = {}
    if "exit_zero" in task.acceptance:
        checks["exit_zero"] = {
            "passed": exit_code == 0,
            "detail": f"exit_code={exit_code}",
            "source": "harness",
        }

    # Driver-asserted checks come only from proof.json.
    for entry in (proof or {}).get("checks", []) or []:
        name = entry.get("name")
        if not name:
            continue
        checks[name] = {
            "passed": bool(entry.get("passed")),
            "detail": entry.get("detail"),
            "source": "proof.json",
        }

    result["checks"] = checks

    missing = [
        name
        for name in task.acceptance
        if name not in checks and name not in HARNESS_CHECKS
    ]
    failed = [name for name, c in checks.items() if not c["passed"]]

    if failed:
        result["state"] = STATE_FAIL
        result["notes"].append("failed checks: " + ", ".join(sorted(failed)))
    elif missing:
        result["state"] = STATE_INCOMPLETE
        result["notes"].append("missing proof for: " + ", ".join(missing))
    elif exit_code not in (0, None):
        result["state"] = STATE_FAIL
        result["notes"].append(f"nonzero exit_code={exit_code}")
    else:
        result["state"] = STATE_PASS

    return result


def _rollup(states: list[str]) -> str:
    for state in STATE_ORDER:
        if state in states:
            return state
    return STATE_INCOMPLETE


def evaluate_campaign(campaign: Campaign, run_id: str) -> dict[str, Any]:
    """Full campaign state. Gating is applied after per-unit evaluation."""
    units: list[dict[str, Any]] = []
    for task in sorted(campaign.tasks.values(), key=lambda t: (t.goal, t.task_id)):
        for env_name in task.environments:
            units.append(evaluate_unit(campaign, task, env_name, run_id))

    # A task's own state is the rollup over the environments it targets.
    task_states: dict[str, str] = {}
    for task in campaign.tasks.values():
        states = [u["state"] for u in units if u["task_id"] == task.task_id]
        task_states[task.task_id] = _rollup(states) if states else STATE_INCOMPLETE

    # Apply gating: a task whose prerequisites are not PASS is GATED, not merely
    # incomplete. This keeps "blocked upstream" distinct from "nobody ran it".
    for task in campaign.tasks.values():
        if not task.requires:
            continue
        unmet = [r for r in task.requires if task_states.get(r) != STATE_PASS]
        if unmet and task_states[task.task_id] != STATE_PASS:
            task_states[task.task_id] = STATE_GATED
            for unit in units:
                if unit["task_id"] == task.task_id and unit["state"] in {
                    STATE_INCOMPLETE,
                    STATE_NEEDS_DRIVER,
                }:
                    unit["state"] = STATE_GATED
                    unit["notes"].append(
                        "gated by: " + ", ".join(f"{r}={task_states.get(r)}" for r in unmet)
                    )

    goal_states: dict[str, str] = {}
    for goal in campaign.goals:
        states = [
            task_states[t.task_id]
            for t in campaign.tasks.values()
            if t.goal == goal
        ]
        goal_states[goal] = _rollup(states) if states else STATE_INCOMPLETE

    return {
        "run_id": run_id,
        "generated": utcnow(),
        "git": git_provenance(),
        "goals": {
            goal: {
                "state": goal_states[goal],
                "description": (campaign.goals[goal] or {}).get("description", ""),
            }
            for goal in campaign.goals
        },
        "tasks": {
            task_id: {
                "state": state,
                "goal": campaign.tasks[task_id].goal,
                "tier": campaign.tasks[task_id].tier,
                "blocker": campaign.tasks[task_id].blocker,
            }
            for task_id, state in task_states.items()
        },
        "units": units,
        "overall": _rollup(list(task_states.values())),
    }

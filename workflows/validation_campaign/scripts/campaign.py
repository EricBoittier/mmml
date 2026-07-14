#!/usr/bin/env python3
"""Prepare, submit, run, and summarize the MMML validation campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


WORKFLOW = Path(__file__).resolve().parents[1]
REPO = WORKFLOW.parents[1]
CAMPAIGN_FILE = WORKFLOW / "campaign.yaml"
ENV_DIR = WORKFLOW / "environments"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def campaign() -> dict[str, Any]:
    return load_yaml(CAMPAIGN_FILE)


def environment(name: str) -> dict[str, Any]:
    path = ENV_DIR / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"unknown environment {name!r}; expected {path}")
    return load_yaml(path)


def selected_tasks(cfg: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for task_id, task in cfg["tasks"].items():
        if getattr(args, "environment", None) and args.environment not in task.get("environments", []):
            continue
        if getattr(args, "tier", None) and args.tier != task.get("tier"):
            continue
        if getattr(args, "goal", None) and args.goal != task.get("goal"):
            continue
        if getattr(args, "task", None) and args.task != task_id:
            continue
        rows.append((task_id, task))
    return rows


def run_id() -> str:
    return os.environ.get("MMML_VALIDATION_RUN_ID") or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def artifact_root(cfg: dict[str, Any]) -> Path:
    return REPO / cfg.get("artifact_root", "artifacts/validation_campaign")


def output_dir(cfg: dict[str, Any], rid: str, task_id: str, env_name: str) -> Path:
    return artifact_root(cfg) / rid / env_name / task_id.replace(".", "/")


def relative_output_dir(cfg: dict[str, Any], rid: str, task_id: str, env_name: str) -> Path:
    return Path(cfg.get("artifact_root", "artifacts/validation_campaign")) / rid / env_name / task_id.replace(".", "/")


def task_command(task: dict[str, Any], out: Path, *, python: str) -> str:
    return str(task["command"]).format(
        output_dir=shlex.quote(str(out)), python=python
    )


def request_payload(task_id: str, task: dict[str, Any], env: dict[str, Any], rid: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": rid,
        "task_id": task_id,
        "goal": task["goal"],
        "tier": task["tier"],
        "environment": env["name"],
        "systems": task.get("systems", []),
        "methods": task.get("methods", []),
        "backends": task.get("backends", []),
        "acceptance": task.get("acceptance", []),
        "command": task["command"],
        "declared_state": task.get("state", "ready"),
        "blocker": task.get("blocker"),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shell_exports(env: dict[str, Any]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in env.get("environment", {}).items()
    )


def slurm_script(task_id: str, task: dict[str, Any], env: dict[str, Any], out: Path) -> str:
    runtime = int(env.get("runtime_min", 120))
    hours, minutes = divmod(runtime, 60)
    directives = [
        f"#SBATCH --job-name=val-{task_id.replace('.', '-')[:90]}",
        f"#SBATCH --partition={env['partition']}",
        f"#SBATCH --nodes={int(env.get('nodes', 1))}",
        f"#SBATCH --ntasks={int(env.get('ntasks', 1))}",
        f"#SBATCH --cpus-per-task={int(env.get('cpus_per_task', 4))}",
        f"#SBATCH --mem-per-cpu={int(env.get('mem_mb_per_cpu', 4000))}M",
        f"#SBATCH --time={hours:02d}:{minutes:02d}:00",
        f"#SBATCH --output={out / 'slurm-%j.out'}",
        f"#SBATCH --error={out / 'slurm-%j.err'}",
    ]
    if int(env.get("gpus", 0)) > 0:
        directives.append(f"#SBATCH --gpus={int(env['gpus'])}")
    for key, flag in (("account", "account"), ("qos", "qos"), ("gpu_constraint", "constraint")):
        if env.get(key):
            directives.append(f"#SBATCH --{flag}={env[key]}")
    command = task_command(task, out, python='"$MMML_PYTHON"')
    repo_root = str(env.get("repo_root", "~/mmml"))
    repo_shell = "$HOME/" + repo_root[2:] if repo_root.startswith("~/") else shlex.quote(repo_root)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "set -euo pipefail",
            f"cd {repo_shell}",
            'source scripts/resolve_mmml_env.sh',
            'mmml_resolve_env "$PWD"',
            shell_exports(env),
            f"mkdir -p {shlex.quote(str(out))}",
            "set +e",
            f"{command} > {shlex.quote(str(out / 'stdout.log'))} 2> {shlex.quote(str(out / 'stderr.log'))}",
            "task_rc=$?",
            "set -e",
            f'"$MMML_PYTHON" workflows/validation_campaign/scripts/finalize_task.py --output-dir {shlex.quote(str(out))} --exit-code "$task_rc"',
            "exit $task_rc",
        ]
    ) + "\n"


def prepare(args: argparse.Namespace, *, submit: bool) -> int:
    cfg = campaign()
    env = environment(args.environment)
    if env.get("kind") != "slurm":
        raise SystemExit(f"{args.environment} is not a Slurm environment")
    rid = args.run_id or run_id()
    count = 0
    for task_id, task in selected_tasks(cfg, args):
        state = task.get("state", "ready")
        if state in {"blocked", "gated", "needs_driver"} and not args.include_not_ready:
            print(f"SKIP {task_id}: {state} — {task.get('blocker', '')}")
            continue
        out = output_dir(cfg, rid, task_id, args.environment)
        write_json(out / "request.json", request_payload(task_id, task, env, rid))
        script = out / "job.slurm"
        job_out = relative_output_dir(cfg, rid, task_id, args.environment)
        script.write_text(slurm_script(task_id, task, env, job_out), encoding="utf-8")
        script.chmod(0o755)
        print(f"PREPARED {task_id}: {script}")
        if submit:
            proc = subprocess.run(["sbatch", str(script)], text=True, capture_output=True)
            write_json(out / "submission.json", {
                "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()
            })
            if proc.returncode != 0:
                print(f"SUBMIT FAILED {task_id}: {proc.stderr.strip()}", file=sys.stderr)
                return proc.returncode
            print(f"SUBMITTED {task_id}: {proc.stdout.strip()}")
        count += 1
    print(f"run_id={rid} tasks={count}")
    return 0


def run_local(args: argparse.Namespace) -> int:
    cfg = campaign()
    env = environment(args.environment)
    if env.get("kind") != "local":
        raise SystemExit(f"{args.environment} is not a local environment")
    rid = args.run_id or run_id()
    failures = 0
    for task_id, task in selected_tasks(cfg, args):
        state = task.get("state", "ready")
        if state in {"blocked", "gated", "needs_driver"} and not args.include_not_ready:
            print(f"SKIP {task_id}: {state}")
            continue
        out = output_dir(cfg, rid, task_id, args.environment)
        write_json(out / "request.json", request_payload(task_id, task, env, rid))
        merged = os.environ.copy()
        merged.update({str(k): str(v) for k, v in env.get("environment", {}).items()})
        started = dt.datetime.now(dt.UTC)
        with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
            proc = subprocess.run(
                task_command(task, out, python=shlex.quote(sys.executable)),
                cwd=REPO, env=merged, shell=True, stdout=stdout, stderr=stderr
            )
        write_json(out / "status.json", {
            "state": "COMPLETED" if proc.returncode == 0 else "FAILED",
            "exit_code": proc.returncode,
            "started_utc": started.isoformat(),
            "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
        })
        if not (out / "proof.json").exists() and task.get("acceptance") == ["exit_zero"]:
            write_json(out / "proof.json", {
                "passed": proc.returncode == 0,
                "checks": {"exit_zero": proc.returncode == 0},
                "sources": ["stdout.log", "stderr.log", "status.json"],
            })
        failures += proc.returncode != 0
        print(f"{'PASS' if proc.returncode == 0 else 'FAIL'} {task_id}: {out}")
    return int(bool(failures))


def proof_state(task: dict[str, Any], roots: list[Path]) -> tuple[str, str]:
    declared = task.get("state", "ready")
    if declared == "blocked":
        return "BLOCKED", str(task.get("blocker", ""))
    receipts = [root for root in roots if (root / "proof.json").is_file()]
    if not receipts:
        if declared in {"gated", "needs_driver"}:
            return declared.upper(), ""
        return "INCOMPLETE", "no proof.json receipt"
    newest = max(receipts, key=lambda p: (p / "proof.json").stat().st_mtime)
    proof = json.loads((newest / "proof.json").read_text())
    return ("PASS" if proof.get("passed") is True else "FAIL"), str(newest)


def status(args: argparse.Namespace) -> int:
    cfg = campaign()
    root = artifact_root(cfg)
    rows = []
    for task_id, task in cfg["tasks"].items():
        candidates = list(root.glob(f"*/*/{task_id.replace('.', '/') }")) if root.exists() else []
        state, detail = proof_state(task, candidates)
        rows.append({"task_id": task_id, "goal": task["goal"], "tier": task["tier"], "state": state, "detail": detail})
        print(f"{state:12s} {task_id:45s} {detail}")
    if args.write:
        payload = {"generated_utc": dt.datetime.now(dt.UTC).isoformat(), "tasks": rows}
        write_json(root / "summary.json", payload)
        lines = ["# Validation campaign status", "", "| State | Goal | Task | Detail |", "|---|---|---|---|"]
        lines.extend(f"| {r['state']} | {r['goal']} | `{r['task_id']}` | {r['detail']} |" for r in rows)
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"WROTE {root / 'summary.json'}")
    return int(any(r["state"] == "FAIL" for r in rows))


def list_tasks(args: argparse.Namespace) -> int:
    for task_id, task in selected_tasks(campaign(), args):
        print(f"{task_id:45s} goal={task['goal']:18s} tier={task['tier']:10s} state={task.get('state', 'ready')}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("list", "status", "prepare", "submit", "run-local"):
        q = sub.add_parser(name)
        q.add_argument("--environment")
        q.add_argument("--tier")
        q.add_argument("--goal")
        q.add_argument("--task")
        q.add_argument("--run-id")
        q.add_argument("--include-not-ready", action="store_true")
        if name == "status":
            q.add_argument("--write", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.action in {"prepare", "submit", "run-local"} and not args.environment:
        raise SystemExit(f"{args.action} requires --environment")
    if args.action == "list":
        return list_tasks(args)
    if args.action == "status":
        return status(args)
    if args.action == "prepare":
        return prepare(args, submit=False)
    if args.action == "submit":
        return prepare(args, submit=True)
    return run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())

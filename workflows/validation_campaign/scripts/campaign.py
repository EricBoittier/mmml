#!/usr/bin/env python3
"""Prepare, submit, run, and summarize the MMML validation campaign.

    campaign.py list      [--environment E] [--tier T] [--goal G]
    campaign.py status    [--write] [--verbose]
    campaign.py prepare   --environment E [filters]     # render scripts, do not run
    campaign.py submit    --environment E [filters]     # sbatch (cluster envs)
    campaign.py run-local --environment E [filters]     # run in foreground (local envs)

Status is derived only from proof receipts. A submitted job, or one that exited
zero without writing the acceptance checks it promised, is reported INCOMPLETE --
never PASS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_lib as lib  # noqa: E402
import run_local  # noqa: E402
import submit_slurm  # noqa: E402


def _resolve(args: argparse.Namespace):
    cfg = lib.campaign()
    env = lib.environment(args.environment)
    rows = lib.select_tasks(cfg, args)
    if not rows:
        raise SystemExit(
            f"no tasks target environment {args.environment!r} with the given filters"
        )
    return cfg, env, rows


def _dispatchable(rows, include_not_ready: bool):
    ready, skipped = [], []
    for task_id, task in rows:
        if lib.is_runnable(task) or include_not_ready:
            ready.append((task_id, task))
        else:
            skipped.append((task_id, task))
    return ready, skipped


def cmd_list(args: argparse.Namespace) -> int:
    cfg = lib.campaign()
    rows = lib.select_tasks(cfg, args)
    if not rows:
        print("no tasks match the given filters")
        return 1

    goal = None
    for task_id, task in rows:
        if task["goal"] != goal:
            goal = task["goal"]
            desc = (cfg["goals"].get(goal) or {}).get("description", "")
            print(f"\n{goal}  --  {desc}")
        print(
            f"  {task_id:<40} tier={task['tier']:<11} state={lib.declared_state(task)}"
        )
        print(f"    environments: {', '.join(task.get('environments', []))}")
        print(f"    acceptance:   {', '.join(task.get('acceptance', []))}")
        if task.get("systems"):
            print(
                f"    matrix:       systems={task['systems']} "
                f"methods={task.get('methods', [])} backends={task.get('backends', [])}"
            )
        if task.get("blocker"):
            print(f"    BLOCKER:      {task['blocker']}")
    print()
    return 0


def _summary_markdown(report: dict) -> str:
    git = report["git"]
    rev = (git.get("revision") or "unknown")[:12]
    dirty = "  **(dirty working tree)**" if git.get("dirty") else ""

    lines = [
        "# MMML validation campaign -- proof-of-work summary",
        "",
        f"- generated: `{report['generated_utc']}`",
        f"- revision: `{rev}`{dirty}",
        f"- **overall: {report['overall']}**",
        "",
        "State is derived **only** from `proof.json` receipts written by a driver.",
        "`INCOMPLETE` means the proof is missing, not that the science failed.",
        "`NEEDS_DRIVER` means the task is catalogued but its driver is not built yet.",
        "",
        "| State | Meaning |",
        "|---|---|",
        "| `PASS` | every declared acceptance check is present and true |",
        "| `FAIL` | a declared check ran and was false |",
        "| `BLOCKED` | a known defect prevents the task from running |",
        "| `GATED` | a prerequisite task has not passed |",
        "| `NEEDS_DRIVER` | no scientific driver implemented yet |",
        "| `INCOMPLETE` | no receipt, or proof missing for a declared check |",
        "",
        "## Goals",
        "",
        "| Goal | State | Description |",
        "|---|---|---|",
    ]
    for goal, info in report["goals"].items():
        lines.append(f"| `{goal}` | **{info['state']}** | {info['description']} |")

    lines += [
        "",
        "## Tasks",
        "",
        "| Task | Goal | Tier | State | Note |",
        "|---|---|---|---|---|",
    ]
    for task_id, info in report["tasks"].items():
        lines.append(
            f"| `{task_id}` | {info['goal']} | {info['tier']} | **{info['state']}** | "
            f"{info.get('blocker') or ''} |"
        )

    lines += [
        "",
        "## Units (task x environment)",
        "",
        "| Task | Environment | State | Note | Receipt |",
        "|---|---|---|---|---|",
    ]
    for unit in report["units"]:
        receipt = f"`{unit['receipt']}`" if unit.get("receipt") else ""
        lines.append(
            f"| `{unit['task_id']}` | {unit['environment']} | {unit['state']} | "
            f"{unit['note']} | {receipt} |"
        )
    return "\n".join(lines) + "\n"


def cmd_status(args: argparse.Namespace) -> int:
    cfg = lib.campaign()
    report = lib.evaluate(cfg)

    print(f"\nMMML validation campaign -- overall: {report['overall']}\n")
    print("GOALS")
    for goal, info in report["goals"].items():
        print(f"  {info['state']:<13} {goal}")

    print("\nTASKS")
    for task_id, info in report["tasks"].items():
        print(f"  {info['state']:<13} {task_id}")
        if info["state"] == lib.BLOCKED and info.get("blocker"):
            print(f"                -> {info['blocker']}")

    if args.verbose:
        print("\nUNITS")
        for unit in report["units"]:
            note = f"  ({unit['note']})" if unit["note"] else ""
            print(
                f"  {unit['state']:<13} {unit['task_id']} @ {unit['environment']}{note}"
            )

    if args.write:
        root = lib.artifact_root(cfg)
        lib.write_json(root / "summary.json", report)
        (root / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
        rel = root.relative_to(lib.REPO)
        print(f"\nwrote {rel}/summary.json")
        print(f"wrote {rel}/summary.md")

    print()
    # Only a genuine FAIL is a nonzero exit; unproven work is not a test failure.
    return 1 if report["overall"] == lib.FAIL else 0


def cmd_prepare(args: argparse.Namespace, *, submit: bool) -> int:
    cfg, env, rows = _resolve(args)
    if env.get("kind") != "slurm":
        raise SystemExit(
            f"{args.environment} is a local environment; use 'run-local' instead."
        )

    ready, skipped = _dispatchable(rows, args.include_not_ready)
    for task_id, task in skipped:
        state = lib.declared_state(task)
        detail = task.get("blocker") or "no scientific driver implemented yet"
        print(f"SKIP  {task_id}  ({state}: {detail})")

    if not ready:
        print("\nnothing dispatchable. Use --include-not-ready to render anyway.")
        return 1

    run_id = args.run_id or lib.new_run_id()
    for task_id, task in ready:
        path = submit_slurm.render(cfg, task_id, task, env, run_id)
        print(f"PREPARED  {task_id:<40} {path.relative_to(lib.REPO)}")
        if submit:
            job_id = submit_slurm.submit(cfg, task_id, task, env, run_id)
            print(f"SUBMITTED {task_id:<40} job {job_id}")

    print(f"\nrun_id={run_id}  tasks={len(ready)}  environment={env['name']}")
    if not submit:
        print("Inspect the scripts above, then re-run with 'submit'.")
    return 0


def cmd_run_local(args: argparse.Namespace) -> int:
    cfg, env, rows = _resolve(args)
    if env.get("kind") != "local":
        raise SystemExit(
            f"{args.environment} is a Slurm environment; use 'submit' instead."
        )

    ready, skipped = _dispatchable(rows, args.include_not_ready)
    for task_id, task in skipped:
        state = lib.declared_state(task)
        detail = task.get("blocker") or "no scientific driver implemented yet"
        print(f"SKIP  {task_id}  ({state}: {detail})")

    if not ready:
        print("\nnothing dispatchable. Use --include-not-ready to run anyway.")
        return 1

    run_id = args.run_id or lib.new_run_id()
    failures = 0
    for task_id, task in ready:
        print(f"\n=== {task_id} @ {env['name']} ===")
        code = run_local.run(cfg, task_id, task, env, run_id)
        failures += code != 0
        print(f"    exit={code}")

    print(f"\nrun_id={run_id}  tasks={len(ready)}  nonzero exits={failures}")
    print("Exit status is not proof -- run 'campaign.py status' to see what was proven.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campaign.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("list", "status", "prepare", "submit", "run-local"):
        p = sub.add_parser(name)
        p.add_argument("--environment")
        p.add_argument("--tier", help="static | smoke | production | report | blocker")
        p.add_argument("--goal")
        p.add_argument("--task")
        p.add_argument("--run-id")
        p.add_argument(
            "--include-not-ready",
            action="store_true",
            help="also dispatch blocked/gated/needs_driver tasks",
        )
        if name == "status":
            p.add_argument("--write", action="store_true")
            p.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action in {"prepare", "submit", "run-local"} and not args.environment:
        raise SystemExit(f"{args.action} requires --environment")

    if args.action == "list":
        return cmd_list(args)
    if args.action == "status":
        return cmd_status(args)
    if args.action == "prepare":
        return cmd_prepare(args, submit=False)
    if args.action == "submit":
        return cmd_prepare(args, submit=True)
    return cmd_run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())

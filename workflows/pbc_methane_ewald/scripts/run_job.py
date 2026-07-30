#!/usr/bin/env python3
"""Run one liquid-methane Ewald campaign cell via mmml md-system --run-all."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    build_md_system_campaign_argv,
    campaign_job_order,
    cell_from_tag,
    cell_run_tag,
    load_config,
    paths_for_run,
    resolve_checkpoint_path,
    validate_checkpoint,
    workflow_root,
)


def _repo_root() -> Path:
    return workflow_root().parents[1]


def _resolve_mmml_cmd(md_argv: list[str]) -> list[str]:
    py = os.environ.get("MMML_PYTHON", sys.executable)
    return [py, "-m", "mmml.cli.__main__", "md-system", *md_argv]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Run cell tag")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Workflow config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cell = cell_from_tag(cfg, args.tag)
    ckpt = resolve_checkpoint_path(cell.checkpoint)
    validate_checkpoint(ckpt)

    paths = paths_for_run(cfg, cell)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    md_argv = build_md_system_campaign_argv(cfg, cell, out_dir=paths["out_dir"])
    os.chdir(_repo_root())
    cmd = _resolve_mmml_cmd(md_argv)

    tag = cell_run_tag(cell, cfg)
    campaign = yaml.safe_load(paths["campaign_yaml"].read_text(encoding="utf-8"))
    defaults = campaign.get("defaults", {})
    print(
        f"methane ewald cell: T={cell.temperature} backend={cell.backend} "
        f"ckpt={cell.checkpoint_slug} lr_solver={defaults.get('lr_solver')} "
        f"mm_nonbond_mode={defaults.get('mm_nonbond_mode')}",
        flush=True,
    )
    print(f"Campaign jobs ({tag}): {campaign_job_order(cfg, cell)}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)

    summary_path = paths["campaign_summary"]
    summary_ok = False
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            jobs = payload.get("jobs", payload if isinstance(payload, list) else [])
            failed = [j.get("job_id") for j in jobs if int(j.get("exit_code", 0)) != 0]
            if failed:
                print(f"Campaign summary reports failed legs: {failed}", file=sys.stderr)
                return 1
            summary_ok = bool(jobs)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"Could not parse {summary_path}: {exc}", file=sys.stderr)
            if rc != 0:
                print(f"{tag} methane ewald campaign failed with exit code {rc}", file=sys.stderr)
                return rc
            return 1
    elif rc == 0:
        print(f"Warning: missing campaign summary {summary_path}", flush=True)

    if rc != 0:
        if summary_ok and paths["final_handoff"].is_file():
            print(
                f"WARN: launcher exit {rc} but campaign legs OK and handoff present; "
                "treating as success",
                flush=True,
            )
        else:
            print(f"{tag} methane ewald campaign failed with exit code {rc}", file=sys.stderr)
            return rc

    if not paths["final_handoff"].is_file():
        print(f"Expected final handoff missing: {paths['final_handoff']}", file=sys.stderr)
        return 1

    paths["done"].write_text(f"ok {tag}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

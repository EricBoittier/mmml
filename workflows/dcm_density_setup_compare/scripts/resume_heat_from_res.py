#!/usr/bin/env python3
"""Resume MLpot heat from an on-disk CHARMM ``.res`` (skip mini / prep ladder)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    build_heat_resume_campaign,
    build_heat_resume_md_argv,
    cell_from_tag,
    cell_run_tag,
    config_for_run_tag,
    discover_heat_resume_restart,
    init_job_id,
    load_config,
    paths_for_run,
    resolve_checkpoint,
    workflow_root,
)


def _repo_root() -> Path:
    return workflow_root().parents[1]


def _resolve_mmml_cmd(md_argv: list[str]) -> list[str]:
    py = os.environ.get("MMML_PYTHON", sys.executable)
    return [py, "-m", "mmml.cli.__main__", "md-system", *md_argv]


def _resolve_restart(
    leg_dir: Path,
    tag: str,
    *,
    explicit: str | None,
    n_heat_segments: int,
) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--restart-from not found: {path}")
        return path
    found = discover_heat_resume_restart(
        leg_dir,
        tag,
        n_heat_segments=n_heat_segments,
    )
    if found is None:
        raise FileNotFoundError(
            f"No valid .res under {leg_dir}. "
            "Pass --restart-from PATH or inspect heat.res / heat.NNNN.res."
        )
    return found.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resume MLpot heat from an existing CHARMM restart under a failed "
            "pycharmm_mini leg (skips mini, liquid prep, and pretreat)."
        )
    )
    parser.add_argument("--tag", required=True, help="Run cell tag")
    parser.add_argument(
        "--restart-from",
        default=None,
        help="CHARMM .res path (default: auto-discover best under the PyCHARMM leg)",
    )
    parser.add_argument(
        "--leg",
        default=None,
        help="Campaign leg id (default: pycharmm_mini or pycharmm_init)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Workflow config YAML (default: tag-matched config like job_shell.sh)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print campaign YAML and md-system command without running",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        from campaign_lib import default_workflow_config_path

        cfg_path = default_workflow_config_path(run_tag=args.tag)
        if not cfg_path.is_absolute():
            cfg_path = workflow_root() / cfg_path

    cfg = load_config(cfg_path)
    cfg = config_for_run_tag(cfg, args.tag)
    cell = cell_from_tag(cfg, args.tag)
    resolve_checkpoint(str(cfg["checkpoint"]))

    paths = paths_for_run(cfg, cell)
    leg_id = args.leg or init_job_id(cell)
    leg_dir = paths["out_dir"] / leg_id
    if not leg_dir.is_dir():
        print(f"PyCHARMM leg directory missing: {leg_dir}", file=sys.stderr)
        return 1

    tag = cell_run_tag(cell, cfg)
    n_heat_segments = max(1, int(cfg.get("n_heat_segments", 1)))
    try:
        restart_path = _resolve_restart(
            leg_dir,
            tag,
            explicit=args.restart_from,
            n_heat_segments=n_heat_segments,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import model_psf

    psf = model_psf(leg_dir)
    if not psf.is_file():
        print(f"Missing topology for skip-cluster-build: {psf}", file=sys.stderr)
        return 1

    campaign = build_heat_resume_campaign(
        cfg,
        cell,
        restart_path=restart_path,
        leg_id=leg_id,
    )
    md_argv = build_heat_resume_md_argv(
        cfg,
        cell,
        restart_path=restart_path,
        leg_id=leg_id,
        out_dir=paths["out_dir"],
    )
    cmd = _resolve_mmml_cmd(md_argv)

    print(f"=== heat resume: {tag} ===", flush=True)
    print(f"leg_dir={leg_dir}", flush=True)
    print(f"restart={restart_path}", flush=True)
    print(f"psf={psf}", flush=True)
    print(f"campaign={paths['out_dir'] / 'resume_heat_campaign.yaml'}", flush=True)
    print(yaml.safe_dump(campaign, sort_keys=False, default_flow_style=False), flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)

    if args.dry_run:
        return 0

    os.chdir(_repo_root())
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"heat resume failed with exit code {rc}", file=sys.stderr)
        return rc

    handoff = paths["final_handoff"]
    if handoff.is_file():
        paths["done"].write_text(f"ok {tag} (heat resume)\n", encoding="utf-8")
        print(f"Handoff OK: {handoff}", flush=True)
        return 0

    leg_handoff = leg_dir / "handoff" / "state.npz"
    if leg_handoff.is_file():
        print(f"Leg handoff OK: {leg_handoff}", flush=True)
        return 0

    print(
        f"Heat run finished but expected handoff missing: {handoff}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one DCM density × setup campaign via mmml md-system --run-all."""

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
    cell_from_cli,
    cell_from_tag,
    cell_run_tag,
    cell_workflow_cfg,
    config_for_run_tag,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Run cell tag, e.g. minimal_dcm_77_t300_l32")
    parser.add_argument("setup_id", nargs="?", help="Setup variant id")
    parser.add_argument("solvent", nargs="?", help="Residue prefix (DCM)")
    parser.add_argument("n_monomers", nargs="?", type=int, help="Monomer count")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (K)")
    parser.add_argument("--box-size", type=float, default=None, help="Cubic box side (Å)")
    parser.add_argument(
        "--heat-thermostat",
        default=None,
        help="Heat thermostat (bussi, hoover, scale) when matrix lists multiple",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Workflow config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = config_for_run_tag(cfg, args.tag or "")
    if args.tag:
        cell = cell_from_tag(cfg, args.tag)
    else:
        if args.setup_id is None or args.solvent is None or args.n_monomers is None:
            parser.error("Provide --tag or SETUP_ID SOLVENT N_MONOMERS")
        cell = cell_from_cli(
            cfg,
            args.setup_id,
            args.solvent,
            args.n_monomers,
            temperature=args.temperature,
            box_size=args.box_size,
            heat_thermostat=args.heat_thermostat,
        )

    resolve_checkpoint(str(cfg["checkpoint"]))
    paths = paths_for_run(cfg, cell)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    md_argv = build_md_system_campaign_argv(cfg, cell, out_dir=paths["out_dir"])
    os.chdir(_repo_root())
    cmd = _resolve_mmml_cmd(md_argv)

    tag = cell_run_tag(cell, cfg)
    cell_cfg = cell_workflow_cfg(cfg, cell)
    campaign = yaml.safe_load(paths["campaign_yaml"].read_text(encoding="utf-8"))
    first_id = init_job_id(cell_cfg)
    init_job = campaign["runs"][first_id]
    ht = cell.heat_thermostat or init_job.get("heat_thermostat")
    print(
        f"{first_id}: setup={cell.setup_id} md_stages={init_job.get('md_stages')} "
        f"heat_thermostat={ht} liquid_prep={init_job.get('liquid_prep')} "
        f"calculator_pre_minimize={init_job.get('calculator_pre_minimize')} "
        f"charmm_mm_pretreat={init_job.get('charmm_mm_pretreat')} "
        f"sweep={cell.sweep_id or '-'}",
        flush=True,
    )
    print(f"Campaign jobs ({tag}): {campaign_job_order(cell_cfg)}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"{tag} setup-compare campaign failed with exit code {rc}", file=sys.stderr)
        return rc

    summary_path = paths["campaign_summary"]
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            jobs = payload.get("jobs", payload if isinstance(payload, list) else [])
            failed = [j for j in jobs if int(j.get("exit_code", 0)) != 0]
            if failed:
                for job in failed:
                    jid = job.get("job_id", "?")
                    backend = job.get("backend", "?")
                    stages = job.get("stages") or []
                    stage_note = ""
                    if stages:
                        last = stages[-1]
                        stage_note = (
                            f" stage={last.get('stage')} status={last.get('status')}"
                        )
                    print(
                        f"Failed leg {jid} backend={backend} "
                        f"exit_code={job.get('exit_code')}{stage_note}",
                        file=sys.stderr,
                    )
                print(
                    f"Campaign summary reports failed legs: "
                    f"{[j.get('job_id') for j in failed]}",
                    file=sys.stderr,
                )
                return 1
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"Could not parse {summary_path}: {exc}", file=sys.stderr)
            return 1
    elif paths["out_dir"].is_dir():
        # Campaign may have aborted before writing summary; hint where to look.
        print(
            f"Warning: missing campaign summary {summary_path}",
            flush=True,
        )
        for leg in reversed(campaign_job_order(cell_cfg)):
            leg_dir = paths["out_dir"] / leg
            if leg_dir.is_dir():
                print(f"Last leg directory present: {leg_dir}", file=sys.stderr)
                break

    if not paths["final_handoff"].is_file():
        print(
            f"Expected final handoff missing: {paths['final_handoff']}",
            file=sys.stderr,
        )
        return 1

    paths["done"].write_text(f"ok {tag}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

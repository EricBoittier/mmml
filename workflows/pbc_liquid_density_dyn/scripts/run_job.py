#!/usr/bin/env python3
"""Run one liquid-density PBC dynamics campaign via mmml-charmm-mpirun.sh."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    build_md_system_campaign_argv,
    campaign_job_order,
    cell_from_cli,
    cell_from_tag,
    cell_run_tag,
    load_config,
    paths_for_run,
    resolve_checkpoint,
    workflow_root,
)


def _repo_root() -> Path:
    return workflow_root().parents[1]


def _resolve_mpirun_wrapper(cfg: dict) -> Path:
    raw = Path(str(cfg.get("mpirun_wrapper", "../../scripts/mmml-charmm-mpirun.sh")))
    if raw.is_absolute():
        return raw
    return (workflow_root() / raw).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Run cell tag, e.g. dcm_277_t300_l32")
    parser.add_argument("solvent", nargs="?", help="Residue prefix (DCM or ACO)")
    parser.add_argument("n_monomers", nargs="?", type=int, help="Monomer count")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--box-size", type=float, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tag:
        cell = cell_from_tag(cfg, args.tag)
    else:
        if args.solvent is None or args.n_monomers is None:
            parser.error("Provide --tag or SOLVENT N_MONOMERS")
        cell = cell_from_cli(
            cfg,
            args.solvent,
            args.n_monomers,
            temperature=args.temperature,
            box_size=args.box_size,
        )

    resolve_checkpoint(str(cfg["checkpoint"]))
    paths = paths_for_run(cfg, cell)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    md_argv = build_md_system_campaign_argv(cfg, cell, out_dir=paths["out_dir"])
    os.chdir(_repo_root())

    mpirun_wrapper = _resolve_mpirun_wrapper(cfg)
    if not mpirun_wrapper.is_file():
        raise SystemExit(f"MPI wrapper not found: {mpirun_wrapper}")

    env = os.environ.copy()
    env.setdefault("MMML_MPI_NP", str(cfg.get("MMML_MPI_NP", 1)))
    env["MMML_NO_MPI_RERUN"] = "1"
    cmd = [str(mpirun_wrapper), "md-system", *md_argv]

    tag = cell_run_tag(cell, cfg)
    print(f"Campaign jobs ({tag}): {campaign_job_order(cfg)}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print(f"{tag} liquid-density dynamics campaign failed: exit {rc}", file=sys.stderr)
        return rc

    summary_path = paths["campaign_summary"]
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            jobs = payload.get("jobs", payload if isinstance(payload, list) else [])
            failed = [j.get("job_id") for j in jobs if int(j.get("exit_code", 0)) != 0]
            if failed:
                print(f"Campaign summary reports failed legs: {failed}", file=sys.stderr)
                return 1
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"Could not parse {summary_path}: {exc}", file=sys.stderr)
            return 1

    if not paths["final_handoff"].is_file():
        print(f"Expected final handoff missing: {paths['final_handoff']}", file=sys.stderr)
        return 1

    paths["done"].write_text(f"ok {tag}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

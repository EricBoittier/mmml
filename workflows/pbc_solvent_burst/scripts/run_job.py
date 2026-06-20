#!/usr/bin/env python3
"""Run one PBC solvent burst campaign via mmml md-system --run-all."""

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


def _resolve_mmml_cmd(md_argv: list[str]) -> list[str]:
    mmml_bin = os.environ.get("MMML_BIN")
    if mmml_bin:
        return [mmml_bin, "md-system", *md_argv]

    venv_mmml = _repo_root() / ".venv" / "bin" / "mmml"
    if venv_mmml.is_file():
        return [str(venv_mmml), "md-system", *md_argv]

    from shutil import which

    on_path = which("mmml")
    if on_path:
        return [on_path, "md-system", *md_argv]

    return [sys.executable, "-m", "mmml.cli.__main__", "md-system", *md_argv]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Run cell tag, e.g. dcm_10_t300_l32")
    parser.add_argument("solvent", nargs="?", help="Residue prefix (DCM or ACO)")
    parser.add_argument("n_monomers", nargs="?", type=int, help="Monomer count")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (K)")
    parser.add_argument("--box-size", type=float, default=None, help="Cubic box side (Å)")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Workflow config YAML",
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
    cmd = _resolve_mmml_cmd(md_argv)

    tag = cell_run_tag(cell)
    print(f"Campaign jobs ({tag}): {campaign_job_order(cfg)}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(
            f"{tag} burst campaign failed with exit code {rc}",
            file=sys.stderr,
        )
        return rc

    summary_path = paths["campaign_summary"]
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            jobs = payload.get("jobs", payload if isinstance(payload, list) else [])
            failed = [
                j.get("job_id")
                for j in jobs
                if int(j.get("exit_code", 0)) != 0
            ]
            if failed:
                print(f"Campaign summary reports failed legs: {failed}", file=sys.stderr)
                return 1
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"Could not parse {summary_path}: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"Warning: missing campaign summary {summary_path}", flush=True)

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

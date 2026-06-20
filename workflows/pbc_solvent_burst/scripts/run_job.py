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
    load_config,
    paths_for_run,
    resolve_checkpoint,
    run_tag,
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
    parser.add_argument("solvent", help="Residue prefix (DCM or ACO)")
    parser.add_argument("n_monomers", type=int, help="Monomer count")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Workflow config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sol = str(args.solvent).strip().upper()
    n = int(args.n_monomers)
    solvents = [str(s).strip().upper() for s in cfg.get("solvents", [])]
    sizes = [int(x) for x in cfg.get("cluster_sizes", [])]

    if sol not in solvents:
        raise SystemExit(f"solvent={sol!r} not in solvents {solvents}")
    if n not in sizes:
        raise SystemExit(f"n_monomers={n} not in cluster_sizes {sizes}")

    resolve_checkpoint(str(cfg["checkpoint"]))
    paths = paths_for_run(cfg, sol, n)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    md_argv = build_md_system_campaign_argv(cfg, sol, n, out_dir=paths["out_dir"])
    os.chdir(_repo_root())
    cmd = _resolve_mmml_cmd(md_argv)

    print(f"Campaign jobs ({run_tag(sol, n)}): {campaign_job_order(cfg)}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(
            f"{sol}:{n} burst campaign failed with exit code {rc}",
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

    paths["done"].write_text(f"ok {run_tag(sol, n)}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

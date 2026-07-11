#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml


NVE_PATTERN = re.compile(
    r"NVE Step\s+(?P<step>\d+).*?Tot Energy:\s+(?P<energy>[-+\d.eE]+) eV"
    r".*?Drift:\s+(?P<drift>[-+\d.eE]+) eV.*?Temp:\s+(?P<temperature>[-+\d.eE]+) K"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-config", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--dt-fs", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workflow_config = yaml.safe_load(args.workflow_config.read_text(encoding="utf-8"))
    mode = workflow_config["energy_modes"][args.mode]
    output_dir = args.output_dir.resolve()
    repo_root = args.repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "checkpoint": str((repo_root / workflow_config["checkpoint"]).resolve()),
        **workflow_config["system"],
        **workflow_config["simulation"],
        "dt_fs": args.dt_fs,
        "use_ml_intramolecular": mode["use_ml_intramolecular"],
        "peptide_water_ml": mode["peptide_water_ml"],
        "output_dir": str(output_dir),
        "workdir": str(output_dir / "build"),
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    command = [sys.executable, str(repo_root / "examples" / "cg_jaxmd.py"), "--config", str(config_path)]
    started = time.monotonic()
    process = subprocess.run(command, cwd=repo_root, env=os.environ.copy(), check=False)
    elapsed = time.monotonic() - started

    log_path = output_dir / "stdout.log"
    final_nve = None
    if log_path.exists():
        for match in NVE_PATTERN.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
            final_nve = match.groupdict()

    status = {
        "mode": args.mode,
        "description": mode["description"],
        "dt_fs": args.dt_fs,
        "returncode": process.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "completed": process.returncode == 0,
        "final_nve": final_nve,
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())


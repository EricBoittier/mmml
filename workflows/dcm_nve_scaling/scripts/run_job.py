#!/usr/bin/env python3
"""Run one DCM:N NVE scaling job via mmml md-system."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scaling_lib import (  # noqa: E402
    build_md_system_argv,
    load_config,
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


def _resolve_mpirun_wrapper(cfg: dict) -> Path:
    raw = Path(str(cfg.get("mpirun_wrapper", "../../scripts/mmml-charmm-mpirun.sh")))
    if raw.is_absolute():
        return raw
    return (workflow_root() / raw).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_monomers", type=int, help="Cluster size N in DCM:N")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Workflow config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sizes = [int(x) for x in cfg.get("cluster_sizes", [])]
    if int(args.n_monomers) not in sizes:
        raise SystemExit(f"n_monomers={args.n_monomers} not in cluster_sizes {sizes}")

    md_argv = build_md_system_argv(cfg, int(args.n_monomers))
    os.chdir(_repo_root())

    mpirun_wrapper = _resolve_mpirun_wrapper(cfg)
    if not mpirun_wrapper.is_file():
        raise SystemExit(f"MPI wrapper not found: {mpirun_wrapper}")
    cmd = [str(mpirun_wrapper), "md-system", *md_argv]

    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"DCM:{args.n_monomers} NVE failed with exit code {rc}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

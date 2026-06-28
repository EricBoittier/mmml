#!/usr/bin/env python3
"""Run one box_electro_compare job (wraps dcm5 benchmark runner + certified box)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "workflows" / "dcm5_md_benchmark" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_lib import build_md_system_argv, load_config  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_mmml_cmd(md_argv: list[str]) -> list[str]:
    root = _repo_root()
    venv_mmml = root / ".venv" / "bin" / "mmml"
    if venv_mmml.is_file():
        return [str(venv_mmml), "md-system", *md_argv]
    return [sys.executable, "-m", "mmml.cli.__main__", "md-system", *md_argv]


def _insert_before_extra_args(argv: list[str], tokens: list[str]) -> list[str]:
    """Insert tokens before ``--extra-args`` (REMAINDER must stay last)."""
    if not tokens:
        return argv
    if "--extra-args" in argv:
        idx = argv.index("--extra-args")
        return [*argv[:idx], *tokens, *argv[idx:]]
    return [*argv, *tokens]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-psf", type=Path, required=True)
    parser.add_argument("--from-crd", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    job = cfg["jobs"][args.job_id]
    backend = str(job["backend"])

    md_argv = build_md_system_argv(
        cfg,
        args.job_id,
        output_dir=args.output_dir,
    )
    if backend == "pycharmm":
        md_argv = _insert_before_extra_args(
            md_argv,
            [
                "--from-psf",
                str(args.from_psf.expanduser().resolve()),
                "--from-crd",
                str(args.from_crd.expanduser().resolve()),
                "--skip-cluster-build",
            ],
        )

    os.chdir(_repo_root())
    if backend == "pycharmm":
        mpirun = (_repo_root() / "scripts" / "mmml-charmm-mpirun.sh").resolve()
        cmd = [str(mpirun), "md-system", *md_argv]
    else:
        cmd = _resolve_mmml_cmd(md_argv)

    print(f"Running: {' '.join(cmd)}", flush=True)
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())

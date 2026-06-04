#!/usr/bin/env python3
"""Run one DCM:5 benchmark job via mmml md-system."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Allow imports from workflow scripts/ when invoked from repo root or workflow dir.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_lib import (  # noqa: E402
    build_md_system_argv,
    job_output_dir,
    load_config,
    namespace_for_job,
    pycharmm_stage_paths,
    workflow_root,
)


def _repo_root() -> Path:
    return workflow_root().parents[1]


def _resolve_mmml_cmd(md_argv: list[str]) -> list[str]:
    """Return argv prefix to invoke ``mmml md-system``."""
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

    # Fallback: module entry (mmml package has no __main__.py).
    return [sys.executable, "-m", "mmml.cli.__main__", "md-system", *md_argv]


def _resolve_mpirun_wrapper(cfg: dict) -> Path:
    raw = Path(str(cfg.get("mpirun_wrapper", "../../scripts/mmml-charmm-mpirun.sh")))
    if raw.is_absolute():
        return raw
    return (workflow_root() / raw).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Job key from config.yaml jobs:")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Benchmark config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override per-job output directory",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.job_id not in cfg["jobs"]:
        raise SystemExit(f"Unknown job_id {args.job_id!r}")

    backend, _, _ = namespace_for_job(
        cfg, args.job_id, output_dir=args.output_dir
    )
    md_argv = build_md_system_argv(
        cfg, args.job_id, output_dir=args.output_dir
    )

    os.chdir(_repo_root())
    mmml_cmd = _resolve_mmml_cmd(md_argv)

    if backend == "pycharmm":
        mpirun_wrapper = _resolve_mpirun_wrapper(cfg)
        if not mpirun_wrapper.is_file():
            raise SystemExit(f"MPI wrapper not found: {mpirun_wrapper}")
        cmd = [str(mpirun_wrapper), "md-system", *md_argv]
    else:
        cmd = mmml_cmd

    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"Job {args.job_id!r} failed with exit code {rc}", file=sys.stderr)
        return rc

    if backend == "pycharmm":
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import dynamics_nstep_from_ps
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
            assert_stage_dynamics_completed,
        )

        job = cfg["jobs"][args.job_id]
        stages = str(job.get("md_stages", ""))
        paths = pycharmm_stage_paths(
            cfg, args.job_id, output_dir=args.output_dir or job_output_dir(cfg, args.job_id)
        )
        try:
            if "nve" in stages:
                ps = float(job.get("ps_nve", cfg["ps"]))
                nstep = int(dynamics_nstep_from_ps(ps, float(cfg["dt_fs"])))
                assert_stage_dynamics_completed(
                    stage="nve",
                    expected_nstep=nstep,
                    nsavc=int(cfg["dcd_nsavc"]),
                    dcd_path=paths.get("nve_dcd"),
                    restart_path=paths.get("nve_res"),
                )
            if "heat" in stages:
                ps = float(job.get("ps_heat", cfg["ps"]))
                nstep = int(dynamics_nstep_from_ps(ps, float(cfg["dt_fs"])))
                assert_stage_dynamics_completed(
                    stage="heat",
                    expected_nstep=nstep,
                    nsavc=int(cfg["dcd_nsavc"]),
                    dcd_path=paths.get("heat_dcd"),
                    restart_path=paths.get("heat_res"),
                )
        except RuntimeError as exc:
            print(f"Job {args.job_id!r} post-run validation failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

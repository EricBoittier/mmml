"""Shared helpers for the DCM:5 cross-backend MD benchmark workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

from mmml.cli.run import md_system


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (workflow_root() / "config.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint(raw: str) -> Path:
    if raw == "${MMML_CKPT}":
        env = os.environ.get("MMML_CKPT", "").strip()
        if not env:
            raise RuntimeError(
                "MMML_CKPT is not set (config checkpoint: ${MMML_CKPT})"
            )
        return Path(env).expanduser().resolve()
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def job_output_dir(cfg: dict[str, Any], job_id: str) -> Path:
    root = workflow_root() / str(cfg.get("output_root", "results"))
    return (root / job_id).resolve()


def build_md_system_argv(
    cfg: dict[str, Any],
    job_id: str,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    """Build flat ``mmml md-system`` CLI argv for one benchmark job."""
    job = cfg["jobs"][job_id]
    out = output_dir or job_output_dir(cfg, job_id)

    argv: list[str] = [
        "--setup",
        str(job["setup"]),
        "--backend",
        str(job["backend"]),
        "--composition",
        str(cfg["composition"]),
        "--checkpoint",
        str(resolve_checkpoint(str(cfg["checkpoint"]))),
        "--output-dir",
        str(out),
        "--job-name",
        job_id,
        "--seed",
        str(cfg["seed"]),
        "--dt-fs",
        str(cfg["dt_fs"]),
        "--ps",
        str(cfg["ps"]),
        "--temperature",
        str(cfg["temperature"]),
        "--spacing",
        str(cfg["spacing"]),
        "--mm-switch-on",
        str(cfg["mm_switch_on"]),
        "--mm-switch-width",
        str(cfg["mm_switch_width"]),
        "--ml-switch-width",
        str(cfg["ml_switch_width"]),
    ]

    if job.get("pbc"):
        argv.extend(["--box-size", str(cfg["box_size"])])
    else:
        argv.extend(
            [
                "--packmol-sphere",
                "--packmol-radius",
                str(cfg["packmol_radius"]),
                "--packmol-tolerance",
                str(cfg["packmol_tolerance"]),
            ]
        )

    if job.get("nvt_integrator"):
        argv.extend(["--nvt-integrator", str(job["nvt_integrator"])])

    if str(job["backend"]) == "pycharmm":
        argv.extend(
            [
                "--mini-nstep",
                str(cfg["mini_nstep"]),
                "--dyn-nprint",
                str(cfg["dyn_nprint"]),
                "--dcd-nsavc",
                str(cfg["dcd_nsavc"]),
                "--dynamics-overlap-action",
                "rescue",
                "--charmm-sd-steps",
                "25",
                "--charmm-abnr-steps",
                "100",
                "--skip-energy-show",
                "--ml-gpu-count",
                "1",
            ]
        )
        if job.get("md_stages"):
            argv.extend(["--md-stages", str(job["md_stages"])])
        if job.get("ps_nve") is not None:
            argv.extend(["--ps-nve", str(job["ps_nve"])])
        if job.get("ps_heat") is not None:
            argv.extend(["--ps-heat", str(job["ps_heat"])])
        if job.get("heat_thermostat"):
            argv.extend(["--heat-thermostat", str(job["heat_thermostat"])])
        heat_ihtfrq = job.get("heat_ihtfrq")
        if heat_ihtfrq is not None:
            argv.extend(["--heat-ihtfrq", str(heat_ihtfrq)])
        argv.extend(["--heat-firstt", str(cfg["heat_firstt"])])
        argv.extend(["--heat-finalt", str(cfg["heat_finalt"])])

    if str(job["setup"]) == "pbc_npt":
        argv.extend(["--pressure", str(cfg["pressure"])])

    extra = list(job.get("extra_args") or [])
    if extra:
        argv.append("--extra-args")
        argv.extend(extra)

    return argv


def namespace_for_job(
    cfg: dict[str, Any],
    job_id: str,
    *,
    output_dir: Path | None = None,
) -> tuple[str, list[str], Any]:
    """Return (backend, backend_argv, argparse.Namespace) for a benchmark job."""
    argv = build_md_system_argv(cfg, job_id, output_dir=output_dir)
    old_argv = sys.argv[:]
    sys.argv = ["md-system", *argv]
    try:
        args = md_system.parse_args()
        md_system._apply_backend_setup_defaults(args)
        backend, backend_argv = md_system.build_command(args)
        return backend, backend_argv, args
    finally:
        sys.argv = old_argv


def job_metadata(cfg: dict[str, Any], job_id: str) -> dict[str, Any]:
    job = cfg["jobs"][job_id]
    return {
        "job_id": job_id,
        "backend": str(job["backend"]),
        "setup": str(job["setup"]),
        "pbc": bool(job.get("pbc")),
        "integrator": str(job.get("integrator", "")),
        "ps": float(cfg["ps"]),
        "nsteps_target": int(cfg.get("nsteps_target", 8000)),
    }

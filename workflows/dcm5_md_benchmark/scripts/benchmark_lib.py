"""Shared helpers for the DCM:5 cross-backend MD benchmark workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return workflow_root().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (workflow_root() / "config.yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint(raw: str) -> Path:
    if raw == "${MMML_CKPT}":
        env = os.environ.get("MMML_CKPT", "").strip()
        if not env:
            raise RuntimeError(
                "MMML_CKPT is not set (config checkpoint: ${MMML_CKPT}). "
                "Export your DCM PhysNet checkpoint directory before running Snakemake."
            )
        path = Path(env).expanduser().resolve()
    else:
        path = Path(os.path.expandvars(raw)).expanduser().resolve()
    validate_checkpoint(path)
    return path


def validate_checkpoint(path: Path) -> None:
    """Fail fast when MMML_CKPT is missing or still a README placeholder."""
    text = str(path)
    placeholders = ("/path/to", "/path/to/dcm", "your/checkpoint", "REPLACE_ME")
    if any(p in text for p in placeholders):
        raise RuntimeError(
            f"Checkpoint path looks like a placeholder: {path}\n"
            "Set a real directory, e.g.\n"
            "  export MMML_CKPT=$HOME/mmml_tutorial/acodcm/ckpts/dcm1-..."
        )
    if not path.exists():
        raise RuntimeError(
            f"Checkpoint not found: {path}\n"
            "Verify MMML_CKPT points at your DCM PhysNet ckpt directory."
        )


def job_output_dir(cfg: dict[str, Any], job_id: str) -> Path:
    root = workflow_root() / str(cfg.get("output_root", "results"))
    return (root / job_id).resolve()


def _append_packmol_argv(argv: list[str], cfg: dict[str, Any]) -> None:
    argv.extend(
        [
            "--packmol-sphere",
            "--packmol-radius",
            str(cfg["packmol_radius"]),
            "--packmol-tolerance",
            str(cfg["packmol_tolerance"]),
        ]
    )


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
        str(job.get("ps", cfg["ps"])),
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
        box_size = job.get("box_size", cfg["box_size"])
        argv.extend(["--box-size", str(box_size)])
        _append_packmol_argv(argv, cfg)
    else:
        _append_packmol_argv(argv, cfg)
        argv.append("--free-space")

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
                "--dynamics-overlap-check-interval",
                str(cfg.get("dynamics_overlap_check_interval", 8000)),
                "--echeck",
                str(cfg.get("echeck", 500.0)),
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
        if cfg.get("bonded_mm_mini"):
            argv.append("--bonded-mm-mini")
            argv.extend(
                ["--bonded-mm-mini-after", str(cfg.get("bonded_mm_mini_after", "mini"))]
            )
            argv.extend(
                [
                    "--bonded-mm-mini-steps",
                    str(int(cfg.get("bonded_mm_mini_steps", 50))),
                ]
            )

    if str(job["setup"]) == "pbc_npt":
        argv.extend(["--pressure", str(job.get("pressure", cfg["pressure"]))])

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
    _ensure_repo_on_path()
    from mmml.cli.run import md_system

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


def composition_tag(composition: str) -> str:
    """``DCM:5`` -> ``dcm_5`` (matches PyCHARMM artifact names)."""
    residue, count = composition.strip().split(":", 1)
    return f"{residue.lower()}_{count}"


def pycharmm_stage_paths(
    cfg: dict[str, Any],
    job_id: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Restart/DCD paths for a PyCHARMM benchmark job."""
    out = output_dir or job_output_dir(cfg, job_id)
    tag = composition_tag(str(cfg["composition"]))
    job = cfg["jobs"][job_id]
    paths: dict[str, Path] = {"out_dir": out, "tag": tag}
    stages = str(job.get("md_stages", ""))
    if "nve" in stages:
        paths["nve_dcd"] = out / f"nve_{tag}.dcd"
        paths["nve_res"] = out / f"nve_{tag}.res"
    if "heat" in stages:
        paths["heat_dcd"] = out / f"heat_{tag}.dcd"
        paths["heat_res"] = out / f"heat_{tag}.res"
    return paths


def job_metadata(cfg: dict[str, Any], job_id: str) -> dict[str, Any]:
    job = cfg["jobs"][job_id]
    ps = float(job.get("ps", cfg["ps"]))
    return {
        "job_id": job_id,
        "backend": str(job["backend"]),
        "setup": str(job["setup"]),
        "pbc": bool(job.get("pbc")),
        "integrator": str(job.get("integrator", "")),
        "ps": ps,
        "nsteps_target": int(round(ps * 1000.0 / float(cfg["dt_fs"]))),
        "optional": bool(job.get("optional")),
    }

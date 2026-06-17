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
    _ensure_repo_on_path()
    from mmml.cli.run.md_campaign import build_benchmark_md_system_argv

    return build_benchmark_md_system_argv(
        cfg,
        job_id,
        output_dir=output_dir,
        resolve_checkpoint=resolve_checkpoint,
        job_output_dir=job_output_dir,
    )


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

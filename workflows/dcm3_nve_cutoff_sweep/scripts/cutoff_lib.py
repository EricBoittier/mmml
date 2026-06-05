"""Shared helpers for the DCM:3 NVE cutoff / geometry sweep workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_STEP_OUTPUT = 1


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return workflow_root().parents[1]


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


def composition_string(cfg: dict[str, Any]) -> str:
    return str(cfg.get("composition", "DCM:3"))


def composition_tag(cfg: dict[str, Any]) -> str:
    comp = composition_string(cfg)
    residue, count = comp.strip().split(":", 1)
    return f"{residue.lower()}_{count}"


def packmol_radius_A(cfg: dict[str, Any]) -> float:
    comp = composition_string(cfg)
    n = int(comp.split(":", 1)[1])
    ref_n = int(cfg.get("packmol_reference_n", 60))
    ref_r = float(cfg.get("packmol_reference_r", 18.0))
    return round(ref_r * (n / float(ref_n)) ** (1.0 / 3.0), 1)


def preset_ids(cfg: dict[str, Any]) -> list[str]:
    return sorted((cfg.get("cutoff_presets") or {}).keys())


def geometry_ids(cfg: dict[str, Any]) -> list[str]:
    return sorted((cfg.get("geometry_variants") or {}).keys())


def preset_config(cfg: dict[str, Any], preset_id: str) -> dict[str, Any]:
    presets = cfg.get("cutoff_presets") or {}
    if preset_id not in presets:
        raise KeyError(f"Unknown cutoff preset {preset_id!r}")
    return dict(presets[preset_id])


def geometry_config(cfg: dict[str, Any], geom_id: str) -> dict[str, Any]:
    variants = cfg.get("geometry_variants") or {}
    if geom_id not in variants:
        raise KeyError(f"Unknown geometry variant {geom_id!r}")
    return dict(variants[geom_id])


def geometry_dir(cfg: dict[str, Any], geom_id: str) -> Path:
    root = workflow_root() / str(cfg.get("output_root", "results"))
    return (root / "geometries" / geom_id).resolve()


def run_dir(cfg: dict[str, Any], preset_id: str, geom_id: str) -> Path:
    root = workflow_root() / str(cfg.get("output_root", "results"))
    return (root / "runs" / preset_id / geom_id).resolve()


def expected_nve_nstep(cfg: dict[str, Any]) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import dynamics_nstep_from_ps

    return int(dynamics_nstep_from_ps(float(cfg["ps_nve"]), float(cfg["dt_fs"])))


def energy_catastrophe_score(cfg: dict[str, Any]) -> float:
    return float(cfg.get("energy_catastrophe_score", 10000.0))


def _assert_per_step_output(cfg: dict[str, Any]) -> None:
    dcd = int(cfg.get("dcd_nsavc", 0))
    dyn = int(cfg.get("dyn_nprint", 0))
    npr = int(cfg.get("nprint", 0))
    if dcd != _REQUIRED_STEP_OUTPUT or dyn != _REQUIRED_STEP_OUTPUT or npr != _REQUIRED_STEP_OUTPUT:
        raise ValueError(
            "Workflow requires dcd_nsavc=dyn_nprint=nprint=1 "
            f"(got dcd_nsavc={dcd}, dyn_nprint={dyn}, nprint={npr})"
        )


def build_md_system_argv(
    cfg: dict[str, Any],
    preset_id: str,
    geom_id: str,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    """Build ``mmml md-system`` argv for one preset × geometry NVE job."""
    _assert_per_step_output(cfg)
    preset = preset_config(cfg, preset_id)
    geom = geometry_config(cfg, geom_id)
    gdir = geometry_dir(cfg, geom_id)
    out = output_dir or run_dir(cfg, preset_id, geom_id)
    tag = composition_tag(cfg)
    psf = gdir / f"cluster_for_vmd_{tag}.psf"
    crd = gdir / "initial.crd"
    if not psf.is_file() or not crd.is_file():
        raise FileNotFoundError(
            f"Geometry artifacts missing for {geom_id!r}: expected {psf} and {crd}"
        )

    job_name = f"dcm3_nve_{preset_id}_{geom_id}"
    argv: list[str] = [
        "--setup",
        "free_nve",
        "--backend",
        "pycharmm",
        "--composition",
        composition_string(cfg),
        "--checkpoint",
        str(resolve_checkpoint(str(cfg["checkpoint"]))),
        "--output-dir",
        str(out),
        "--job-name",
        job_name,
        "--tag",
        tag,
        "--seed",
        str(cfg["seed"]),
        "--dt-fs",
        str(cfg["dt_fs"]),
        "--ps-nve",
        str(cfg["ps_nve"]),
        "--temperature",
        str(cfg["temperature"]),
        "--nve-boltzmann-temp",
        str(cfg.get("nve_boltzmann_temp", float(cfg["temperature"]) * 0.2)),
        "--spacing",
        str(cfg["spacing"]),
        "--mm-switch-on",
        str(preset["mm_switch_on"]),
        "--mm-switch-width",
        str(preset["mm_switch_width"]),
        "--ml-switch-width",
        str(preset["ml_switch_width"]),
        "--free-space",
        "--skip-cluster-build",
        "--from-psf",
        str(psf),
        "--from-crd",
        str(crd),
        "--md-stages",
        "mini,nve",
        "--mini-nstep",
        str(cfg["mini_nstep"]),
        "--nprint",
        str(cfg["nprint"]),
        "--dyn-nprint",
        str(cfg["dyn_nprint"]),
        "--dcd-nsavc",
        str(cfg["dcd_nsavc"]),
        "--dyn-inbfrq",
        str(int(cfg.get("dyn_inbfrq", -1))),
        "--dynamics-overlap-action",
        str(cfg.get("dynamics_overlap_action", "rescue")),
        "--dynamics-overlap-min-distance",
        str(cfg.get("dynamics_overlap_min_distance", 0.4)),
        "--dynamics-overlap-check-interval",
        str(cfg.get("dynamics_overlap_check_interval", 1)),
        "--charmm-sd-steps",
        str(cfg.get("charmm_sd_steps", 25)),
        "--charmm-abnr-steps",
        str(cfg.get("charmm_abnr_steps", 100)),
        "--skip-energy-show",
        "--ml-gpu-count",
        str(cfg.get("ml_gpu_count", 1)),
    ]

    if cfg.get("echeck") is not None:
        argv.extend(["--echeck", str(cfg["echeck"])])
    if bool(cfg.get("no_scale_echeck", False)):
        argv.append("--no-scale-echeck")
    if bool(cfg.get("no_echeck", False)):
        argv.append("--no-echeck")
    if bool(cfg.get("pre_nve_charmm_update", True)):
        argv.append("--pre-nve-charmm-update")
    else:
        argv.append("--no-pre-nve-charmm-update")
    if bool(cfg.get("bonded_mm_mini", False)):
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

    # Metadata for collectors (not md-system flags)
    _ = geom
    return argv


def ensure_repo_on_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

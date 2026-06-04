"""Shared helpers for the DCM NVE scaling Snakemake workflow."""

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


def packmol_radius_A(n_monomers: int, *, reference_n: int = 60, reference_r: float = 18.0) -> float:
    """Sphere radius scaling used by run_dcm9 / run_dcm90 scripts."""
    n = max(1, int(n_monomers))
    return round(float(reference_r) * (n / float(reference_n)) ** (1.0 / 3.0), 1)


def composition_tag(n_monomers: int, *, prefix: str = "DCM") -> str:
    return f"{prefix.lower()}_{int(n_monomers)}"


def composition_string(n_monomers: int, *, prefix: str = "DCM") -> str:
    return f"{prefix}:{int(n_monomers)}"


def nve_inbfrq_values(cfg: dict[str, Any]) -> list[int]:
    raw = cfg.get("nve_inbfrq_values")
    if raw is None:
        return [int(cfg.get("default_nve_inbfrq", -1))]
    return [int(x) for x in raw]


def inbfrq_slug(value: int) -> str:
    v = int(value)
    return f"neg{abs(v)}" if v < 0 else str(v)


def inbfrq_from_slug(slug: str) -> int:
    s = str(slug).strip()
    if s.startswith("neg"):
        return -int(s[3:])
    return int(s)


def job_output_dir(cfg: dict[str, Any], n_monomers: int) -> Path:
    tag = composition_tag(
        n_monomers,
        prefix=str(cfg.get("composition_prefix", "DCM")),
    )
    root = workflow_root() / str(cfg.get("output_root", "results"))
    return (root / f"{tag}_nve").resolve()


def run_variant_dir(cfg: dict[str, Any], n_monomers: int, inbfrq: int) -> Path:
    """Per-``inbfrq`` output directory under ``{tag}_nve/inbfrq_<slug>/``."""
    base = job_output_dir(cfg, n_monomers)
    return (base / f"inbfrq_{inbfrq_slug(inbfrq)}").resolve()


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
    n_monomers: int,
    *,
    output_dir: Path | None = None,
    inbfrq: int | None = None,
) -> list[str]:
    """Build ``mmml md-system`` argv for one cluster size and NVE ``inbfrq``."""
    _assert_per_step_output(cfg)
    n = int(n_monomers)
    prefix = str(cfg.get("composition_prefix", "DCM"))
    comp = composition_string(n, prefix=prefix)
    tag = composition_tag(n, prefix=prefix)
    if inbfrq is None:
        inbfrq = int(nve_inbfrq_values(cfg)[0])
    out = output_dir or run_variant_dir(cfg, n, int(inbfrq))
    job_name = f"{tag}_nve_ib{inbfrq_slug(int(inbfrq))}"
    packmol_r = packmol_radius_A(
        n,
        reference_n=int(cfg.get("packmol_reference_n", 60)),
        reference_r=float(cfg.get("packmol_reference_r", 18.0)),
    )

    argv: list[str] = [
        "--setup",
        "free_nve",
        "--backend",
        "pycharmm",
        "--composition",
        comp,
        "--checkpoint",
        str(resolve_checkpoint(str(cfg["checkpoint"]))),
        "--output-dir",
        str(out),
        "--job-name",
        job_name,
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
        str(cfg["mm_switch_on"]),
        "--mm-switch-width",
        str(cfg["mm_switch_width"]),
        "--ml-switch-width",
        str(cfg["ml_switch_width"]),
        "--free-space",
        "--packmol-sphere",
        "--packmol-radius",
        str(packmol_r),
        "--packmol-tolerance",
        str(cfg["packmol_tolerance"]),
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
        str(int(inbfrq)),
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

    if bool(cfg.get("save_forces_npz", False)):
        argv.append("--save-forces-npz")
        argv.extend(
            [
                "--forces-npz-interval",
                str(int(cfg.get("forces_npz_interval", 1))),
            ]
        )

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

    if bool(cfg.get("pre_nve_charmm_update", True)):
        argv.append("--pre-nve-charmm-update")
    else:
        argv.append("--no-pre-nve-charmm-update")

    return argv


def paths_for_size(
    cfg: dict[str, Any],
    n_monomers: int,
    *,
    inbfrq: int | None = None,
) -> dict[str, Path]:
    if inbfrq is None:
        inbfrq = int(nve_inbfrq_values(cfg)[0])
    out = run_variant_dir(cfg, n_monomers, int(inbfrq))
    tag = composition_tag(n_monomers, prefix=str(cfg.get("composition_prefix", "DCM")))
    return {
        "out_dir": out,
        "tag": tag,
        "inbfrq": int(inbfrq),
        "mini_crd": out / f"mini_full_mlpot_{tag}.crd",
        "nve_dcd": out / f"nve_{tag}.dcd",
        "nve_res": out / f"nve_{tag}.res",
        "audit_json": out / "audit.json",
        "com_npz": out / "com_analysis.npz",
        "forces_npz": out / "forces.npz",
    }


def expected_nve_nstep(cfg: dict[str, Any]) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import dynamics_nstep_from_ps

    return int(dynamics_nstep_from_ps(float(cfg["ps_nve"]), float(cfg["dt_fs"])))

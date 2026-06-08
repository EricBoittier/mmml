"""Shared helpers for the DCM heat scaling Snakemake workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


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


def composition_tag(n_monomers: int, *, prefix: str = "DCM") -> str:
    return f"{prefix.lower()}_{int(n_monomers)}"


def composition_string(n_monomers: int, *, prefix: str = "DCM") -> str:
    return f"{prefix}:{int(n_monomers)}"


def dt_fs_slug(dt_fs: float) -> str:
    v = float(dt_fs)
    if v == 0.25:
        return "dt025"
    if v == 0.125:
        return "dt0125"
    text = f"{v:g}".replace(".", "p")
    return f"dt{text}"


def dt_fs_from_slug(slug: str) -> float:
    s = str(slug)
    if s == "dt025":
        return 0.25
    if s == "dt0125":
        return 0.125
    if s.startswith("dt"):
        body = s[2:].replace("p", ".")
        return float(body)
    raise ValueError(f"unknown dt slug: {slug!r}")


def run_seed(
    n_monomers: int,
    repeat: int,
    dt_fs: float,
    *,
    seed_base: int = 123456,
) -> int:
    """Deterministic unique seed per (N, repeat, dt_fs)."""
    n = int(n_monomers)
    rep = int(repeat)
    dt_off = 0 if float(dt_fs) == 0.25 else 1
    return int(seed_base) + n * 10000 + rep * 100 + dt_off


def run_output_dir(
    cfg: dict[str, Any],
    n_monomers: int,
    repeat: int,
    dt_fs: float,
) -> Path:
    n = int(n_monomers)
    rep = int(repeat)
    slug = dt_fs_slug(dt_fs)
    root = repo_root() / str(cfg.get("output_root", "artifacts/pycharmm_mlpot"))
    return (root / f"dcm{n}_npt_x64_{rep}" / slug).resolve()


def build_md_system_argv(
    cfg: dict[str, Any],
    n_monomers: int,
    repeat: int,
    dt_fs: float,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    n = int(n_monomers)
    rep = int(repeat)
    dt = float(dt_fs)
    prefix = str(cfg.get("composition_prefix", "DCM"))
    comp = composition_string(n, prefix=prefix)
    tag = composition_tag(n, prefix=prefix)
    out = output_dir or run_output_dir(cfg, n, rep, dt)
    job_name = f"dcm{n}_npt_x64_{rep}_{dt_fs_slug(dt)}"
    seed = run_seed(n, rep, dt, seed_base=int(cfg.get("seed_base", 123456)))

    argv: list[str] = [
        "--setup",
        str(cfg.get("setup", "pycharmm_full")),
        "--backend",
        str(cfg.get("backend", "pycharmm")),
        "--composition",
        comp,
        "--checkpoint",
        str(resolve_checkpoint(str(cfg["checkpoint"]))),
        "--output-dir",
        str(out),
        "--job-name",
        job_name,
        "--box-size",
        str(cfg["box_size"]),
    ]

    md_stages = cfg.get("md_stages")
    if md_stages:
        stages = md_stages if isinstance(md_stages, str) else ",".join(str(s) for s in md_stages)
        argv.extend(["--md-stages", stages])
    else:
        argv.extend(["--md-stage", str(cfg.get("md_stage", "heat"))])

    argv.extend(
        [
        "--ps-heat",
        str(cfg["ps_heat"]),
        "--n-heat-segments",
        str(int(cfg["n_heat_segments"])),
        "--heat-thermostat",
        str(cfg["heat_thermostat"]),
        "--dt-fs",
        str(dt),
        "--flat-bottom-radius",
        str(cfg["flat_bottom_radius"]),
        "--packmol-radius",
        str(cfg["packmol_radius"]),
        "--temperature",
        str(cfg["temperature"]),
        "--dynamics-overlap-action",
        str(cfg.get("dynamics_overlap_action", "rescue")),
        "--seed",
        str(seed),
        "--dcd-nsavc",
        str(int(cfg["dcd_nsavc"])),
        "--dynamics-intra-min-distance",
        str(cfg.get("dynamics_intra_min_distance", 0.5)),
        "--ml-gpu-count",
        str(int(cfg.get("ml_gpu_count", 1))),
        "--ml-batch-size",
        str(int(cfg.get("ml_batch_size", 2056))),
        ]
    )

    if bool(cfg.get("no_echeck", False)):
        argv.append("--no-echeck")

    return argv


def paths_for_run(
    cfg: dict[str, Any],
    n_monomers: int,
    repeat: int,
    dt_fs: float,
) -> dict[str, Path]:
    out = run_output_dir(cfg, n_monomers, repeat, dt_fs)
    tag = composition_tag(n_monomers, prefix=str(cfg.get("composition_prefix", "DCM")))
    return {
        "out_dir": out,
        "tag": tag,
        "heat_dcd": out / f"heat_{tag}.dcd",
        "log": out / f"dcm{int(n_monomers)}.log",
    }


def _stable_slot(key: str, n_slots: int) -> int:
    import zlib

    return int(zlib.crc32(key.encode("utf-8"))) % max(1, int(n_slots))


def pick_slurm_gpu_node(cfg: dict[str, Any], n_monomers: int, repeat: int, dt_slug: str) -> str:
    nodes = [str(x) for x in (cfg.get("slurm_gpu_nodes") or ["gpu08", "gpu09"])]
    key = f"{int(n_monomers)}_{int(repeat)}_{dt_slug}"
    return nodes[_stable_slot(key, len(nodes))]

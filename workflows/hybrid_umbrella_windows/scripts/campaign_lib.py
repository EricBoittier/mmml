"""Config helpers for hybrid_umbrella_windows Snakemake workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(
        path
        or os.environ.get("MMML_WORKFLOW_CONFIG")
        or Path(__file__).resolve().parents[1] / "config.yaml"
    )
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")
    data["_config_path"] = str(cfg_path.resolve())
    return data


def resolve_path(repo: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo / p).resolve()


def slurm_value(cfg: dict[str, Any], name: str, default: Any) -> Any:
    return (cfg.get("slurm") or {}).get(name, default)


def slurm_max_concurrent(cfg: dict[str, Any]) -> int:
    slurm = cfg.get("slurm") or {}
    if "max_jobs" in slurm:
        return int(slurm["max_jobs"])
    return int(slurm.get("jobs", 8))


def slurm_launch_jobs(cfg: dict[str, Any]) -> int:
    return slurm_max_concurrent(cfg)


def slurm_resources_cli(cfg: dict[str, Any]) -> str:
    n = slurm_max_concurrent(cfg)
    return f"gpu={n} charmm_slot={n}"


def checkpoint_path(cfg: dict[str, Any]) -> str:
    return str(cfg.get("checkpoint", "examples/m/model_ext.json"))


def window_ids(cfg: dict[str, Any]) -> list[int]:
    if "window_ids" in cfg and cfg["window_ids"] is not None:
        return [int(x) for x in cfg["window_ids"]]
    return list(range(int(cfg.get("n_windows", 30))))

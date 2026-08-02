"""Config helpers for the nh3_ch3cl_reaction_path Snakemake campaign."""

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


def enabled(cfg: dict[str, Any], key: str) -> bool:
    return bool((cfg.get("enable") or {}).get(key, False))


def checkpoint_path(cfg: dict[str, Any]) -> str:
    return str(cfg.get("checkpoint", "examples/m/model_ext.json"))


def umbrella_variants(cfg: dict[str, Any]) -> list[str]:
    umb = cfg.get("umbrella") or {}
    active = list(umb.get("active") or [])
    variants = umb.get("variants") or {}
    missing = [name for name in active if name not in variants]
    if missing:
        raise KeyError(f"umbrella.active names missing from variants: {missing}")
    return active


def solvents(cfg: dict[str, Any]) -> list[str]:
    return [str(s).lower() for s in (cfg.get("solvents") or ["tip3"])]


def dmc_basins(cfg: dict[str, Any]) -> list[str]:
    return [str(b) for b in ((cfg.get("dmc") or {}).get("basins") or ["react", "product"])]


def seeds(cfg: dict[str, Any]) -> list[int]:
    raw = cfg.get("seeds")
    if raw is None and "seed" in cfg:
        return [int(cfg["seed"])]
    return [int(s) for s in (raw or [0])]


def temperatures(cfg: dict[str, Any]) -> list[int]:
    return [int(t) for t in (cfg.get("temperatures") or [300])]


def slurm_value(cfg: dict[str, Any], name: str, default: Any) -> Any:
    return (cfg.get("slurm") or {}).get(name, default)


def slurm_max_concurrent(cfg: dict[str, Any]) -> int:
    return int(slurm_value(cfg, "max_jobs", 8))


def slurm_launch_jobs(cfg: dict[str, Any]) -> int:
    return slurm_max_concurrent(cfg)


def slurm_resources_cli(cfg: dict[str, Any]) -> str:
    n = slurm_max_concurrent(cfg)
    return f"gpu={n} charmm_slot={n}"


def job_runtime_min(cfg: dict[str, Any], job: str, *, variant: str | None = None) -> int:
    if job.startswith("umbrella") and variant is not None:
        umb = ((cfg.get("umbrella") or {}).get("variants") or {}).get(variant) or {}
        if "runtime_min" in umb:
            return int(umb["runtime_min"])
    section = {
        "neb": "neb",
        "dmc": "dmc",
        "adumb_gas": "adumb",
        "adumb_sol": "adumb",
        "make_boxes": "make_boxes",
        "endpoints": None,
        "mbar": None,
        "umbrella_gas": "umbrella",
        "umbrella_sol": "umbrella",
    }.get(job)
    if section and isinstance(cfg.get(section), dict) and "runtime_min" in cfg[section]:
        return int(cfg[section]["runtime_min"])
    return int(slurm_value(cfg, "runtime_min", 120))


def expand_targets(cfg: dict[str, Any], output_root: str) -> list[str]:
    """Return status.json paths that rule ``all`` should wait on."""
    targets: list[str] = []
    root = output_root.rstrip("/")
    temps = temperatures(cfg)
    seed_list = seeds(cfg)

    if enabled(cfg, "endpoints"):
        targets.append(f"{root}/endpoints/status.json")
    if enabled(cfg, "make_boxes"):
        targets.append(f"{root}/boxes/status.json")
    if enabled(cfg, "neb"):
        targets.append(f"{root}/neb/status.json")
    if enabled(cfg, "dmc"):
        for basin in dmc_basins(cfg):
            for seed in seed_list:
                targets.append(f"{root}/dmc/{basin}/seed{seed}/status.json")
    if enabled(cfg, "umbrella_gas"):
        for variant in umbrella_variants(cfg):
            for temp in temps:
                for seed in seed_list:
                    targets.append(
                        f"{root}/umbrella_gas/{variant}/T{temp}/seed{seed}/status.json"
                    )
    if enabled(cfg, "umbrella_sol"):
        for solvent in solvents(cfg):
            for variant in umbrella_variants(cfg):
                for temp in temps:
                    for seed in seed_list:
                        targets.append(
                            f"{root}/umbrella_sol/{solvent}/{variant}/T{temp}/seed{seed}/status.json"
                        )
    if enabled(cfg, "adumb_gas"):
        for temp in temps:
            for seed in seed_list:
                targets.append(f"{root}/adumb_gas/T{temp}/seed{seed}/status.json")
    if enabled(cfg, "adumb_sol"):
        for solvent in solvents(cfg):
            for temp in temps:
                for seed in seed_list:
                    targets.append(
                        f"{root}/adumb_sol/{solvent}/T{temp}/seed{seed}/status.json"
                    )
    if enabled(cfg, "mbar"):
        if enabled(cfg, "umbrella_gas"):
            for variant in umbrella_variants(cfg):
                for temp in temps:
                    for seed in seed_list:
                        targets.append(
                            f"{root}/umbrella_gas/{variant}/T{temp}/seed{seed}/mbar/status.json"
                        )
        if enabled(cfg, "umbrella_sol"):
            for solvent in solvents(cfg):
                for variant in umbrella_variants(cfg):
                    for temp in temps:
                        for seed in seed_list:
                            targets.append(
                                f"{root}/umbrella_sol/{solvent}/{variant}/T{temp}/seed{seed}/mbar/status.json"
                            )
    return targets

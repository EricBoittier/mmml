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


def slurm_extra_string(cfg: dict[str, Any]) -> str:
    """Extra ``sbatch`` flags for every rule.

    GRES is deliberately absent. snakemake-executor-plugin-slurm builds
    ``--gres`` / ``--gpus`` itself from the job resources and raises
    ``The --generic-resources-(GRES) option is not allowed in the 'slurm_extra'
    parameter`` if it finds either one here.
    """
    parts: list[str] = []
    nodelist = str(slurm_value(cfg, "nodelist", "") or "").strip()
    if nodelist:
        parts.append(f"--nodelist={nodelist}")
    mail = str(slurm_value(cfg, "mail_user", "") or "").strip()
    if mail:
        parts.append(f"--mail-user={mail} --mail-type=FAIL")
    return " ".join(parts)


def slurm_exclude_nodes(cfg: dict[str, Any]) -> str:
    """Nodes to keep jobs off, as ``--slurm-exclude-failed-nodes`` wants them.

    Accepts a list or a comma string. The executor only auto-excludes nodes on
    Slurm status ``NODE_FAIL``; a node whose GPU is wedged fails jobs with plain
    ``FAILED``, so retries land right back on it unless it is listed here.
    """
    value = slurm_value(cfg, "exclude_nodes", "")
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = str(value or "").split(",")
    return ",".join(p.strip() for p in parts if p and p.strip())


def gpu_request_resources(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resources that make the executor ask Slurm for a GPU.

    ``slurm.gres`` (default ``gpu:1``) becomes ``--gres=gpu:1``. Setting it
    empty switches to the plugin's ``gpu`` resource, i.e. ``--gpus=1`` — the two
    forms are not interchangeable on every site, so this stays configurable.
    """
    gres = str(slurm_value(cfg, "gres", "gpu:1") or "").strip()
    if gres:
        return {"gres": gres}
    return {"gpu": int(slurm_value(cfg, "gpu", 1))}


def slurm_max_concurrent(cfg: dict[str, Any]) -> int:
    slurm = cfg.get("slurm") or {}
    if "max_jobs" in slurm:
        return int(slurm["max_jobs"])
    return int(slurm.get("jobs", 8))


def slurm_launch_jobs(cfg: dict[str, Any]) -> int:
    return slurm_max_concurrent(cfg)


def slurm_resources_cli(cfg: dict[str, Any]) -> str:
    """``--resources`` throttles.

    ``gpu_slot`` rather than ``gpu``: any resource literally named ``gpu`` is
    consumed by the Slurm plugin and turned into ``--gpus=N``, which would
    duplicate the ``gres`` request.
    """
    n = slurm_max_concurrent(cfg)
    return f"gpu_slot={n} charmm_slot={n}"


def checkpoint_path(cfg: dict[str, Any]) -> str:
    return str(cfg.get("checkpoint", "examples/m/model_ext.json"))


def window_ids(cfg: dict[str, Any]) -> list[int]:
    if "window_ids" in cfg and cfg["window_ids"] is not None:
        return [int(x) for x in cfg["window_ids"]]
    return list(range(int(cfg.get("n_windows", 30))))

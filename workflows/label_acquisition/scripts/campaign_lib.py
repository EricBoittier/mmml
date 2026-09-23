"""Shared helpers for the label_acquisition Snakemake workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_METHODS = (
    "stratified_random",
    "activation_fps",
    "activation_largest_norm",
    "output_grad_fps",
    "output_grad_largest_norm",
    "force_doptimal",
    "loss_grad_fps",
    "loss_grad_largest_norm",
    "unmodified_student",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} is not a mapping")
    return cfg


def output_root(cfg: dict[str, Any], workflow_basedir: str | Path) -> str:
    raw = cfg.get("output_root") or "artifacts/label_acquisition/default"
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    repo = Path(workflow_basedir).resolve().parents[1]
    return str((repo / path).resolve())


def methods(cfg: dict[str, Any]) -> list[str]:
    sel = cfg.get("selection") or {}
    return list(sel.get("methods") or DEFAULT_METHODS)


def budgets(cfg: dict[str, Any]) -> list[int]:
    sel = cfg.get("selection") or {}
    return [int(b) for b in (sel.get("budgets") or [8])]


def seeds(cfg: dict[str, Any]) -> list[int]:
    sel = cfg.get("selection") or {}
    return [int(s) for s in (sel.get("seeds") or [0])]


def tune_modes(cfg: dict[str, Any]) -> list[str]:
    t = cfg.get("training") or {}
    return list(t.get("modes") or ["readout", "full"])

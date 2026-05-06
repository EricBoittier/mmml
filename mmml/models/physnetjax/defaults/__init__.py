"""Bundled portable PhysNetJax weights (JSON) for MMML/PhysNet quickstarts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULTS_DIR = Path(__file__).resolve().parent
HF_JSON_DIR = DEFAULTS_DIR / "hf_json"
HF_JSON_MANIFEST_PATH = HF_JSON_DIR / "manifest.json"
JOINT_TRAINING_CATEGORY = "joint-training-defaults"
MMML_DEFAULT_ALIAS = "mmml-default"
BEST_FORCE_ALIAS = "best-forces"


def load_hf_physnet_manifest() -> dict[str, Any]:
    """Load the bundled Hugging Face PhysNet checkpoint manifest."""
    with HF_JSON_MANIFEST_PATH.open() as f:
        return json.load(f)


def list_hf_physnet_models(category: str | None = None) -> list[dict[str, Any]]:
    """Return bundled PhysNet model metadata, optionally filtered by category."""
    manifest = load_hf_physnet_manifest()
    checkpoints = manifest.get("checkpoints", [])
    if category is None:
        return list(checkpoints)

    category_info = manifest.get("categories", {}).get(category)
    if category_info is None:
        raise KeyError(f"Unknown PhysNet model category: {category}")

    ids = set(category_info.get("models", []))
    return [entry for entry in checkpoints if entry.get("id") in ids]


def default_hf_physnet_model_id(category: str = JOINT_TRAINING_CATEGORY) -> str:
    """Return the default bundled model ID for a category."""
    manifest = load_hf_physnet_manifest()
    if category == JOINT_TRAINING_CATEGORY:
        default = manifest.get("default_joint_training_model")
        if isinstance(default, str):
            return default

    category_info = manifest.get("categories", {}).get(category, {})
    default = category_info.get("default")
    if isinstance(default, str):
        return default

    models = category_info.get("models") or []
    if models:
        return models[0]

    raise KeyError(f"No default PhysNet model configured for category: {category}")


def best_force_hf_physnet_model_id() -> str:
    """Return the bundled model ID with the lowest validation force MAE."""
    checkpoints = load_hf_physnet_manifest().get("checkpoints", [])
    candidates = [
        entry
        for entry in checkpoints
        if entry.get("metadata", {}).get("objectives", {}).get("valid_forces_mae") is not None
    ]
    if not candidates:
        raise KeyError("No bundled PhysNet model has metadata.objectives.valid_forces_mae")
    best = min(
        candidates,
        key=lambda entry: float(entry["metadata"]["objectives"]["valid_forces_mae"]),
    )
    return str(best["id"])


def resolve_hf_physnet_model(selection: str | None = None) -> dict[str, Any]:
    """Resolve a bundled PhysNet model by ID, file name, category, or default alias."""
    manifest = load_hf_physnet_manifest()
    categories = manifest.get("categories", {})
    checkpoints = manifest.get("checkpoints", [])
    selected = selection or "default"

    if selected in {"default", "joint-default", "joint-training-default"}:
        selected = default_hf_physnet_model_id()
    elif selected in {MMML_DEFAULT_ALIAS, BEST_FORCE_ALIAS, "lowest-force-mae"}:
        selected = best_force_hf_physnet_model_id()
    elif selected in categories:
        selected = default_hf_physnet_model_id(selected)

    for entry in checkpoints:
        if selected in {entry.get("id"), entry.get("file"), Path(str(entry.get("file"))).stem}:
            resolved = dict(entry)
            resolved["path"] = HF_JSON_DIR / str(entry["file"])
            return resolved

    raise KeyError(f"Unknown bundled PhysNet model selection: {selection}")


def resolve_hf_physnet_checkpoint(selection: str | None = None) -> Path:
    """Return the checkpoint path for a bundled PhysNet model selection."""
    return Path(resolve_hf_physnet_model(selection)["path"])

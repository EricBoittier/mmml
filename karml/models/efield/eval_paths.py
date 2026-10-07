"""Output path helpers for EF evaluation (no JAX dependency)."""

from __future__ import annotations

from pathlib import Path


def resolve_evaluation_output_dir(
    base_dir: str | Path,
    *,
    rot_augment: bool = False,
    rot_perturbation: float = 1.0,
) -> Path:
    """Return output directory; use a rotaug_pert* subdir when augmentation is on."""
    base = Path(base_dir)
    if not rot_augment:
        return base
    pert_tag = f"{rot_perturbation:g}".replace(".", "p")
    return base / f"rotaug_pert{pert_tag}"

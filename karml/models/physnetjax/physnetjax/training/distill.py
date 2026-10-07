"""Pure helpers for PhysNet distillation loss blending (unit-testable without training loop)."""

from __future__ import annotations


def blend_regression_loss(gt_loss: float, teacher_loss: float, alpha: float) -> float:
    """Blend ground-truth and teacher regression losses."""
    return alpha * gt_loss + (1.0 - alpha) * teacher_loss


def blend_component_loss(
    gt_loss: float,
    teacher_loss: float,
    alpha: float,
    distill: bool,
) -> float:
    if not distill:
        return gt_loss
    return blend_regression_loss(gt_loss, teacher_loss, alpha)


def parse_distill_targets(targets) -> tuple[bool, bool, bool]:
    """Return (energy, forces, dipole) flags from a sequence of target names."""
    if targets is None:
        return True, True, True
    normalized = {str(t).lower() for t in targets}
    return (
        "energy" in normalized,
        "forces" in normalized,
        "dipole" in normalized,
    )

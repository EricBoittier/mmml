"""Unit tests for PhysNet distillation loss helpers."""

import pytest

from mmml.models.physnetjax.physnetjax.training.distill import (
    blend_component_loss,
    blend_regression_loss,
    parse_distill_targets,
)


def test_blend_regression_loss_endpoints():
    assert blend_regression_loss(10.0, 4.0, 1.0) == pytest.approx(10.0)
    assert blend_regression_loss(10.0, 4.0, 0.0) == pytest.approx(4.0)
    assert blend_regression_loss(10.0, 4.0, 0.5) == pytest.approx(7.0)


def test_blend_component_loss_disabled():
    assert blend_component_loss(3.0, 9.0, 0.25, distill=False) == pytest.approx(3.0)


def test_blend_component_loss_enabled():
    assert blend_component_loss(3.0, 9.0, 0.5, distill=True) == pytest.approx(6.0)


def test_parse_distill_targets_defaults():
    assert parse_distill_targets(None) == (True, True, True)


def test_parse_distill_targets_subset():
    assert parse_distill_targets(["energy", "forces"]) == (True, True, False)
    assert parse_distill_targets(["dipole"]) == (False, False, True)

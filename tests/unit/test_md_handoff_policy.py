"""Unit tests for handoff policy helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.cli.run.md_handoff import (
    MdHandoffState,
    handoff_skip_pre_min,
    resolve_handoff_box,
    resolve_handoff_velocity_policy,
    summarize_handoff_policy,
    write_handoff_policy_json,
)


def test_handoff_skip_pre_min_defaults_true_without_flag() -> None:
    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1], dtype=int),
    )
    assert handoff_skip_pre_min(handoff, handoff_pre_minimize=False) is True
    assert handoff_skip_pre_min(handoff, handoff_pre_minimize=True) is False
    assert handoff_skip_pre_min(None, handoff_pre_minimize=False) is False


def test_resolve_handoff_box_prefers_handoff_cell() -> None:
    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1], dtype=int),
        cell=np.diag([32.0, 32.0, 32.0]),
        pbc=True,
    )
    side, source, warnings = resolve_handoff_box(
        handoff,
        yaml_box_size=38.0,
        free_space=False,
        auto_box_from_geometry=40.0,
    )
    assert side == pytest.approx(32.0)
    assert source == "handoff_cell"
    assert warnings


def test_resolve_handoff_box_yaml_fallback_when_no_cell() -> None:
    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1], dtype=int),
    )
    side, source, warnings = resolve_handoff_box(
        handoff,
        yaml_box_size=38.0,
        free_space=False,
        auto_box_from_geometry=40.0,
    )
    assert side == pytest.approx(38.0)
    assert source == "yaml_fallback"
    assert warnings


def test_resolve_handoff_velocity_policy() -> None:
    handoff = MdHandoffState(
        positions=np.zeros((3, 3)),
        atomic_numbers=np.array([1, 1, 1], dtype=int),
        velocities=np.ones((3, 3)),
    )
    policy, use = resolve_handoff_velocity_policy(
        handoff, continue_velocities=True, pre_min_ran=False
    )
    assert policy == "continue"
    assert use is True

    policy2, use2 = resolve_handoff_velocity_policy(
        handoff, continue_velocities=True, pre_min_ran=True
    )
    assert policy2 == "rethermalize_after_pre_min"
    assert use2 is False


def test_write_handoff_policy_json(tmp_path: Path) -> None:
    summary = summarize_handoff_policy(
        None,
        skip_pre_min=False,
        handoff_pre_minimize=False,
        continue_velocities=True,
        velocity_policy="maxwell_boltzmann",
        use_handoff_velocities=False,
        box_side_A=32.0,
        box_source="yaml",
    )
    path = write_handoff_policy_json(summary, tmp_path / "handoff_policy.json")
    data = json.loads(path.read_text())
    assert data["box_side_A"] == 32.0
    assert data["velocity_policy"] == "maxwell_boltzmann"

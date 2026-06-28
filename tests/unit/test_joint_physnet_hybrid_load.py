"""Unit tests for PhysNet extraction from joint checkpoints."""

from __future__ import annotations

import pytest

from mmml.interfaces.calculators.checkpoint_loading import (
    extract_physnet_params_for_hybrid,
    is_joint_checkpoint_config,
)


def test_is_joint_checkpoint_config() -> None:
    assert is_joint_checkpoint_config({"physnet_config": {}, "dcmnet_config": {}})
    assert is_joint_checkpoint_config({"physnet_config": {}, "noneq_config": {}})
    assert not is_joint_checkpoint_config({"features": 32, "cutoff": 6.0})


def test_extract_physnet_params_from_joint_tree() -> None:
    joint = {
        "params": {
            "physnet": {"Dense_0": {"kernel": [1.0]}},
            "dcmnet": {"Dense_0": {"kernel": [2.0]}},
        }
    }
    out = extract_physnet_params_for_hybrid(joint)
    assert "params" in out
    assert "Dense_0" in out["params"]
    assert "dcmnet" not in out.get("params", {})


def test_extract_physnet_params_standalone_passthrough() -> None:
    standalone = {"params": {"Dense_0": {"kernel": [1.0]}}}
    out = extract_physnet_params_for_hybrid(standalone)
    assert "params" in out
    assert "Dense_0" in out["params"]


def test_extract_physnet_params_joint_missing_physnet_raises() -> None:
    joint = {"params": {"dcmnet": {"Dense_0": {"kernel": [2.0]}}}}
    with pytest.raises(ValueError, match="missing 'physnet' subtree"):
        extract_physnet_params_for_hybrid(joint)

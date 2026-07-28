"""Tests for umbrella checkpoint loading (JSON + Orbax-compatible)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mmml.interfaces.calculators.checkpoint_loading import LoadedCheckpoint
from mmml.umbrella.checkpoint import load_params_and_model


def test_load_params_and_model_rejects_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_params_and_model(tmp_path / "missing.json")


def test_load_params_and_model_json_builds_physnet(tmp_path: Path):
    path = tmp_path / "portable.json"
    path.write_text("{}", encoding="utf-8")
    config = {
        "features": 8,
        "max_degree": 1,
        "num_iterations": 1,
        "num_basis_functions": 4,
        "cutoff": 5.0,
        "max_atomic_number": 10,
        "natoms": 2,
        "max_padded_atoms": 2,
        "charges": False,
        "include_electrostatics": False,
        "zbl": False,
    }
    fake_bundle = LoadedCheckpoint(
        params={"params": {}},
        config=config,
        source=path,
        format="json",
    )
    with patch(
        "mmml.interfaces.calculators.checkpoint_loading.load_checkpoint_bundle",
        return_value=fake_bundle,
    ):
        with patch(
            "mmml.utils.model_checkpoint.build_physnet_from_config",
            return_value=MagicMock(max_padded_atoms=2),
        ) as build:
            with patch(
                "mmml.utils.model_checkpoint.infer_trainable_zbl_config",
                side_effect=lambda cfg, _p: cfg,
            ):
                with patch(
                    "mmml.utils.model_checkpoint.normalize_physnet_config",
                    side_effect=lambda cfg: cfg,
                ):
                    params, model = load_params_and_model(path, natoms=2)
    assert params == {"params": {}}
    assert model.max_padded_atoms == 2
    build.assert_called_once()
    assert build.call_args[0][0]["max_padded_atoms"] == 2


def test_load_params_and_model_rejects_joint(tmp_path: Path):
    path = tmp_path / "joint.json"
    path.write_text("{}", encoding="utf-8")
    fake_bundle = LoadedCheckpoint(
        params={"params": {}},
        config={"physnet_config": {}, "dcmnet_config": {}},
        source=path,
        format="json",
    )
    with patch(
        "mmml.interfaces.calculators.checkpoint_loading.load_checkpoint_bundle",
        return_value=fake_bundle,
    ):
        with pytest.raises(ValueError, match="joint"):
            load_params_and_model(path)

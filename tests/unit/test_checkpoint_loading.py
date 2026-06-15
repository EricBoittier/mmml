"""Unit tests for multi-format checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.calculators.checkpoint_loading import (
    detect_checkpoint_format,
    load_checkpoint_bundle,
    validate_checkpoint_path,
)
from mmml.utils.model_checkpoint import to_jsonable


def test_detect_checkpoint_format_json_file(tmp_path: Path) -> None:
    json_path = tmp_path / "params.json"
    json_path.write_text(json.dumps({"params": {"x": [1.0]}}))
    assert detect_checkpoint_format(json_path) == "json"


def test_detect_checkpoint_format_json_directory(tmp_path: Path) -> None:
    (tmp_path / "params.json").write_text("{}")
    (tmp_path / "model_config.json").write_text("{}")
    assert detect_checkpoint_format(tmp_path) == "json"


def test_detect_checkpoint_format_pickle_directory(tmp_path: Path) -> None:
    (tmp_path / "best_params.pkl").write_bytes(b"")
    (tmp_path / "model_config.pkl").write_bytes(b"")
    assert detect_checkpoint_format(tmp_path) == "pickle_joint"


def test_detect_checkpoint_format_orbax_epoch(tmp_path: Path) -> None:
    pytest.importorskip("orbax")
    import orbax.checkpoint as ocp

    params = {"dense": {"kernel": np.array([[0.1]], dtype=np.float32)}}
    epoch_dir = tmp_path / "epoch-1"
    ocp.PyTreeCheckpointer().save(str(epoch_dir), params)
    assert detect_checkpoint_format(tmp_path) == "orbax"


def test_validate_checkpoint_path_accepts_directory(tmp_path: Path) -> None:
    (tmp_path / "params.json").write_text("{}")
    validate_checkpoint_path(tmp_path)


def test_load_json_checkpoint_bundle_with_config(tmp_path: Path) -> None:
    config = {
        "features": 16,
        "max_degree": 2,
        "num_iterations": 2,
        "num_basis_functions": 8,
        "cutoff": 6.0,
        "max_atomic_number": 10,
        "natoms": 4,
        "charges": False,
        "include_electrostatics": False,
    }
    payload = {
        "params": to_jsonable(
            {
                "params": {
                    "embedding": np.zeros((11, 16), dtype=np.float32),
                }
            }
        ),
        "config": config,
    }
    json_path = tmp_path / "portable.json"
    json_path.write_text(json.dumps(payload))

    bundle = load_checkpoint_bundle(json_path)
    assert bundle.format == "json"
    assert bundle.config["natoms"] == 4
    assert "params" in bundle.params

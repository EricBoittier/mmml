from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from mmml.models.physnetjax.physnetjax.restart.restart import (
    get_last,
    get_params_model,
    restart_training,
)
from mmml.models.physnetjax.physnetjax.utils.utils import get_files


def test_get_files_keeps_epochs_under_system_tmp(tmp_path: Path) -> None:
    """Regression: parents named ``tmp`` must not drop valid ``epoch-N`` dirs."""
    # Mimic CI's /tmp/pytest-of-runner/... layout (substring "tmp" in the path).
    run = tmp_path / "tmp" / "pytest-of-runner" / "run"
    good = run / "epoch-10"
    junk = run / "epoch-11-tmp"
    good.mkdir(parents=True)
    junk.mkdir(parents=True)
    assert "tmp" in str(good)
    found = get_files(str(run))
    assert [p.name for p in found] == ["epoch-10"]
    assert get_last(str(run)).name == "epoch-10"

def test_get_params_model_robustness(tmp_path: Path) -> None:
    # Set up a dummy folder for restart
    restart_dir = tmp_path / "run"
    epoch_dir = restart_dir / "epoch-10"
    epoch_dir.mkdir(parents=True)
    
    # Mock restored dict with missing epoch and best_loss
    restored_mock = {
        "params": {"weights": 1.0},
        "model_attributes": {
            "features": 32,
            "max_degree": 2,
            "num_iterations": 2,
            "num_basis_functions": 32,
            "cutoff": 8.0,
            "max_atomic_number": 35,
            "charges": False,
            "natoms": 20,
            "total_charge": 0.0,
            "zbl": False,
        }
    }
    
    with patch("mmml.models.physnetjax.physnetjax.restart.restart.orbax_checkpointer") as mock_checkpointer:
        mock_checkpointer.restore.return_value = restored_mock
        
        # Test get_params_model
        params, model, meta = get_params_model(
            str(epoch_dir),
            natoms=20,
            return_meta=True,
            quiet=True
        )
        
        assert params == {"weights": 1.0}
        assert model is not None
        assert meta is not None
        assert meta["epoch"] == 0
        assert meta["best_loss"] == float("inf")


def test_get_params_model_prefers_ema_by_default(tmp_path: Path) -> None:
    epoch_dir = tmp_path / "epoch-10"
    epoch_dir.mkdir(parents=True)
    restored_mock = {
        "params": {"weights": 1.0},
        "ema_params": {"weights": 9.0},
        "model_attributes": {
            "features": 32,
            "max_degree": 2,
            "num_iterations": 2,
            "num_basis_functions": 32,
            "cutoff": 8.0,
            "max_atomic_number": 35,
            "charges": False,
            "natoms": 20,
            "total_charge": 0.0,
            "zbl": False,
        },
    }
    with patch(
        "mmml.models.physnetjax.physnetjax.restart.restart.orbax_checkpointer"
    ) as mock_checkpointer:
        mock_checkpointer.restore.return_value = restored_mock
        params, _model = get_params_model(str(epoch_dir), natoms=20, quiet=True)
        assert params == {"weights": 9.0}

        live, _model = get_params_model(
            str(epoch_dir), natoms=20, quiet=True, prefer_ema=False
        )
        assert live == {"weights": 1.0}


def test_get_last_accepts_params_json_file(tmp_path: Path) -> None:
    json_path = tmp_path / "params_acem1_2026-07-29_04-54-20.json"
    json_path.write_text('{"params": {}, "config": {"features": 32}}\n', encoding="utf-8")
    assert get_last(str(json_path)) == json_path.resolve()


def test_get_last_finds_params_json_in_directory(tmp_path: Path) -> None:
    older = tmp_path / "params_run_2020-01-01_00-00-00.json"
    newer = tmp_path / "params_run_2026-07-29_04-54-20.json"
    older.write_text('{"params": {}}\n', encoding="utf-8")
    newer.write_text('{"params": {}}\n', encoding="utf-8")
    # Ensure mtime ordering even on coarse filesystems.
    os.utime(older, (1_000_000_000, 1_000_000_000))
    os.utime(newer, (1_800_000_000, 1_800_000_000))
    assert get_last(str(tmp_path)) == newer.resolve()


def test_get_params_model_loads_portable_json(tmp_path: Path) -> None:
    json_path = tmp_path / "params_acem1_test.json"
    payload = {
        "params": {"Dense_0": {"kernel": [[1.0, 0.0], [0.0, 1.0]]}},
        "config": {
            "features": 32,
            "max_degree": 0,
            "num_iterations": 1,
            "num_basis_functions": 8,
            "cutoff": 5.0,
            "max_atomic_number": 9,
            "charges": False,
            "natoms": 9,
            "total_charge": 0.0,
            "zbl": False,
        },
        "metadata": {"epoch": 12, "best_loss": 0.5},
    }
    import json as _json

    json_path.write_text(_json.dumps(payload), encoding="utf-8")

    fake_model = MagicMock()
    fake_model.zbl = False
    with patch(
        "mmml.utils.model_checkpoint.build_physnet_from_config",
        return_value=fake_model,
    ):
        params, model, restored = get_params_model(
            str(json_path),
            natoms=9,
            return_everything=True,
            quiet=True,
        )
    assert model is fake_model
    assert isinstance(params, dict) and "params" in params
    assert restored["_checkpoint_format"] == "json"
    assert restored.get("epoch") == 12
    assert float(restored.get("best_loss")) == 0.5


def test_restart_training_from_json(tmp_path: Path) -> None:
    json_path = tmp_path / "params_tag_2026-07-29_04-54-20.json"
    import json as _json

    json_path.write_text(
        _json.dumps(
            {
                "params": {"w": [1.0]},
                "config": {
                    "features": 8,
                    "max_degree": 0,
                    "num_iterations": 1,
                    "num_basis_functions": 4,
                    "cutoff": 5.0,
                    "max_atomic_number": 9,
                    "charges": False,
                    "natoms": 9,
                    "total_charge": 0.0,
                    "zbl": False,
                },
                "metadata": {"epoch": 3, "best_loss": 1.25},
            }
        ),
        encoding="utf-8",
    )
    fake_model = MagicMock()
    fake_model.zbl = False
    mock_optimizer = MagicMock()
    mock_optimizer.init.return_value = "opt"
    mock_transform = MagicMock()
    mock_transform.init.return_value = "tx"
    with patch(
        "mmml.utils.model_checkpoint.build_physnet_from_config",
        return_value=fake_model,
    ):
        (
            ema_params,
            model,
            opt_state,
            params,
            transform_state,
            step,
            best_loss,
            CKPT_DIR,
            state,
        ) = restart_training(str(json_path), mock_transform, mock_optimizer, 9)
    assert model is fake_model
    assert step == 4
    assert best_loss == 1.25
    assert CKPT_DIR == tmp_path.resolve()
    assert opt_state == "opt"
    assert transform_state == "tx"
    assert ema_params is not None and params is not None


def test_restart_training_robustness(tmp_path: Path) -> None:
    # Set up a dummy folder for restart
    restart_dir = tmp_path / "run"
    epoch_dir = restart_dir / "epoch-10"
    epoch_dir.mkdir(parents=True)
    
    # Mock restored dict with missing epoch and best_loss, and only model (no params)
    restored_mock = {
        "model": MagicMock(params={"weights": 2.0}),
        "model_attributes": {
            "features": 32,
            "max_degree": 2,
            "num_iterations": 2,
            "num_basis_functions": 32,
            "cutoff": 8.0,
            "max_atomic_number": 35,
            "charges": False,
            "natoms": 20,
            "total_charge": 0.0,
            "zbl": False,
        }
    }
    
    # Mock optimizer and transform
    mock_optimizer = MagicMock()
    mock_optimizer.init.return_value = "opt_state_dummy"
    mock_transform = MagicMock()
    mock_transform.init.return_value = "transform_state_dummy"

    with patch("mmml.models.physnetjax.physnetjax.restart.restart.orbax_checkpointer") as mock_checkpointer:
        mock_checkpointer.restore.return_value = restored_mock
        
        # Test restart_training (which searches the parent directory for epoch folders)
        (
            ema_params,
            model,
            opt_state,
            params,
            transform_state,
            step,
            best_loss,
            CKPT_DIR,
            state,
        ) = restart_training(
            str(restart_dir),
            mock_transform,
            mock_optimizer,
            num_atoms=20
        )
        
        assert params == {"weights": 2.0}
        assert ema_params == {"weights": 2.0}
        assert step == 1
        assert best_loss == float("inf")
        assert opt_state == "opt_state_dummy"
        assert transform_state == "transform_state_dummy"

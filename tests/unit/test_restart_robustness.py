from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from mmml.models.physnetjax.physnetjax.restart.restart import (
    get_params_model,
    restart_training,
)

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

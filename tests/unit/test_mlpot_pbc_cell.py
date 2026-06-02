"""Tests for MIC PBC cell threading in PyCHARMM MLpot backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("jax")

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
    DecomposedMlpotModel,
    build_decomposed_mlpot_model,
    warmup_decomposed_mlpot,
)


def test_build_decomposed_mlpot_passes_cell_to_setup_calculator():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    factory = MagicMock(
        return_value=(
            None,
            MagicMock(),
            None,
        )
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ) as mock_setup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), None),
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        model = build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
            cell=20.0,
        )
    mock_setup.assert_called_once()
    assert mock_setup.call_args.kwargs["cell"] == 20.0
    assert model._cell == 20.0


def test_build_decomposed_mlpot_vacuum_cell_false():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    factory = MagicMock(return_value=(None, MagicMock(), None))
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ) as mock_setup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), None),
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ):
        model = build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
        )
    assert mock_setup.call_args.kwargs["cell"] is False
    assert model._cell is False


def test_register_mlpot_context_forwards_cell():
    from mmml.interfaces.pycharmmInterface.mlpot import run_workflow

    z = np.zeros(8, dtype=int)
    r = np.zeros((8, 3), dtype=float)
    ckpt = Path("/tmp/fake_ckpt.json")
    fake_model = DecomposedMlpotModel(MagicMock(), CutoffParameters(), 2, z, cell=20.0)
    fake_ctx = MagicMock()
    fake_sel = MagicMock()

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.load_physnet_mlpot_bundle",
        return_value=(None, None, fake_model),
    ) as mock_load, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.register_mlpot",
        return_value=fake_ctx,
    ) as mock_register, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.select_all_atoms",
        return_value=fake_sel,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.warmup_decomposed_mlpot",
    ) as mock_warmup, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.refresh_nbonds_after_mlpot_pbc",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.sync_charmm_positions",
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.run_workflow.get_charmm_positions_array",
        return_value=r,
    ):
        ctx, model = run_workflow._register_mlpot_context(
            z,
            r,
            ckpt,
            len(z),
            2,
            cubic_box_side_A=20.0,
            verbose=True,
        )

    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs["cell"] == 20.0
    mock_register.assert_called_once_with(fake_model, z, fake_sel)
    mock_warmup.assert_called_once()
    assert mock_warmup.call_args.kwargs["cell"] == 20.0
    assert ctx is fake_ctx
    assert model is fake_model


def test_warmup_decomposed_mlpot_passes_box_when_cell_set():
    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(MagicMock(), CutoffParameters(), 2, z, cell=20.0)
    r = np.zeros((8, 3), dtype=float)

    with patch(
        "mmml.utils.jax_gpu_warmup.warmup_hybrid_spherical_cutoff",
    ) as mock_warmup, patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        warmup_decomposed_mlpot(model, r, verbose=False)

    mock_warmup.assert_called_once()
    box = mock_warmup.call_args.kwargs["box"]
    assert box is not None
    assert float(box[0, 0]) == pytest.approx(20.0)

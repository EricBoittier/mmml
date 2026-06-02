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


def test_charmm_ctypes_scalar_accepts_int_and_ctypes_wrapper():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import _charmm_ctypes_scalar

    assert _charmm_ctypes_scalar(1) == pytest.approx(1.0)
    wrapper = MagicMock(value=39.5)
    assert _charmm_ctypes_scalar(wrapper) == pytest.approx(39.5)


def test_cubic_box_matrix_from_side():
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_matrix_from_side

    m = cubic_box_matrix_from_side(40.0)
    assert m.shape == (3, 3)
    assert float(m[0, 0]) == pytest.approx(40.0)
    assert float(m[1, 1]) == pytest.approx(40.0)


def test_sync_mlpot_pbc_cell_from_charmm_updates_model():
    from mmml.interfaces.pycharmmInterface.mlpot import run_workflow

    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(MagicMock(), CutoffParameters(), 2, z, cell=40.0)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.get_charmm_cubic_box_side_A",
        return_value=39.25,
    ):
        side = run_workflow.sync_mlpot_pbc_cell_from_charmm(model, verbose=False)
    assert side == pytest.approx(39.25)
    assert model._cell == pytest.approx(39.25)


def test_decomposed_calculator_passes_charmm_box_to_spherical_fn():
    z = np.zeros(8, dtype=int)
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        cell=40.0,
    )
    mock_out = MagicMock(energy=0.0, forces=np.zeros((8, 3)))
    calc.spherical_fn = MagicMock(return_value=mock_out)
    n = 8
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.get_charmm_cubic_box_side_A",
        return_value=39.0,
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None
        )
    assert calc._cell == pytest.approx(39.0)
    box = calc.spherical_fn.call_args.kwargs["box"]
    assert box is not None
    assert float(box[0, 0]) == pytest.approx(39.0)

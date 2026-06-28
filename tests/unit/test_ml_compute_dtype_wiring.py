"""Tests for ml_compute_dtype wiring in CLI and hybrid MLpot paths."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.cli.run.md_pbc_suite.pycharmm_mlpot import parse_args as parse_pycharmm_mlpot_args
from mmml.cli.run.md_system import build_pycharmm_command
from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
    DecomposedMlpotCalculator,
    DecomposedMlpotModel,
    _DeferredDecomposedMlpotCalculator,
    build_decomposed_mlpot_model,
)


def _mock_build_patches():
    factory = MagicMock(return_value=(None, MagicMock(), None))
    return patch(
        "mmml.interfaces.energy_forces.ml.assert_hybrid_ml_compatible",
        return_value=MagicMock(),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.setup_calculator",
        return_value=factory,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.unpack_factory_result",
        return_value=(None, MagicMock(), None),
    ), patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ), factory


def test_build_decomposed_mlpot_passes_ml_compute_dtype_to_setup_calculator():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    compat_patch, setup_patch, unpack_patch, ctx_patch, factory = _mock_build_patches()
    with compat_patch, setup_patch as mock_setup, unpack_patch, ctx_patch:
        build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
            ml_compute_dtype="float64",
        )
    mock_setup.assert_called_once()
    assert mock_setup.call_args.kwargs["ml_compute_dtype"] == "float64"


def test_build_decomposed_mlpot_reads_ml_compute_dtype_from_args():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    per = [4, 4]
    args = argparse.Namespace(ml_compute_dtype="float32")
    compat_patch, setup_patch, unpack_patch, ctx_patch, _factory = _mock_build_patches()
    with compat_patch, setup_patch as mock_setup, unpack_patch, ctx_patch:
        model = build_decomposed_mlpot_model(
            "/tmp/fake_ckpt.json",
            z,
            per,
            2,
            args=args,
        )
    assert mock_setup.call_args.kwargs["ml_compute_dtype"] == "float32"
    assert model._ml_compute_dtype == "float32"


def test_decomposed_mlpot_model_forwards_dtype_to_calculator():
    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        ml_compute_dtype="float64",
    )
    calc = model.get_pycharmm_calculator()
    assert calc._ml_compute_dtype == "float64"


def test_deferred_pycharmm_calculator_skips_jax_until_first_ener():
    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(
        None,
        CutoffParameters(),
        2,
        z,
        pending_factory=MagicMock(),
        pending_factory_z=z,
        defer_jax_until_after_sd=True,
    )
    with patch.object(model, "_finalize_jax_factory") as mock_finalize:
        calc = model.get_pycharmm_calculator(ml_atomic_numbers=z)
    mock_finalize.assert_not_called()
    assert isinstance(calc, _DeferredDecomposedMlpotCalculator)
    np.testing.assert_array_equal(calc.ml_atomic_numbers, z)


def test_decomposed_calculator_casts_positions_with_configured_dtype():
    z = np.zeros(8, dtype=int)
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        do_mm=False,
        ml_compute_dtype="float32",
    )
    mock_energy = jnp.array(0.0, dtype=jnp.float32)
    mock_forces = jnp.zeros((8, 3), dtype=jnp.float32)
    mock_forward = MagicMock(return_value=(mock_energy, mock_forces))
    calc._get_spherical_forward_fn = MagicMock(return_value=mock_forward)
    n = 8
    x = np.ones(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)

    with patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
        )

    positions = mock_forward.call_args.args[0]
    assert positions.dtype == jnp.float32
    assert np.asarray(dz, dtype=np.float64).dtype == np.float64


def test_calculate_charmm_uses_cpu_context_while_jax_deferred():
    z = np.zeros(8, dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        defer_jax_until_after_sd=True,
    )
    model._jax_on_gpu = False
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        do_mm=False,
    )
    calc._parent_model = model
    mock_energy = jnp.array(0.0, dtype=jnp.float32)
    mock_forces = jnp.zeros((8, 3), dtype=jnp.float32)
    calc._get_spherical_forward_fn = MagicMock(return_value=MagicMock(return_value=(mock_energy, mock_forces)))
    cpu_ctx = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    gpu_ctx = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    n = 8
    x = np.ones(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.jax_cpu_until_mlpot_registered",
        return_value=cpu_ctx,
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.mlpot_jax_device_context",
        return_value=gpu_ctx,
    ), patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
        )

    cpu_ctx.__enter__.assert_called_once()
    gpu_ctx.__enter__.assert_not_called()


def test_pycharmm_mlpot_parse_args_accepts_ml_compute_dtype():
    args = parse_pycharmm_mlpot_args(
        [
            "--checkpoint",
            "/tmp/fake.json",
            "--composition",
            "DCM:2",
            "--ml-compute-dtype",
            "float64",
        ]
    )
    assert args.ml_compute_dtype == "float64"


def test_build_pycharmm_command_omits_ml_compute_dtype_when_unset():
    from tests.unit.test_md_system_pycharmm_cmd import _pycharmm_args

    cmd = build_pycharmm_command(_pycharmm_args())
    assert "--ml-compute-dtype" not in cmd

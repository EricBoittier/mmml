"""Integration tests: spatial MPI path through MLpot callback (mocked JAX/MPI)."""

from __future__ import annotations

import os
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotCalculator


def _make_calculator(*, spatial_mpi: bool = True, cell: float = 40.0) -> DecomposedMlpotCalculator:
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1] * 4, dtype=int)  # 32 atoms, 4 monomers
    return DecomposedMlpotCalculator(
        mock.MagicMock(),
        CutoffParameters(),
        4,
        z,
        cell=cell,
        spatial_mpi=spatial_mpi,
    )


def _invoke_calculate_charmm(calc: DecomposedMlpotCalculator) -> dict:
    captured: dict = {}
    n = len(calc.atomic_numbers)

    def _fake_forward_fn(*, n_atoms, atomic_numbers_jax, box_jax):
        def _eval(
            positions_jax,
            mm_pair_idx,
            mm_pair_mask,
            use_mm_pairs,
            spatial_monomer_indices,
            spatial_dimer_indices,
            use_spatial,
        ):
            captured["use_spatial"] = bool(use_spatial)
            captured["mono_len"] = int(spatial_monomer_indices.shape[0])
            captured["dimer_len"] = int(spatial_dimer_indices.shape[0])
            rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
            forces = jnp.ones((n_atoms, 3)) * float(rank + 1)
            return jnp.array(float(rank + 1)), forces

        return _eval

    calc._get_spherical_forward_fn = mock.MagicMock(side_effect=_fake_forward_fn)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zc = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
        return_value=(40.0, "test"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock()),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.as_ml_array",
        side_effect=lambda arr, dtype=None: jnp.asarray(arr),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.resolve_ml_compute_dtype",
        return_value=jnp.float32,
    ):
        calc.calculate_charmm(
            n, 0, 0, None, x, y, zc, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
        )
    return captured


def test_spatial_mpi_callback_passes_batch_indices(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    calc = _make_calculator(spatial_mpi=True)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 2),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange.mpi_allreduce_forces",
        side_effect=lambda f, comm=None: np.asarray(f, dtype=np.float64),
    ) as mock_f, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange.mpi_allreduce_energy",
        side_effect=lambda e, comm=None: float(e),
    ) as mock_e:
        captured = _invoke_calculate_charmm(calc)
    assert captured["use_spatial"] is True
    assert captured["mono_len"] >= 1
    mock_f.assert_called_once()
    mock_e.assert_called_once()


def test_spatial_mpi_disabled_at_np1(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    calc = _make_calculator(spatial_mpi=True)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 1),
    ):
        captured = _invoke_calculate_charmm(calc)
    assert captured["use_spatial"] is False


def test_rank0_bridge_skips_ml_on_nonzero_rank(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    calc = _make_calculator(spatial_mpi=False)
    calls: list[int] = []

    def _fake_forward_fn(*, n_atoms, atomic_numbers_jax, box_jax):
        calls.append(1)

        def _eval(*_a, **_k):
            return jnp.array(0.0), jnp.zeros((n_atoms, 3))

        return _eval

    calc._get_spherical_forward_fn = mock.MagicMock(side_effect=_fake_forward_fn)
    n = len(calc.atomic_numbers)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(2, 4),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.broadcast_mlpot_result",
        side_effect=lambda f, e, n: (np.zeros((n, 3)), 0.0),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.resolve_mlpot_mic_box_side_A",
        return_value=(40.0, "test"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock()),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
    ):
        calc.calculate_charmm(
            n,
            0,
            0,
            None,
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    assert calls == []


def test_broadcast_spatial_uses_allreduce_not_bcast(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import mpi_bridge

    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    forces = np.ones((5, 3), dtype=np.float64)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(1, 2),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange.mpi_allreduce_forces",
        return_value=forces * 2.0,
    ) as mock_arf, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.force_exchange.mpi_allreduce_energy",
        return_value=3.0,
    ) as mock_are:
        out_f, out_e = mpi_bridge.broadcast_mlpot_result(forces, 1.5, 5)
    mock_arf.assert_called_once()
    mock_are.assert_called_once()
    np.testing.assert_allclose(out_f, forces * 2.0)
    assert out_e == 3.0

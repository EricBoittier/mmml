"""Unit tests for Tier 2 spatial MPI environment validation."""

from __future__ import annotations

import contextlib
import os
from unittest import mock

from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_validate import (
    validate_tier2_spatial_mpi_env,
)


@contextlib.contextmanager
def _tier2_patches(**overrides):
    defaults = {
        "charmm_lib_links_mpi": True,
        "charmm_mpirun_path": "/usr/bin/mpirun",
        "_under_mpirun": True,
        "mpi_rank_size": (0, 2),
        "spatial_mpi_enabled": True,
        "mlpot_jax_device_name": "gpu",
        "mlpot_local_gpu_count": 1,
        "defer_jax_warmup_until_after_mlpot_sd": True,
    }
    defaults.update(overrides)
    patches = [
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
            return_value=defaults["charmm_lib_links_mpi"],
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpirun_path",
            return_value=defaults["charmm_mpirun_path"],
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
            return_value=defaults["_under_mpirun"],
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
            return_value=defaults["mpi_rank_size"],
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.spatial_mpi_enabled",
            return_value=defaults["spatial_mpi_enabled"],
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_name",
            return_value=defaults["mlpot_jax_device_name"],
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_local_gpu_count",
            return_value=defaults["mlpot_local_gpu_count"],
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.defer_jax_warmup_until_after_mlpot_sd",
            return_value=defaults["defer_jax_warmup_until_after_mlpot_sd"],
        ),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def test_tier2_fails_without_gpu_when_spatial_and_gpu_mode(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    with _tier2_patches(mlpot_local_gpu_count=0):
        report = validate_tier2_spatial_mpi_env()
    assert report.ok is False
    assert any("no GPU" in e for e in report.errors)


def test_tier2_errors_on_ml_gpu_count_gt1_with_mpi(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    monkeypatch.setenv("MMML_MLPOT_N_GPUS", "2")
    with _tier2_patches(mlpot_local_gpu_count=2):
        report = validate_tier2_spatial_mpi_env()
    assert report.ok is False
    assert any("MMML_MLPOT_N_GPUS" in e for e in report.errors)


def test_tier2_ok_serial_spatial_disabled(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    with _tier2_patches(spatial_mpi_enabled=False, mpi_rank_size=(0, 1)):
        report = validate_tier2_spatial_mpi_env()
    assert report.ok is True


def test_tier2_prelaunch_strict_ok_without_spatial_env(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    with _tier2_patches(spatial_mpi_enabled=False, mpi_rank_size=(0, 1)):
        report = validate_tier2_spatial_mpi_env(strict=True, prelaunch=True)
    assert report.ok is True


def test_tier2_strict_without_prelaunch_fails_on_spatial_off(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    with _tier2_patches(spatial_mpi_enabled=False, mpi_rank_size=(0, 1)):
        report = validate_tier2_spatial_mpi_env(strict=True, prelaunch=False)
    assert report.ok is False

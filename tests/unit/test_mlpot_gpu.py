"""Tests for MLpot multi-GPU chunk helpers."""

from __future__ import annotations

import os

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_batch_policy import resolve_ml_batch_size
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy import (
    effective_ml_gpu_count,
    resolve_ml_gpu_count,
)


def test_resolve_ml_gpu_count_explicit():
    assert resolve_ml_gpu_count(3) == 3
    assert resolve_ml_gpu_count(0) == 1


def test_resolve_ml_gpu_count_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_N_GPUS", "2")
    assert resolve_ml_gpu_count(None) == 2


def test_effective_ml_gpu_count_clamps(monkeypatch):
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy.mlpot_local_gpu_count",
        lambda: 4,
    )
    assert effective_ml_gpu_count(8, n_chunks=2) == 2
    assert effective_ml_gpu_count(2, n_chunks=10) == 2
    assert effective_ml_gpu_count(2, n_chunks=1) == 1


def test_resolve_ml_batch_size_cpu_default(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "cpu")
    assert resolve_ml_batch_size(90, None) == 64


def test_resolve_ml_batch_size_gpu_default(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "gpu")
    assert resolve_ml_batch_size(90, None) == 256
    assert resolve_ml_batch_size(25, None) == 256

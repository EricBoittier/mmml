"""Tests for MLpot PhysNet batch chunking defaults."""

from __future__ import annotations

import os

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_batch_policy import resolve_ml_batch_size


def test_resolve_ml_batch_size_explicit():
    assert resolve_ml_batch_size(90, 32) == 32


def test_resolve_ml_batch_size_small_cluster():
    assert resolve_ml_batch_size(5, None) is None


def test_resolve_ml_batch_size_large_cluster_cpu(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "cpu")
    assert resolve_ml_batch_size(90, None) == 64
    assert resolve_ml_batch_size(40, None) == 64
    assert resolve_ml_batch_size(25, None) == 128


def test_resolve_ml_batch_size_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_ML_BATCH_SIZE", "48")
    assert resolve_ml_batch_size(90, None) == 48

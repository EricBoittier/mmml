"""Unit tests for scripts/train_md_embedding_converge.py helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "train_md_embedding_converge.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_md_embedding_converge", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def converge():
    return _load_module()


def test_target_num_epochs_fresh_start(converge):
    assert converge._target_num_epochs(0, 40) == 40


def test_target_num_epochs_after_restart(converge):
    assert converge._target_num_epochs(40, 40) == 80
    assert converge._target_num_epochs(80, 40) == 120


def test_epochs_per_round_prefers_config_key(converge):
    cfg = {"epochs_per_round": 25, "num_epochs": 40}
    assert converge._epochs_per_round(cfg, None) == 25


def test_epochs_per_round_falls_back_to_num_epochs(converge):
    cfg = {"num_epochs": 30}
    assert converge._epochs_per_round(cfg, None) == 30


def test_epochs_per_round_cli_overrides(converge):
    cfg = {"epochs_per_round": 25}
    assert converge._epochs_per_round(cfg, 10) == 10


def test_current_epoch_from_run_dir(converge, tmp_path):
    run = tmp_path / "aaa_long-abc"
    (run / "epoch-38").mkdir(parents=True)
    (run / "epoch-40").mkdir(parents=True)
    assert converge._current_epoch(run) == 40
    assert converge._current_epoch(None) == 0

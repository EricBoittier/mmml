"""Tests for MLpot callback / ASE calculator profiling."""

from __future__ import annotations

import json
import os

from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
    enable_mlpot_profiling,
    get_mlpot_profile_stats,
    mlpot_profiling_enabled,
    reset_mlpot_profile_stats,
    write_mlpot_profile_summary,
    write_profile_git_metadata,
)


def test_mlpot_profile_accumulates(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_PROFILE", "1")
    assert mlpot_profiling_enabled()
    reset_mlpot_profile_stats()
    stats = get_mlpot_profile_stats()
    stats.record_ml(0.5)
    stats.record_charmm_gap()
    stats.record_ml(0.3)
    line = stats.summary_line()
    assert "2 ML callbacks" in line
    assert "ML=" in line


def test_ase_calculate_and_chunk_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("MMML_MLPOT_PROFILE", "1")
    reset_mlpot_profile_stats()
    stats = get_mlpot_profile_stats()
    stats.record_calculate(0.020)
    stats.record_calculate(0.030)
    stats.record_chunk_apply(
        0.015,
        n_gpus=2,
        n_chunks=8,
        chunk_size=256,
        effective_batch_size=1800,
    )
    line = stats.summary_line()
    assert "2 ASE calculate" in line
    assert "chunk-apply" in line
    assert "n_gpus=2" in line
    payload = stats.to_dict()
    assert payload["calculate_calls"] == 2
    assert payload["max_n_gpus"] == 2
    assert payload["last_effective_batch_size"] == 1800

    path = write_mlpot_profile_summary(tmp_path, extra={"backend": "jaxmd"})
    assert path is not None
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["calculate_calls"] == 2
    assert written["extra"]["backend"] == "jaxmd"


def test_enable_mlpot_profiling_sets_env(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_PROFILE", raising=False)
    monkeypatch.delenv("MMML_JAX_COMPILE_TIMERS", raising=False)
    enable_mlpot_profiling()
    assert os.environ["MMML_MLPOT_PROFILE"] == "1"
    assert os.environ["MMML_JAX_COMPILE_TIMERS"] == "1"
    assert mlpot_profiling_enabled()


def test_profile_git_metadata_sidecar(tmp_path):
    path = write_profile_git_metadata(
        tmp_path,
        argv=["md-system", "--mlpot-profile"],
        extra={"effective_update_interval_steps": 10},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "profile_git_metadata.json"
    assert payload["argv"] == ["md-system", "--mlpot-profile"]
    assert payload["effective_update_interval_steps"] == 10
    assert "timestamp_utc" in payload
    assert "repo_root" in payload
    assert "git_commit" in payload or "git_error" in payload

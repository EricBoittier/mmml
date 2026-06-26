"""Tests for MLpot callback profiling."""

from __future__ import annotations

import json

from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
    get_mlpot_profile_stats,
    mlpot_profiling_enabled,
    reset_mlpot_profile_stats,
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

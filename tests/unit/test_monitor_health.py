"""Unit tests for pbc_liquid_density_dyn monitor failure classification."""

from __future__ import annotations

import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "pbc_liquid_density_dyn"
sys.path.insert(0, str(WORKFLOW / "scripts"))

import monitor_health as mh  # noqa: E402


def _first_match(text: str) -> mh.FailureRule | None:
    hits = mh._classify_failures(text)
    return hits[0] if hits else None


def test_classify_config_path():
    rule = _first_match("FileNotFoundError: config.yaml")
    assert rule is not None
    assert rule.id == "config_path"
    assert rule.action == "rerun_fresh"


def test_classify_warmup_pmi():
    rule = _first_match("refuse to run under mpirun/PMI launcher env")
    assert rule is not None
    assert rule.id == "warmup_pmi"
    assert rule.action == "rerun_fresh"


def test_classify_charmm_tier_exceeded():
    text = "395 ML atoms need max_Npr>=60200000 pairs; largest tier xlarge=56000000 is insufficient"
    rule = _first_match(text)
    assert rule is not None
    assert rule.id == "charmm_tier_exceeded"
    assert rule.action == "skip_manual"


def test_classify_charmm_tier_prebuild():
    rule = _first_match("ERROR: could not resolve CHARMM NPR tier for N_ML=395")
    assert rule is not None
    assert rule.id == "charmm_tier"
    assert rule.action == "prebuild_tier_rerun"


def test_classify_echeck():
    rule = _first_match("dynamics incomplete (echeck)")
    assert rule is not None
    assert rule.id == "echeck_abort"
    assert rule.action == "rerun_resume"


def test_retry_budget(tmp_path: Path):
    tracker = mh.RetryTracker(tmp_path / "retries.json")
    rule = next(r for r in mh.FAILURE_RULES if r.id == "config_path")
    assert tracker.can_retry("prod", "tag1", rule) is True
    tracker.record("prod", "tag1", rule.id)
    assert tracker.count("prod", "tag1", rule.id) == 1
    assert tracker.can_retry("prod", "tag1", rule) is True
    for _ in range(rule.max_retries - 1):
        tracker.record("prod", "tag1", rule.id)
    assert tracker.can_retry("prod", "tag1", rule) is False

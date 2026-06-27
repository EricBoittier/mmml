"""Tests for epoch timing helpers."""

from __future__ import annotations

from mmml.models.physnetjax.physnetjax.training.epoch_timing import (
    EpochTiming,
    EpochTimingSummary,
)


def test_epoch_timing_summary_means():
    summary = EpochTimingSummary()
    summary.record(EpochTiming(batch_prep_s=0.1, train_s=2.0, valid_s=0.2, checkpoint_s=0.5))
    summary.record(EpochTiming(batch_prep_s=0.2, train_s=2.2, valid_s=0.3, checkpoint_s=0.4))
    means = summary.means()
    assert means["batch_prep_s"] == 0.15
    assert means["train_s"] == 2.1
    assert "avg epoch" in summary.format_means()

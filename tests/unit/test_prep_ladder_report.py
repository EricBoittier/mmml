"""Unit tests for prep/ladder Rich reporting helpers."""

from __future__ import annotations

import pytest

from mmml.data.units import format_grms_kcal_ev_a
from mmml.utils import prep_ladder_report, rich_report


@pytest.fixture(autouse=True)
def _no_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMML_NO_RICH", "1")
    monkeypatch.delenv("MMML_QUIET", raising=False)
    rich_report._console.cache_clear()


def test_format_grms_dual_units() -> None:
    text = format_grms_kcal_ev_a(2241.4970)
    assert "2241.4970 kcal/mol/Å" in text
    assert "eV/Å" in text


def test_emit_hybrid_grms_diag_plain_geometry_stress(capsys) -> None:
    prep_ladder_report.emit_hybrid_grms_diag(
        "gate check",
        hybrid=80.0,
        charmm=2.5,
        kind="geometry_stress",
    )
    out = capsys.readouterr().out
    assert "hybrid GRMS=80.0000" in out
    assert "geometry stress" in out


def test_emit_hybrid_grms_diag_plain_desync(capsys) -> None:
    prep_ladder_report.emit_hybrid_grms_diag(
        "gate check",
        hybrid=30.0,
        charmm=10.0,
        kind="desync_suspected",
        ratio=3.0,
    )
    out = capsys.readouterr().out
    assert "possible hybrid/CHARMM desync" in out


def test_prep_ladder_journal_plain(capsys) -> None:
    journal = prep_ladder_report.PrepLadderJournal(quiet=False)
    journal.begin(initial_grms=100.0, max_grms=30.0, max_rounds=2)
    journal.begin_round(0, 100.0)
    metrics = prep_ladder_report.PrepMetrics(
        hybrid_grms=50.0,
        charmm_grms=2.0,
        user_kcal=1234.5,
        diag_kind="geometry_stress",
    )
    journal.record_step("round1:monomer_repack", metrics)
    journal.finish(45.0, reason="grms_still_high")
    out = capsys.readouterr().out
    assert "Density prep ladder" in out
    assert "round1:monomer_repack" in out
    assert "45.0000 kcal/mol/Å" in out
    assert "1234.5000 kcal/mol" in out


def test_emit_sd_pass_header_plain(capsys) -> None:
    metrics = prep_ladder_report.PrepMetrics(hybrid_grms=2241.0, user_kcal=5000.0)
    prep_ladder_report.emit_sd_pass_header(
        "SD",
        "pass 1 (free, all atoms)",
        remaining=20000,
        n_chunks=2000,
        chunk_cap=25,
        min_chunk=10,
        metrics=metrics,
    )
    out = capsys.readouterr().out
    assert "pass 1 (free, all atoms)" in out
    assert "20000 steps" in out
    assert "2241.0000 kcal/mol/Å" in out

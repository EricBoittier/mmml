"""Tests for the CI coverage floor.

The floor exists because the interesting failure is not "coverage is low" --
it is "coverage fell". A full run sits near 46% of ~121k statements and the
CI-reachable ceiling is around 70%, so nothing here chases a target; the gate
only rejects a slide backwards, and it rejects a report that measured nothing
(which otherwise reads as a clean 0-line pass).
"""

from __future__ import annotations

import pytest

from scripts.ci.check_coverage_floor import (
    CoverageTotals,
    check_totals,
    main,
    parse_coverage,
)


def _cobertura(*, covered: int, valid: int) -> str:
    rate = covered / valid if valid else 0.0
    return (
        '<?xml version="1.0" ?>\n'
        f'<coverage line-rate="{rate}" lines-covered="{covered}" '
        f'lines-valid="{valid}" version="7.1.0">'
        "<packages/></coverage>"
    )


def _write(tmp_path, *, covered: int, valid: int, name: str = "coverage.xml"):
    path = tmp_path / name
    path.write_text(_cobertura(covered=covered, valid=valid))
    return path


# --- totals arithmetic ------------------------------------------------------


def test_percent_is_covered_over_valid():
    assert CoverageTotals(covered=46, valid=100).percent == pytest.approx(46.0)


def test_percent_of_an_empty_report_is_zero_not_an_error():
    assert CoverageTotals().percent == 0.0


# --- parsing ----------------------------------------------------------------


def test_parse_reads_the_root_counters(tmp_path):
    totals = parse_coverage(_write(tmp_path, covered=55_600, valid=121_150))
    assert (totals.covered, totals.valid) == (55_600, 121_150)
    assert totals.percent == pytest.approx(45.9, abs=0.1)


def test_parse_falls_back_to_counting_lines(tmp_path):
    """A writer that omits the root counters must not read as 0%."""
    path = tmp_path / "coverage.xml"
    path.write_text(
        '<coverage line-rate="0.5"><packages><package><classes><class>'
        "<lines>"
        '<line number="1" hits="1"/>'
        '<line number="2" hits="0"/>'
        '<line number="3" hits="4"/>'
        "</lines></class></classes></package></packages></coverage>"
    )

    totals = parse_coverage(path)

    assert (totals.covered, totals.valid) == (2, 3)


# --- the floors -------------------------------------------------------------


def test_a_report_above_both_floors_passes():
    totals = CoverageTotals(covered=55_600, valid=121_150)
    assert check_totals(totals, min_percent=40.0, min_covered_lines=50_000) == []


def test_a_percentage_slide_is_rejected():
    totals = CoverageTotals(covered=40_000, valid=121_150)
    problems = check_totals(totals, min_percent=40.0)
    assert len(problems) == 1
    assert "below the floor" in problems[0]


def test_deleting_covered_code_cannot_buy_a_pass():
    """33% -> 80% by deleting the untested half still loses covered lines."""
    totals = CoverageTotals(covered=40_000, valid=50_000)
    assert check_totals(totals, min_percent=40.0) == []
    assert check_totals(totals, min_percent=40.0, min_covered_lines=50_000)


def test_a_report_that_measured_nothing_is_rejected():
    problems = check_totals(CoverageTotals(), min_percent=40.0)
    assert len(problems) == 1
    assert "no measurable lines" in problems[0]


def test_no_floors_configured_means_only_the_empty_check_runs():
    assert check_totals(CoverageTotals(covered=1, valid=100)) == []
    assert check_totals(CoverageTotals())


def test_the_floor_is_inclusive():
    totals = CoverageTotals(covered=40, valid=100)
    assert check_totals(totals, min_percent=40.0, min_covered_lines=40) == []


# --- the command line -------------------------------------------------------


def test_main_returns_zero_for_a_healthy_report(tmp_path, capsys):
    path = _write(tmp_path, covered=55_600, valid=121_150)
    assert main([str(path), "--min-percent", "40"]) == 0
    assert "45.89% (55600 of 121150 lines)" in capsys.readouterr().out


def test_main_returns_one_and_annotates_on_a_slide(tmp_path, capsys):
    path = _write(tmp_path, covered=30_000, valid=121_150)
    assert main([str(path), "--min-percent", "40"]) == 1
    assert "::error::" in capsys.readouterr().err


def test_main_reports_a_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.xml"), "--min-percent", "40"]) == 1
    assert "report not found" in capsys.readouterr().err


def test_main_reports_unparseable_xml(tmp_path, capsys):
    path = tmp_path / "coverage.xml"
    path.write_text("<coverage")
    assert main([str(path)]) == 1
    assert "not valid XML" in capsys.readouterr().err


def test_the_label_appears_in_the_annotation(tmp_path, capsys):
    path = _write(tmp_path, covered=1, valid=121_150)
    main([str(path), "--min-percent", "40", "--label", "unit coverage"])
    assert "unit coverage" in capsys.readouterr().err

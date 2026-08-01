"""Tests for the CI gate that rejects a green-but-empty pytest run.

The gate exists because ``pytest`` exits 0 when every selected test skips, so a
job guarding a live dependency (libcharmm, an ML checkpoint, MPI) can report
success having validated nothing.  These tests pin the two directions that
matter: an all-skipped report must be rejected, and a healthy report must not
be.
"""

from __future__ import annotations

import pytest

from scripts.ci.check_test_report import (
    ReportTotals,
    check_totals,
    collect_report_paths,
    main,
    parse_report,
)


def _junit_xml(*, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites>"
        f'<testsuite name="pytest" tests="{tests}" failures="{failures}" '
        f'errors="{errors}" skipped="{skipped}" time="1.0"/>'
        "</testsuites>"
    )


def _write_report(tmp_path, name: str = "junit.xml", **counts) -> "object":
    path = tmp_path / name
    path.write_text(_junit_xml(**counts))
    return path


# --- totals arithmetic ------------------------------------------------------


def test_passed_excludes_skips_failures_and_errors():
    totals = ReportTotals(tests=10, failures=1, errors=2, skipped=3)
    assert totals.passed == 4


def test_skipped_frac_is_zero_for_an_empty_report():
    assert ReportTotals().skipped_frac == 0.0


def test_totals_add_componentwise():
    combined = ReportTotals(tests=3, skipped=1) + ReportTotals(tests=5, failures=2)
    assert (combined.tests, combined.skipped, combined.failures) == (8, 1, 2)


# --- parsing ----------------------------------------------------------------


def test_parse_report_reads_pytest_counters(tmp_path):
    path = _write_report(tmp_path, tests=7, failures=1, errors=0, skipped=2)
    totals = parse_report(path)
    assert (totals.tests, totals.failures, totals.skipped, totals.passed) == (7, 1, 2, 4)


def test_parse_report_sums_multiple_testsuites(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        "<testsuites>"
        '<testsuite tests="2" failures="0" errors="0" skipped="0"/>'
        '<testsuite tests="3" failures="1" errors="0" skipped="1"/>'
        "</testsuites>"
    )
    totals = parse_report(path)
    assert (totals.tests, totals.failures, totals.skipped) == (5, 1, 1)


def test_collect_report_paths_expands_a_directory(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_report(reports, "a.xml", tests=1)
    _write_report(reports, "b.xml", tests=1)
    (reports / "not-a-report.txt").write_text("noise")
    found = collect_report_paths([str(reports)])
    assert [p.name for p in found] == ["a.xml", "b.xml"]


# --- the actual gate --------------------------------------------------------


def test_all_skipped_run_is_rejected():
    """The headline silent-failure mode: pytest exit 0 with zero real coverage."""
    problems = check_totals(ReportTotals(tests=40, skipped=40), min_passed=1)
    assert problems
    assert "only 0 test(s) passed" in problems[0]


def test_empty_report_is_rejected():
    problems = check_totals(ReportTotals(), min_passed=1)
    assert any("no tests were recorded" in p for p in problems)


def test_healthy_run_passes_the_gate():
    assert check_totals(ReportTotals(tests=100, skipped=5), min_passed=90) == []


def test_min_passed_counts_only_real_passes():
    totals = ReportTotals(tests=100, skipped=95, failures=0)
    assert check_totals(totals, min_passed=50)
    assert check_totals(totals, min_passed=5) == []


def test_failures_are_reported_unless_explicitly_allowed():
    totals = ReportTotals(tests=10, failures=2)
    assert any("failure(s)" in p for p in check_totals(totals, min_passed=1))
    assert check_totals(totals, min_passed=1, allow_failures=True) == []


def test_max_skipped_absolute_bound():
    totals = ReportTotals(tests=100, skipped=20)
    assert check_totals(totals, min_passed=1, max_skipped=10)
    assert check_totals(totals, min_passed=1, max_skipped=20) == []


def test_max_skipped_fraction_bound():
    totals = ReportTotals(tests=100, skipped=20)
    assert check_totals(totals, min_passed=1, max_skipped_frac=0.10)
    assert check_totals(totals, min_passed=1, max_skipped_frac=0.20) == []


# --- CLI --------------------------------------------------------------------


def test_main_exits_zero_on_a_healthy_report(tmp_path, capsys):
    path = _write_report(tmp_path, tests=50, skipped=2)
    assert main([str(path), "--min-passed", "40"]) == 0
    assert "48 passed" in capsys.readouterr().out


def test_main_exits_nonzero_when_everything_skipped(tmp_path, capsys):
    path = _write_report(tmp_path, tests=50, skipped=50)
    assert main([str(path), "--min-passed", "1"]) == 1
    assert "::error::" in capsys.readouterr().err


def test_main_exits_nonzero_when_the_report_is_missing(tmp_path, capsys):
    assert main([str(tmp_path / "absent.xml")]) == 1
    assert "report not found" in capsys.readouterr().err


def test_main_exits_nonzero_on_malformed_xml(tmp_path, capsys):
    path = tmp_path / "junit.xml"
    path.write_text("<testsuite tests=")
    assert main([str(path)]) == 1
    assert "not valid XML" in capsys.readouterr().err


def test_main_aggregates_several_reports(tmp_path, capsys):
    _write_report(tmp_path, "a.xml", tests=10, skipped=10)
    _write_report(tmp_path, "b.xml", tests=10, skipped=0)
    assert main([str(tmp_path), "--min-passed", "10"]) == 0
    out = capsys.readouterr().out
    assert "10 passed" in out and "10 skipped" in out


def test_main_rejects_an_aggregate_that_is_entirely_skips(tmp_path, capsys):
    """Per-module smoke runs that each skip must not add up to a green job."""
    _write_report(tmp_path, "a.xml", tests=10, skipped=10)
    _write_report(tmp_path, "b.xml", tests=6, skipped=6)
    assert main([str(tmp_path), "--min-passed", "1"]) == 1
    assert "only 0 test(s) passed" in capsys.readouterr().err


@pytest.mark.parametrize("frac", [0.0, 0.5, 1.0])
def test_skipped_frac_matches_hand_computation(frac):
    tests = 20
    skipped = int(tests * frac)
    assert ReportTotals(tests=tests, skipped=skipped).skipped_frac == pytest.approx(frac)

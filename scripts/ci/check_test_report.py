#!/usr/bin/env python3
"""Fail CI when a pytest run was green *because nothing actually ran*.

``pytest`` exits 0 when every selected test skips, so a job whose entire point
is exercising a live dependency (libcharmm, an ML checkpoint, MPI) reports
success after validating nothing.  Every skip guard in this repo is written as
``pytest.skip(...)`` / ``@pytest.mark.skipif(...)``, so one broken build
artifact silently converts the whole suite into a no-op.

This gate reads the JUnit XML pytest already knows how to emit and asserts that
a run met the *shape* it is supposed to have: enough tests passed, not too many
skipped, no collection errors.  Point it at one file, several files, or a
directory of them::

    pytest --junitxml=reports/junit-unit.xml tests/
    python scripts/ci/check_test_report.py reports/junit-unit.xml \\
        --min-passed 3000 --max-skipped-frac 0.15

Violations print GitHub ``::error::`` annotations and exit non-zero.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportTotals:
    """Aggregated counts across every ``<testsuite>`` in the parsed reports."""

    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0

    @property
    def passed(self) -> int:
        return self.tests - self.failures - self.errors - self.skipped

    @property
    def skipped_frac(self) -> float:
        return self.skipped / self.tests if self.tests else 0.0

    def __add__(self, other: "ReportTotals") -> "ReportTotals":
        return ReportTotals(
            tests=self.tests + other.tests,
            failures=self.failures + other.failures,
            errors=self.errors + other.errors,
            skipped=self.skipped + other.skipped,
        )


def parse_report(path: Path) -> ReportTotals:
    """Sum the counters of every ``<testsuite>`` element in a JUnit XML file.

    pytest writes a ``<testsuites>`` wrapper around a single ``<testsuite>``,
    but other producers nest several; walking all of them handles both without
    double-counting the wrapper (which carries no counters of its own in the
    pytest layout, and is skipped explicitly when it does).
    """
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    if not suites and root.tag == "testsuite":  # pragma: no cover - defensive
        suites = [root]
    totals = ReportTotals()
    for suite in suites:
        totals = totals + ReportTotals(
            tests=int(suite.get("tests", 0) or 0),
            failures=int(suite.get("failures", 0) or 0),
            errors=int(suite.get("errors", 0) or 0),
            skipped=int(suite.get("skipped", 0) or 0),
        )
    return totals


def collect_report_paths(inputs: list[str]) -> list[Path]:
    """Expand each input into concrete XML files (directories recurse)."""
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.xml")))
        else:
            paths.append(path)
    return paths


def check_totals(
    totals: ReportTotals,
    *,
    min_passed: int = 1,
    max_skipped: int | None = None,
    max_skipped_frac: float | None = None,
    allow_failures: bool = False,
) -> list[str]:
    """Return one message per violated expectation (empty list == healthy)."""
    problems: list[str] = []
    if totals.tests == 0:
        problems.append("no tests were recorded in the report(s) at all")
    if totals.passed < min_passed:
        problems.append(
            f"only {totals.passed} test(s) passed, expected at least {min_passed} "
            f"({totals.skipped} skipped, {totals.failures} failed, {totals.errors} errored)"
        )
    if not allow_failures and (totals.failures or totals.errors):
        problems.append(
            f"{totals.failures} failure(s) and {totals.errors} error(s) recorded"
        )
    if max_skipped is not None and totals.skipped > max_skipped:
        problems.append(
            f"{totals.skipped} test(s) skipped, more than the allowed {max_skipped}"
        )
    if max_skipped_frac is not None and totals.skipped_frac > max_skipped_frac:
        problems.append(
            f"{totals.skipped_frac:.1%} of tests skipped, more than the allowed "
            f"{max_skipped_frac:.1%}"
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", help="JUnit XML files or directories")
    parser.add_argument(
        "--min-passed",
        type=int,
        default=1,
        help="minimum number of tests that must have actually passed (default: 1)",
    )
    parser.add_argument(
        "--max-skipped", type=int, default=None, help="maximum absolute skip count"
    )
    parser.add_argument(
        "--max-skipped-frac",
        type=float,
        default=None,
        help="maximum fraction (0-1) of collected tests that may skip",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="do not fail on recorded failures/errors (pytest's own exit code covers them)",
    )
    parser.add_argument(
        "--label", default="pytest", help="name used in the printed summary"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = collect_report_paths(args.reports)
    missing = [p for p in paths if not p.is_file()]
    if not paths or missing:
        for path in missing or [Path(r) for r in args.reports]:
            print(f"::error::check_test_report: report not found: {path}", file=sys.stderr)
        return 1

    totals = ReportTotals()
    for path in paths:
        try:
            totals = totals + parse_report(path)
        except ET.ParseError as exc:
            print(f"::error::check_test_report: {path} is not valid XML: {exc}", file=sys.stderr)
            return 1

    print(
        f"{args.label}: {totals.passed} passed, {totals.skipped} skipped, "
        f"{totals.failures} failed, {totals.errors} errored "
        f"({totals.tests} collected across {len(paths)} report(s))"
    )

    problems = check_totals(
        totals,
        min_passed=args.min_passed,
        max_skipped=args.max_skipped,
        max_skipped_frac=args.max_skipped_frac,
        allow_failures=args.allow_failures,
    )
    for problem in problems:
        print(f"::error::{args.label}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

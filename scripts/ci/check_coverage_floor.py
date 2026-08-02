#!/usr/bin/env python3
"""Fail CI when line coverage falls below a floor.

A full ``MMML_DISABLE_CHARMM=1`` run covers ~46% of ~121k statements. Most of
what is left is not reachable from CI at all: ~18.7k statements need live
CHARMM, ~10.9k are plotting, ~5.3k need PySCF/torch/GPU. Covering every
CI-safe statement lands near 70%, so a *target* here would be a number nobody
can hit and everybody learns to ignore.

A floor is the useful shape. It does not ask anyone to chase a percentage; it
stops the number sliding backwards when a module stops importing, a fixture
tree goes missing, or a large untested subsystem lands.

Percentage alone is gameable in the wrong direction -- deleting covered code
raises it -- so ``--min-covered-lines`` pins the absolute count too. Both must
hold::

    pytest --cov=mmml --cov-report=xml tests/
    python scripts/ci/check_coverage_floor.py coverage.xml \\
        --min-percent 40 --min-covered-lines 50000

Violations print GitHub ``::error::`` annotations and exit non-zero.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoverageTotals:
    """Line totals read from a Cobertura ``coverage.xml``."""

    covered: int = 0
    valid: int = 0

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.valid if self.valid else 0.0


def parse_coverage(path: Path) -> CoverageTotals:
    """Read the line totals from a Cobertura report.

    ``coverage.py`` writes ``lines-covered`` / ``lines-valid`` on the root
    element. Older writers omit them but always carry ``line-rate``, so fall
    back to summing the per-class counters rather than reporting 0% -- a
    silent zero here would fail every build for the wrong reason.
    """
    root = ET.parse(path).getroot()
    covered = root.get("lines-covered")
    valid = root.get("lines-valid")
    if covered is not None and valid is not None:
        return CoverageTotals(covered=int(covered), valid=int(valid))

    total_valid = 0
    total_covered = 0
    for line in root.iter("line"):
        total_valid += 1
        if int(line.get("hits", 0) or 0) > 0:
            total_covered += 1
    return CoverageTotals(covered=total_covered, valid=total_valid)


def check_totals(
    totals: CoverageTotals,
    *,
    min_percent: float | None = None,
    min_covered_lines: int | None = None,
) -> list[str]:
    """Return one message per violated floor (empty list == healthy)."""
    problems: list[str] = []
    if totals.valid == 0:
        problems.append(
            "the report contains no measurable lines at all -- coverage was "
            "not collected, or it measured the wrong package"
        )
        return problems
    if min_percent is not None and totals.percent < min_percent:
        problems.append(
            f"line coverage is {totals.percent:.2f}%, below the floor of "
            f"{min_percent:.2f}% ({totals.covered} of {totals.valid} lines)"
        )
    if min_covered_lines is not None and totals.covered < min_covered_lines:
        problems.append(
            f"only {totals.covered} lines are covered, below the floor of "
            f"{min_covered_lines} (the percentage can rise while this falls, "
            f"so both are checked)"
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", help="path to a Cobertura coverage.xml")
    parser.add_argument(
        "--min-percent",
        type=float,
        default=None,
        help="minimum line-coverage percentage",
    )
    parser.add_argument(
        "--min-covered-lines",
        type=int,
        default=None,
        help="minimum absolute number of covered lines",
    )
    parser.add_argument(
        "--label", default="coverage", help="name used in the printed summary"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.report)
    if not path.is_file():
        print(
            f"::error::check_coverage_floor: report not found: {path}",
            file=sys.stderr,
        )
        return 1
    try:
        totals = parse_coverage(path)
    except ET.ParseError as exc:
        print(
            f"::error::check_coverage_floor: {path} is not valid XML: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"{args.label}: {totals.percent:.2f}% "
        f"({totals.covered} of {totals.valid} lines)"
    )

    problems = check_totals(
        totals,
        min_percent=args.min_percent,
        min_covered_lines=args.min_covered_lines,
    )
    for problem in problems:
        print(f"::error::{args.label}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

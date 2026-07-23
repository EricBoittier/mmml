#!/usr/bin/env python3
"""Refresh/check current-tree counts in docs/package-architecture.md."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/package-architecture.md"


def _count_python(relative: str) -> int:
    return sum(1 for _ in (REPO / relative).rglob("*.py"))


def generated_text() -> str:
    text = DOC.read_text(encoding="utf-8")
    counts = {
        "CLI": _count_python("mmml/cli"),
        "IFACE": _count_python("mmml/interfaces"),
        "MODELS": _count_python("mmml/models"),
        "UTILS": _count_python("mmml/utils"),
        "DATA": _count_python("mmml/data"),
        "GEN": _count_python("mmml/generate"),
        "GUI": _count_python("mmml/gui"),
        "SPEC": _count_python("mmml/spectra"),
    }
    total = _count_python("mmml")
    from mmml.cli.registry import COMMAND_REGISTRY

    text = re.sub(r"\(~\d+ Python modules\)", f"({total} Python modules)", text, count=1)
    for node, count in counts.items():
        text = re.sub(rf'({node}\["[^"\n]*\\n)\d+( modules"\])', rf"\g<1>{count}\2", text, count=1)
    text = re.sub(
        r'(MAIN\["cli/__main__\.py\\n)\d+( subcommands"\])',
        rf"\g<1>{len(COMMAND_REGISTRY)}\2",
        text,
        count=1,
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = DOC.read_text(encoding="utf-8")
    expected = generated_text()
    if args.check:
        if current != expected:
            print(f"stale: {DOC.relative_to(REPO)}")
            return 1
        return 0
    DOC.write_text(expected, encoding="utf-8")
    print(f"updated: {DOC.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

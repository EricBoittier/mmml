#!/usr/bin/env python3
"""Reject high-risk scientific recommendations without evidence annotation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RISK = re.compile(
    r"safe default|recommended hyperparameters|gpu\d+-validated|"
    r"default validated|validated default|generally optimal",
    re.IGNORECASE,
)
ANNOTATION = re.compile(r"\[evidence:\s*[a-z0-9_]+\]|\bUNVERIFIED\b", re.IGNORECASE)
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md"}
SKIP_PARTS = {
    ".claude",
    ".git",
    ".venv",
    "build",
    "node_modules",
    "packmol",
    "site",
    "tests",
}


def findings() -> list[str]:
    errors: list[str] = []
    for path in sorted(REPO.rglob("*")):
        if path == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in SUFFIXES or SKIP_PARTS.intersection(path.parts):
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not RISK.search(line):
                continue
            context = "\n".join(lines[max(0, index - 4) : index + 2])
            if not ANNOTATION.search(context):
                errors.append(f"{path.relative_to(REPO)}:{index + 1}: {line.strip()}")
    return errors


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    errors = findings()
    if errors:
        print("Hard-coded scientific recommendations require [evidence: id] or UNVERIFIED:")
        print("\n".join(errors))
        return 1
    print("hard-coded recommendation audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

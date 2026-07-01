#!/usr/bin/env python3
"""Patch literature comparison tables into docs/cli/structure-building.md."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "cli" / "structure-building.md"
START = "<!-- CRYSTAL_LIT_COMPARE_START -->"
END = "<!-- CRYSTAL_LIT_COMPARE_END -->"


def _patch(text: str, body: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(START)}.*?^{re.escape(END)}\n?",
        re.MULTILINE | re.DOTALL,
    )
    block = f"{START}\n{body.rstrip()}\n{END}\n"
    if not pattern.search(text):
        raise SystemExit(
            f"{DOC} missing {START} / {END} markers under the build-crystal section."
        )
    return pattern.sub(block, text)


def generate(*, check: bool = False, use_live_pyxtal: bool = False) -> int:
    sys.path.insert(0, str(REPO))
    from mmml.interfaces.crystal_reference import literature_comparison_markdown

    body = literature_comparison_markdown(use_live_pyxtal=use_live_pyxtal)
    if not DOC.is_file():
        raise SystemExit(f"missing {DOC}")

    current = DOC.read_text(encoding="utf-8")
    new_text = _patch(current, body)
    if check:
        if new_text != current:
            print("generate_crystal_lit_compare: docs/cli/structure-building.md is stale")
            return 1
        print("generate_crystal_lit_compare: OK")
        return 0

    if new_text != current:
        DOC.write_text(new_text, encoding="utf-8")
        print("generate_crystal_lit_compare: updated docs/cli/structure-building.md")
    else:
        print("generate_crystal_lit_compare: no changes")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if structure-building.md comparison block is stale.",
    )
    parser.add_argument(
        "--live-pyxtal",
        action="store_true",
        help="Use live PyXtal builds (default: frozen reference metrics for reproducible docs).",
    )
    args = parser.parse_args()
    return generate(check=args.check, use_live_pyxtal=args.live_pyxtal)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate PBC / CHARMM IMAGE pedagogy figures for docs/pbc-super-system.md.

Run from repo root::

    uv run python scripts/plot_pbc_super_system.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "images" / "pbc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"PNG output directory (default: {DEFAULT_OUT.relative_to(REPO)})",
    )
    args = parser.parse_args()

    from mmml.utils.pbc_super_system_plot import generate_pbc_doc_figures

    paths = generate_pbc_doc_figures(args.output_dir)
    for name, path in paths.items():
        print(f"wrote {path} ({name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

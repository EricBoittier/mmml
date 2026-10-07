#!/usr/bin/env python
"""CLI for build-crystal: PyXtal structure generation with optional ASE optimization."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from mmml.cli.misc.build_crystal import main as run
    except ModuleNotFoundError as exc:
        if "pyxtal" in str(exc).lower():
            print(
                "Error: build-crystal requires PyXtal. Install with: uv sync --extra chem",
                file=sys.stderr,
            )
            return 1
        raise
    return int(run())


if __name__ == "__main__":
    raise SystemExit(main())

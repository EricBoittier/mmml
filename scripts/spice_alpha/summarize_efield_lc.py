#!/usr/bin/env python3
"""Print a polar learning-curve table from efield ckpt dirs (history.jsonl)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmml.data.efield_lc import format_lc_table, summarize_lc_runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt_dir", nargs="+", type=Path)
    args = p.parse_args(argv)
    print(format_lc_table(summarize_lc_runs(args.ckpt_dir)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``mmml validate`` — NPZ schema validation (argparse wrapper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml validate",
        description="Validate NPZ files against the MMML schema.",
    )
    parser.add_argument(
        "npz_files",
        nargs="+",
        type=Path,
        help="One or more NPZ files to validate",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print pass/fail summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from mmml.data.npz_schema import validate_npz

    args = build_parser().parse_args(argv)
    all_valid = True
    for npz_file in args.npz_files:
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print(f"Validating: {npz_file}")
            print("=" * 60)
        is_valid, _info = validate_npz(str(npz_file), verbose=not args.quiet)
        if not is_valid:
            all_valid = False
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())

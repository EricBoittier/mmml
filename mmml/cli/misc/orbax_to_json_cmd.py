#!/usr/bin/env python
"""
CLI: export an Orbax checkpoint to a portable JSON file.

Usage:
    mmml orbax-to-json path/to/epoch-1985 -o DESdimers_params.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an Orbax checkpoint to a portable JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Orbax checkpoint directory (epoch-* dir or experiment root)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSON path (e.g. DESdimers_params.json)",
    )
    parser.add_argument(
        "--params-key",
        default="params",
        help='Key to extract from restored checkpoint dict (default: "params")',
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> int:
    args = parse_args()

    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.utils.model_checkpoint import orbax_to_json

    _, epoch_dir = resolve_checkpoint_paths(args.checkpoint)
    if epoch_dir.is_file() and epoch_dir.suffix == ".json":
        print(
            f"Error: checkpoint is already JSON: {epoch_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        output_path = orbax_to_json(
            orbax_checkpoint_dir=epoch_dir,
            output_path=args.output,
            params_key=args.params_key,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"Error: failed to convert checkpoint: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote portable checkpoint to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

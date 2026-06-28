"""``mmml commands`` and ``mmml examples`` — browse help without wall-of-text ``-h``."""

from __future__ import annotations

import argparse
import sys

from mmml.cli.help_text import format_commands_help, format_examples_help
from mmml.cli.registry import format_audit_report


def build_commands_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml commands",
        description="List MMML subcommands by category.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Per-command flags: mmml <command> --help",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Show deprecated/legacy commands and tab-completion coverage",
    )
    return parser


def build_examples_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="mmml examples",
        description="Copy-paste example invocations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Interactive setup: mmml configure",
    )


def commands_main(argv: list[str] | None = None) -> int:
    args = build_commands_parser().parse_args(argv if argv is not None else [])
    if args.audit:
        sys.stdout.write(format_audit_report() + "\n")
        return 0
    sys.stdout.write(format_commands_help() + "\n")
    return 0


def examples_main(argv: list[str] | None = None) -> int:
    build_examples_parser().parse_args(argv if argv is not None else [])
    sys.stdout.write(format_examples_help() + "\n")
    return 0

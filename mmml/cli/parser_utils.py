"""Lazy ``build_parser()`` loading for MMML subcommands."""

from __future__ import annotations

import argparse
import importlib
from typing import Any

from mmml.cli.registry import COMMAND_REGISTRY, command_by_name


def _import_parser_builder(module_path: str):
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return None
    return getattr(mod, "build_parser", None)


def get_subcommand_parser(command: str) -> argparse.ArgumentParser | None:
    """Return an ``ArgumentParser`` for ``mmml <command>`` (completion / introspection)."""
    spec = command_by_name(command)
    if spec is None:
        return None
    mod_path = spec.parser_module or spec.module
    builder = _import_parser_builder(mod_path)
    if builder is None:
        return None
    try:
        return builder()
    except Exception:
        return None


def parser_available(command: str) -> bool:
    return get_subcommand_parser(command) is not None


def parsers_with_flags() -> list[str]:
    return [spec.name for spec in COMMAND_REGISTRY if parser_available(spec.name)]


def parse_subcommand_args(command: str, argv: list[str] | None = None) -> argparse.Namespace | None:
    parser = get_subcommand_parser(command)
    if parser is None:
        return None
    return parser.parse_args(argv)

"""Lazy ``build_parser()`` loading for MMML subcommands."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
from pathlib import Path

from mmml.cli.registry import COMMAND_REGISTRY, command_by_name


def _module_defines_build_parser(module_path: str) -> bool:
    """Fast static check (no import) — used by ``mmml commands --audit``."""
    try:
        if module_path == "mmml" or module_path.startswith("mmml."):
            package_root = Path(__file__).resolve().parents[1]
            relative = module_path.split(".")[1:]
            module_file = package_root.joinpath(*relative).with_suffix(".py")
            if not module_file.is_file():
                module_file = package_root.joinpath(*relative, "__init__.py")
            if not module_file.is_file():
                return False
        else:
            spec = importlib.util.find_spec(module_path)
            if spec is None or spec.origin is None:
                return False
            module_file = Path(spec.origin)
        source = module_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_file))
    except Exception:
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "build_parser"
        for node in ast.walk(tree)
    )


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


def parser_available(command: str, *, import_module: bool = False) -> bool:
    """Return True when ``mmml <command> --help`` is wired via ``build_parser()``."""
    spec = command_by_name(command)
    if spec is None:
        return False
    mod_path = spec.parser_module or spec.module
    if import_module:
        return get_subcommand_parser(command) is not None
    return _module_defines_build_parser(mod_path)


def parsers_with_flags() -> list[str]:
    return [spec.name for spec in COMMAND_REGISTRY if parser_available(spec.name)]


def parse_subcommand_args(command: str, argv: list[str] | None = None) -> argparse.Namespace | None:
    parser = get_subcommand_parser(command)
    if parser is None:
        return None
    return parser.parse_args(argv)

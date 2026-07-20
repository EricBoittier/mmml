"""Shared functional grouping and terminal color for argparse help."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import TextIO

_GROUP_ORDER = (
    "Input & configuration",
    "Scientific model",
    "Execution",
    "Output & artifacts",
    "Diagnostics & safety",
    "Other options",
)

_TOKENS: dict[str, frozenset[str]] = {
    "Input & configuration": frozenset(
        {
            "config", "input", "checkpoint", "residue", "residues", "composition",
            "template", "dataset", "data", "structure", "pdb", "psf", "crd", "manifest",
            "interaction_policy", "calculator_config", "from_psf", "from_crd",
        }
    ),
    "Scientific model": frozenset(
        {
            "calculator", "method", "basis", "xc", "charge", "spin", "temperature",
            "pressure", "cutoff", "switch", "distance", "energy", "force", "field",
            "ensemble", "integrator", "thermostat", "barostat", "model", "ff", "units",
            "density", "restraint", "lambda", "multipole", "mbd", "ewald", "pme",
        }
    ),
    "Execution": frozenset(
        {
            "backend", "device", "seed", "steps", "nsteps", "ps", "dt", "workers",
            "threads", "mpi", "ranks", "batch", "epochs", "sampler", "builder", "setup",
            "resume", "continue", "job", "workflow", "preset", "stage", "run_all",
        }
    ),
    "Output & artifacts": frozenset(
        {
            "output", "output_dir", "out", "plot", "trajectory", "traj", "report", "json",
            "csv", "extxyz", "save", "write", "overwrite", "workdir", "log", "prefix",
        }
    ),
    "Diagnostics & safety": frozenset(
        {
            "help", "verbose", "quiet", "debug", "strict", "dry_run", "check", "validate",
            "diagnose", "audit", "allow_partial", "fail", "warning", "profile", "health",
            "list", "non_interactive", "no_rich",
        }
    ),
}

_HEADING_RE = re.compile(
    r"^(usage|options|optional arguments|positional arguments|subcommands|commands|"
    r"Input & configuration|Scientific model|Execution|Output & artifacts|"
    r"Diagnostics & safety|Other options):\s*$",
    re.IGNORECASE,
)
_FLAG_RE = re.compile(r"(?<![\w-])--?[A-Za-z0-9][A-Za-z0-9_-]*")
_CHOICE_RE = re.compile(r"\{[^{}\n]+\}")
_METAVAR_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:[A-Z][A-Z0-9_]{1,})(?![A-Za-z0-9_-])")
_DEFAULT_RE = re.compile(r"\(default:[^)]+\)", re.IGNORECASE)

_INSTALLED = False
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help
_ORIGINAL_PRINT_MESSAGE = argparse.ArgumentParser._print_message


def _words(action: argparse.Action) -> set[str]:
    values = [str(action.dest), *action.option_strings]
    words: set[str] = set()
    for value in values:
        normalized = value.lstrip("-").replace("-", "_").lower()
        words.add(normalized)
        words.update(part for part in normalized.split("_") if part)
    return words


def _classify(action: argparse.Action) -> str:
    words = _words(action)
    for group in _GROUP_ORDER[:-1]:
        tokens = _TOKENS[group]
        if words & tokens or any(token in word for token in tokens for word in words):
            return group
    return "Other options"


def group_parser_options(parser: argparse.ArgumentParser) -> None:
    """Group a flat parser once; preserve parsers with explicit custom groups."""

    if getattr(parser, "_mmml_functionally_grouped", False):
        return
    parser._mmml_functionally_grouped = True
    custom = [group for group in parser._action_groups if group not in {parser._positionals, parser._optionals}]
    if custom:
        return
    actions = list(parser._optionals._group_actions)
    if len(actions) < 5:
        return
    parser._optionals._group_actions.clear()
    groups = {name: parser.add_argument_group(name) for name in _GROUP_ORDER}
    for action in actions:
        groups[_classify(action)]._group_actions.append(action)


def styled_help_text(message: str):
    """Return Rich Text for already-formatted argparse/help catalog text."""

    from rich.text import Text

    text = Text(message)
    offset = 0
    for line in message.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if _HEADING_RE.match(bare.strip()):
            start = offset + len(bare) - len(bare.lstrip())
            text.stylize("bold cyan", start, offset + len(bare))
        elif bare.lower().startswith("usage:"):
            text.stylize("bold cyan", offset, offset + len("usage:"))
        for pattern, style in (
            (_FLAG_RE, "bold green"),
            (_CHOICE_RE, "yellow"),
            (_METAVAR_RE, "magenta"),
            (_DEFAULT_RE, "dim"),
        ):
            for match in pattern.finditer(bare):
                text.stylize(style, offset + match.start(), offset + match.end())
        offset += len(line)
    return text


def _use_color(stream: TextIO) -> bool:
    forced = (os.environ.get("MMML_RICH") or "").strip().lower() in {"1", "yes", "true"}
    disabled = (os.environ.get("MMML_NO_RICH") or "").strip().lower() in {"1", "yes", "true"}
    return not disabled and (forced or bool(getattr(stream, "isatty", lambda: False)()))


def print_cli_text(message: str, *, stream: TextIO | None = None) -> None:
    """Print grouped help/catalog text with color on terminals and plain otherwise."""

    target = stream or sys.stdout
    if not _use_color(target):
        target.write(message)
        if message and not message.endswith("\n"):
            target.write("\n")
        return
    from rich.console import Console

    Console(file=target, force_terminal=True).print(styled_help_text(message), end="")
    if message and not message.endswith("\n"):
        target.write("\n")


def install_colored_argparse() -> None:
    """Install the shared formatter once for parsers dispatched by ``mmml``."""

    global _INSTALLED
    if _INSTALLED:
        return

    def format_help(parser: argparse.ArgumentParser) -> str:
        group_parser_options(parser)
        return _ORIGINAL_FORMAT_HELP(parser)

    def print_message(parser: argparse.ArgumentParser, message=None, file=None) -> None:
        if not message:
            return
        target = file or sys.stderr
        if _use_color(target):
            print_cli_text(message, stream=target)
        else:
            _ORIGINAL_PRINT_MESSAGE(parser, message, target)

    argparse.ArgumentParser.format_help = format_help
    argparse.ArgumentParser._print_message = print_message
    _INSTALLED = True


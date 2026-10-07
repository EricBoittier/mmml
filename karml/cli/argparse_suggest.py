"""Fuzzy suggestions for argparse unrecognized-option errors."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections.abc import Iterable


_UNRECOGNIZED_RE = re.compile(
    r"unrecognized arguments:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def option_strings_from_parser(parser: argparse.ArgumentParser) -> list[str]:
    """Collect long/short option strings registered on ``parser`` (recursive)."""
    seen: set[str] = set()
    out: list[str] = []

    def _walk(p: argparse.ArgumentParser) -> None:
        for action in getattr(p, "_actions", []):
            for opt in getattr(action, "option_strings", []) or []:
                if opt not in seen:
                    seen.add(opt)
                    out.append(opt)
            if action.__class__.__name__ == "_SubParsersAction":
                for sub in getattr(action, "choices", {}).values():
                    if isinstance(sub, argparse.ArgumentParser):
                        _walk(sub)

    _walk(parser)
    return out


def suggest_close_option_flags(
    unknown_tokens: Iterable[str],
    known_flags: Iterable[str],
    *,
    n: int = 3,
    cutoff: float = 0.55,
) -> list[tuple[str, list[str]]]:
    """Return ``(bad_flag, [suggestions])`` for tokens that look like options."""
    known = [str(f) for f in known_flags]
    pairs: list[tuple[str, list[str]]] = []
    for tok in unknown_tokens:
        text = str(tok).strip()
        if not text.startswith("-"):
            continue
        # Drop attached values: --flag=800 → --flag
        flag = text.split("=", 1)[0]
        close = difflib.get_close_matches(flag, known, n=n, cutoff=cutoff)
        if close:
            pairs.append((flag, close))
    return pairs


def format_unrecognized_suggestions(
    message: str,
    known_flags: Iterable[str],
) -> str | None:
    """If ``message`` is an unrecognized-args error, return a suggestion suffix."""
    match = _UNRECOGNIZED_RE.search(message.strip())
    if match is None:
        return None
    tokens = match.group(1).split()
    pairs = suggest_close_option_flags(tokens, known_flags)
    if not pairs:
        return None
    lines = ["Did you mean:"]
    for bad, close in pairs:
        lines.append(f"  {bad} → {', '.join(close)}")
    return "\n".join(lines)


class SuggestingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that fuzzy-matches unknown option flags."""

    def error(self, message: str) -> None:  # type: ignore[override]
        hint = format_unrecognized_suggestions(
            message,
            option_strings_from_parser(self),
        )
        if hint:
            message = f"{message}\n{hint}"
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")

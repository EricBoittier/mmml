"""Allowlisted mmml subcommands the MCP server may invoke."""

from __future__ import annotations

from mmml.cli.registry import COMMAND_REGISTRY, CommandSpec

# Commands safe for agent-driven orchestration (no raw shell, no deprecated unless noted).
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "health-check",
        "make-res",
        "make-box",
        "liquid-box",
        "sample-diverse-xyz",
        "normal-mode-sample",
        "interpolate-xyz",
        "pyscf-dft",
        "pyscf-evaluate",
        "fix-and-split",
        "physnet-train",
        "physnet-evaluate",
        "train-joint",
        "efield-train",
        "md-system",
        "physnet-md",
        "efield-md",
        "warmup-mlpot-jax",
        "validate",
        "cross-check",
        "configure",
    }
)

# Console scripts invoked outside ``mmml`` dispatch.
ALLOWED_CONSOLE_SCRIPTS: frozenset[str] = frozenset({"mmml-spectra-md"})


def command_spec(name: str) -> CommandSpec | None:
    for spec in COMMAND_REGISTRY:
        if spec.name == name:
            return spec
    return None


def is_allowed_mmml_command(name: str) -> bool:
    return name in ALLOWED_COMMANDS


def is_allowed_console_script(name: str) -> bool:
    return name in ALLOWED_CONSOLE_SCRIPTS


def validate_cli_args(args: list[str]) -> list[str]:
    """Reject shell injection patterns in forwarded CLI args."""
    cleaned: list[str] = []
    for arg in args:
        text = str(arg)
        if not text:
            raise ValueError("empty CLI argument")
        if any(ch in text for ch in (";", "|", "&", "$", "`", "\n", "\r")):
            raise ValueError(f"disallowed shell metacharacter in argument: {text!r}")
        cleaned.append(text)
    return cleaned

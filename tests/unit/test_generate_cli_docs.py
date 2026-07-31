"""Tests for CLI docs generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_generate_cli_docs_writes_all_registry_commands():
    from mmml.cli.registry import COMMAND_REGISTRY

    commands_dir = REPO / "docs" / "cli" / "commands"
    for spec in COMMAND_REGISTRY:
        path = commands_dir / f"{spec.name}.md"
        assert path.is_file(), f"missing generated page: {path}"
        text = path.read_text(encoding="utf-8")
        assert f"# `mmml {spec.name}`" in text
        assert spec.summary in text


def test_generate_cli_docs_check_is_clean():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_cli_docs.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_mkdocs_has_cli_nav_markers():
    text = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert "# CLI_NAV_START" in text
    assert "# CLI_NAV_END" in text
    assert "cli/commands/md-system.md" in text


def _nav_groups():
    sys.path.insert(0, str(REPO / "scripts"))
    from generate_cli_docs import CLI_NAV_GROUPS

    return CLI_NAV_GROUPS


def test_every_registry_command_has_exactly_one_nav_group():
    from mmml.cli.registry import COMMAND_REGISTRY

    placements: dict[str, list[str]] = {}
    for group, names in _nav_groups():
        for name in names:
            placements.setdefault(name, []).append(group)

    duplicated = {n: g for n, g in placements.items() if len(g) > 1}
    assert not duplicated, f"commands in multiple nav groups: {duplicated}"

    missing = sorted({spec.name for spec in COMMAND_REGISTRY} - set(placements))
    assert not missing, (
        "registry commands with no docs nav group (add them to CLI_NAV_GROUPS "
        f"in scripts/generate_cli_docs.py): {missing}"
    )


def test_docs_nav_groups_track_cli_command_groups():
    """The sidebar must read like `mmml commands` — same task-group names."""
    from mmml.cli.help_text import COMMAND_GROUPS

    nav_names = {group for group, _ in _nav_groups()}
    # "Other" is the CLI's catch-all; docs place those commands in real groups.
    cli_names = {group for group, _ in COMMAND_GROUPS} - {"Other"}
    drifted = sorted(cli_names - nav_names)
    assert not drifted, (
        "CLI task groups with no matching docs nav section: "
        f"{drifted}. Rename the section in scripts/generate_cli_docs.py "
        "and mkdocs.yml, or rename the CLI group."
    )


def test_no_nav_group_is_empty():
    """An empty marker block yields a null nav value and breaks `mkdocs build`."""
    from mmml.cli.registry import COMMAND_REGISTRY

    registry_names = {spec.name for spec in COMMAND_REGISTRY}
    for group, names in _nav_groups():
        assert [n for n in names if n in registry_names], (
            f"nav group {group!r} matches no registry command"
        )

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

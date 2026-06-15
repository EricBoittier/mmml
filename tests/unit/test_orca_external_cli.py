"""CLI dispatch tests for ORCA external-tool commands."""

from __future__ import annotations

import subprocess
import sys


def test_orca_server_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mmml.cli.__main__", "orca-server", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--checkpoint" in proc.stdout
    assert "--warmup" in proc.stdout


def test_orca_client_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mmml.cli.__main__", "orca-client", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--bind" in proc.stdout


def test_orca_external_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mmml.cli.__main__", "orca-external", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "inputfile" in proc.stdout

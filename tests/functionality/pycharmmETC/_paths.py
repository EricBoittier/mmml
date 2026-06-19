"""Shared paths for PyCHARMM integration tests."""

from __future__ import annotations

from pathlib import Path

PYCHARMMETC_DIR = Path(__file__).resolve().parent


def fixture_pdb(name: str) -> Path:
    """Return a committed read-only PDB fixture under pycharmmETC/pdb/."""
    return PYCHARMMETC_DIR / "pdb" / name

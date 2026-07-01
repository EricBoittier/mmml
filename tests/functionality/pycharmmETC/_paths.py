"""Shared paths for PyCHARMM integration tests."""

from __future__ import annotations

from pathlib import Path

PYCHARMMETC_DIR = Path(__file__).resolve().parent


def fixture_pdb(name: str) -> Path:
    """Return a committed read-only PDB fixture under pycharmmETC/pdb/."""
    return PYCHARMMETC_DIR / "pdb" / name


def workdir_pdb(name: str) -> str:
    """Relative PDB path for CHARMM OPEN inside ``pycharmm_workdir`` (copied seed)."""
    return f"pdb/{name}"


def workdir_psf(name: str) -> str:
    """Relative PSF path for CHARMM OPEN inside ``pycharmm_workdir`` (copied seed)."""
    return f"psf/{name}"

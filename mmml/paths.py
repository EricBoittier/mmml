"""Paths to non-Python files bundled with the ``mmml`` package."""

from __future__ import annotations

from pathlib import Path

from importlib.resources import files


def bundled_file(*parts: str) -> Path:
    """Return an on-disk path to a file declared in setuptools package-data."""
    return Path(files("mmml").joinpath(*parts))


def default_meoh_template_pdb() -> Path:
    """Default monomer PDB for Packmol / cluster builders."""
    return bundled_file("generate", "sample", "pdb", "meoh.pdb")


def crystal_image_str_source() -> Path:
    """CHARMM periodic-image helper copied into the working directory when needed."""
    return bundled_file("data", "charmm", "crystal_image.str")

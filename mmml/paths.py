"""Paths to non-Python files bundled with the ``mmml`` package."""

from __future__ import annotations

from pathlib import Path

def _package_dir() -> Path:
    """Directory containing ``paths.py`` (the installed ``mmml`` package root)."""
    return Path(__file__).resolve().parent


def bundled_file(*parts: str) -> Path:
    """Return an on-disk path to a file declared in setuptools package-data."""
    return _package_dir().joinpath(*parts)


def default_meoh_template_pdb() -> Path:
    """Default monomer PDB for Packmol / cluster builders."""
    return bundled_file("generate", "sample", "pdb", "meoh.pdb")


def default_aco_template_pdb() -> Path:
    """Default acetone (CGenFF ``ACO``) monomer geometry for cluster builders."""
    return bundled_file("generate", "sample", "pdb", "aco_monomer.pdb")


def crystal_image_str_source() -> Path:
    """CHARMM periodic-image helper copied into the working directory when needed."""
    return bundled_file("data", "charmm", "crystal_image.str")


def default_dcm_molecule_xyz() -> Path:
    """Bundled DCM (CH2Cl2) monomer XYZ for ``build-crystal`` / PyXtal."""
    return bundled_file("data", "molecules", "dcm.xyz")


def default_dcm_crystal_cif() -> Path:
    """Experimental DCM crystal (Pbcn, COD 2100015 / CCDC doi:10.5517/cc9lyjb)."""
    return bundled_file("data", "structures", "dcm_pbcn_cod2100015.cif")


def default_benzene_crystal_cif() -> Path:
    """Experimental benzene crystal (P2₁/c, COD 4501704)."""
    return bundled_file("data", "structures", "benzene_p21c_cod4501704.cif")


def default_trialanine_water_smoke_extxyz() -> Path:
    """Bundled illustrative tri-alanine + TIP3 grid (docs / figure CI)."""
    return bundled_file("data", "charmm", "trialanine-water-smoke.extxyz")

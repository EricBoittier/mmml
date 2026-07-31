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


def default_tip3_template_pdb() -> Path:
    """Default TIP3 water monomer PDB (OH2, H1, H2) for cluster builders."""
    return bundled_file("data", "charmm", "tip3.pdb")


def crystal_image_str_source() -> Path:
    """CHARMM periodic-image helper copied into the working directory when needed."""
    return bundled_file("data", "charmm", "crystal_image.str")


def default_dcm_molecule_xyz() -> Path:
    """Bundled DCM (CH2Cl2) monomer XYZ for ``build-crystal`` / PyXtal."""
    return bundled_file("data", "molecules", "dcm.xyz")


# The two pressure points of Podsiadło, Dziubek & Katrusiak, Acta Crystallogr. B
# 61, 595 (2005) (doi:10.1107/S0108768105017374), redistributed by the
# Crystallography Open Database.  These are the only pure CH2Cl2 structures in
# COD; the ambient-pressure phase (Kawaguchi et al. 1973) has no open CIF.
DCM_CRYSTAL_CIFS: dict[str, str] = {
    "pbcn_133gpa": "dcm_pbcn_133gpa_cod2100014.cif",
    "pbcn_163gpa": "dcm_pbcn_cod2100015.cif",
}


def default_dcm_crystal_cif(phase: str = "pbcn_163gpa") -> Path:
    """Experimental DCM crystal for one pressure point of Podsiadło et al. (2005).

    Defaults to the 1.63 GPa structure (COD 2100015 / CCDC doi:10.5517/cc9lyjb)
    because that is what the ``dcm`` ``build-crystal`` preset and the literature
    cross-check table have always used.
    """
    key = phase.strip().lower()
    try:
        filename = DCM_CRYSTAL_CIFS[key]
    except KeyError:
        known = ", ".join(sorted(DCM_CRYSTAL_CIFS))
        raise KeyError(f"Unknown DCM phase {phase!r}; known phases: {known}") from None
    return bundled_file("data", "structures", filename)


def default_benzene_crystal_cif() -> Path:
    """Experimental benzene crystal (P2₁/c, COD 4501704)."""
    return bundled_file("data", "structures", "benzene_p21c_cod4501704.cif")


# The five acetone phases of Allan, Clark, Ibberson, Parsons, Pulham & Sawyer,
# Chem. Commun. 1999, 751 (doi:10.1039/a900558g), deposited as CCDC 182/1197 and
# redistributed by the Crystallography Open Database.  Keys are the phase labels
# used by ``ACETONE_CRYSTAL_PHASES``.
ACETONE_CRYSTAL_CIFS: dict[str, str] = {
    "pbca_5k": "acetone_pbca_5k_cod7110465.cif",
    "pbca_110k": "acetone_pbca_110k_cod7110466.cif",
    "pbca_150k": "acetone_pbca_150k_cod7110464.cif",
    "cmcm_160k": "acetone_cmcm_160k_cod7110463.cif",
    "cmcm_15kbar": "acetone_cmcm_15kbar_cod7110462.cif",
}


def default_acetone_crystal_cif(phase: str = "pbca_150k") -> Path:
    """Experimental acetone crystal for one phase of Allan et al. (1999).

    ``pbca_150k`` is the default because it is the stable low-temperature phase
    with ordered, refined hydrogens -- the 5 K entry is the deuterated neutron
    refinement and the 15 kbar entry has rotationally disordered methyls.
    """
    key = phase.strip().lower()
    try:
        filename = ACETONE_CRYSTAL_CIFS[key]
    except KeyError:
        known = ", ".join(sorted(ACETONE_CRYSTAL_CIFS))
        raise KeyError(f"Unknown acetone phase {phase!r}; known phases: {known}") from None
    return bundled_file("data", "structures", filename)


def default_trialanine_water_smoke_extxyz() -> Path:
    """Bundled CHARMM-built tri-alanine + TIP3 box (docs / figure CI)."""
    return bundled_file("data", "charmm", "trialanine-water-smoke.extxyz")


def default_trialanine_water_smoke_pdb() -> Path:
    """CHARMM PDB for the bundled trialanine water smoke box."""
    return bundled_file("data", "charmm", "trialanine-water-smoke.pdb")


def default_make_box_aco_pdb() -> Path:
    """Packmol-packed 8× ACO in a 22 Å cube (``make-box`` docs figure)."""
    return bundled_file("data", "structures", "make-box-aco-8x22A.pdb")


def default_alad_reference_pdb() -> Path:
    """CHARMM36 ACE–ALA–CT3 reference (protein docs figure)."""
    return bundled_file("data", "charmm", "alad_reference.pdb")


def default_trialanine_cgenff_rtf() -> Path:
    """Bundled ``RESI TRIA`` (TRIALANINE) supplemental CGENFF topology."""
    return bundled_file("data", "charmm", "top_trialanine_cgenff.rtf")


def default_trialanine_backbone_cmap_prm() -> Path:
    """Bundled CMAP grid for TRIA backbone (CGENFF atom-type headers)."""
    return bundled_file("data", "charmm", "par_trialanine_backbone_cmap.prm")

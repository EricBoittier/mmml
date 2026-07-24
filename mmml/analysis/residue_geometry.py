"""Resolve monomer geometries for CGenFF residue names.

Lookup order
------------
1. Campaign dimers in ``mmml.analysis.dimer_molecules.MOLECULES`` (plus ACE↔ACO).
2. Working-directory ``pdb/<resi>.pdb`` (from a prior ``make-res``).
3. Bundled package templates (TIP3, OCOH, ACO, MEOH, DCM, BENZ).
4. Optional ``make-res`` generation via PyCHARMM (when ``generate=True``).
"""

from __future__ import annotations

from pathlib import Path

import ase.io
from ase import Atoms

from mmml.interfaces.pycharmmInterface.cgenff_residues import (
    normalize_cgenff_residue_name,
)
from mmml.paths import (
    bundled_file,
    default_aco_template_pdb,
    default_meoh_template_pdb,
    default_tip3_template_pdb,
)

# CGenFF RESI → campaign dimer label when geometries are shared.
_CAMPAIGN_GEOMETRY_ALIASES: dict[str, str] = {
    "ACO": "ACE",
}

# Approximate liquid densities (kg/m³) for common solvents.
KNOWN_SOLVENT_DENSITY_KG_M3: dict[str, float] = {
    "TIP3": 1000.0,
    "OCOH": 824.0,
}


def _correct_pdb_symbols(atoms: Atoms) -> Atoms:
    """Fix CHARMM-style element symbols that ASE misreads (CL, HO, …)."""
    problem_symbols = {"CL", "HO"}
    symbols = atoms.get_chemical_symbols()
    fixed = [
        s[:1] if s.upper() in problem_symbols else ("H" if s[0] == "H" else s)
        for s in symbols
    ]
    atoms = atoms.copy()
    atoms.set_chemical_symbols(fixed)
    return atoms


def _read_monomer_pdb(path: Path) -> Atoms:
    atoms = ase.io.read(str(path))
    if not isinstance(atoms, Atoms):
        raise TypeError(f"expected a single ASE Atoms from {path}")
    return _correct_pdb_symbols(atoms)


def bundled_monomer_pdb(residue: str) -> Path | None:
    """Return a bundled monomer PDB for *residue*, or ``None`` if unavailable."""
    name = normalize_cgenff_residue_name(residue)
    mapping: dict[str, Path] = {
        "TIP3": default_tip3_template_pdb(),
        "OCOH": bundled_file("data", "charmm", "ocoh.pdb"),
        "ACO": default_aco_template_pdb(),
        "MEOH": default_meoh_template_pdb(),
        "DCM": bundled_file("data", "molecules", "dcm_monomer.pdb"),
        "BENZ": bundled_file("data", "molecules", "benz_monomer.pdb"),
    }
    path = mapping.get(name)
    if path is not None and path.is_file():
        return path
    return None


def known_solvent_density_kg_m3(residue: str) -> float | None:
    """Built-in solvent density in kg/m³, or ``None`` if unknown."""
    return KNOWN_SOLVENT_DENSITY_KG_M3.get(normalize_cgenff_residue_name(residue))


def resolve_solvent_density_kg_m3(
    residue: str,
    density: float | None = None,
) -> float:
    """Return density in kg/m³; require an explicit value when not built-in."""
    if density is not None:
        value = float(density)
        if value <= 0.0:
            raise ValueError(f"density must be positive, got {value}")
        return value
    known = known_solvent_density_kg_m3(residue)
    if known is not None:
        return known
    name = normalize_cgenff_residue_name(residue)
    raise ValueError(
        f"No built-in density for solvent {name!r}; pass --density in kg/m³ "
        f"(e.g. water≈1000, methanol≈792)."
    )


def load_residue_monomer_atoms(
    residue: str,
    *,
    generate: bool = False,
) -> Atoms:
    """Load a single-residue geometry for *residue*.

    Parameters
    ----------
    residue:
        CGenFF RESI name (aliases like ``water`` → ``TIP3`` are accepted).
    generate:
        If True and no template/cwd PDB is found, call ``make-res`` (needs PyCHARMM).
    """
    name = normalize_cgenff_residue_name(residue)

    # Lazy import avoids circular import with dimer_molecules.
    from mmml.analysis.dimer_molecules import MOLECULES

    if name in MOLECULES:
        return MOLECULES[name].copy()
    campaign = _CAMPAIGN_GEOMETRY_ALIASES.get(name)
    if campaign is not None and campaign in MOLECULES:
        return MOLECULES[campaign].copy()

    for candidate in (
        Path(f"pdb/{name.lower()}.pdb"),
        Path(f"xyz/{name.lower()}.xyz"),
    ):
        if candidate.is_file():
            return _read_monomer_pdb(candidate)

    bundled = bundled_monomer_pdb(name)
    if bundled is not None:
        return _read_monomer_pdb(bundled)

    if generate:
        return _generate_monomer_via_make_res(name)

    raise FileNotFoundError(
        f"No monomer geometry for residue {name!r}. Run "
        f"'mmml make-res --res {name} --skip-energy-show' or pass a bundled/"
        "working-directory PDB."
    )


def ensure_residue_pdb(
    residue: str,
    *,
    generate: bool = True,
    dest: Path | str | None = None,
) -> Path:
    """Ensure ``pdb/<resi>.pdb`` exists; return its path.

    Preserves an existing ``pdb/initial.pdb`` when generating via ``make-res``.
    """
    name = normalize_cgenff_residue_name(residue)
    out = Path(dest) if dest is not None else Path("pdb") / f"{name.lower()}.pdb"
    if out.is_file():
        return out.resolve()

    bundled = bundled_monomer_pdb(name)
    if bundled is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(bundled.read_bytes())
        return out.resolve()

    atoms = load_residue_monomer_atoms(name, generate=generate)
    out.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(out), atoms)
    return out.resolve()


def _generate_monomer_via_make_res(name: str) -> Atoms:
    """Generate a residue with ``make-res``, restoring ``pdb/initial.pdb`` afterward."""
    import argparse

    from mmml.cli.make import make_res

    initial = Path("pdb/initial.pdb")
    backup = initial.read_bytes() if initial.is_file() else None
    try:
        atoms = make_res.main_loop(
            argparse.Namespace(res=name, skip_energy_show=True)
        )
        out = Path("pdb") / f"{name.lower()}.pdb"
        out.parent.mkdir(parents=True, exist_ok=True)
        ase.io.write(str(out), atoms)
        return atoms
    finally:
        if backup is not None:
            initial.parent.mkdir(parents=True, exist_ok=True)
            initial.write_bytes(backup)

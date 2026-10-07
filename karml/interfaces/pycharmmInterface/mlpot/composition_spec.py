"""Composition string parsing for md-system / liquid-box / Packmol.

Supports CGenFF ``RES:N`` tokens and PDB path tokens:

* ``DCM:60`` / ``ACO:4,MEOH:2`` — CGenFF residues (validated against the RTF)
* ``solute.pdb:1,DCM:200`` — Packmol mix (PDB is a single-residue monomer template)
* ``system.pdb`` — full-system cold start (CHARMM ``READ SEQU PDB``)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mmml.interfaces.pycharmmInterface.cgenff_residues import require_cgenff_residue_name

CompositionMode = Literal["cgenff", "packmol_pdb", "full_system_pdb"]


@dataclass(frozen=True, slots=True)
class CompositionEntry:
    """One composition species after parsing.

    ``residue`` is the resolved CGenFF RESN.  For full-system PDBs (multi-residue),
    ``residue`` is a tag derived from the file stem (not used as a PSF sequence).
    """

    residue: str
    count: int
    pdb_path: Path | None = None


def is_composition_pdb_token(token: str) -> bool:
    """True when a composition left-hand token refers to a PDB path."""
    text = str(token).strip()
    if not text:
        return False
    lower = text.lower()
    if lower.endswith(".pdb"):
        return True
    if "/" in text or "\\" in text or text.startswith("."):
        return True
    return False


def _split_composition_token(tok: str) -> tuple[str, int]:
    """Split ``RES:N`` / ``path.pdb:N``; bare token means count 1."""
    tok = tok.strip()
    if not tok:
        raise ValueError("Empty composition token")
    if ":" in tok:
        left, right = tok.rsplit(":", 1)
        right = right.strip()
        if right.isdigit():
            count = int(right)
            left = left.strip()
            if not left or count <= 0:
                raise ValueError(f"Invalid composition token: '{tok}'")
            return left, count
    return tok, 1


def read_pdb_residue_names(pdb_path: Path | str) -> list[str]:
    """Return RESN for each ATOM/HETATM record (uppercase, stripped)."""
    path = Path(pdb_path)
    resnames: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if len(line) < 20:
                raise RuntimeError(f"Truncated PDB ATOM record in {path}")
            resnames.append(line[17:20].strip().upper())
    if not resnames:
        raise RuntimeError(f"No ATOM/HETATM records found in {path}")
    return resnames


def read_pdb_cryst1_side_A(pdb_path: Path | str) -> float | None:
    """Return cubic CRYST1 a-axis (Å) when present and positive; else None."""
    path = Path(pdb_path)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("CRYST1"):
                continue
            try:
                a = float(line[6:15])
            except ValueError:
                return None
            return a if a > 0.0 else None
    return None


def _element_symbol_from_pdb_atom_name(atom_name: str) -> str:
    """Best-effort element symbol from a CHARMM PDB atom name."""
    letters = "".join(c for c in str(atom_name) if c.isalpha())
    if not letters:
        return "C"
    upper = letters.upper()
    if upper.startswith("CL"):
        return "Cl"
    if upper.startswith("BR"):
        return "Br"
    if upper.startswith("H"):
        return "H"
    return letters[0].upper()


def load_monomer_geometry_from_pdb(
    pdb_path: Path | str,
) -> tuple[str, object, list[str], object]:
    """Load a single-residue CGenFF monomer PDB for Packmol templates.

    Returns ``(resname, coords, atom_names, atomic_numbers)``.
    """
    import numpy as np
    from ase.data import atomic_numbers, chemical_symbols

    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        _parse_pdb_atom_records,
    )

    path = Path(pdb_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Composition PDB not found: {path}")

    names, resids, positions = _parse_pdb_atom_records(path)
    resnames = read_pdb_residue_names(path)
    if len(resnames) != len(names):
        raise RuntimeError(f"RESN/atom count mismatch in {path}")

    unique_resn = {r for r in resnames if r}
    unique_resid = set(int(x) for x in resids)
    if len(unique_resn) != 1:
        raise ValueError(
            f"Packmol monomer PDB must contain a single residue name; "
            f"found {sorted(unique_resn)} in {path}"
        )
    if len(unique_resid) != 1:
        raise ValueError(
            f"Packmol monomer PDB must contain a single residue number; "
            f"found {sorted(unique_resid)} in {path}"
        )

    resname = require_cgenff_residue_name(next(iter(unique_resn)))

    elements: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            elem = line[76:78].strip() if len(line) >= 78 else ""
            if elem:
                if len(elem) == 2:
                    elements.append(elem[0].upper() + elem[1].lower())
                else:
                    elements.append(elem.upper())
            else:
                elements.append(
                    _element_symbol_from_pdb_atom_name(line[12:16].strip())
                )

    sym_map = {s.upper(): n for n, s in enumerate(chemical_symbols) if s}
    z_list: list[int] = []
    for elem in elements:
        key = str(elem).strip()
        if key in atomic_numbers:
            z_list.append(int(atomic_numbers[key]))
        else:
            z_list.append(int(sym_map.get(key.upper(), 6)))

    coords = np.asarray(positions, dtype=float)
    z_arr = np.asarray(z_list, dtype=int)
    if int(z_arr.shape[0]) != int(coords.shape[0]):
        raise RuntimeError(
            f"Element count ({z_arr.shape[0]}) != coords ({coords.shape[0]}) in {path}"
        )
    return resname, coords, [str(n) for n in names], z_arr


def parse_composition_entries(
    spec: str,
    *,
    validate_cgenff: bool = True,
    resolve_pdb_files: bool = True,
) -> list[CompositionEntry]:
    """Parse ``RES:N`` / ``path.pdb:N`` composition into structured entries."""
    out: list[CompositionEntry] = []
    for raw in str(spec).split(","):
        tok = raw.strip()
        if not tok:
            continue
        left, count = _split_composition_token(tok)
        if is_composition_pdb_token(left):
            path = Path(left).expanduser()
            if resolve_pdb_files:
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Composition PDB not found: {path}")
            # Mode is decided after all tokens are collected; for monomers we
            # resolve RESN now when the file exists. Full-system (lone PDB:1)
            # may be multi-residue — defer strict single-res check to mode.
            residue = path.stem.upper()[:6] or "SYS"
            if resolve_pdb_files and path.is_file():
                try:
                    resnames = read_pdb_residue_names(path)
                    unique = {r for r in resnames if r}
                    if len(unique) == 1:
                        residue = (
                            require_cgenff_residue_name(next(iter(unique)))
                            if validate_cgenff
                            else next(iter(unique)).upper()
                        )
                except RuntimeError:
                    pass
            out.append(CompositionEntry(residue=residue, count=count, pdb_path=path))
        else:
            residue = left.strip().upper()
            if validate_cgenff:
                residue = require_cgenff_residue_name(residue)
            else:
                residue = residue.upper()
            if not residue or count <= 0:
                raise ValueError(f"Invalid composition token: '{tok}'")
            out.append(CompositionEntry(residue=residue, count=count, pdb_path=None))
    if not out:
        raise ValueError("Empty composition")
    return out


def composition_mode(entries: list[CompositionEntry]) -> CompositionMode:
    """Classify composition as CGenFF-only, Packmol PDB mix, or full-system PDB."""
    pdb_entries = [e for e in entries if e.pdb_path is not None]
    if not pdb_entries:
        return "cgenff"
    if (
        len(entries) == 1
        and entries[0].pdb_path is not None
        and int(entries[0].count) == 1
    ):
        return "full_system_pdb"
    return "packmol_pdb"


def composition_as_pairs(entries: list[CompositionEntry]) -> list[tuple[str, int]]:
    """``[(RES, N), ...]`` for PSF / Packmol sequence builders."""
    return [(e.residue, int(e.count)) for e in entries]


def composition_pdb_templates(entries: list[CompositionEntry]) -> dict[str, Path]:
    """Map resolved RESN → monomer PDB path for Packmol templates."""
    out: dict[str, Path] = {}
    for entry in entries:
        if entry.pdb_path is None:
            continue
        key = str(entry.residue).upper()
        path = Path(entry.pdb_path)
        if key in out and out[key].resolve() != path.resolve():
            raise ValueError(
                f"Conflicting monomer PDBs for residue {key}: {out[key]} vs {path}"
            )
        out[key] = path
    return out


def ensure_packmol_pdb_monomers(entries: list[CompositionEntry]) -> list[CompositionEntry]:
    """Validate Packmol-mix PDB tokens are single-residue CGenFF monomers.

    Returns entries with RESN refreshed from each monomer PDB.
    """
    if composition_mode(entries) != "packmol_pdb":
        return entries
    refreshed: list[CompositionEntry] = []
    for entry in entries:
        if entry.pdb_path is None:
            refreshed.append(entry)
            continue
        resname, _coords, _names, _z = load_monomer_geometry_from_pdb(entry.pdb_path)
        refreshed.append(
            CompositionEntry(
                residue=resname,
                count=int(entry.count),
                pdb_path=Path(entry.pdb_path),
            )
        )
    return refreshed


def reject_pdb_composition_for_builder(
    entries: list[CompositionEntry],
    *,
    builder: str | None = None,
    packmol: bool | None = None,
    pyxtal: bool | None = None,
) -> None:
    """Raise when Packmol-mix PDB tokens are incompatible with the selected builder."""
    mode = composition_mode(entries)
    if mode != "packmol_pdb":
        return
    if pyxtal is True or (builder or "").lower() == "crystal":
        raise ValueError(
            "PDB composition tokens require Packmol (or a lone full-system PDB); "
            "not compatible with PyXtal / --builder crystal"
        )
    if packmol is False:
        raise ValueError(
            "PDB composition tokens require Packmol; remove --no-packmol"
        )


def apply_from_pdb_alias(args: Any) -> None:
    """If ``--from-pdb`` is set, treat it as a lone full-system composition PDB."""
    from_pdb = getattr(args, "from_pdb", None)
    if from_pdb is None:
        return
    path = Path(str(from_pdb)).expanduser()
    if getattr(args, "from_psf", None) or getattr(args, "from_crd", None):
        raise ValueError("--from-pdb is mutually exclusive with --from-psf/--from-crd")
    comp = getattr(args, "composition", None)
    if comp is not None and str(comp).strip():
        # Allow identical lone-PDB composition; reject mixes.
        entries = parse_composition_entries(str(comp), resolve_pdb_files=True)
        mode = composition_mode(entries)
        if mode != "full_system_pdb":
            raise ValueError(
                "--from-pdb cannot be combined with a multi-token / Packmol composition; "
                "use composition path tokens alone (e.g. solute.pdb:1,DCM:200) without --from-pdb"
            )
        # Prefer explicit --from-pdb path
    setattr(args, "composition", str(path))
    setattr(args, "from_pdb", path)


def resolve_composition_plan(
    composition_spec: str,
    *,
    builder: str | None = None,
    packmol: bool | None = None,
    pyxtal: bool | None = None,
) -> tuple[list[CompositionEntry], CompositionMode, list[tuple[str, int]], dict[str, Path] | None]:
    """Parse composition and return entries, mode, RES:N pairs, and PDB templates."""
    entries = parse_composition_entries(composition_spec)
    mode = composition_mode(entries)
    if mode == "packmol_pdb":
        entries = ensure_packmol_pdb_monomers(entries)
        reject_pdb_composition_for_builder(
            entries, builder=builder, packmol=packmol, pyxtal=pyxtal
        )
    elif mode == "full_system_pdb":
        reject_pdb_composition_for_builder(
            entries, builder=builder, packmol=packmol, pyxtal=pyxtal
        )
    elif mode == "cgenff":
        pass
    pairs = composition_as_pairs(entries)
    templates = composition_pdb_templates(entries) if mode == "packmol_pdb" else None
    return entries, mode, pairs, templates


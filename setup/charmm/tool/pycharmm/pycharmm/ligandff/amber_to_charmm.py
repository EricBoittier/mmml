"""
Parse an AMBER ``prmtop`` (via parmed) into a CHARMM :class:`~model.ForceField`.

This is the rewritten core of C. L. Brooks III's ``amber_2_charmm_utils``: it
reads a parmed ``AmberParm`` and builds the force-field data model, doing only
extraction (no formatting -- the writer handles that). It fixes three problems
in the original while preserving its chemistry:

* impropers are identified by parmed's ``dihedral.improper`` flag rather than by
  matching against a raw ``DIHEDRALS_*`` index array, which crashed on molecules
  with zero impropers (empty-array broadcast);
* a bond to a *non-adjacent* residue (a disulfide SG-SG cross-link) is no longer
  emitted inside a ``RESI`` as a bogus ``+SG`` -- cross-links belong to a patch;
* parameters are accumulated in plain lists instead of ``pd.concat`` inside a
  loop (which was O(n^2)).

De-duplication and sorting are deliberately *not* done here; the writer owns
them (parameters recur across residues by design).

Notes
-----
Two hardcoded pieces of Brooks's parm19sb protein workflow are exposed as
helpers rather than baked into the parser: :func:`disulfide_parameters` and
:func:`opc_water_angle`. The protein driver injects them; the ligand path does
not use them.
"""

from __future__ import annotations

from .model import (
    AngleType,
    AtomType,
    BondType,
    CmapType,
    DihedralType,
    ForceField,
    ImproperType,
    LonePair,
    Residue,
    TopologyAtom,
)

# CA atom-type remap that enables the parm19sb phi/psi CMAP in CHARMM, keyed by
# residue name (Brooks's cmap_at).
_CMAP_CA_TYPE = {
    "ALA": "XC1",
    "ARG": "XC2",
    "ASH": "XC3",
    "ASN": "XC4",
    "ASP": "XC5",
    "CYM": "XC6",
    "CYS": "XC7",
    "CYX": "XC8",
    "GLH": "XC9",
    "GLN": "XC10",
    "GLU": "XC11",
    "GLY": "XC12",
    "HID": "XC6",
    "HIE": "XC6",
    "HIP": "XC6",
    "HYP": "XC6",
    "ILE": "XC13",
    "LEU": "XC6",
    "LYN": "XC14",
    "LYS": "XC14",
    "MET": "XC6",
    "PHE": "XC6",
    "PRO": "XC15",
    "SER": "XC16",
    "THR": "XC17",
    "TRP": "XC6",
    "TYR": "XC6",
    "VAL": "XC13",
}

# Atom-type replacements by residue: TRP C* and OPC-water lone pair (Brooks's
# atom_map); index [1] is the replacement type.
_ATOM_TYPE_REMAP = {"TRP": ["C*", "CZ0"], "WAT": ["EP", "LP"]}

# sigma (angstrom) -> CHARMM Rmin/2 (angstrom): Rmin = 2**(1/6) * sigma.
_SIGMA_TO_RMIN_HALF = 2 ** (1 / 6) / 2


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_forcefield(parm, *, ligand: bool) -> ForceField:
    """Build a :class:`~model.ForceField` from a parmed ``AmberParm``.

    Extraction only: one record per source item, one :class:`~model.Residue`
    per prmtop residue, no de-duplication and no terminal renaming (the writer
    de-duplicates; the protein driver renames terminals).

    Parameters
    ----------
    parm : parmed.amber.AmberParm
        A loaded AMBER topology (``parmed.load_file("...prmtop")``).
    ligand : bool
        If True, prepend ``_`` to every atom-type name so a GAFF ligand's types
        cannot collide with protein types loaded alongside it.

    Returns
    -------
    ForceField
        Atom types, bonded/CMAP parameters, and residue topologies. Records may
        contain duplicates; the writer de-duplicates by CHARMM key.

    Raises
    ------
    ValueError
        If an improper or CMAP term spans residues so widely that no residue can
        own it (and it would otherwise be dropped from the topology silently).
    """
    atom_types = _charmm_atom_types(parm, ligand)
    ff = ForceField()
    ff.atom_types = _atom_type_records(parm, atom_types)
    ff.bonds = _bond_types(parm, atom_types)
    ff.angles = _angle_types(parm, atom_types)
    ff.dihedrals = _dihedral_types(parm, atom_types)
    ff.impropers = _improper_types(parm, atom_types)
    ff.cmaps = _cmap_types(parm, atom_types)
    ff.residues = _residues(parm, atom_types)
    return ff


def disulfide_parameters() -> tuple[BondType, AngleType, list[DihedralType]]:
    """Return the hardcoded parm19sb disulfide S-S parameters.

    Returns
    -------
    bond : BondType
        The ``S-S`` bond.
    angle : AngleType
        The ``2C-S-S`` angle.
    dihedrals : list of DihedralType
        The S-S torsion terms (and the terminal backbone ``X-C-N-X`` term).
    """
    bond = BondType(("S", "S"), 166.0, 2.038)
    angle = AngleType(("2C", "S", "S"), 68.0, 103.7)
    # (a1, a2, a3, a4, k, multiplicity, phase)
    terms = [
        ("2C", "S", "S", "2C", 0.379, 4, 0.0),
        ("2C", "S", "S", "2C", 0.682, 3, 0.0),
        ("2C", "S", "S", "2C", 4.48, 2, 0.0),
        ("2C", "S", "S", "2C", 0.42, 1, 0.0),
        ("XC8", "2C", "S", "S", 0.135, 4, 180.0),
        ("XC8", "2C", "S", "S", 0.302, 3, 0.0),
        ("XC8", "2C", "S", "S", 0.666, 2, 0.0),
        ("XC8", "2C", "S", "S", 0.056, 1, 0.0),
        ("H1", "2C", "S", "S", 0.333, 3, 0.0),
        ("X", "C", "N", "X", 2.5, 2, 180.0),
    ]
    dihedrals = [DihedralType((a, b, c, d), k, n, ph) for a, b, c, d, k, n, ph in terms]
    return bond, angle, dihedrals


def opc_water_angle() -> AngleType:
    """Return the OPC water ``HW-OW-HW`` angle parameter (absent from the prmtop)."""
    return AngleType(("HW", "OW", "HW"), 55.0, 103.60)


# --------------------------------------------------------------------------- #
# atom types
# --------------------------------------------------------------------------- #
def _charmm_atom_types(parm, ligand: bool) -> list[str]:
    """Return the CHARMM atom-type name for each atom, positioned by ``atom.idx``.

    Iterating ``parm.atoms`` (which is in ``atom.idx`` order) guarantees the
    returned list is indexable as ``types[atom.idx]`` elsewhere.

    Applies the CMAP CA remap, the TRP ``C*`` / OPC-water ``EP`` replacements,
    and the ligand ``_`` prefix. The original's replacement test had a
    precedence quirk (``(residue in remap and type=='C*') or type=='EP'``) that
    raised KeyError for an ``EP`` atom in a residue not in the remap; we require
    ``residue in remap`` for both cases, which is identical for real inputs
    (only WAT carries ``EP``, only TRP carries ``C*``) but cannot crash.

    The ligand prefix is applied last, to whichever base type was selected, so a
    remapped type cannot escape it -- an unprefixed name in a ligand file could
    collide with the protein force field's own types, which is exactly what the
    prefix exists to prevent.
    """
    has_cmap = "CMAP_COUNT" in getattr(parm, "parm_data", {}) and (
        parm.parm_data["CMAP_COUNT"][0] > 0
    )
    types: list[str] = []
    for atom in parm.atoms:
        resname = atom.residue.name
        if has_cmap and resname in _CMAP_CA_TYPE and atom.name == "CA":
            base = _CMAP_CA_TYPE[resname]
        elif resname in _ATOM_TYPE_REMAP and atom.type in ("C*", "EP"):
            base = _ATOM_TYPE_REMAP[resname][1]
        else:
            base = atom.type
        types.append(f"_{base}" if ligand else base)
    return types


def _atom_type_records(parm, types: list[str]) -> list[AtomType]:
    """One :class:`~model.AtomType` per atom (mass + Lennard-Jones)."""
    return [
        AtomType(
            name=types[atom.idx],
            mass=atom.mass,
            comment=atom.name,
            epsilon=atom.epsilon,
            rmin_half=_SIGMA_TO_RMIN_HALF * atom.sigma,
        )
        for atom in parm.atoms
    ]


# --------------------------------------------------------------------------- #
# bonded parameters
# --------------------------------------------------------------------------- #
def _bond_types(parm, types: list[str]) -> list[BondType]:
    out = []
    for bond in parm.bonds:
        a, b = sorted((types[bond.atom1.idx], types[bond.atom2.idx]))
        out.append(BondType((a, b), bond.type.k, bond.type.req))
    return out


def _angle_types(parm, types: list[str]) -> list[AngleType]:
    out = []
    for angle in parm.angles:
        a, c = sorted((types[angle.atom1.idx], types[angle.atom3.idx]))
        out.append(AngleType((a, types[angle.atom2.idx], c), angle.type.k, angle.type.theteq))
    return out


def _torsion_atom_types(dih, types: list[str]) -> tuple[str, str, str, str]:
    """Return the four CHARMM atom-type names of a dihedral, in atom order."""
    return (
        types[dih.atom1.idx],
        types[dih.atom2.idx],
        types[dih.atom3.idx],
        types[dih.atom4.idx],
    )


def _dihedral_types(parm, types: list[str]) -> list[DihedralType]:
    """Proper dihedrals only (skip impropers via ``dih.improper``)."""
    return [
        DihedralType(
            _torsion_atom_types(dih, types),
            dih.type.phi_k,
            dih.type.per,
            dih.type.phase,
        )
        for dih in parm.dihedrals
        if not dih.improper
    ]


def _improper_types(parm, types: list[str]) -> list[ImproperType]:
    """Improper dihedrals only.

    Matches the original selection: ``dih.improper`` AND periodicity 2. AMBER
    and GAFF impropers are periodicity-2; the original dropped any other, so we
    do too.
    """
    return [
        ImproperType(
            _torsion_atom_types(dih, types),
            dih.type.phi_k,
            dih.type.per,
            dih.type.phase,
        )
        for dih in parm.dihedrals
        if dih.improper and dih.type.per == 2
    ]


def _cmap_types(parm, types: list[str]) -> list[CmapType]:
    out = []
    for cmap in getattr(parm, "cmaps", []):
        atoms = (
            types[cmap.atom1.idx],
            types[cmap.atom2.idx],
            types[cmap.atom3.idx],
            types[cmap.atom4.idx],
            types[cmap.atom5.idx],
        )
        comment = cmap.type.comments
        if isinstance(comment, (list, tuple)):
            comment = " ".join(str(c) for c in comment)
        out.append(CmapType(atoms, cmap.type.resolution, tuple(cmap.type.grid), str(comment)))
    return out


# --------------------------------------------------------------------------- #
# residue topology
# --------------------------------------------------------------------------- #
def _prefixed(atom, owner_idx: int) -> str:
    """Atom name with a ``-``/``+`` prefix for the previous/next residue."""
    ridx = atom.residue.idx
    if ridx < owner_idx:
        return f"-{atom.name}"
    if ridx > owner_idx:
        return f"+{atom.name}"
    return atom.name


def _owner_residue(atoms, threshold: int) -> int | None:
    """Residue index that contains at least `threshold` of `atoms`, else None."""
    counts: dict[int, int] = {}
    for atom in atoms:
        counts[atom.residue.idx] = counts.get(atom.residue.idx, 0) + 1
    for ridx, n in counts.items():
        if n >= threshold:
            return ridx
    return None


def _unownable(atoms, kind: str, threshold: int) -> str:
    """Build the error message for a term no single residue can own."""
    spread = ", ".join(f"{a.name}({a.residue.name}{a.residue.idx + 1})" for a in atoms)
    return (
        f"{kind} term [{spread}] spans residues with no residue holding at least "
        f"{threshold} of its atoms, so it cannot be written into any RESI. This "
        "is not representable as a CHARMM residue topology; the term would "
        "otherwise be dropped silently."
    )


def _residues(parm, types: list[str]) -> list[Residue]:
    """Build one :class:`~model.Residue` per prmtop residue.

    No de-duplication or terminal (N-/C-) renaming happens here: the writer
    de-duplicates residues by name (first-wins), and the protein driver applies
    any terminal renaming before that. Keeping one Residue per prmtop residue is
    what lets the driver rename same-named terminal residues (e.g. two ``ALA``
    into ``NALA``/``CALA``) without the parser having collapsed them first.
    """
    # Pre-group impropers and CMAPs by the residue that owns them (>=3 of the
    # improper's 4 atoms, or >=3 of the CMAP's 5 atoms) -- one pass, not one
    # scan of every dihedral per residue.
    impropers_by_res: dict[int, list] = {}
    for dih in parm.dihedrals:
        if not dih.improper:
            continue
        atoms = (dih.atom1, dih.atom2, dih.atom3, dih.atom4)
        owner = _owner_residue(atoms, threshold=3)
        if owner is None:
            raise ValueError(_unownable(atoms, "improper", 3))
        impropers_by_res.setdefault(owner, []).append(dih)
    cmaps_by_res: dict[int, list] = {}
    for cmap in getattr(parm, "cmaps", []):
        atoms = (cmap.atom1, cmap.atom2, cmap.atom3, cmap.atom4, cmap.atom5)
        owner = _owner_residue(atoms, threshold=3)
        if owner is None:
            raise ValueError(_unownable(atoms, "CMAP", 3))
        cmaps_by_res.setdefault(owner, []).append(cmap)

    residues: list[Residue] = []
    for i, residue in enumerate(parm.residues):
        res = Residue(
            name=residue.name,
            total_charge=sum(atom.charge for atom in residue.atoms),
        )
        res.atoms = [
            TopologyAtom(atom.name, types[atom.idx], atom.charge) for atom in residue.atoms
        ]
        res.bonds = _residue_bonds(residue, i)
        if residue.name == "WAT":
            res.angles.append(("H1", "O", "H2"))
        res.impropers = [
            tuple(_prefixed(a, i) for a in (d.atom1, d.atom2, d.atom3, d.atom4))
            for d in impropers_by_res.get(i, [])
        ]
        res.cmaps = [_cmap_tokens(cmap, i) for cmap in cmaps_by_res.get(i, [])]
        if residue.name == "WAT":
            res.acceptors.append("O")
            res.lonepairs.append(LonePair("bisector", "EPW", ("O", "H1", "H2"), 0.1594, 0.0, 0.0))
        residues.append(res)
    return residues


def _residue_bonds(residue, owner_idx: int) -> list[tuple[str, str]]:
    """Intra-residue bonds plus bonds to the next residue (with ``+``).

    A bond is owned by the lower-indexed residue it touches; bonds to the
    previous residue are left for that residue to emit. Bonds to a *non-adjacent*
    residue (a disulfide SG-SG cross-link) are skipped entirely -- they belong to
    a patch (``PRES DISU``), not to the residue template.
    """
    bonds: list[tuple[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for atom in residue.atoms:
        for bond in atom.bonds:
            key = (bond.atom1.idx, bond.atom2.idx)
            if key in seen:
                continue
            seen.add(key)
            r1, r2 = bond.atom1.residue.idx, bond.atom2.residue.idx
            if min(r1, r2) != owner_idx:  # owned by the lower residue
                continue
            if abs(r1 - r2) > 1:  # non-adjacent cross-link -> PRES DISU, not here
                continue
            bonds.append((_prefixed(bond.atom1, owner_idx), _prefixed(bond.atom2, owner_idx)))
    return bonds


def _cmap_tokens(cmap, owner_idx: int) -> tuple[str, ...]:
    """Return the eight CMAP atom-name tokens (phi atoms 0-3, psi atoms 1-4)."""
    atoms = (cmap.atom1, cmap.atom2, cmap.atom3, cmap.atom4, cmap.atom5)
    phi = [f"-{a.name}" if a.residue.idx < owner_idx else a.name for a in atoms[:4]]
    psi = [f"+{a.name}" if a.residue.idx > owner_idx else a.name for a in atoms[1:]]
    return tuple(phi + psi)

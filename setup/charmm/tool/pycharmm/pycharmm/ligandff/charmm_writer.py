"""
Serialize a :class:`~model.ForceField` to CHARMM rtf and prm text.

This replaces the original converter's ``write_*`` functions. It owns the two
jobs the model deliberately left to the writer:

* **de-duplication by CHARMM key** -- atom types by name, bonded parameters by
  their atom-type tuple, torsions by their *orientation-independent* tuple plus
  periodicity (the original's ``remove_palindromes``) -- always keeping the
  first occurrence. Frozen-record set equality is *not* used, because the same
  key recurs with different ``comment``/insertion order across residues.
* **formatting** into CHARMM-standard sections.

The output format is tidied relative to the original (no line-continuation
whitespace in torsions, no debug prints), but stays chemically identical --
verified by building each result in CHARMM and comparing energies.

Notes
-----
``write_rtf`` and ``write_prm`` take the flavor-specific header/declaration text
as arguments; the two canonical profiles are exposed as module constants
(``LIGAND_*`` / ``PROTEIN_*``). Patches (e.g. the disulfide ``PRES DISU``) are
emitted from ``ForceField.patches`` rather than hardcoded.
"""

from __future__ import annotations

import math

from .model import ForceField

# --------------------------------------------------------------------------- #
# flavor profiles: rtf/prm headers and the rtf pre-RESI declarations
# --------------------------------------------------------------------------- #
LIGAND_RTF_TITLE = (
    "*>>>>>>>> All-Hydrogen Topology File for a Ligand from AMBER GAFF2 <<<<<<\n"
    "*>>>>>>>>>>>>>>>>>>>>>>> C. L. Brooks III <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    "*\n"
    "36  1"
)
PROTEIN_RTF_TITLE = (
    "*>>>>>>>> All-Hydrogen Topology File for Proteins from AMBER parm19sb <<<<<<\n"
    "*>>>>>>>>>>>>>> Includes phi, psi cross-term map (CMAP) <<<<<<<<<<<<<<<<\n"
    "*>>>>>>>>>>>>>>>>>>>>>>> C. L. Brooks III <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    "*\n"
    "36  1"
)
LIGAND_RTF_DECL = "AUTOGENERATE ANGLES DIHEDRALS\nDEFA FIRS NONE LAST NONE"
PROTEIN_RTF_DECL = (
    "DECL -CA\nDECL -C\nDECL -O\nDECL +N\nDECL +HN\nDECL +CA\n"
    "DEFA FIRS NONE LAST NONE\nAUTO ANGLES DIHE PATCH"
)
LIGAND_PRM_TITLE = (
    "*>>>>>>>> All-Hydrogen Parameter File for a Ligand from AMBER GAFF2 <<<<<<\n"
    "*>>>>>>>>>>>>>>>>>>>>>>> C. L. Brooks III <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    "*"
)
PROTEIN_PRM_TITLE = (
    "*>>>>>>>> All-Hydrogen Parameter File for Proteins from AMBER parm19sb <<<<<<\n"
    "*>>>>>>>>>>>>>> Includes phi, psi cross-term map (CMAP) <<<<<<<<<<<<<<<<\n"
    "*>>>>>>>>>>>>>>>>>>>>>>> C. L. Brooks III <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    "*"
)

_NONBONDED_HEADER = (
    "NONBONDED nbxmod 5 atom cdiel switch vatom vdistance vswitch -\n"
    "cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 0.8333 wmin 1.5"
)


# --------------------------------------------------------------------------- #
# de-duplication helpers (key-based, first-wins)
# --------------------------------------------------------------------------- #
def _dedup_first(items, key, *, value=None, what=None):
    """Yield `items` with later entries whose `key` was already seen removed.

    When `value` is given, a repeat of a key must carry the same value: a CHARMM
    parameter table holds one entry per key, so two records sharing a key but
    disagreeing on the numbers cannot both be honoured and first-wins would
    silently pick one. That happens when force fields built separately are
    merged -- parmchk2 estimates missing parameters per molecule, so two ligands
    can arrive with different values for the same atom-type key.

    Parameters
    ----------
    items : iterable
        Records to de-duplicate, in precedence order.
    key : callable
        Maps a record to its CHARMM key.
    value : callable, optional
        Maps a record to the values that must agree across a repeated key.
    what : str, optional
        Record description, for the error message.

    Returns
    -------
    list
        The first record for each distinct key, in input order.

    Raises
    ------
    ValueError
        If `value` is given and two records share a key but not their values.
    """
    seen = {}
    out = []
    for item in items:
        k = key(item)
        if k in seen:
            if value is not None and not _values_agree(seen[k], value(item)):
                raise ValueError(
                    f"conflicting {what or 'parameter'} for {k}: "
                    f"{seen[k]} and {value(item)}. A CHARMM parameter table holds "
                    "one entry per key, so these cannot be combined -- they most "
                    "likely come from separately built force fields whose missing "
                    "parameters parmchk2 estimated differently."
                )
            continue
        seen[k] = value(item) if value is not None else None
        out.append(item)
    return out


# Two parameters agreeing this closely are the same parameter written to
# different precision, not a conflict. Both cases occur between the tabulated
# parm19sb values and the same parameters carried in a prmtop: an angle held in
# radians comes back as 103.700044 where the table says 103.7, and a barrier
# tabulated to three significant figures reads 0.333 against the prmtop's
# 0.333333333. Estimates that genuinely disagree -- parmchk2 filling the same
# gap differently for two molecules -- differ by percent or more, well outside
# this band.
_VALUE_REL_TOL = 5e-3


def _values(*values):
    """Collect the values that must agree across a repeated parameter key."""
    return tuple(values)


def _values_agree(left, right):
    """Compare two value tuples, treating floats as equal within the tolerance."""
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=_VALUE_REL_TOL, abs_tol=1e-8):
                return False
        elif isinstance(a, tuple) and isinstance(b, tuple):
            if not _values_agree(a, b):
                return False
        elif a != b:
            return False
    return True


def _torsion_key(t):
    """Orientation-independent key for a torsion: canonical atoms + periodicity.

    Folds a reversed duplicate (CHARMM ``remove_palindromes``) and an exact
    duplicate onto the same key.
    """
    return (min(t.atoms, t.atoms[::-1]), t.multiplicity)


def _unique_sorted(items, *, dedup_key, sort_key, value=None, what=None):
    """De-duplicate `items` by `dedup_key` (first-wins), then sort by `sort_key`.

    `value`/`what` are passed through to :func:`_dedup_first` to reject a
    repeated key whose values disagree.
    """
    return sorted(_dedup_first(items, dedup_key, value=value, what=what), key=sort_key)


def _improper_key(imp):
    """Key an improper by its central atom plus its (unordered) peripherals.

    AMBER puts the central atom third. The prmtop can list the same improper
    with its *peripheral* atoms permuted between two instances of one residue
    (e.g. ``-C CA N H`` vs ``-C H N CA``), which is not a real difference -- but
    a different central atom is, so that position stays significant.
    """
    return (imp[2], frozenset((imp[0], imp[1], imp[3])))


def _residue_signature(res):
    """Return a hashable topology signature for a residue (for conflict detection).

    Atom sequence is order-sensitive (it defines the RESI's ATOM order). The
    derived bonded terms are compared as unordered collections, and each term is
    normalized only as far as is chemically meaningless: a bond's two atoms are
    interchangeable, and an improper's peripheral atoms are (see
    :func:`_improper_key`), but an angle's vertex and an improper's central atom
    are not.
    """
    return (
        tuple((a.name, a.atom_type, round(a.charge, 6)) for a in res.atoms),
        frozenset(frozenset(b) for b in res.bonds),
        frozenset(_improper_key(imp) for imp in res.impropers),
        frozenset(tuple(cm) for cm in res.cmaps),
        frozenset((ang[1], frozenset((ang[0], ang[2]))) for ang in res.angles),
        frozenset(res.acceptors),
        frozenset(
            (lp.kind, lp.atom, lp.reference_atoms, lp.distance, lp.angle, lp.dihedral)
            for lp in res.lonepairs
        ),
    )


def _unique_residues(residues):
    """Keep one residue per name (first-wins); raise if a name has two topologies.

    A CHARMM rtf defines each residue *type* once, so repeated instances of a
    residue (a peptide's many ALA) collapse to one RESI. Standard force-field
    naming makes a name uniquely identify a topology; if two residues share a
    name but differ (e.g. an unrenamed terminal vs. internal form, or two
    distinct molecules both named ``MOL``), folding them would silently drop
    one, so raise instead.

    Parameters
    ----------
    residues : list of ~ligandff.model.Residue
        Residues in emission order.

    Returns
    -------
    list of ~ligandff.model.Residue
        One residue per distinct name, first occurrence kept.

    Raises
    ------
    ValueError
        If two residues share a name but have different topology.
    """
    seen = {}
    out = []
    for res in residues:
        sig = _residue_signature(res)
        if res.name in seen:
            if seen[res.name] != sig:
                raise ValueError(
                    f"two residues named {res.name!r} have different topology; "
                    "give each distinct residue a unique name (e.g. terminal "
                    "N-/C- forms) before writing"
                )
            continue
        seen[res.name] = sig
        out.append(res)
    return out


# --------------------------------------------------------------------------- #
# rtf
# --------------------------------------------------------------------------- #
def _mass_records(ff, width) -> list[str]:
    """MASS records, one per unique atom type, ordered by (mass, name)."""
    unique = _unique_sorted(
        ff.atom_types,
        dedup_key=lambda a: a.name,
        sort_key=lambda a: (a.mass, a.name),
        value=lambda a: _values(a.mass, a.epsilon, a.rmin_half),
        what="atom type",
    )
    return [f"MASS  -1  {a.name:<{width}} {a.mass:>9.5f} ! {a.comment}" for a in unique]


def _residue_block(res) -> str:
    charge = f"{res.total_charge:.2f}"
    if charge == "-0.00":  # a tiny-negative rounding sum; show plain zero
        charge = "0.00"
    lines = [f"RESI {res.name} {charge}"]
    lines += [f"ATOM {a.name:<4} {a.atom_type:<4} {a.charge:>10.5f}" for a in res.atoms]
    lines += [f"BOND {a:<4} {b:<4}" for a, b in res.bonds]
    lines += [f"ANGL {a} {b} {c}" for a, b, c in res.angles]
    lines += [f"IMPROPER {a:<4} {b:<4} {c:<4} {d:<4}" for a, b, c, d in res.impropers]
    lines += ["CMAP " + " ".join(f"{t:<4}" for t in cm) for cm in res.cmaps]
    lines += [f"ACCE {a}" for a in res.acceptors]
    lines += [
        f"LONEPAIR {lp.kind} {lp.atom} {' '.join(lp.reference_atoms)}  "
        f"distance {lp.distance:.4f} angle {lp.angle:.1f} dihe {lp.dihedral:.1f}"
        for lp in res.lonepairs
    ]
    return "\n".join(lines)


def _patch_block(patch) -> str:
    lines = [f"PRES {patch.name} {patch.total_charge:.2f}"]
    lines += [f"BOND {a:<4} {b:<4}" for a, b in patch.bonds]
    return "\n".join(lines)


def write_rtf(ff: ForceField, *, title: str, declarations: str) -> str:
    """Serialize `ff`'s atom types, residues, and patches to CHARMM rtf text.

    Parameters
    ----------
    ff : ForceField
        The force field to serialize.
    title : str
        The rtf header, including the ``36  1`` version line (see
        ``LIGAND_RTF_TITLE`` / ``PROTEIN_RTF_TITLE``).
    declarations : str
        The pre-RESI directive block (``AUTOGENERATE``/``DECL``/``AUTO``; see
        ``LIGAND_RTF_DECL`` / ``PROTEIN_RTF_DECL``).

    Returns
    -------
    str
        The rtf file contents, ending in a single trailing newline.
    """
    residues = _unique_residues(ff.residues)
    blocks = [title, "\n".join(_mass_records(ff, width=6)), declarations]
    blocks += [_residue_block(r) for r in residues]
    blocks += [_patch_block(p) for p in ff.patches]
    blocks.append("END")
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# prm
# --------------------------------------------------------------------------- #
def _bond_records(ff) -> list[str]:
    bonds = _unique_sorted(
        ff.bonds,
        dedup_key=lambda b: b.atoms,
        sort_key=lambda b: b.atoms,
        value=lambda b: _values(b.k, b.b0),
        what="bond",
    )
    return [f"{b.atoms[0]:<4} {b.atoms[1]:<4} {b.k:>8.3f} {b.b0:>8.4f}" for b in bonds]


def _angle_records(ff) -> list[str]:
    angles = _unique_sorted(
        ff.angles,
        dedup_key=lambda a: a.atoms,
        sort_key=lambda a: a.atoms,
        value=lambda a: _values(a.k, a.theta0),
        what="angle",
    )
    return [
        f"{a.atoms[0]:<4} {a.atoms[1]:<4} {a.atoms[2]:<4} {a.k:>8.3f} {a.theta0:>8.2f}"
        for a in angles
    ]


def _torsion_records(torsions) -> list[str]:
    # Sort by (atoms, multiplicity): atoms alone is not a total order for a
    # multi-periodicity series, which would leave those lines in insertion order
    # -- i.e. dependent on where the parameters came from rather than on content.
    kept = _unique_sorted(
        torsions,
        dedup_key=_torsion_key,
        sort_key=lambda t: (t.atoms, t.multiplicity),
        value=lambda t: _values(t.k, t.phase),
        what="torsion",
    )
    return [
        f"{t.atoms[0]:<4} {t.atoms[1]:<4} {t.atoms[2]:<4} {t.atoms[3]:<4} "
        f"{t.k:>8.3f} {t.multiplicity:>d} {t.phase:>8.1f}"
        for t in kept
    ]


def _cmap_records(ff) -> list[str]:
    """CMAP blocks: the eight atom-type tokens, resolution, then the grid."""
    cmaps = _unique_sorted(
        ff.cmaps,
        dedup_key=lambda c: c.atoms,
        sort_key=lambda c: c.atoms,
        value=lambda c: (c.resolution, _values(*c.grid)),
        what="CMAP",
    )
    blocks = []
    for cm in cmaps:
        tokens = list(cm.atoms[:4]) + list(cm.atoms[1:])
        header = " ".join(f"{t:<4}" for t in tokens) + f" {cm.resolution}"
        lines = [f"! {cm.comment}".rstrip(), header]
        step = 360 // cm.resolution
        for i in range(cm.resolution):
            phi = -180 + i * step
            lines.append(f"! phi = {phi}")
            row = cm.grid[i * cm.resolution : (i + 1) * cm.resolution]
            lines.extend(
                " ".join(f"{x:10.6f}" for x in row[j : j + 5]) for j in range(0, cm.resolution, 5)
            )
        blocks.append("\n".join(lines))
    return blocks


def _nonbonded_records(ff) -> list[str]:
    """NONBONDED lines. CHARMM epsilon is negative; 1-4 uses epsilon/2."""
    types = _unique_sorted(
        ff.atom_types,
        dedup_key=lambda a: a.name,
        sort_key=lambda a: a.name,
        value=lambda a: _values(a.mass, a.epsilon, a.rmin_half),
        what="atom type",
    )
    return [
        f"{a.name:<4} {0.0:>10.6f} {-a.epsilon:>10.6f} {a.rmin_half:>10.6f} "
        f"{0.0:>10.6f} {-a.epsilon / 2:>10.6f} {a.rmin_half:>10.6f}"
        for a in types
    ]


def write_prm(ff: ForceField, *, title: str) -> str:
    """Serialize `ff`'s parameters to CHARMM prm text.

    Parameters
    ----------
    ff : ForceField
        The force field to serialize.
    title : str
        The prm header (see ``LIGAND_PRM_TITLE`` / ``PROTEIN_PRM_TITLE``).

    Returns
    -------
    str
        The prm file contents, ending in ``END`` and a single trailing newline.
    """
    sections = [title, "ATOMS\n" + "\n".join(_mass_records(ff, width=4))]
    sections.append("BONDS\n! atom types   Kb    b0\n" + "\n".join(_bond_records(ff)))
    sections.append("ANGLES\n! atom types     Ktheta    Theta0\n" + "\n".join(_angle_records(ff)))
    sections.append(
        "DIHEDRALS\n! atom types     Kchi    n    delta\n"
        + "\n".join(_torsion_records(ff.dihedrals))
    )
    sections.append(
        "IMPROPERS\n! atom types     Kpsi    n    delta\n"
        + "\n".join(_torsion_records(ff.impropers))
    )
    cmap = _cmap_records(ff)
    if cmap:
        sections.append("CMAP\n\n" + "\n\n".join(cmap))
    header = (
        _NONBONDED_HEADER + "\n"
        "!type  ignored   epsilon  Rmin/2   ignored   epsilon,1-4    Rmin/2,1-4"
    )
    sections.append(header + "\n" + "\n".join(_nonbonded_records(ff)))
    sections.append("END")
    return "\n\n".join(sections) + "\n"

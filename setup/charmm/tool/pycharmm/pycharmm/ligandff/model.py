"""
Data model for a CHARMM force field.

Plain dataclasses that represent the pieces of a CHARMM residue-topology (rtf)
and parameter (prm) file: atom types, the bonded/nonbonded parameter tables,
per-residue topologies, and patches. A single :class:`ForceField` instance
holds them all.

This replaces the seven parallel pandas DataFrames and the eleven-value tuple
that the original converter threaded through every function. The parser
(``amber_to_charmm``) builds a :class:`ForceField` from an AMBER ``prmtop``,
and the writer (``charmm_writer``) serializes it to rtf/prm text -- so parsing
and formatting no longer share mutable state.

Units follow CHARMM conventions: masses in amu, lengths in angstrom, angles in
degrees, energies in kcal/mol, force constants in kcal/mol/A^2 (bonds) or
kcal/mol/rad^2 (angles). Lennard-Jones parameters are stored CHARMM-style as a
well depth ``epsilon`` (kcal/mol, positive) and ``rmin_half`` (Rmin/2, in
angstrom); the parser converts from the AMBER sigma/epsilon.

Notes
-----
The immutable parameter records (:class:`AtomType`, :class:`BondType`, ...) are
frozen dataclasses, so they are hashable and usable as set/dict members. Their
identity spans *all* fields, though (including ``comment`` and the parameter
values), so the writer must de-duplicate by the CHARMM *key* -- the atom-type
name for :class:`AtomType`, the atom-type tuple for the bonded records --
keeping the first occurrence, rather than relying on plain set equality (the
same type appears with many representative ``comment`` names, and the same
atom-type key may recur). The topology containers (:class:`Residue`,
:class:`ForceField`) are mutable and built up incrementally.

Parmed ships its own parameter classes, but this model is deliberately a small,
CHARMM-facing target (atom-type-name strings, Rmin/2 + epsilon) decoupled from
parmed's AMBER-oriented objects and their parent ``Structure``; the parser
reads parmed and converts into this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# parameter records (frozen -> hashable, de-dupable)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AtomType:
    """A CHARMM atom type: its mass and Lennard-Jones parameters.

    Attributes
    ----------
    name : str
        CHARMM atom-type name (e.g. ``"_c3"``, ``"XC1"``, ``"LP"``).
    mass : float
        Atomic mass in amu.
    comment : str
        Representative atom name, written as the ``MASS`` record comment.
    epsilon : float
        Lennard-Jones well depth in kcal/mol (positive magnitude).
    rmin_half : float
        Lennard-Jones ``Rmin/2`` in angstrom.
    """

    name: str
    mass: float
    comment: str
    epsilon: float
    rmin_half: float


@dataclass(frozen=True)
class BondType:
    """A harmonic bond parameter.

    Attributes
    ----------
    atoms : tuple of str
        The two atom-type names.
    k : float
        Force constant in kcal/mol/A^2.
    b0 : float
        Equilibrium length in angstrom.
    """

    atoms: tuple[str, str]
    k: float
    b0: float


@dataclass(frozen=True)
class AngleType:
    """A harmonic angle parameter.

    Attributes
    ----------
    atoms : tuple of str
        The three atom-type names (vertex second).
    k : float
        Force constant in kcal/mol/rad^2.
    theta0 : float
        Equilibrium angle in degrees.
    """

    atoms: tuple[str, str, str]
    k: float
    theta0: float


@dataclass(frozen=True)
class DihedralType:
    """A proper dihedral (torsion) parameter, one periodicity term.

    Attributes
    ----------
    atoms : tuple of str
        The four atom-type names.
    k : float
        Barrier height in kcal/mol.
    multiplicity : int
        Periodicity ``n``.
    phase : float
        Phase offset in degrees.
    """

    atoms: tuple[str, str, str, str]
    k: float
    multiplicity: int
    phase: float


@dataclass(frozen=True)
class ImproperType:
    """An improper dihedral parameter.

    Same shape as :class:`DihedralType`, kept distinct because CHARMM writes
    impropers in their own ``IMPROPERS`` section.

    Attributes
    ----------
    atoms : tuple of str
        The four atom-type names.
    k : float
        Force constant in kcal/mol.
    multiplicity : int
        Periodicity ``n``.
    phase : float
        Phase offset in degrees.
    """

    atoms: tuple[str, str, str, str]
    k: float
    multiplicity: int
    phase: float


@dataclass(frozen=True)
class CmapType:
    """A CHARMM CMAP (phi/psi cross-term) parameter.

    Attributes
    ----------
    atoms : tuple of str
        The five atom-type names spanning the overlapping phi/psi dihedrals
        (phi = atoms 0-3, psi = atoms 1-4); the writer expands these to the
        eight-token CHARMM CMAP header.
    resolution : int
        Grid dimension along each axis (e.g. ``24``).
    grid : tuple of float
        The ``resolution * resolution`` energy values, row-major.
    comment : str, optional
        Free-text label written above the CMAP block.
    """

    atoms: tuple[str, str, str, str, str]
    resolution: int
    grid: tuple[float, ...]
    comment: str = ""

    def __post_init__(self) -> None:
        """Validate that the grid holds ``resolution ** 2`` energy values.

        Raises
        ------
        ValueError
            If ``len(grid) != resolution ** 2`` (a malformed CMAP that would
            otherwise silently produce a garbled parameter block).
        """
        expected = self.resolution * self.resolution
        if len(self.grid) != expected:
            raise ValueError(
                f"CmapType grid has {len(self.grid)} values, expected resolution**2 = {expected}"
            )


# --------------------------------------------------------------------------- #
# residue topology
# --------------------------------------------------------------------------- #


@dataclass
class TopologyAtom:
    """One atom in a residue topology (an rtf ``ATOM`` record).

    Attributes
    ----------
    name : str
        Atom name (e.g. ``"CA"``, ``"H1"``).
    atom_type : str
        CHARMM atom-type name, matching an :class:`AtomType`.
    charge : float
        Partial charge in elementary charge units.
    """

    name: str
    atom_type: str
    charge: float


@dataclass
class LonePair:
    """A lone-pair / virtual-site construction (an rtf ``LONEPAIR`` record).

    Attributes
    ----------
    kind : str
        Construction geometry, e.g. ``"bisector"``.
    atom : str
        The lone-pair (virtual site) atom name, e.g. ``"EPW"``.
    reference_atoms : tuple of str
        The atoms defining the placement, e.g. ``("O", "H1", "H2")``.
    distance : float
        Distance parameter in angstrom.
    angle : float
        Angle parameter in degrees.
    dihedral : float
        Dihedral parameter in degrees.
    """

    kind: str
    atom: str
    reference_atoms: tuple[str, ...]
    distance: float
    angle: float
    dihedral: float


@dataclass
class Residue:
    """A CHARMM residue topology (an rtf ``RESI`` block).

    Bond/improper/cmap/angle atom references are atom *names* (not types), and
    may carry a leading ``-``/``+`` to denote the previous/next residue in the
    chain.

    Attributes
    ----------
    name : str
        Residue name (e.g. ``"ALA"``, ``"AIN"``).
    total_charge : float
        Sum of the atom partial charges.
    atoms : list of TopologyAtom
        The residue's atoms, in order.
    bonds : list of tuple of str
        Bonds as ``(name1, name2)`` atom-name pairs.
    impropers : list of tuple of str
        Impropers as four atom names.
    cmaps : list of tuple of str
        CMAP terms as eight atom names.
    angles : list of tuple of str
        Explicit angles as three atom names (e.g. the water H-O-H).
    acceptors : list of str
        Hydrogen-bond acceptor atom names (rtf ``ACCE``).
    lonepairs : list of LonePair
        Lone-pair / virtual-site constructions (rtf ``LONEPAIR``).
    """

    name: str
    total_charge: float
    atoms: list[TopologyAtom] = field(default_factory=list)
    bonds: list[tuple[str, str]] = field(default_factory=list)
    impropers: list[tuple[str, str, str, str]] = field(default_factory=list)
    cmaps: list[tuple[str, ...]] = field(default_factory=list)
    angles: list[tuple[str, str, str]] = field(default_factory=list)
    acceptors: list[str] = field(default_factory=list)
    lonepairs: list[LonePair] = field(default_factory=list)


@dataclass
class Patch:
    """A CHARMM patch residue (an rtf ``PRES`` block).

    Attributes
    ----------
    name : str
        Patch name (e.g. ``"DISU"``).
    total_charge : float, optional
        Net charge change applied by the patch. Default ``0.0``.
    bonds : list of tuple of str
        Bonds the patch adds, as atom-name pairs (names may be prefixed by the
        patched-residue index, e.g. ``"1SG"``/``"2SG"``).
    """

    name: str
    total_charge: float = 0.0
    bonds: list[tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# top-level container
# --------------------------------------------------------------------------- #


@dataclass
class ForceField:
    """A complete CHARMM force field: atom types, parameters, and topologies.

    This is the single object passed from the parser to the writer, replacing
    the original converter's seven DataFrames and eleven-tuple. Collections are
    accumulated in insertion order and may contain duplicates; de-duplication
    and sorting are the writer's responsibility.

    Attributes
    ----------
    atom_types : list of AtomType
        Atom-type (mass + Lennard-Jones) records.
    bonds : list of BondType
        Bond parameters.
    angles : list of AngleType
        Angle parameters.
    dihedrals : list of DihedralType
        Proper dihedral parameters.
    impropers : list of ImproperType
        Improper dihedral parameters.
    cmaps : list of CmapType
        CMAP cross-term parameters.
    residues : list of Residue
        Residue topologies.
    patches : list of Patch
        Patch residues.
    """

    atom_types: list[AtomType] = field(default_factory=list)
    bonds: list[BondType] = field(default_factory=list)
    angles: list[AngleType] = field(default_factory=list)
    dihedrals: list[DihedralType] = field(default_factory=list)
    impropers: list[ImproperType] = field(default_factory=list)
    cmaps: list[CmapType] = field(default_factory=list)
    residues: list[Residue] = field(default_factory=list)
    patches: list[Patch] = field(default_factory=list)

    def extend(self, other: ForceField) -> None:
        """Append every record from `other` into this force field, in place.

        Used to accumulate parameters and topologies from several ``prmtop``
        files (e.g. building a protein force field residue by residue).

        Parameters
        ----------
        other : ForceField
            The force field whose records are appended to this one.
        """
        self.atom_types.extend(other.atom_types)
        self.bonds.extend(other.bonds)
        self.angles.extend(other.angles)
        self.dihedrals.extend(other.dihedrals)
        self.impropers.extend(other.impropers)
        self.cmaps.extend(other.cmaps)
        self.residues.extend(other.residues)
        self.patches.extend(other.patches)

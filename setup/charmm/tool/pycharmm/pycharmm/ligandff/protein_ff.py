"""
Assemble a protein (parm19sb) CHARMM :class:`~model.ForceField` from a prmtop.

This is the protein driver: it wraps the general parser
(:func:`amber_to_charmm.build_forcefield`) and layers on the protein-specific
policy the parser deliberately omits -- terminal-residue naming, the disulfide
``PRES DISU`` patch plus its hardcoded S-S parameters, and the OPC water angle.
Ligands use none of this (see :func:`ligandff.build_ligand_ff`).

Notes
-----
Terminal naming lives here (not in the parser) so that same-named terminal
residues (e.g. two ``ALA`` in a bare N-/C-terminal dipeptide) can be renamed
``NALA``/``CALA`` before the writer de-duplicates residues by name.
"""

from __future__ import annotations

from .amber_to_charmm import build_forcefield, disulfide_parameters, opc_water_angle
from .model import ForceField, Patch


def build_protein_forcefield(
    parm, *, ncter: bool = False, disulfide: bool = False, opc: bool = False
) -> ForceField:
    """Build a protein :class:`~model.ForceField` from a parmed ``AmberParm``.

    Parameters
    ----------
    parm : parmed.amber.AmberParm
        A loaded AMBER protein topology.
    ncter : bool, optional
        If True, rename residue 0 ``N<name>`` and residue 1 ``C<name>`` (the
        bare two-residue N-/C-terminal dipeptide workflow). Requires exactly two
        residues. Default ``False``.
    disulfide : bool, optional
        If True, add the hardcoded S-S bond/angle/dihedral parameters and a
        ``PRES DISU`` patch. Default ``False``.
    opc : bool, optional
        If True, add the OPC water ``HW-OW-HW`` angle parameter (absent from the
        prmtop). Default ``False``.

    Returns
    -------
    ForceField
        The protein force field, ready for :func:`charmm_writer.write_rtf` /
        :func:`charmm_writer.write_prm` with the ``PROTEIN_*`` headers.

    Raises
    ------
    ValueError
        If ``ncter=True`` and the topology does not have exactly two residues.
    """
    ff = build_forcefield(parm, ligand=False)
    if ncter:
        _apply_terminal_names(ff)
    if disulfide:
        bond, angle, dihedrals = disulfide_parameters()
        # PREPEND, don't append: the writer de-duplicates parameters first-wins,
        # and these curated parm19sb values are authoritative. Appending would
        # let a value carried in the prmtop (an input that already contains an
        # SG-SG bond) silently take precedence over them.
        ff.bonds.insert(0, bond)
        ff.angles.insert(0, angle)
        ff.dihedrals[:0] = dihedrals
        ff.patches.append(Patch("DISU", 0.0, [("1SG", "2SG")]))
    if opc:
        ff.angles.insert(0, opc_water_angle())
    return ff


def _apply_terminal_names(ff: ForceField) -> None:
    """Rename residue 0 ``N<name>`` and residue 1 ``C<name>``, in place."""
    if len(ff.residues) != 2:
        raise ValueError(
            "ncter=True expects exactly two residues (an N-terminal + a "
            f"C-terminal amino acid); got {len(ff.residues)}"
        )
    ff.residues[0].name = f"N{ff.residues[0].name}"
    ff.residues[1].name = f"C{ff.residues[1].name}"

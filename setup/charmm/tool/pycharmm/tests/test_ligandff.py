"""Tests for pycharmm.ligandff: the AMBER GAFF2 -> CHARMM ligand converter.

Pure-Python: these exercise the conversion and its input validation against
committed AMBER topologies in ``data/ligandff/``, and need neither a built
CHARMM library nor AmberTools. The in-process CHARMM energy checks live in
test_ligandff_charmm.py.

``parmed`` is an optional pycharmm dependency, so the tests that need it skip
when it is absent.
"""

import os
from pathlib import Path

import pytest

from pycharmm import ligandff
from pycharmm.ligandff import charmm_writer as writer
from pycharmm.ligandff.amber_to_charmm import build_forcefield
from pycharmm.ligandff.model import (
    AtomType,
    BondType,
    DihedralType,
    ForceField,
    Patch,
    Residue,
    TopologyAtom,
)
from pycharmm.ligandff.protein_ff import build_protein_forcefield

pmd = pytest.importorskip("parmed", reason="parmed is an optional dependency")

# Absolute so the paths survive a chdir (conftest chdir's to tests/, and the
# sibling CHARMM tests chdir again into a scratch directory).
_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data" / "ligandff"
# Goldens live OUTSIDE tests/data/: conftest's autouse guard fails any test
# that writes under tests/data/, which would break LIGANDFF_REGEN_GOLDEN=1.
GOLDEN = _HERE / "ligandff_golden"
_REGEN = os.environ.get("LIGANDFF_REGEN_GOLDEN") == "1"

LIGANDS = ["aspirin", "tylenol", "chlorobenzene"]
ZERO_IMPROPER = "methylammonium"  # also carries a +1 charge
PROTEIN_CMAP = "ala_ace_nme"
PROTEIN_NCTER = "ala_nc"
PROTEIN_DISULFIDE = "disulfide"
WATER_OPC = "opc_water"


def _load(name):
    return pmd.load_file(str(DATA / f"{name}.prmtop"))


def _convert_ligand(name):
    """prmtop -> (rtf text, prm text) through the ligand path."""
    ff = build_forcefield(_load(name), ligand=True)
    return (
        writer.write_rtf(ff, title=writer.LIGAND_RTF_TITLE, declarations=writer.LIGAND_RTF_DECL),
        writer.write_prm(ff, title=writer.LIGAND_PRM_TITLE),
    )


def _convert_protein(name, *, ncter=False, opc=False, disu=False):
    """prmtop -> (rtf text, prm text) through the protein driver."""
    ff = build_protein_forcefield(_load(name), ncter=ncter, disulfide=disu, opc=opc)
    return (
        writer.write_rtf(ff, title=writer.PROTEIN_RTF_TITLE, declarations=writer.PROTEIN_RTF_DECL),
        writer.write_prm(ff, title=writer.PROTEIN_PRM_TITLE),
    )


def _check_golden(text, name, ext):
    gold = GOLDEN / f"{name}.{ext}"
    if _REGEN:
        gold.write_text(text)
        return
    assert gold.is_file(), f"missing golden {gold}"
    assert text == gold.read_text(), f"{name}.{ext} differs from {gold}"


# --------------------------------------------------------------------------- #
# public API surface
# --------------------------------------------------------------------------- #
def test_public_api():
    """The documented entry points are exported and callable."""
    assert ligandff.__all__ == [
        "LigandFF",
        "LigandSet",
        "build_ligand_ff",
        "combine_ligand_ffs",
        "load_ligand_ff",
        "load_ligand_ffs",
    ]
    assert callable(ligandff.build_ligand_ff)
    assert callable(ligandff.load_ligand_ff)
    assert callable(ligandff.combine_ligand_ffs)
    assert callable(ligandff.load_ligand_ffs)
    assert "estimated_parameters" in ligandff.LigandFF._fields
    assert "estimated_parameters" in ligandff.LigandSet._fields


def test_import_does_not_pull_optional_dependencies():
    """Importing pycharmm must not require parmed/rdkit (they are optional)."""
    import subprocess
    import sys

    import pycharmm

    code = (
        "import sys; import pycharmm; "
        "assert 'parmed' not in sys.modules, 'parmed imported eagerly'; "
        "assert 'rdkit' not in sys.modules, 'rdkit imported eagerly'; "
        "print('ok')"
    )
    # Resolve the package's own location: the suite chdir's to tests/, so a
    # relative PYTHONPATH would not find it in a source tree.
    pkg_parent = str(Path(pycharmm.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


# --------------------------------------------------------------------------- #
# ligand conversion + goldens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", LIGANDS)
def test_ligand_golden(name):
    rtf_txt, prm_txt = _convert_ligand(name)
    _check_golden(rtf_txt, name, "rtf")
    _check_golden(prm_txt, name, "prm")


@pytest.mark.parametrize("name", LIGANDS)
def test_ligand_rtf_well_formed(name):
    rtf_txt, _ = _convert_ligand(name)
    # Without AUTOGENERATE, CHARMM builds 0 angles/dihedrals and the energy is
    # silently wrong -- the original converter's ligand path omitted this.
    assert "AUTOGENERATE ANGLES DIHEDRALS" in rtf_txt
    assert rtf_txt.count("\nRESI ") == 1
    assert rtf_txt.rstrip().endswith("END")


def test_ligand_atom_types_are_prefixed():
    """GAFF types carry `_` so they cannot collide with protein atom types."""
    ff = build_forcefield(_load("aspirin"), ligand=True)
    assert all(t.name.startswith("_") for t in ff.atom_types)


def test_zero_improper_ligand_converts():
    """Regression: molecules with no impropers used to crash the converter."""
    rtf_txt, prm_txt = _convert_ligand(ZERO_IMPROPER)
    assert rtf_txt.rstrip().endswith("END")
    assert "IMPROPERS" in prm_txt


def test_remapped_type_still_gets_the_ligand_prefix():
    """A remapped type (WAT EP -> LP) must not escape the `_` prefix."""
    ff = build_forcefield(_load(WATER_OPC), ligand=True)
    names = {t.name for t in ff.atom_types}
    assert "_LP" in names and "LP" not in names


# --------------------------------------------------------------------------- #
# protein path (parm19sb): CMAP, terminals, water, disulfide
# --------------------------------------------------------------------------- #
def test_cmap_protein_conversion():
    rtf_txt, prm_txt = _convert_protein(PROTEIN_CMAP)
    _check_golden(rtf_txt, PROTEIN_CMAP, "rtf")
    _check_golden(prm_txt, PROTEIN_CMAP, "prm")
    assert "XC1" in rtf_txt  # ALA CA remapped for the phi/psi CMAP
    assert "\nCMAP\n" in prm_txt


def test_ncter_residue_naming():
    rtf_txt, _ = _convert_protein(PROTEIN_NCTER, ncter=True)
    _check_golden(rtf_txt, PROTEIN_NCTER, "rtf")
    assert "\nRESI NALA " in rtf_txt and "\nRESI CALA " in rtf_txt


def test_ncter_requires_two_residues():
    with pytest.raises(ValueError, match="two residues"):
        build_protein_forcefield(_load(PROTEIN_CMAP), ncter=True)


def test_opc_water_lonepair():
    rtf_txt, prm_txt = _convert_protein(WATER_OPC, opc=True)
    _check_golden(rtf_txt, WATER_OPC, "rtf")
    _check_golden(prm_txt, WATER_OPC, "prm")
    assert "LONEPAIR bisector EPW O H1 H2" in rtf_txt
    assert "\nACCE O\n" in rtf_txt
    assert "HW   OW   HW" in prm_txt


def test_disulfide_uses_patch_not_a_cross_residue_bond():
    """The SG-SG link belongs to PRES DISU, not to the CYX residue template."""
    rtf_txt, prm_txt = _convert_protein(PROTEIN_DISULFIDE, disu=True)
    _check_golden(rtf_txt, PROTEIN_DISULFIDE, "rtf")
    _check_golden(prm_txt, PROTEIN_DISULFIDE, "prm")
    assert "PRES DISU" in rtf_txt
    assert "BOND 1SG  2SG" in rtf_txt
    assert "BOND SG   +SG" not in rtf_txt  # regression: used to be emitted
    assert any(ln.startswith("S    S") and "166" in ln for ln in prm_txt.splitlines())


def test_unownable_term_raises(monkeypatch):
    """A term no residue can hold is reported, not silently dropped."""
    import pycharmm.ligandff.amber_to_charmm as a2c

    monkeypatch.setattr(a2c, "_owner_residue", lambda atoms, threshold: None)
    with pytest.raises(ValueError, match="cannot be written into any RESI"):
        build_forcefield(_load(PROTEIN_CMAP), ligand=False)


# --------------------------------------------------------------------------- #
# writer: de-duplication and formatting
# --------------------------------------------------------------------------- #
def test_parameters_dedup_by_charmm_key():
    ff = ForceField(
        atom_types=[
            AtomType("_ca", 12.01, "C3", 0.086, 1.908),
            AtomType("_ca", 12.01, "C4", 0.086, 1.908),  # same type, other comment
        ],
        bonds=[BondType(("_c", "_o"), 570.0, 1.229)] * 2,
    )
    prm = writer.write_prm(ff, title="*\n*")
    assert prm.count("\nMASS  -1  _ca") == 1
    assert prm.count("_c   _o") == 1


def test_torsion_palindromes_fold_and_series_is_ordered():
    terms = [
        DihedralType(("a", "b", "c", "d"), 1.0, 3, 0.0),
        DihedralType(("d", "c", "b", "a"), 1.0, 3, 0.0),  # reversed duplicate
        DihedralType(("a", "b", "c", "d"), 2.0, 1, 0.0),
    ]
    prm = writer.write_prm(ForceField(dihedrals=terms), title="*\n*")
    lines = [ln for ln in prm.splitlines() if ln.startswith("a    b    c    d")]
    assert len(lines) == 2  # the palindrome folded, both periodicities kept
    assert [int(ln.split()[5]) for ln in lines] == [1, 3]  # ordered by periodicity


def test_nonbonded_epsilon_sign_and_14_halving():
    ff = ForceField(atom_types=[AtomType("_c", 12.01, "C", 0.10, 1.9)])
    line = next(
        ln for ln in writer.write_prm(ff, title="*\n*").splitlines() if ln.startswith("_c ")
    )
    cols = line.split()
    assert float(cols[2]) == pytest.approx(-0.10)  # CHARMM epsilon is negative
    assert float(cols[5]) == pytest.approx(-0.05)  # 1-4 epsilon is halved


def test_patch_is_written():
    rtf = writer.write_rtf(
        ForceField(patches=[Patch("DISU", 0.0, [("1SG", "2SG")])]),
        title=writer.LIGAND_RTF_TITLE,
        declarations=writer.LIGAND_RTF_DECL,
    )
    assert "PRES DISU 0.00" in rtf and "BOND 1SG  2SG" in rtf


def test_same_name_different_topology_raises():
    """Two residues sharing a name but not a topology must not fold silently."""
    ff = ForceField(
        residues=[
            Residue("LIG", 0.0, atoms=[TopologyAtom("C1", "_c3", 0.0)]),
            Residue("LIG", 0.0, atoms=[TopologyAtom("N1", "_n3", 0.0)]),
        ]
    )
    with pytest.raises(ValueError, match="different topology"):
        writer.write_rtf(ff, title=writer.LIGAND_RTF_TITLE, declarations=writer.LIGAND_RTF_DECL)


def test_improper_peripheral_permutation_folds_but_central_atom_does_not():
    def res(imp):
        return Residue("R", 0.0, atoms=[TopologyAtom("A", "_a", 0.0)], impropers=[imp])

    kw = {"title": writer.LIGAND_RTF_TITLE, "declarations": writer.LIGAND_RTF_DECL}
    # peripheral swap (positions 2 and 4) is meaningless -> folds
    writer.write_rtf(
        ForceField(residues=[res(("W", "X", "Y", "Z")), res(("W", "Z", "Y", "X"))]), **kw
    )
    # a different central atom (position 3) is a real difference -> raises
    with pytest.raises(ValueError, match="different topology"):
        writer.write_rtf(
            ForceField(residues=[res(("W", "X", "Y", "Z")), res(("W", "X", "Z", "Y"))]),
            **kw,
        )


# --------------------------------------------------------------------------- #
# merging several ligands into one force field
# --------------------------------------------------------------------------- #
def _one_bond_ff(k, b0, *, name="LIG"):
    """A minimal force field carrying a single bond parameter."""
    return ForceField(
        atom_types=[AtomType("_c3", 12.01, "C1", 0.1094, 1.9080)],
        bonds=[BondType(("_c3", "_c3"), k, b0)],
        residues=[Residue(name, 0.0, atoms=[TopologyAtom("C1", "_c3", 0.0)])],
    )


def _merged(*ffs):
    """Accumulate force fields the way combine_ligand_ffs does (extend is in-place)."""
    merged = ForceField()
    for ff in ffs:
        merged.extend(ff)
    return merged


def test_conflicting_parameter_for_one_key_raises():
    """Two ligands disagreeing on the same bond must not silently first-win.

    parmchk2 estimates a missing parameter per molecule, so merged ligands can
    arrive with genuinely different numbers for one atom-type pair. CHARMM's
    table holds a single entry per key, so the disagreement has to surface.
    """
    ff = _merged(_one_bond_ff(300.0, 1.50, name="AAA"), _one_bond_ff(410.0, 1.50, name="BBB"))
    with pytest.raises(ValueError, match="bond"):
        writer.write_prm(ff, title=writer.LIGAND_PRM_TITLE)


def test_parameters_agreeing_within_tolerance_are_folded():
    """The same parameter tabulated to different precision is not a conflict.

    AMBER writes some values to a few significant figures and others through a
    degree/radian conversion, so identical parameters can differ in the last
    digits. Only a real disagreement should raise.
    """
    ff = _merged(
        _one_bond_ff(300.0, 1.50, name="AAA"),
        _one_bond_ff(300.00004, 1.500001, name="BBB"),
    )
    prm = writer.write_prm(ff, title=writer.LIGAND_PRM_TITLE)
    assert prm.count("_c3  _c3") == 1


def test_combine_ligand_ffs_merges_two_real_ligands(scratch_dir):
    """Two converted prmtops merge into one rtf/prm holding both residues."""
    from pycharmm.ligandff.pipeline import LigandFF, combine_ligand_ffs

    ligands = []
    for name in ("aspirin", "tylenol"):
        prmtop = scratch_dir / f"{name}.prmtop"
        prmtop.write_bytes((DATA / f"{name}.prmtop").read_bytes())
        # Only prmtop is read by the merge; the rest of a real build's paths are
        # irrelevant here.
        ligands.append(
            LigandFF(
                rtf=None,
                prm=None,
                pdb=None,
                prmtop=prmtop,
                inpcrd=None,
                mol2=None,
                frcmod=None,
                sdf=None,
                estimated_parameters=(),
            )
        )

    combined = combine_ligand_ffs(
        ligands, rtf=scratch_dir / "both.rtf", prm=scratch_dir / "both.prm"
    )
    rtf_txt = Path(combined.rtf).read_text()
    assert "RESI AIN" in rtf_txt and "RESI TYL" in rtf_txt
    assert list(combined.resnames) == ["AIN", "TYL"]
    # One shared parameter table, and every type still carries the ligand prefix.
    prm_txt = Path(combined.prm).read_text()
    assert prm_txt.count("NONBONDED") == 1
    assert "_ca" in prm_txt


# --------------------------------------------------------------------------- #
# input validation and parameter provenance
# --------------------------------------------------------------------------- #
def test_net_charge_detected_from_structure():
    pytest.importorskip("rdkit", reason="rdkit is an optional dependency")
    from pycharmm.ligandff.gaff import detect_formal_charge

    assert detect_formal_charge("C[NH3+]", from_smiles=True) == 1
    assert detect_formal_charge("CC(=O)[O-]", from_smiles=True) == -1
    assert detect_formal_charge("CC(=O)Oc1ccccc1C(O)=O", from_smiles=True) == 0


def test_wrong_net_charge_is_refused(scratch_dir):
    """A declared charge that disagrees with the structure must not be used."""
    pytest.importorskip("rdkit", reason="rdkit is an optional dependency")
    with pytest.raises(ValueError, match="disagrees with the structure"):
        ligandff.build_ligand_ff("C[NH3+]", resname="MAM", net_charge=0, workdir=str(scratch_dir))


def test_missing_sdf_is_a_file_error(scratch_dir):
    with pytest.raises(FileNotFoundError, match="no such sdf file"):
        ligandff.build_ligand_ff(str(scratch_dir / "nope.sdf"), workdir=str(scratch_dir))


def test_estimated_parameters_are_reported(scratch_dir):
    """parmchk2's guessed terms must be visible, not silently shipped."""
    from pycharmm.ligandff.gaff import estimated_parameters

    frcmod = scratch_dir / "x.frcmod"
    frcmod.write_text(
        "Remark\nBOND\nca-ca  478.4  1.387\nDIHE\n"
        "c3-o -c -os  1.1  180.0  2.0  Same as X -o -c -o , penalty score= 49.6\n"
        "ca-ca-ca-os  1.1  180.0  2.0  Using the default value\n"
    )
    flagged = estimated_parameters(frcmod)
    assert len(flagged) == 2
    assert not any("478.4" in line for line in flagged)  # real GAFF2 term kept quiet


def test_missing_tool_output_is_reported(scratch_dir):
    from pycharmm.ligandff.gaff import _require_output

    with pytest.raises(RuntimeError, match="did not write"):
        _require_output(scratch_dir / "absent.prmtop", "tleap")

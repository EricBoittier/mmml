"""In-process CHARMM checks for pycharmm.ligandff.

Builds each converted force field inside the running CHARMM (no subprocess) and
compares the potential energy against a recorded reference, which is what makes
this the real chemical-equivalence check: the references were measured with
C. L. Brooks III's original converter, so a drift here means the rewritten
conversion changed the chemistry.

Also covers :func:`load_ligand_ff`, the opt-in step that reads a built ligand
into CHARMM.
"""

import tempfile
from pathlib import Path

import pytest

from pycharmm import generate, lingo, psf, read, settings
from pycharmm.ligandff import charmm_writer as writer
from pycharmm.ligandff.amber_to_charmm import build_forcefield
from pycharmm.ligandff.protein_ff import build_protein_forcefield

pmd = pytest.importorskip("parmed", reason="parmed is an optional dependency")

# Absolute: these tests chdir into a scratch directory (see below), so relative
# data paths would not resolve.
DATA = (Path(__file__).resolve().parent / "data" / "ligandff").resolve()

# Reference CHARMM potential energies (kcal/mol) from the original converter.
ENERGY_REFERENCE = {
    "aspirin": (-2.57564, True),
    "tylenol": (-9.55035, True),
    "chlorobenzene": (4.93378, True),
    "ala_ace_nme": (-20.98780, False),
}
ENERGY_TOL = 0.05


def _empty_ff_files():
    """Write a topology and parameter file that define nothing, and cache them.

    Reading these replaces whatever CHARMM currently holds, which is the only way
    to empty the tables (there is no "clear rtf" command). Kept in one temporary
    directory for the whole module, short enough for CHARMM's file-name limit.
    """
    global _EMPTY_FF
    if _EMPTY_FF is None:
        empty = Path(tempfile.mkdtemp(prefix="ligandff_empty_"))
        rtf_path = empty / "empty.rtf"
        prm_path = empty / "empty.prm"
        rtf_path.write_text("* empty\n*\n36  1\n\nEND\n")
        prm_path.write_text("* empty\n*\n\nEND\n")
        _EMPTY_FF = (str(rtf_path), str(prm_path))
    return _EMPTY_FF


_EMPTY_FF = None


@pytest.fixture(autouse=True)
def _clean_charmm():
    """Start each test from an empty PSF *and* empty topology/parameter tables.

    Clearing the PSF alone is not enough: CHARMM keeps the residue topology and
    parameter tables across tests, so a test that appends adds its atom types on
    top of a previous test's. Two GAFF ligands share type names (``_ca``,
    ``_ha``, ...), which CHARMM rejects with "ATOM already exists with the same
    name" and then aborts the interpreter -- taking the whole pytest process with
    it. Without this reset the file passes or fails depending on the order it
    happens to run in, which is exactly the order coupling tracked in issue #45.
    """
    from pycharmm.select_atoms import SelectAtoms

    def reset():
        if psf.get_natom() > 0:
            psf.delete_atoms(SelectAtoms(select_all=True))
        empty_rtf, empty_prm = _empty_ff_files()
        read.rtf(empty_rtf)
        read.prm(empty_prm, flex=True)

    settings.set_bomb_level(-2)
    reset()
    yield
    reset()


def _write_files(name, ligand, workdir):
    """Convert data/ligandff/<name>.prmtop and write the rtf/prm/pdb."""
    parm = pmd.load_file(str(DATA / f"{name}.prmtop"))
    if ligand:
        ff = build_forcefield(parm, ligand=True)
        rtf_txt = writer.write_rtf(
            ff, title=writer.LIGAND_RTF_TITLE, declarations=writer.LIGAND_RTF_DECL
        )
        prm_txt = writer.write_prm(ff, title=writer.LIGAND_PRM_TITLE)
    else:
        ff = build_protein_forcefield(parm)
        rtf_txt = writer.write_rtf(
            ff, title=writer.PROTEIN_RTF_TITLE, declarations=writer.PROTEIN_RTF_DECL
        )
        prm_txt = writer.write_prm(ff, title=writer.PROTEIN_PRM_TITLE)

    rtf = workdir / f"{name}.rtf"
    prm = workdir / f"{name}.prm"
    pdb = workdir / f"{name}.pdb"
    rtf.write_text(rtf_txt)
    prm.write_text(prm_txt)
    struct = pmd.load_file(str(DATA / f"{name}.prmtop"), str(DATA / f"{name}.inpcrd"))
    pmd.formats.PDBFile.write(struct, str(pdb), charmm=True, use_hetatoms=False, renumber=True)
    return rtf, prm, pdb


def _energy_in_charmm(rtf, prm, pdb, segid="MOL"):
    """Read the force field + coordinates into CHARMM and return the energy.

    Files are passed by *basename*: CHARMM keeps a filename in a fixed-length
    buffer and silently truncates a long path, so an absolute path under a
    pytest scratch directory fails to open with only a level-0 warning and the
    energy then comes out wrong. Callers run with the cwd set to the files'
    directory (the ``scratch_chdir`` fixture).
    """
    settings.set_bomb_level(-2)
    read.rtf(rtf.name)
    read.prm(prm.name, flex=True)
    read.sequence_pdb(pdb.name)
    generate.new_segment(segid, setup_ic=True, warn=True)
    read.pdb(pdb.name)  # no resid=True: see the note in charmm_load.load_ligand_ff
    # energy.show() only prints the last computed energy; evaluate explicitly.
    lingo.charmm_script("energy")
    return {
        "total": lingo.get_energy_value("ENER"),
        "angle": lingo.get_energy_value("ANGL"),
        "dihe": lingo.get_energy_value("DIHE"),
    }


@pytest.mark.parametrize("name", list(ENERGY_REFERENCE))
def test_energy_matches_reference(name, scratch_chdir):
    """The converted force field reproduces the original converter's energy."""
    reference, ligand = ENERGY_REFERENCE[name]
    rtf, prm, pdb = _write_files(name, ligand, scratch_chdir)
    terms = _energy_in_charmm(rtf, prm, pdb)
    # Angles must exist: without AUTOGENERATE in the rtf, CHARMM builds none and
    # reports exactly zero, with a plausible-looking total.
    assert terms["angle"] > 0.0, f"{name}: no angle energy -- angles not generated"
    assert terms["total"] == pytest.approx(reference, abs=ENERGY_TOL), (
        f"{name}: energy {terms['total']:.5f} drifted from the reference {reference:.5f}"
    )


def test_load_ligand_ff_reads_rtf_prm_and_coordinates(scratch_chdir):
    """load_ligand_ff puts a built ligand into CHARMM ready to energize.

    Uses the default ``append`` (None): nothing is loaded yet here, so it must
    replace rather than append -- CHARMM treats appending to an empty topology
    as fatal ("Cant append to zero RTF") and would abort the interpreter.
    """
    from pycharmm.ligandff import LigandFF, load_ligand_ff

    rtf, prm, pdb = _write_files("aspirin", True, scratch_chdir)
    built = LigandFF(
        rtf=rtf,
        prm=prm,
        pdb=pdb,
        prmtop=None,
        inpcrd=None,
        mol2=None,
        frcmod=None,
        sdf=None,
        estimated_parameters=(),
    )
    settings.set_bomb_level(-2)
    err = load_ligand_ff(built, segid="AIN")  # default append
    assert err == 1
    assert psf.get_natom() == 21  # aspirin, all atoms present
    lingo.charmm_script("energy")
    assert lingo.get_energy_value("ANGL") > 0.0  # coordinates really were read
    assert lingo.get_energy_value("ENER") == pytest.approx(-2.57564, abs=ENERGY_TOL)


def test_load_ligand_ff_appends_to_a_protein_force_field(scratch_chdir):
    """The real use case: add a ligand to a protein force field already loaded.

    This is what the ligand atom-type ``_`` prefix exists for -- the ligand's
    GAFF types cannot collide with the protein's, so appending is safe.
    """
    from pycharmm.ligandff import LigandFF, load_ligand_ff

    # protein first, on its own
    p_rtf, p_prm, p_pdb = _write_files("ala_ace_nme", False, scratch_chdir)
    settings.set_bomb_level(-2)
    read.rtf(p_rtf.name)
    read.prm(p_prm.name, flex=True)
    read.sequence_pdb(p_pdb.name)
    generate.new_segment("PROA", setup_ic=True, warn=True)
    read.pdb(p_pdb.name)
    protein_atoms = psf.get_natom()
    assert protein_atoms == 22

    # then the ligand, appended by the default policy
    l_rtf, l_prm, l_pdb = _write_files("aspirin", True, scratch_chdir)
    built = LigandFF(l_rtf, l_prm, l_pdb, None, None, None, None, None, ())
    assert load_ligand_ff(built, segid="AIN") == 1
    assert psf.get_natom() == protein_atoms + 21  # both are present
    lingo.charmm_script("energy")
    assert lingo.get_energy_value("ANGL") > 0.0


def test_load_ligand_ff_missing_file_is_reported(scratch_chdir):
    """A missing rtf must raise, not read as an empty force field."""
    from pycharmm.ligandff import LigandFF, load_ligand_ff

    rtf, prm, pdb = _write_files("aspirin", True, scratch_chdir)
    rtf.unlink()
    built = LigandFF(rtf, prm, pdb, None, None, None, None, None, ())
    with pytest.raises(FileNotFoundError, match="ligand rtf not found"):
        load_ligand_ff(built)


def test_load_ligand_ff_overlong_path_is_reported(scratch_chdir):
    """A path CHARMM cannot hold is refused with an actionable message."""
    from pycharmm.ligandff import LigandFF, load_ligand_ff

    deep = scratch_chdir / ("d" * 120) / ("e" * 120)
    deep.mkdir(parents=True)
    rtf, prm, pdb = _write_files("aspirin", True, scratch_chdir)
    moved = deep / rtf.name
    rtf.replace(moved)
    built = LigandFF(moved, prm, pdb, None, None, None, None, None, ())
    with pytest.raises(ValueError, match="too long for CHARMM"):
        load_ligand_ff(built)


def test_load_ligand_ff_without_pdb_is_refused(scratch_dir):
    """Asking for coordinates when none were built is an error, not a crash."""
    from pycharmm.ligandff import LigandFF, load_ligand_ff

    rtf, prm, _pdb = _write_files("aspirin", True, scratch_dir)
    built = LigandFF(
        rtf=rtf,
        prm=prm,
        pdb=None,
        prmtop=None,
        inpcrd=None,
        mol2=None,
        frcmod=None,
        sdf=None,
        estimated_parameters=(),
    )
    with pytest.raises(ValueError, match="make_pdb=False"):
        load_ligand_ff(built)


def _built(name, workdir):
    """A LigandFF for a data fixture, as build_ligand_ff would return."""
    from pycharmm.ligandff import LigandFF

    rtf, prm, pdb = _write_files(name, True, workdir)
    return LigandFF(
        rtf=rtf,
        prm=prm,
        pdb=pdb,
        prmtop=DATA / f"{name}.prmtop",
        inpcrd=DATA / f"{name}.inpcrd",
        mol2=None,
        frcmod=None,
        sdf=None,
        estimated_parameters=(),
    )


def test_two_ligands_combined_load_together(scratch_chdir):
    """Two ligands merged into one force field coexist in CHARMM.

    Loading them one after another cannot work -- they share GAFF atom types --
    so combine_ligand_ffs merges them first. This is the canonical route for a
    system with more than one ligand.
    """
    from pycharmm.ligandff import combine_ligand_ffs, load_ligand_ffs

    a = _built("aspirin", scratch_chdir)
    b = _built("tylenol", scratch_chdir)
    both = combine_ligand_ffs([a, b], rtf="ligs.rtf", prm="ligs.prm")

    # one RESI per ligand, and the shared types appear once
    rtf_txt = both.rtf.read_text()
    assert both.resnames == ("AIN", "TYL")
    assert rtf_txt.count("\nRESI ") == 2
    prm_txt = both.prm.read_text()
    assert prm_txt.count("\nMASS  -1  _ca") == 1  # shared type merged

    settings.set_bomb_level(-2)
    codes = load_ligand_ffs(both)
    assert codes == [1, 1]
    assert psf.get_natom() == 21 + 20  # aspirin + acetaminophen
    lingo.charmm_script("energy")
    assert lingo.get_energy_value("ANGL") > 0.0


def test_combined_set_appends_to_a_protein(scratch_chdir):
    """A combined ligand set can still join a protein force field in memory."""
    from pycharmm.ligandff import combine_ligand_ffs, load_ligand_ffs

    p_rtf, p_prm, p_pdb = _write_files("ala_ace_nme", False, scratch_chdir)
    settings.set_bomb_level(-2)
    read.rtf(p_rtf.name)
    read.prm(p_prm.name, flex=True)
    read.sequence_pdb(p_pdb.name)
    generate.new_segment("PROA", setup_ic=True, warn=True)
    read.pdb(p_pdb.name)

    both = combine_ligand_ffs(
        [_built("aspirin", scratch_chdir), _built("chlorobenzene", scratch_chdir)],
        rtf="ligs.rtf",
        prm="ligs.prm",
    )
    assert load_ligand_ffs(both) == [1, 1]
    assert psf.get_natom() == 22 + 21 + 12

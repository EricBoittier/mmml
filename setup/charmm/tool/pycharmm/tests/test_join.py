"""Tests for gen.join (segment joining) and gen.replica (segment replication).

Builds a 6-residue peptide, places it in a water box (introducing
clashes), confirms the same selection counts as test_select_atoms,
then joins all PEPT residues into one segment and replicates the whole
system 5 times.

Pass criteria (carried over from the original procedural script):
  - 1.1: 3 atoms within 1.5 Å of sulfur atoms
  - 1.2: 28 atoms by-residue within 1.5 Å of sulfur
  - 1.3: 6 carbons within 10 Å of sulfur
  - 1.4: 92 atoms by-residue (carbons within 10 Å of sulfur)
  - 1.5: 28 atoms by-residue containing sulfur
  - 2.1: 73 water oxygens within 2.8 Å of peptide
  - join: PEPT segment still resolves after gen.join
  - replica: FARB segment count == nreplica * <atoms in PEPT>

Original by Thanh Lai, October 2021 (extended for join/replica).
"""

import pytest

from pycharmm import (
    SelectAtoms,
    coor,
    gen,
    ic,
    psf,
    read,
)
from pycharmm.lingo import charmm_script


@pytest.fixture(autouse=True)
def _reset_replicas():
    """Undo the replica setup this module creates, for the whole process.

    ``gen.replica`` raises CHARMM's replica count above one, and that lives in
    module data for the life of the process -- deleting every atom does not
    clear it. While it is set, any later ``energy omm`` aborts the interpreter
    with "Replicas not supported with OpenMM", which ends the pytest run rather
    than failing one test. Every OpenMM test file happens to sort before this
    one, so the trap only springs when a new one sorts after it.

    ``replica reset`` is CHARMM's own way back to a single primary subsystem,
    and it does restore ``energy omm``.
    """
    yield
    charmm_script("replica reset")


def _translate(dpos, sel=None):
    """Translate selected atoms by (dx, dy, dz). Default: all atoms."""
    if sel is None:
        sel = SelectAtoms(select_all=True).get_selection()
    else:
        sel = sel.get_selection()
    pos = coor.get_positions()
    pos_sel = pos[list(sel)]
    pos_sel = pos_sel.apply(
        lambda p: p + dpos[0] if p.name == "x" else (p + dpos[1] if p.name == "y" else p + dpos[2])
    )
    pos.update(pos_sel)
    coor.set_positions(pos)


def test_join_segments_workflow():
    """Full workflow: peptide build, selection counts, join, replica."""
    # Build peptide centered at origin
    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)
    charmm_script("stream data/toppar_water_ions.str")

    read.sequence_string("ALA SER CYS TYR MET ALA")
    gen.new_segment(seg_name="PEPT", first_patch="ACE", last_patch="CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    stats = coor.stat()
    _translate((-stats["xave"], -stats["yave"], -stats["zave"]))

    # Selection-count regression checks
    assert sum(SelectAtoms(chem_type="S").around(1.5)) == 3
    assert sum(SelectAtoms(chem_type="S").around(1.5).whole_residues()) == 28
    assert sum(SelectAtoms(chem_type="S").around(10) & SelectAtoms(chem_type="C")) == 6
    assert (
        sum((SelectAtoms(chem_type="S").around(10) & SelectAtoms(chem_type="C")).whole_residues())
        == 92
    )
    assert sum(SelectAtoms(chem_type="S").whole_residues()) == 28

    # Add water box and verify the clash count
    read.psf_card("data/water_cube.psf", append=True)
    read.coor_card("data/water_cube.crd", append=True)

    sel_clashes = (SelectAtoms(atom_type="OH2") & SelectAtoms(seg_id="TIP3")) & SelectAtoms(
        seg_id="PEPT"
    ).around(2.8)
    assert sum(sel_clashes) == 73

    # gen.join and gen.replica are the actual subjects of this test
    pept_atoms_before = sum(SelectAtoms(seg_id="PEPT"))
    assert pept_atoms_before > 0

    gen.join("PEPT", renumber=True)
    pept_atoms_after = sum(SelectAtoms(seg_id="PEPT"))
    assert pept_atoms_after == pept_atoms_before, (
        f"gen.join changed PEPT atom count: {pept_atoms_before} → {pept_atoms_after}"
    )

    natom_before_replica = psf.get_natom()
    gen.replica(segid="FARB", selection=SelectAtoms(select_all=True), nreplica=5)
    natom_after_replica = psf.get_natom()
    # gen.replica should add 5 copies of every atom currently in the system
    expected_added = 5 * natom_before_replica
    actual_added = natom_after_replica - natom_before_replica
    assert actual_added == expected_added, (
        f"gen.replica(nreplica=5) added {actual_added} atoms; expected {expected_added}"
    )

"""Tests for SelectAtoms.around and SelectAtoms.whole_residues.

The first half of the test suite uses an arbitrary 6-residue peptide
and verifies pyCHARMM atom selections agree with the canonical CHARMM
selection-language counts (".around.", ".byres.").

The second half places the peptide in a water box (introducing many
peptide-water clashes) and verifies that pyCHARMM picks the same
clashing-water sets as the equivalent CHARMM script.

Original by Thanh Lai, October 2021.
"""

import pytest

from pycharmm import SelectAtoms, coor, gen, ic, read
from pycharmm.lingo import charmm_script


def _translate(dpos, sel=None):
    """Translate a selection (or all atoms) by (dx, dy, dz)."""
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


@pytest.fixture(scope="module")
def peptide_at_origin():
    """Build ALA SER CYS TYR MET ALA peptide and recenter at origin."""
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


@pytest.fixture(scope="module")
def peptide_in_water_box(peptide_at_origin):
    """Add a water-cube box on top of the centered peptide."""
    read.psf_card("data/water_cube.psf", append=True)
    read.coor_card("data/water_cube.crd", append=True)


# Reference counts established in the original test (Thanh Lai 2021)
class TestAroundAndWholeResidues:
    def test_around_atoms_within_15_of_sulfur(self, peptide_at_origin):
        sel = SelectAtoms(chem_type="S").around(1.5)
        assert sum(sel) == 3

    def test_whole_residues_around_15_of_sulfur(self, peptide_at_origin):
        sel = SelectAtoms(chem_type="S").around(1.5).whole_residues()
        assert sum(sel) == 28

    def test_carbons_within_10_of_sulfur(self, peptide_at_origin):
        sel = SelectAtoms(chem_type="S").around(10) & SelectAtoms(chem_type="C")
        assert sum(sel) == 6

    def test_carbon_residues_within_10_of_sulfur(self, peptide_at_origin):
        sel = (SelectAtoms(chem_type="S").around(10) & SelectAtoms(chem_type="C")).whole_residues()
        assert sum(sel) == 92

    def test_all_sulfur_containing_residues(self, peptide_at_origin):
        sel = SelectAtoms(chem_type="S").whole_residues()
        assert sum(sel) == 28


class TestWaterBoxOverlap:
    def test_water_oxygens_within_28_of_peptide(self, peptide_in_water_box):
        sel = (SelectAtoms(atom_type="OH2") & SelectAtoms(seg_id="TIP3")) & SelectAtoms(
            seg_id="PEPT"
        ).around(2.8)
        assert sum(sel) == 73

    def test_water_residues_within_28_of_peptide(self, peptide_in_water_box):
        sel = (
            (SelectAtoms(atom_type="OH2") & SelectAtoms(seg_id="TIP3"))
            & SelectAtoms(seg_id="PEPT").around(2.8)
        ).whole_residues()
        assert sum(sel) == 219

    def test_waters_near_S_or_OH_residues(self, peptide_in_water_box):
        """Whole-residue selection across nested .byres. + .around. clauses."""
        sel = (
            SelectAtoms(seg_id="TIP3")
            & (SelectAtoms(chem_type="S") | SelectAtoms(chem_type="OH"))
            .whole_residues()
            .around(2.8)
        ).whole_residues()
        assert sum(sel) == 102

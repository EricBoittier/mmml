# This script provides a simple example of using SelectAtoms.around
# and SelectAtoms.byres. The methods are tested on an arbitrary peptide
# In the second part of the example, the peptide is placed in a water box to introduce
# atomic collisions. The two methods are tested to select colliding water molecules
# Thanh Lai, October 26, 2021

import sys

from pycharmm import *


def translate(dpos, sel=None):
    """Helper function for this script. Translate position of selected atoms. The default selection is all atoms.

    Args:
        dpos (tuple[float,float,float]): Change atoms by dx, dy, dz
        sel (pycharmm.Select.Atoms): Selection of atoms to translate. Default is all atoms
    """
    if sel is None:
        sel = SelectAtoms(select_all=True).get_selection()
    else:
        sel = sel.get_selection()
    pos = coor.get_positions()  # position dataframe of all atoms
    pos_sel = pos[list(sel)]  # position dataframe of selected atoms
    pos_sel = pos_sel.apply(lambda p: p + dpos[0] if p.name == "x" else (p+dpos[1] if p.name == "y" else p + dpos[2]))
    pos.update(pos_sel)
    coor.set_positions(pos)


def main():

    passed = True

    ########################################
    # LOAD IN TOPPAR AND ARBITRARY PEPTIDE #
    ########################################

    # load in toppar
    chm_read.rtf("toppar/top_all36_prot.rtf")
    chm_read.prm("toppar/par_all36_prot.prm", flex=True)
    charmm_script("stream toppar/toppar_water_ions.str")

    # load in an arbitrary peptide and center it at the origin
    chm_read.sequence_string("ALA SER CYS TYR MET ALA")
    gen.new_segment(seg_name="PEPT", first_patch="ACE", last_patch="CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, 'CAY', 1, 'CY', 1, 'N')  # specify the first 3 atoms to place in space
    ic.build()  # then from the initial placements of the three atoms we can build the seg
    stats = coor.stat()
    translate((-stats["xave"], -stats["yave"], -stats["zave"]))

    ############################
    # TESTING AROUND AND WHOLE_RESIDUES #
    ############################

    # EXAMPLE 1: counting neighboring atoms around sulfur atoms in peptide

    # 1.1 test around: select all ATOMS within 1.5 angstrom of sulfur atoms (there are two sulfur atoms in the system)
    print("\n\n!!!! 1.1 ATOMS WITHIN 1.5 ANGSTROM OF SULFURS !!!!")
    sel = SelectAtoms(chem_type="S").around(1.5)
    charmm_script("define sel select ( chem S .around. 1.5 ) end")
    print(f"PyCharmm selection {sum(sel)}")

    if sum(sel) != 3:
        passed = False
        print(f"Test 1.1 failed: {sum(sel)}")

    # 1.2 test whole_residues: now we select all RESIDUES within 1.5 angstrom of sulfur atoms
    print("\n\n!!!! 1.2 ATOM (BY-RESIDUE) WITHIN 1.5 ANGSTROM OF SULFURS !!!!")
    sel = (SelectAtoms(chem_type="S").around(1.5)).whole_residues()
    charmm_script("define sel select .byres. ( chem S .around. 1.5 ) end")
    print(f"PyCharmm selection {sum(sel)}")

    if sum(sel) != 28:
        passed = False
        print(f"Test 1.2 failed: {sum(sel)}")

    # 1.3 test around: let's select only carbon atoms that are around sulfur atoms
    print("\n\n!!!! 1.3 CARBON ATOMS WITHIN 10 ANGSTROM OF SULFURS !!!!")
    sel = SelectAtoms(chem_type="S").around(10) & SelectAtoms(chem_type="C")
    charmm_script("define sel select ( chem S .around. 10 ) .and. ( type C ) end")
    print(f"PyCharmm selection {sum(sel)}")

    if sum(sel) != 6:
        passed = False
        print(f"Test 1.3 failed: {sum(sel)}")

    # 1.4 test whole_residues: let's select only carbon atoms that are around sulfur atoms
    # note the python selection: we encase our selection in a parentheses and then call whole_residues()
    # which is equivalent to calling .byres. on CHARMM on the entire selection
    print("\n\n!!!! 1.4 CARBON ATOMS (BY-RESIDUE) WITHIN 10 ANGSTROM OF SULFURS !!!!")
    sel = (SelectAtoms(chem_type="S").around(10) & SelectAtoms(chem_type="C")).whole_residues()
    charmm_script("define sel select .byres. ( ( chem S .around. 10 ) .and. ( type C ) ) end")
    print(f"PyCharmm selection {sum(sel)}")

    if sum(sel) != 92:
        passed = False
        print(f"Test 1.4 failed: {sum(sel)}")

    # 1.5 test whole_residues: let's select all the sulfur atoms in the system (2 total) and then
    # convert our selection to a by-residue basis, thus we expect to select all atoms from cysteine and methionine
    print("\n\n!!!! 1.5 SULFUR ATOMS (BY-RESIDUE) !!!!")
    sel = SelectAtoms(chem_type="S").whole_residues()
    charmm_script("define sel select .byres. ( chem S ) end")
    print(f"PyCharmm selection {sum(sel)}")

    if sum(sel) != 28:
        passed = False
        print(f"Test 1.5 failed: {sum(sel)}")

    # we load in a water box to introduce collisions. water box is already centered at origin.
    chm_read.psf_card("data/water_cube.psf", append=True)  # seg_id = TIP3
    chm_read.coor_card("data/water_cube.crd", append=True)

    # select oxygen atoms of water within 2.8 angstroms of peptide
    print("\n\n!!!! 2.1 WATER OXYGENS WITHIN 2.8 ANGSTROM OF PEPTIDE !!!!")
    sel = (SelectAtoms(atom_type="OH2") & SelectAtoms(seg_id="TIP3")) & SelectAtoms(seg_id="PEPT").around(2.8)
    charmm_script("define sel select ( segid TIP3 .and. type OH2 ) .and. ( segid PEPT .around. 2.8 ) end")
    print(f"PyCharmm selection {sum(sel)}")

    if sum(sel) != 73:
        passed = False
        print(f"Test 2.1 failed: {sum(sel)}")

    gen.join('PEPT', renumber=True)
    # gen.join(segid_1='TIP3', segid_2='PEPT', renumber=True)
    gen.replica(segid='FARB', selection=SelectAtoms(select_all=True), nreplica=5)

if __name__ == "__main__":
    main()

# This script tests the Coordinates class from the pycharmm pip package
# Written by Josh Buckner 24 November 2021

from pycharmm import *


chm_read.rtf('toppar/top_all22_prot.inp')
chm_read.prm('toppar/par_all22_prot.inp')

chm_read.sequence_string('ALA GLY')
gen.new_segment(seg_name='ALAD',
                    first_patch='ACE',
                    last_patch='CT3',
                    setup_ic=True)

chm_read.sequence_string('GLU')
gen.new_segment(seg_name='GLAD',
                    first_patch='ACE',
                    last_patch='CT3',
                    setup_ic=True)

chm_read.sequence_string('SER')
gen.new_segment(seg_name='SAD',
                    first_patch='ACE',
                    last_patch='CT3',
                    setup_ic=True)

ic.prm_fill(False)
ic.seed(1, 'CAY', 1, 'CY', 1, 'N')
ic.build()

coor.orient(by_rms=False,by_mass=False,by_noro=False)

coor.show()

main_pos = Coordinates('main')
print('MAIN ==========')
print(main_pos.coords.to_markdown())
print('==========')

comp_pos = Coordinates('comp')
print('COMP ==========')
print(main_pos.coords.to_markdown())
print('==========')

comp2_pos = Coordinates('comp2')
print('COMP2 ==========')
print(main_pos.coords.to_markdown())
print('==========')

# test for coor.dist()

# 1. residue-to-residue min distance (COOR DIST RESI)
print('\n\n==== CONTACTS (reisude-to-residue min distance) ======')
rcontacts = coor.dist(selection1= (SelectAtoms(seg_id='ALAD') & SelectAtoms(res_id='1')),\
                      selection2= (SelectAtoms(seg_id='ALAD') & SelectAtoms(res_id='2')),\
                      cutoff = 6.)
print(rcontacts)
print('==========')

# any pair of atoms not in the exclusion list (COOR DIST)
print('\n\n==== CONTACTS (any pair of atoms not in exclusion list) ======')
rcontacts = coor.dist(selection1= (SelectAtoms(seg_id='ALAD') & SelectAtoms(res_id='1')),\
                      selection2= (SelectAtoms(seg_id='ALAD') & SelectAtoms(res_id='1')),\
                      cutoff = 6., \
                      resi = False)
print(rcontacts)
print('==========')


# any pair of atoms regardless of the exclusion list (COOR DIST NOEXCLusion NO14Exclusion)
print('\n\n==== CONTACTS (any pair of atoms) ======')
rcontacts = coor.dist(selection1= (SelectAtoms(seg_id='ALAD') & SelectAtoms(res_id='1')),\
                      selection2= (SelectAtoms(seg_id='ALAD') & SelectAtoms(res_id='1')),\
                      cutoff = 6., \
                      resi = False, omit_excl = False, omit_14excl = False)
print(rcontacts)
print('==========')

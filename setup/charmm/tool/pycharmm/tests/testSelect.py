# This script implements that c37test/domdec_gpu.inp test case
# This script generates and
# plots the alanine dipeptide phi/psi (f/y) map using pyCHARMM
# Written by C.L. Brooks III, November 10, 2020

import numpy as np
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

passed = True
atoms = SelectAtoms(atom_type='CA')
if np.sum(list(atoms)) != 4:
    passed = False
    print('CA atoms failed',np.sum(list(atoms)))
atoms = SelectAtoms().by_atom_type('CA')
if np.sum(list(atoms)) != 4:
    passed = False
    print('CA atoms failed',np.sum(list(atoms)))
atoms_in_seg = atoms & SelectAtoms(seg_id='ALAD')
if np.sum(list(atoms_in_seg)) != 2:
    passed = False
    print('CA atoms in seg ALAD failed',np.sum(list(atoms_in_seg)))
atoms = SelectAtoms().by_atom_type('CA').by_atom_type('C')
if np.sum(list(atoms)) != 8:
    passed = False
    print('CA or C atoms failed',np.sum(list(atoms)))

# added ability to select multiple segments, resids and atom_types by
# generalizing by_res_and_type

atoms_in_seg = SelectAtoms().by_res_and_type('ALAD','1','CA')
if np.sum(list(atoms_in_seg)) != 1:
    passed = False
    print('CA atoms in seg ALAD and resid 1 failed',np.sum(list(atoms_in_seg)))

atoms = SelectAtoms().by_res_and_type('ALAD','1','CY N CA C')
if np.sum(list(atoms)) != 4:
    passed = False
    print('by_res_and_type failed on multiple type',np.sum(list(atoms)))

atoms = SelectAtoms().by_res_and_type('ALAD GLAD','1','CY N CA C')
if np.sum(list(atoms)) != 8:
    passed = False
    print('by_res_and_type failed on multiple segid',np.sum(list(atoms)))

atoms = SelectAtoms().by_res_and_type('ALAD','1 2','CY N CA C')
if np.sum(list(atoms)) != 7:
    passed = False
    print('by_res_and_type failed on multiple resid',np.sum(list(atoms)))

atoms = SelectAtoms().by_res_and_type('ALAD','1','CA')
bondedAtoms = SelectAtoms(select.bonded(atoms))
if np.sum(list(bondedAtoms)) != 5:
    passed = False
    print(f'selection using select.bonded() failed. Selected {np.sum(list(bondedAtoms))} atoms (expected 5)')

if passed: print('\n\n\nSelection Test PASSED')
else: print('\n\n\nSelection Test FAILED')

lingo.charmm_script('stop')

exit()

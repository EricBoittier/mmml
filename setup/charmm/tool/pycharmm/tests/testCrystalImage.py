# This script implements that c37test/domdec_gpu.inp test caseThis script generates and
# plots the alanine dipeptide phi/psi (f/y) map using pyCHARMM
# Written by C.L. Brooks III, November 10, 2020

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import shlex

from pycharmm import *

# Read in the topology and parameter files
chm_read.rtf('toppar/top_all36_prot.rtf')
chm_read.prm('toppar/par_all36_prot.prm', flex=True)
charmm_script('stream toppar/toppar_water_ions.str')

chm_read.sequence_string('ALA')
gen.new_segment(seg_name='PRO0',
                    first_patch='ACE',
                    last_patch='CT3',
                    setup_ic=True)

# Note would be great to have this way of adding a repeating sequence
charmm_script('read sequ tip3 350')
gen.new_segment(seg_name='WT00',
                angle=False,
                dihedral=False)

chm_read.pdb('data/ws0.pdb',resid=True)

# set the nonbonds dictionary and set-up nonbonds
# Nonbonded dictionary values
nbonds = {'elec': True,
          'atom': True,
          'cdie': True,
          'eps': 1,
          'switch': True,
          'pmewald': True,
          'kappa': 0.32,
          'fftx': 24,
          'ffty': 24,
          'fftz': 24,
          'order': 4,
          'vdw': True,
          'vatom': True,
          'vswitch': True,
          'cutnb': 11,
          'ctofnb': 10,
          'ctonnb': 10,
      }
nbonds['cutim'] = nbonds['cutnb']
nbond_noewald = NonBondedScript(**nbonds)
nbond_noewald.run()
energy.show()

stats = coor.stat()
size = ( stats['xmax'] - stats['xmin']\
       + stats['ymax'] - stats['ymin']\
       + stats['zmax'] - stats['zmin'] ) / 3
print('Cubic Box Size',size)
offset = size/2
xyz = coor.get_positions()
print(np.average(xyz))
xyz += size/2
print(np.average(xyz))
coor.set_positions(xyz)
# Now set up crystal image calculation
crystal.define_cubic(length=size)
crystal.build(cutoff=nbonds['cutim'])
for i in 'PRO0'.strip().split():
    image.setup_segment(offset,offset,offset,i)
for i in 'TIP3'.strip().split():
    image.setup_residue(offset,offset,offset,i)

energy.show()

# Now try Ewald interactions
nbonds['ewald']=True
nbond_ewald = NonBondedScript(**nbonds)
nbond_ewald.run()
energy.show()

charmm_script('stop')

exit()

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
lingo.charmm_script('stream toppar/toppar_water_ions.str')

chm_read.sequence_string('ALA')
gen.new_segment(seg_name='PRO0',
                    first_patch='ACE',
                    last_patch='CT3',
                    setup_ic=True)

# Note would be great to have this way of adding a repeating sequence
lingo.charmm_script('read sequ tip3 350')
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
print('Cubic Box Size: ', size)
offset = size / 2.0
xyz = coor.get_positions()
print('Average position: ', np.average(xyz))
xyz += size / 2.0
print('Average position after translation: ', np.average(xyz))
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

# Run some langevin dynamics using CPU
dynamics_dict = {
    'ktable': True,
    'velos': True,
    'lambdata': True,
    'leap': True,
    'verlet': False,
    'cpt': False,
    'new': False,
    'langevin': True,
    'omm': False, #'langevin gamma 5',
    'timestep': 0.002,
    'start': True,
    'nstep': 100,
    'nsavc': 0,
    'nsavv': 10,
    'inbfrq':-1,
    'ihbfrq':0,
    'ilbfrq':0,
    'imgfrq':0,
    'iunrea':-1,
    'iunwri':-1,
    'iuncrd':-1,
    'nsavl': 0,  # frequency for saving lambda values in lamda-dynamics
    'iunldm':-1,
    'ilap': -1,
    'ilaf': -1,
    'nprint': 10, # Frequency to write to output
    'iprfrq': 50, # Frequency to calculate averages
    'isvfrq': 100, # Frequency to save restart file
    'ntrfrq': 100,
    'firstt': 298,
    'finalt': 298,
    'tstruct': 298,
    'tbath': 298,
    'iasors': 1,
    'ichecw': 0,
    'iscale': 0,  # scale velocities on a restart
    'scale': 1,  # scaling factor for velocity scaling
    'echeck': -1}

# set the values of fbeta for atoms in system
dyn.set_fbetas(np.full(psf.get_natom(),5.0))
# table = dyn.run(**dynamics_dict)
# print(table)
# dyn_lang = DynamicsScript(**dynamics_dict)
# dyn_lang.run()


# Run some langevin dynamics using OpenMM/GPU
# dynamics_dict['omm'] = 'gamma 5'
dyn_lang = DynamicsScript(**dynamics_dict)
dyn_lang.run()
print(dyn_lang.ktable)
print(dyn_lang.velos)
# lingo.charmm_script('stop')

exit()

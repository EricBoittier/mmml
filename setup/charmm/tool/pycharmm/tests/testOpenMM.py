# This script provides a simple example of building an
# alanine dipeptide and minimizing the structure and then
# calculating the energy to illustrate functionality to be
# exposed in PyCHARMM.
#  copyright C.L. Brooks III, April 15, 2019

import numpy as np
import openmm
import os
import subprocess
import sys


try:
    charmm_lib_dir = os.environ['CHARMM_LIB_DIR']
    charmm_data_dir = os.environ['CHARMM_DATA_DIR']
except KeyError:
    print('please set environment variables',
          'CHARMM_LIB_DIR and CHARMM_DATA_DIR',
          file=sys.stderr)
    sys.exit(1)


from pycharmm import *


chm_settings.set_verbosity(0)
rtf_fn = charmm_data_dir + '/top_all36_prot.rtf'
chm_read.rtf(rtf_fn)

prm_fn = charmm_data_dir + '/par_all36_prot.prm'
chm_read.prm(prm_fn, flex=True)

# begin toppar/toppar_water_ions.str
chm_read.rtf('data/water_ions.rtf', append=True)
chm_read.prm('data/water_ions.prm', append=True, flex=True)

old_warn_level = chm_settings.set_warn_level(-1)
old_bomb_level = chm_settings.set_bomb_level(-1)

chm_read.prm('data/sodium_oxygen_nbfixes.prm', append=True, flex=True)

chm_settings.set_warn_level(old_warn_level)
chm_settings.set_bomb_level(old_bomb_level)
# end toppar/toppar_water_ions.str

# read in the sequence of the protein to be generated
# only useful for the same residue
chm_read.sequence_string('ALA')

gen.new_segment('ADP', 'ACE', 'CT3', setup_ic=True)

ic.prm_fill(False)
ic.seed(1, 'CAY', 1, 'CY', 1, 'N')
ic.build()

# The coor orie command is useful to expose since it allows one to
# orient the system in preparation for other calculations
coor.orient()
coor.show()

# The nbonds command is where most of the control of how the
# energy is calculated to compute forces for minimization, dynamics
# or simply energy commands, this should be exposed to the pyCHARMM
# interface so that one can set-up exactly how the forces/energy is
# calculated.

nbonds_script = NonBondedScript(
    cutnb=18.0,
    ctonnb=15.0,
    ctofnb=13.0,
    eps=1.0,
    cdie=True,
    atom=True,
    fswitch=True,
    vatom=True,
    vfswitch=True
).run()

xyz = coor.get_positions()
charmm_script('energy omm')
minimize.run_omm(nstep=500,tolgrd=0)
charmm_script('energy omm')
energy.show()
eomm = energy.get_total()
grmsomm  = energy.get_grms()
coor.set_positions(xyz)
energy.show()
minimize.run_abnr(nstep=700, tolgrd=1e-3)
# chm_settings.set_verbosity(5)
energy.show()
eabnr = energy.get_total()
grmsabnr = energy.get_grms()
coor.set_positions(xyz)
energy.show()

tol = 3e-4
passed = True
if np.abs(eomm-eabnr)>tol:
    passed = False
    print('energy FAIL for OpenMM minimizer')
else:
    print('energy PASS for OpenMM minimizer')
if passed:
    print('PASSED for OpenMM Minimizer')
else:
    print('FAILED for OpenMM Minimizer')

serial_sys = chm_omm.get_system_serial()
sys = openmm.XmlSerializer.deserialize(serial_sys)

for force in sys.getForces():
    print(force.getName())

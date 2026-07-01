# This script provides a simple example of building an
# alanine dipeptide and minimizing the structure and then
# calculating the energy to illustrate functionality to be
# exposed in PyCHARMM.
#  copyright C.L. Brooks III, April 15, 2019

import os
import sys
import subprocess
import shlex

# import pandas

try:
    charmm_lib_dir = os.environ['CHARMM_LIB_DIR']
    charmm_data_dir = os.environ['CHARMM_DATA_DIR']
except KeyError:
    print('please set environment variables',
          'CHARMM_LIB_DIR and CHARMM_DATA_DIR',
          file=sys.stderr)
    sys.exit(1)


from pycharmm import *


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
coor.orient(by_rms=False,by_mass=False,by_noro=False)
coor.show()

# The nbonds command is where most of the control of how the
# energy is calculated to compute forces for minimization, dynamics
# or simply energy commands, this should be exposed to the pyCHARMM
# interface so that one can set-up exactly how the forces/energy is
# calculated.

my_nbonds = NonBondedScript(
    cutnb=18.0, ctonnb=15.0, ctofnb=13.0,
    eps=1.0,
    cdie=True,
    atom=True, vatom=True,
    fswitch=True, vfswitch=True)
my_nbonds.run()

# Exposing the minimizer is helpful since it will be a widely
# useful function. Note that if nbonds is available to the PyCHARMM API
# then this sort of basic minimize command is sufficient, where abnr could
# be substituted by sd or conj as well.
#
# minimize abnr nstep 1000 tole 1e-3 tolgr 1e-3
minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
energy.show()

print('prop names')
print(energy.get_property_names())

print('prop statuses')
print(energy.get_property_statuses())

print('props')
print(energy.get_properties())

print('term names')
print(energy.get_term_names())

print('term statuses')
print(energy.get_term_statuses())

print('terms')
print(energy.get_terms())

etable = energy.get_energy()
print(etable.to_markdown())

print('ENER ' +
        str(energy.get_property_by_name('ener')) + ' ' +
        str(etable.ENER))

print('GRMS ' +
        str(energy.get_property_by_name('grms')) + ' ' +
        str(etable.GRMS))

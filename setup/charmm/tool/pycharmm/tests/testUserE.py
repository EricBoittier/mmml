# This script provides a simple example of
# a python energy function that runs during each step of
# energy calculation (minimization, dynamics, a simple energy call ...)

import os
import sys
import numpy as np


try:
    charmm_lib_dir = os.environ['CHARMM_LIB_DIR']
    charmm_data_dir = os.environ['CHARMM_DATA_DIR']
except KeyError:
    print('please set environment variables',
          'CHARMM_LIB_DIR and CHARMM_DATA_DIR',
          file=sys.stderr)
    sys.exit(1)


from pycharmm import *


# Set-up blocked alanne residue in CHARMM
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

chm_read.sequence_string('ALA')

gen.new_segment('ADP', 'ACE', 'CT3', setup_ic=True)

ic.prm_fill(False)
ic.seed(1, 'CAY', 1, 'CY', 1, 'N')
ic.build()

coor.orient()
coor.show()

NonBondedScript(
    cutnb=18.0,
    ctonnb=15.0,
    ctofnb=13.0,
    eps=1.0,
    cdie=True,
    atom=True,
    fswitch=True,
    vatom=True,
    vfswitch=True).run()

# Impose harmonic restraints via external python function,
# check against harmonic restraints
# Set current build coordinates as reference coordinates
xref = coor.get_main()

# Note the variables x_pos, y_pos, z_pos, dx, dy and dz are
# C-type pointers and thus need some indexing
def harm(natoms,
         x_pos, y_pos, z_pos,
         dx, dy, dz):
    eharm = np.sum(np.power((x_pos[:natoms] - xref.x),2)+
                   np.power((y_pos[:natoms] - xref.y),2)+
                   np.power((z_pos[:natoms] - xref.z),2))

    for i in range(natoms):
        dx[i] += 2.0 * ( x_pos[i] - xref.x[i] )
        dy[i] += 2.0 * ( y_pos[i] - xref.y[i] )
        dz[i] += 2.0 * ( z_pos[i] - xref.z[i] )

    return eharm


energy.show()
minimize.run_abnr(nstep = 1000,
                   tolenr = 1e-3,
                   tolgrd = 1e-3)
energy.show()

# Test the usere harmonic restraint
charmm_script('skipe incl all excl harm user')
e_func = EnergyFunc(harm)
energy.show()
grms_harm = lingo.get_energy_value('GRMS')
e_harm = lingo.get_energy_value('ENER')

# Now get restraint energy with conventional CHARMM harmonic restraints
# Turn off usere function
e_func.unset_func()
coor.set_comparison(xref)
cons_harm.setup_absolute(force_const = 1.0,
                         q_mass = False,
                         comparison=True)
energy.show()
grms_cons = lingo.get_energy_value('GRMS')
e_cons = lingo.get_energy_value('ENER')
cons_harm.turn_off()

tol = 1e-5
passed = True
if np.abs(e_harm-e_cons)>tol:
    passed = False
    print('energy FAIL for function harm')
else:
    print('energy PASS for function harm')

if np.abs(grms_harm-grms_cons)>tol:
    passed = False
    print('force FAIL for function harm')
else:
    print('force PASS for function harm')

# Now test a selection of atoms
rcs = SelectAtoms().by_res_and_type('ADP','1','CY N CA C NT')
# and make an natoms boolean list from this
rcs_bool = list(rcs)
print('Number of selected atoms',np.sum(rcs_bool))
cons_harm.setup_absolute(force_const = 10.0, q_mass = False,
                         selection=rcs, comparison=True)
energy.show()
grms_cons = lingo.get_energy_value('GRMS')
e_cons = lingo.get_energy_value('ENER')
cons_harm.turn_off()

# Note the variables x_pos, y_pos, z_pos, dx, dy and dz are
# C-type pointers and thus need some indexing
def selharm(natoms,
         x_pos, y_pos, z_pos,
         dx, dy, dz):
    eharm = 0.0
    for i,b in enumerate(rcs_bool):
        if b:
            d =  ( x_pos[i] - xref.x[i] )
            eharm += 10.0 * d*d
            dx[i] += 2.0 * 10.0 * d
            d =  ( y_pos[i] - xref.y[i] )
            eharm += 10.0 * d*d
            dy[i] += 2.0 * 10.0 * d
            d =  ( z_pos[i] - xref.z[i] )
            eharm += 10.0 * d*d
            dz[i] += 2.0 * 10.0 * d
    return eharm

e_func.set_func(selharm)
energy.show()
grms_harm = lingo.get_energy_value('GRMS')
e_harm = lingo.get_energy_value('ENER')

if np.abs(e_harm-e_cons) > tol:
    passed = False
    print('energy FAIL for function selharm')
else:
    print('energy PASS for function selharm')

if np.abs(grms_harm-grms_cons)>tol:
    Passed = False
    print('force FAIL for function selharm')
else:
    print('force PASS for function selharm')

if passed:
    print('\n\n\nUser Energy Test PASSED')
else:
    print('\n\n\nUser Energy Test FAILED')

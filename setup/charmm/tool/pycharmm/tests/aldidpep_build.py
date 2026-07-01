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

# Here it would be useful to have the coordinates available
# to the python script with the full precision by directly accessing
# the coordinate arrays in source/ltm/coord_ltm.F90
# and coordc_ltm.F90 (comparison coordinates).
coor.show()

# my_atoms = SelectAtoms(seg_id='ADP')
# flags = list(my_atoms)
# print('DEBUG: SelectAtoms res_name test')
# print(flags)

# my_atoms = SelectAtoms(chem_type='HA3')
# flags = list(my_atoms)
# print('DEBUG: SelectAtoms chem_type test')
# print(flags)

# natc = param.get_natc()
# atc = param.get_atc()
# n_atoms = psf.get_natom()
# iac = psf.get_iac()

# print('DEBUG: begin chem types ' + str(natc))
# for i in range(n_atoms):
#     print('    ', i, ' : ', atc[iac[i]])

# print('DEBUG: end chem types')

# print('DEBUG: begin atom types')
# atypes = psf.get_atype()
# for i, atom_type in enumerate(atypes):
#     print('    ', i, ' : ', atom_type)

# print('DEBUG: end atom types')

# my_atoms = SelectAtoms(atom_type='HA')
# flags = list(my_atoms)
# print('DEBUG: SelectAtoms atom_type test')
# print(flags)

# my_atoms = SelectAtoms().by_res_and_type('ADP', '1', 'HA')
# flags = list(my_atoms)
# for i, flag in enumerate(flags):
#     if flag:
#         print('    atom i ', i, ' : ', flag)

# sys.exit(0)

# Here it would be useful to have the forces available
# to the python script, this could be done through
# an interface to coor force and then access to the coordinates
# or by directly accessing the force arrays in source/ltm/dreiv_ltm.F90.
# coor force comp
# print coor comp
coor.copy_forces()
coor.show_comp()

# write coor pdb name pdb/adp.pdb
# write psf card name pdb/adp.psf
if not os.path.isdir('pdb'): os.system('mkdir pdb')
chm_write.coor_pdb('pdb/adp.pdb')
chm_write.psf_card('pdb/adp.psf')

# Just to build on this example and provide examples for reading
# psf and coordinates, the delete isn't necessary initially, unless its
# easy.
# delete atom select all end
psf.delete_atoms(SelectAtoms().all_atoms())

# read psf card name pdb/adp.psf
chm_read.psf_card('pdb/adp.psf')

# read coor pdb name pdb/adp.pdb resid
adp_pdb_file = 'pdb/adp.pdb'
chm_read.pdb(adp_pdb_file, resid=True)

# This is not needed, just here to enable this processing in this script.
# system "convpdb.pl -solvate -cutoff 10 -cubic -out charmm22 pdb/adp.pdb
# | convpdb.pl -segnames adp ala -nsel TIP3 > pdb/wt00.pdb"

convpdb_command = 'convpdb.pl'
convpdb_str = (convpdb_command + ' -solvate -cutoff 10 -cubic -out charmm22 ' +
               adp_pdb_file)
p1 = subprocess.Popen(shlex.split(convpdb_str), stdout=subprocess.PIPE)

convpdb_str = convpdb_command + ' -segnames -nsel TIP3'
wt00_pdb_fn = 'pdb/wt00.pdb'
with open(wt00_pdb_fn, 'w') as wt00_pdb_file:
    p2 = subprocess.Popen(shlex.split(convpdb_str),
                          stdin=p1.stdout, stdout=wt00_pdb_file)

p2.communicate()

# Here is an alternative means of reading a sequence
# read sequ pdb name pdb/wt00.pdb
chm_read.sequence_pdb(wt00_pdb_fn)

# Another example of the generate command
# generate wt00 noangle nodihedral
# TODO: add no_angle, no_dihedral opts
gen.new_segment('WT00', angle=False, dihedral=False)

chm_read.pdb(wt00_pdb_fn, resid=True)

# This command gives info that is useful in setting up
# simulation though could be gotten in separate CHARMM
# preparation.
# coor stat
stats = coor.stat()

xsize = stats['xmax'] - stats['xmin']
ysize = stats['ymax'] - stats['ymin']
zsize = stats['zmax'] - stats['zmin']

boxsize = max(xsize, ysize, zsize)
boxhalf = boxsize / 2.0
# To here ^^^^^^^^^

# These next 4 commands are likely needed in setting up
# a system using PyCHARMM
# crystal define cubic @boxsize @boxsize @boxsize 90 90 90
# crystal build cutoff @boxhalf noper 0
crystal.define_cubic(boxsize)
crystal.build(boxhalf)

# image byseg xcen 0 ycen 0 zcen 0 select segid adp end
# image byres xcen 0 ycen 0 zcen 0 select resname tip3 end
image.setup_segment(0.0, 0.0, 0.0, 'ADP')
image.setup_residue(0.0, 0.0, 0.0, 'TIP3')
# To here ^^^^^^^^^

# Again this stuff is not essential
cutnb = boxhalf
cutim = cutnb
ctofnb = cutnb - 2.0
ctonnb = cutnb - 4.0

# Another nbonds example
# nbonds cutnb @cutnb cutim @cutim ctofnb @ctofnb ctonnb @ctonnb -
#        inbfrq -1 imgfrq -1
NonBondedScript(
    cutnb=cutnb, cutim=cutim, ctonnb=ctonnb, ctofnb=ctofnb,
    eps=1.0,
    cdie=True,
    atom=True, vatom=True,
    fswitch=True, vfswitch=True,
    inbfrq=-1, imgfrq=-1).run()

# cons fix, cons harm are two sets of commands that could
# be useful, but not essential on first passes
# cons fix select segid adp end
cons_fix.setup(SelectAtoms(seg_id='ADP'))

# Another example of minimization
# mini sd nstep 1000 tole 1e-3 tolgrd 1e-3
minimize.run_sd(nstep=500, tolenr=1e-3, tolgrd=1e-3)

# cons fix, cons harm are two sets of commands that could
# be useful, but not essential on first passes
# cons fix select none end
cons_fix.turn_off()

# write psf card name pdb/adp+wat.psf
# write coor pdb name pdb/adp+wat_min.pdb
chm_write.psf_card('pdb/adp+wat.psf')
chm_write.coor_pdb('pdb/adp+wat_min.pdb')

# It will also be essential to be able to set SHAKE up
# shake bonh tol 1e-7
shake.on(bonh=True, tol=1e-7)

# This would be required to run Langevin dynamics with
# PyCHARMM. Note there are two pieces here, the scalar command
# and its repertoir of sub-commands and the selection command and
# its syntax
# scalar fbeta set 5 select .not. hydrogen end
# fbetas = scalar.get(fbeta=[not x for x in select.hydrogen()])
# n = psf.get_natom()
# scalar.set_fbetas([0.0] * n, [not x for x in select.hydrogen()])
# fbetas = scalar.get_fbetas()
# print(fbetas[0:100])

# Basic file i/o to open files for writing by other CHARMM processes

# open unit 10 write form name res/adp.res
# This is the dynamics command that contains all the variables
# that are not set/cannot be set via nbonds
# dynamics langevin start nstep 1000 timestep 0.002 - ! Basic dynam descriptors
#          firstt 298 finalt 298 tbath 298 tstruct 298 - ! Temp specifiers
# 	 teminc 0 twindh 0 twindl 0 - ! Temperature control variables
# 	 iasors 0 iasvel 1 ichew 0 iscale 0 iscvel 0 - ! Temperature set-up
# 	                                            - ! and checking variables
# 	 echeck -1 - ! energy checking criterion
# 	 iuncrd 0 iunvel 0 iunwri 10 kunit 0 - !unit numbers for writing files
# 	 nsavc 0 nsavv 0 ntrfrq 100 isvfrq 100 - ! frequencies for saving/writing
# 	 iprfrq 200 nprint 100 ihtfrq 0 ieqfrq 0 - ! frequencies for output
# 	                                         - ! and heating/equilibration
# 	 ilbfrq 0 iseed 2393 121923 12239 143569   !random seed
n = psf.get_natom()
fbetas = [0.0] * n
for i, v in enumerate(SelectAtoms(hydrogens=True)):
    if not v:
        fbetas[i] = 5.0

dyn.set_fbetas(fbetas)
if not os.path.isdir('res'): os.system('mkdir res')
if not os.path.isdir('dcd'): os.system('mkdir dcd')
res_file = CharmmFile(file_name='res/adp.res', file_unit=2,
                                   formatted=True,read_only=False)
my_dyn = DynamicsScript(lang=True, start=True, nstep=500, timest=0.002,
                                 firstt=298.0, finalt=298.0, tbath=298.0, tstruc=298.0,
                                 teminc=0.0, twindh=0.0, twindl=0.0,
                                 iunwri=res_file.file_unit,
                                 inbfrq=-1, imgfrq=-1,
                                 iasors=0, iasvel=1, ichecw=0, iscale=0, iscvel=0,
                                 echeck=-1.0, nsavc=10, nsavv=0, ntrfrq=100, isvfrq=100,
                                 iprfrq=200, nprint=100, ihtfrq=0, ieqfrq=0,
                                 ilbfrq=0)
my_dyn.run()
res_file.close()
res_file = CharmmFile(file_name='res/adp.res', file_unit=2,
                                   formatted=True,read_only=False)
dcd_file = CharmmFile(file_name='dcd/dcd.res', file_unit=1,
                                   formatted=False,read_only=False)

DynamicsScript(lang=True, restart=True, nstep=500, timest=0.002,
                        firstt=298.0, finalt=298.0, tbath=298.0, tstruc=298.0,
                        teminc=0.0, twindh=0.0, twindl=0.0,
                        iunwri=res_file.file_unit,iunrea=res_file.file_unit,
                        inbfrq=-1, imgfrq=-1,
                        iuncrd=dcd_file.file_unit,
                        iasors=0, iasvel=1, ichecw=0, iscale=0, iscvel=0,
                        echeck=-1.0, nsavc=10, nsavv=0, ntrfrq=100, isvfrq=100,
                        iprfrq=50, nprint=10, ihtfrq=0, ieqfrq=0,
                        ilbfrq=0).run()

# dopts = dyn.configure(lang=True, start=True, nstep=1000, timest=0.002,
#                       firstt=298.0, finalt=298.0, tbath=298.0, tstruc=298.0,
#                       teminc=0.0, twindh=0.0, twindl=0.0,
#                       iasors=0, iasvel=1, ichecw=0, iscale=0, iscvel=0,
#                       echeck=-1.0,
#                       iuncrd='dcd/adp.dcd', iunwri='res/adp.res',
#                       nsavc=0, nsavv=0, ntrfrq=100, isvfrq=100,
#                       iprfrq=200, nprint=100, ihtfrq=0, ieqfrq=0,
#                       ilbfrq=0, iseed=[2393, 121923, 12239, 143569],
#                       fbeta=fbetas)
# dyn.run(dopts)
# dyn.set_fbetas(fbetas)
# dyn.set_iuncrd('dcd/adp.dcd')
# dyn.set_iunwri('res/adp.res')
# my_dyn = DynamicsScript(lang=True, start=True, nstep=1000, timest=0.002,
#                                  firstt=298.0, finalt=298.0, tbath=298.0, tstruc=298.0,
#                                  teminc=0.0, twindh=0.0, twindl=0.0,
#                                  iasors=0, iasvel=1, ichecw=0, iscale=0, iscvel=0,
#                                  echeck=-1.0, nsavc=10, nsavv=0, ntrfrq=100, isvfrq=100,
#                                  iprfrq=200, nprint=100, ihtfrq=0, ieqfrq=0,
#                                  ilbfrq=0)
# my_dyn.run()

# TODO: iuncrd 0 iunvel 0 iunwri 10 kunit 0 - !unit numbers for writing files
#       iseed 2393 121923 12239 143569   !random seed
#       restart

# close unit 10
# open unit 10 read form name res/adp.res
# open unit 11 write unform name dcd/adp.dcd
# This is the dynamics command that contains all the variables
# that are not set/cannot be set via nbonds
# dynamics langevin restart nstep 1000 timestep 0.002 - ! Basic dynam descript
#          firstt 298 finalt 298 tbath 298 tstruct 298 - ! Temp specifiers
# 	 teminc 0 twindh 0 twindl 0 - ! Temperature control variables
# 	 iasors 0 iasvel 1 ichew 0 iscale 0 iscvel 0 - ! Temperature set-up
# 	                                            - ! and checking variables
# 	 echeck -1 - ! energy checking criterion
# 	 iuncrd 11 iunvel 0 iunwri 10 iunrea 10 kunit 0 - !unit nums for writing
# 	 nsavc 10 nsavv 0 ntrfrq 100 isvfrq 100 - ! frequencies for saving/writing
# 	 iprfrq 200 nprint 100 ihtfrq 0 ieqfrq 0 - ! frequencies for output
# 	                                         - ! and heating/equilibration
# 	 ilbfrq 0
# END

# cons_harm.setup_absolute(force_const=10.0)
# cons_harm.setup_best_fit(force_const=10.0)

# natom = psf.get_natom()

# isel = select.none_selection(natom)
# isel = select.by_elt(range(0, 5), isel)
# isel = SelectAtoms().set_selection(isel)

# jsel = select.none_selection(natom)
# jsel = select.by_elt(range(5, 10), jsel)
# jsel = SelectAtoms().set_selection(jsel)

# cons_harm.setup_relative(isel, jsel, force_const=10.0)

# delete bond sele type h1 end sele type h2 end
# psf.delete_bonds(select.atom_type('H1'), select.atom_type('H2'))

crystal.define_cubic(50.0)
crystal.build(25.0, ['(-X,Y+1/2,-Z)', ])
# crystal.build(25.0, ['(X+1/2,-Y+1/2,-Z)',
#                      '(-X,Y+1/2,-Z+1/2)',
#                      '(-X+1/2,-Y,Z+1/2)', ])

image.setup_segment(0.0, 0.0, 0.0, 'ADP')
image.setup_residue(0.0, 0.0, 0.0, 'ALA')

# print('DEBUG: natom ' + str(psf.get_natom()))
# print('DEBUG: nres ' + str(psf.get_nres()))
# print('DEBUG: nseg ' + str(psf.get_nseg()))
# print('DEBUG: ngrp ' + str(psf.get_ngrp()))

# print('DEBUG: iac')
# for i in psf.get_iac():
#     print('|' + str(i) + '|')

# print('DEBUG: amass')
# for i in psf.get_amass():
#     print('|' + str(i) + '|')

# print('DEBUG: ibase')
# for i in psf.get_ibase():
#     print('|' + str(i) + '|')

# print('DEBUG atype')
# for a in psf.get_atype():
#     print('|' + a + '|')

# print('DEBUG res')
# for a in psf.get_res():
#     print('|' + a + '|')

# print('DEBUG resid')
# for a in psf.get_resid():
#     print('|' + a + '|')

# print('DEBUG segid')
# for a in psf.get_segid():
#     print('|' + a + '|')

# print('DEBUG nictot')
# for a in psf.get_nictot():
#     print('|' + str(a) + '|')

# print('DEBUG igpbs')
# for a in psf.get_igpbs():
#     print('|' + str(a) + '|')

# print('DEBUG igptyp')
# for a in psf.get_igptyp():
#     print('|' + str(a) + '|')

# print('DEBUG flags1 flags2')
# flags1 = select.residues('ALA')
# flags2 = select.residue('ALA')
# for f1, f2 in zip(flags1, flags2):
#     print(str(f1) + ':' + str(f2))

# print('DEBUG flags1 flags2')
# flags1 = select.segments('ADP')
# flags2 = select.segment('ADP')
# for f1, f2 in zip(flags1, flags2):
#     print(str(f1) + ':' + str(f2))

# natc = param.get_natc()
# ctypes = param.get_atc()
# print('DEBUG: chem types ' + str(natc))
# print(ctypes)

# iac = psf.get_iac()
# print('DEBUG: iac')
# for i in psf.get_iac():
#     print('|' + str(ctypes[i]) + '|')

# flags = select.chem_type('HA3')
# print('DEBUG: select by chem type')
# print(flags)

# flags = select.atom('ADP', '1', 'CY')
# print('DEBUG: select atom')
# print(flags)

# my_atoms = SelectAtoms(chem_type='HA3')
# flags = list(my_atoms)
# print('DEBUG: SelectAtoms chem_type test')
# print(flags)

# coor.show()

# flags = select.point(0.0, 0.0, 0.0, 1.0)
# print('DEBUG: select point')
# print(flags)

# flags = select.hydrogen()
# print('DEBUG: select hydrogen')
# print(flags)

# flags = select.lone()
# print('DEBUG: select lone')
# print(flags)

# flags = select.initial()
# print('DEBUG: select initial')
# print(flags)

# print('DEBUG: select get_property charge')
# flags = select.get_property('charge')
# print(flags)

# print('DEBUG: select prop charge < 0.5')
# flags = select.prop('charge', lambda p, t: p < t, 0.5)
# print(flags)

# image.setup_selection(0.0, 0.0, 0.0, flags)

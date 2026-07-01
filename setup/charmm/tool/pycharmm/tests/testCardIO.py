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

# this test is meant to replicate test/c31test/ioformat.inp

rtf_fn = charmm_data_dir + '/toph19.rtf'
chm_read.rtf(rtf_fn)

prm_fn = charmm_data_dir + '/param19.prm'
chm_read.prm(prm_fn)

lingo.charmm_script('''
  read sequence tip3 1
  gene wat nodihe noangle
  coor set xdir 0.0 ydir 0.0 zdir 0.0
  print coor''')

# write coor card name  @9ioform1.crd
write.coor_card('ioform1.crd', title='one water molecule, standard format')
# write.coor_card('ioform1.crd')

lingo.charmm_script('''
  rename segid bigwater sele segid wat end
  q 1 2
  print coor
  q 1 2
  coor trans xdir 1.0
''')

# read coor card name  @9ioform1.crd
chm_read.coor_card('ioform1.crd')
coor.show()

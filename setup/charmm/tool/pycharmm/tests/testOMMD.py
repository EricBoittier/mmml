# pycharmm grid test case

## Import module
import numpy as np
import pandas as pd

from pycharmm import *


## Read in grid box information
xcen = 26.9114167
ycen = 6.126
zcen = 4.179
maxlen = 8

## Build system
chm_settings.set_bomb_level(-1)
chm_read.rtf('toppar/top_all36_prot.rtf')
chm_read.rtf('toppar/top_all36_cgenff.rtf', append = True)
chm_read.rtf('toppar/probes.rtf', append = True)
chm_read.prm('toppar/par_all36m_prot.prm', flex = True)
chm_read.prm('toppar/par_all36_cgenff.prm', append = True, flex = True)
chm_read.prm('toppar/probes.prm', append = True, flex = True)
chm_settings.set_bomb_level(0)
lingo.charmm_script('stream data/benzene.rtf')

## Build protein
chm_read.psf_card('data/t4.psf', append = True)
chm_read.pdb('data/t4.pdb', resid = True)

## Show ommd system can generate grids, soft grid and hard grid
## I am using CPU, GPU testing is in testCDOCKER.py
ommd = grid.OMMD()
settings = {'xCen' : xcen, 'yCen' : ycen, 'zCen' : zcen,
            'xMax' : maxlen, 'yMax' : maxlen, 'zMax' : maxlen,
            'emax' : 3, 'maxe' : 30, 'mine' : -30, 'flag_gpu' : False,
            'flag_grhb' : True, 'gridFile' : 'scratch/soft_grid.bin'}
ommd.setVar(settings)
ommd.generate()
settings = {'emax' : 100, 'maxe' : 100, 'mine' : -100,
            'gridFile' : 'scratch/hard_grid.bin'}
ommd.setVar(settings)
ommd.generate()

## Prepare system
psf.delete_atoms(SelectAtoms().all_atoms())
chm_read.sequence_pdb("data/benzene.pdb")
gen.new_segment(seg_name = "LIGA")
chm_read.pdb("data/benzene.pdb", resid = True)
xyz = coor.get_positions().to_numpy()
energy.show()
nbonds_script = NonBondedScript(
  atom = True, switch = True, vswitch = True,
  cutnb = 12, ctofnb = 10, ctonnb = 8, vdwe = True,
  elec = True, rdie = True, epsilon = 3).run()

## Set up OMMD system
numCopy = 5
ligand = SelectAtoms(seg_id = "LIGA")
settings = {'softGridFile' : 'scratch/soft_grid.bin',
            'hardGridFile' : 'scratch/hard_grid.bin',
            'flex_select' : ligand,
            'flag_grhb' : True, 'numCopy' : numCopy}
ommd.setVar(settings)
status = ommd.create()
print("OMMD set up is", status)

## Set OMMD flexible particles coordinates from main coor
idx = 1
while idx <= numCopy :
	new_xyz = pd.DataFrame(xyz + np.random.rand(len(xyz), 3), columns = ['x', 'y', 'z'])
	coor.show()
	status = ommd.set_coor(idxCopy = idx)
	print("OMMD coor set for idx ", idx, 'is', status)
	idx += 1

## Change OMMD system energy softness
settings = {'soft' : 1, 'hard' : 0, 'emax' : 3, 'mine' : -20, 'maxe' : 40, 'eps' : 3}
ommd.setVar(settings)
status = ommd.change_softness()
print("OMMD change softness is", status)

## Run simulated annealing
settings = {'steps' : 3000, 'heatFrq' : 50, 'startTemp' : 300, 'endTemp' : 700, 'incrTemp' : 1}
ommd.setVar(settings)
status = ommd.simulated_annealing()
print("OMMD simulated annealing is", status)

## Copy OMMD flexible particles coordinates to main cooor
idx = 1
while idx <= numCopy :
	ommd.copy_coor(idxCopy = idx)
	coor.show()
	print("OMMD coor set for idx ", idx, 'is', status)
	idx += 1

## Print OMMD system Energy
status = ommd.energy()
print("OMMD energy print is", status)

exit()

# pycharmm grid test case

## Import module
from pycharmm import *

## Read in grid box information
xcen = 26.9114167
ycen = 6.126
zcen = 4.179
maxlen = 8

## Read in the topology and parameter file
settings.set_bomb_level(-1)
chm_read.rtf('toppar/top_all36_prot.rtf')
chm_read.rtf('toppar/top_all36_cgenff.rtf', append = True)
chm_read.rtf('toppar/probes.rtf', append = True)
chm_read.prm('toppar/par_all36m_prot.prm', flex = True)
chm_read.prm('toppar/par_all36_cgenff.prm', append = True, flex = True)
chm_read.prm('toppar/probes.prm', append = True, flex = True)
settings.set_bomb_level(0)
lingo.charmm_script('stream data/benzene.rtf')

## Build protein
chm_read.psf_card('data/t4.psf', append = True)
chm_read.pdb('data/t4.pdb', resid = True)

## Generate grids
cdocker = chm_grid.CDOCKER()
settings = {'xCen' : xcen, 'yCen' : ycen, 'zCen' : zcen,
            'xMax' : maxlen, 'yMax' : maxlen, 'zMax' : maxlen,
            'emax' : 3, 'maxe' : 30, 'mine' : -30, 'flag_gpu' : True,
            'flag_grhb' : True, 'gridFile' : 'scratch/test_grid.bin',
            'probeFile' : 'toppar/fftdock_c36prot_cgenff_probes.txt'}
cdocker.setVar(settings)
cdocker.generate()

## Prepare system
psf.delete_atoms(SelectAtoms().all_atoms())
chm_read.sequence_pdb("data/benzene.pdb")
gen.new_segment(seg_name = "LIGA")
chm_read.pdb("data/benzene.pdb", resid = True)

## Check CDOCKER grid function with specified selection
print("Now check CDOCKER grid function with specific selection")
ligand = SelectAtoms(seg_id = "LIGA")
settings = {'selection' : ligand}
cdocker.setVar(settings)
status = cdocker.read()
energy.show()
print("Grid read is", status)

status = cdocker.off()
energy.show()
print("Grid off is", status)

status = cdocker.on()
energy.show()
print("Grid on is", status)

status = cdocker.clear()
energy.show()
print("Grid clear is", status)

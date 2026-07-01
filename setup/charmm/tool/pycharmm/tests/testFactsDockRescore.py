# pycharmm rigid cdocker test case

## Import module
from pycharmm import *


################################################################
##
##		Begin of pyCHARMM Rigid CDOCKER
##
################################################################

## Topology and parameter files
chm_settings.set_bomb_level(-1)
chm_read.rtf('toppar/top_all36_prot.rtf')
chm_read.rtf('toppar/top_all36_cgenff.rtf', append = True)
chm_read.prm('toppar/par_all36m_prot.prm', flex = True)
chm_read.prm('toppar/par_all36_cgenff.prm', append = True, flex = True)
chm_settings.set_bomb_level(0)
charmm_script('stream data/benzene.rtf')

## Build system
chm_read.psf_card("data/t4.psf", append = True)
chm_read.pdb("data/t4.pdb", resid = True)
chm_read.sequence_pdb("data/benzene.pdb")
gen.new_segment(seg_name = "LIGA")
chm_read.pdb("data/benzene.pdb", resid = True)

## Prepare for minimization with FACTS
ligand = SelectAtoms().by_seg_id("LIGA")
receptor = ligand.__invert__()

## FACTS rescoring
facts_ener = cdocker.FACTS_rescore(fixAtomSel = receptor)
if abs(facts_ener - (-4218.12223)) <= 0.01 :
    print("Testcase result: PASS")
else:
    print("Testcase result: FAIL")

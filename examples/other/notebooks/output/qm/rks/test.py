from pyscf import gto, dft, scf
from pyscf.tools import cubegen
 
benzene_coordinates = '''
C         -0.65914       -1.21034        3.98683
C          0.73798       -1.21034        4.02059
C         -1.35771       -0.00006        3.96990
C          1.43653       -0.00004        4.03741
C         -0.65915        1.21024        3.98685
C          0.73797        1.21024        4.02061
H         -1.20447       -2.15520        3.97369
H          1.28332       -2.15517        4.03382
H         -2.44839       -0.00006        3.94342
H          2.52722       -0.00004        4.06369
H         -1.20448        2.15509        3.97373
H          1.28330        2.15508        4.03386
'''
 
mol = gto.Mole()
mol.atom = benzene_coordinates
mol.basis = 'def2-SVP'
mol.build()
 
mf = dft.RKS(mol)
# mf = scf.RHF(mol) # For Hartree-Fock
mf.kernel()
 
# 1st MO
cubegen.orbital(mol, 'benzene_mo_1.cub', mf.mo_coeff[:,0])
# 22nd MO (LUMO)
cubegen.orbital(mol, 'benzene_mo_22.cub', mf.mo_coeff[:,21])

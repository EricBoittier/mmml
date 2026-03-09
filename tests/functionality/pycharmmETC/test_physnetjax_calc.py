import mmml
from mmml.interfaces.pycharmmInterface import import_pycharmm

import ase
from ase.io import read

atoms = read("pdb/init-packmol.pdb")

print(atoms)

calc = mmml.PhysNetJaxCalculator(chcseckpoint="mmml/physnetjax/ckpts/test-9af0d71b-4140-4d4b-83e3-ce07c652d048")
atoms.calc = calc

print(atoms.get_potential_energy())
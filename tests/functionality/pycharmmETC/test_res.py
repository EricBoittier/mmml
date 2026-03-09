import mmml
from mmml.interfaces.pycharmmInterface import import_pycharmm
from mmml.interfaces.pycharmmInterface import setupRes
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    reset_block,
    reset_block_no_internal,
    energy,
)
atoms = setupRes.main("TIP3")
atoms = setupRes.generate_coordinates()
_ = setupRes.coor.get_positions()
atoms.set_positions(_)
reset_block()
reset_block_no_internal()
reset_block()
atoms = setupRes.generate_coordinates()
_ = setupRes.coor.get_positions()
atoms.set_positions(_)
reset_block()
reset_block_no_internal()
reset_block()

print(atoms)
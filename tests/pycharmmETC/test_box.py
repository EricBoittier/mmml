import mmml
from mmml.interfaces.pycharmmInterface import import_pycharmm
from mmml.interfaces.pycharmmInterface import setupBox
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    reset_block,
    reset_block_no_internal,
    energy,
)

setupBox.setup_box_generic("pdb/init-packmol.pdb", side_length=10.0, tag="tip3")
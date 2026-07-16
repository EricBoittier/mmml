import pycharmm
from mmml.interfaces.pycharmmInterface.setup import init_charmm
from pycharmm import read, write
init_charmm()
read.sequence_string("ALA")
pycharmm.generate.new_segment("PROT", setup_ic=True)
pycharmm.ic.prm_fill()
pycharmm.ic.build()
pycharmm.energy.show()
pycharmm.lingo.charmm_script("""
open write formatted unit 10 name dummy.res
write coord restart unit 10
close unit 10
""")
print(open("dummy.res").read())

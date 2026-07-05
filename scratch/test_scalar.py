import pycharmm
from pycharmm.lingo import charmm_script
import pycharmm.lib as lib
import ctypes

charmm_script("""
bomlev -5
read rtf card name /Users/ericboittier/mmml/mmml/data/charmm/top_all36_cgenff.rtf
read param card flex name /Users/ericboittier/mmml/mmml/data/charmm/par_all36_cgenff.prm
bomlev 0

read sequence card
* DCM
*
1
DCM

generate DCM setup

coor set xdir 0.0 ydir 0.0 zdir 0.0
coor set xdir 3.0 ydir 0.0 zdir 0.0 select atom DCM 1 * end

update
ener
""")

charmm_script("scalar vdw set 0.0 sele all end")
charmm_script("scalar vdw14 set 0.0 sele all end")
charmm_script("update")
charmm_script("ener")

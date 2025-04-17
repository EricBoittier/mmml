import os
import subprocess
from pathlib import Path
import sys

# current directory
cwd = os.getcwd()
cwd = Path(cwd)

chmh = os.environ["CHARMM_HOME"] #= "/pchem-data/meuwly/boittier/home/charmm"
chml = os.environ["CHARMM_LIB_DIR"] #= "/pchem-data/meuwly/boittier/home/charmm/build/cmake"

if chmh is None:
    raise ValueError("CHARMM_HOME is not set")
if chml is None:
    raise ValueError("CHARMM_LIB_DIR is not set")

print(chmh)
print(chml)
chmhp = Path(chmh) 
pych = chmhp / "tool" / "pycharmm"
sys.path.append(str(pych))

import pycharmm
CGENFF_RTF = cwd / "top_all36_cgenff.rtf"
print(CGENFF_RTF)
CGGENFF_PRM = cwd / "par_all36_cgenff.prm"
print(CGGENFF_PRM)

CGENFF_RTF = str(CGENFF_RTF)
CGGENFF_PRM = str(CGGENFF_PRM)

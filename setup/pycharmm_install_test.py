import os
import subprocess
from pathlib import Path
import sys

chmh = os.environ["CHARMM_HOME"] #= "/pchem-data/meuwly/boittier/home/charmm"
chml = os.environ["CHARMM_LIB_DIR"] #= "/pchem-data/meuwly/boittier/home/charmm/build/cmake"
print(chmh)
print(chml)
chmhp = Path(chmh) 
pych = chmhp / "tool" / "pycharmm"
sys.path.append(str(pych))

import pycharmm



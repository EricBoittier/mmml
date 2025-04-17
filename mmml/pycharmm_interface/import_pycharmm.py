import os
import subprocess
from pathlib import Path
import sys

# current directory of current file
cwd = Path(__file__).parent

chmh = None
chml = None
with open(cwd / ".." / ".." / "CHARMMSETUP") as f:
    lines = f.readlines()
    for line in lines:
        if "CHARMM_HOME" in line:
            chmh = line.split("=")[1].strip()
        if "CHARMM_LIB_DIR" in line:
            chml = line.split("=")[1].strip()
if chmh is None:
    raise ValueError("CHARMM_HOME is not set")
if chml is None:
    raise ValueError("CHARMM_LIB_DIR is not set")

os.environ["CHARMM_HOME"] = chmh
os.environ["CHARMM_LIB_DIR"] = chml

chmhp = Path(chmh) / "tool" / "pycharmm"
sys.path.append(str(chmhp))

import pycharmm
CGENFF_RTF = cwd / "top_all36_cgenff.rtf"
print(CGENFF_RTF)
CGENFF_PRM = cwd / "par_all36_cgenff.prm"
print(CGENFF_PRM)

CGENFF_RTF = str(CGENFF_RTF)
CGENFF_PRM = str(CGENFF_PRM)

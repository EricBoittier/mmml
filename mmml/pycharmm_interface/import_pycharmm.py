import os
import subprocess
from pathlib import Path
import sys

# current directory of current file
cwd = Path(__file__).parent

chmh = None
chml = None
with open(cwd / ".." / ".." / "CHARMMSETUP") as f:
    if "CHARMM_HOME" in f.read():
        chmh = f.read().split("=")[1].strip()
    if "CHARMM_LIB_DIR" in f.read():
        chml = f.read().split("=")[1].strip()
    else:
        raise ValueError("CHARMMSETUP does not contain CHARMM_HOME or CHARMM_LIB_DIR")
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

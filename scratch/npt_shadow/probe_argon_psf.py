"""Feasibility probe v2: build argon PSF via pycharmm C-API module functions."""
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pycharmm
import pycharmm.read as read
import pycharmm.generate as gen
import pycharmm.psf as psf
import pycharmm.settings as settings

settings.set_bomb_level(-2); settings.set_warn_level(-2)
N = 64
open("ar.rtf","w").write("""* argon
   36  1
MASS  -1  AR   39.94800
RESI AR  0.00
ATOM AR AR 0.00
END
""")
open("ar.prm","w").write("""* argon params
ATOMS
MASS -1 AR 39.948
NONBONDED
AR 0.0 0.0 1.9075
END
""")
try:
    read.rtf("ar.rtf"); print("OK read.rtf")
    read.prm("ar.prm", flex=True); print("OK read.prm")
    read.sequence_string(" ".join(["AR"]*N)); print("OK sequence")
    gen.new_segment(seg_name="AR", setup_ic=False, warn=False); print("OK generate")
    print("natom =", psf.get_natom())
except Exception as e:
    import traceback; traceback.print_exc()
    print("BUILD FAILED:", type(e).__name__, str(e)[:80])

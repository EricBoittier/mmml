import sys
import os

os.environ['CHARMM_LIB_DIR'] = '/Users/ericboittier/mmml/setup/charmm'

import pycharmm
import pycharmm.dynamics as charm_dyn

kw = {"start": False, "restart": True, "iasvel": 1, "iunrea": 88}
dyn = pycharmm.DynamicsScript(**kw)
script = dyn.create_script_string()
print(f"SCRIPT: {script!r}")
command_line = charm_dyn.flatten_dynamics_script(script)
print(f"COMMAND LINE: {command_line!r}")

# Basics
import os

# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ASE
from ase import io

# PyCHARMM
os.environ["CHARMM_HOME"] = "/pchem-data/meuwly/boittier/home/charmm/"
os.environ["CHARMM_LIB_DIR"] = "/pchem-data/meuwly/boittier/home/charmm/build/cmake"

# from pycharmm_calculator import PyCharmm_Calculator

import jax

devices = jax.local_devices()
print(devices)
print(jax.default_backend())
print(jax.devices())

from physnetjax.calc.helper_mlp import *

# from helper_mlp import Model

# os.sleep(1)
with open("i_", "w") as f:
    print("...")

import ase.units as units
import pandas as pd

# PyCHARMM
import pycharmm
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.lingo as stream
import pycharmm.minimize as minimize
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.write as write

# ASE
from ase import io

params = pd.read_pickle("checkpoints/test2-q.pkl")
NATOMS = 34
model = EF(
    # attributes
    features=64,
    max_degree=2,
    num_iterations=2,
    num_basis_functions=32,
    cutoff=5.0,
    max_atomic_number=32,
    charges=True,
    natoms=NATOMS,
    total_charge=1,
)

with open("t_", "w") as f:
    print("")
    f.close()

# Step 0: Load parameter files
# -----------------------------------------------------------

# Read in the topology (rtf) and parameter file (prm) for proteins
# equivalent to the CHARMM scripting command: read rtf card name toppar/top_all36_prot.rtf
read.rtf("/pchem-data/meuwly/boittier/home/charmm/toppar/top_all36_prot.rtf")
# equivalent to the CHARMM scripting command: read param card flexible name toppar/par_all36m_prot.prm
read.prm(
    "/pchem-data/meuwly/boittier/home/charmm/toppar/par_all36m_prot.prm", flex=True
)
pycharmm.lingo.charmm_script(
    "stream /pchem-data/meuwly/boittier/home/charmm/toppar/toppar_water_ions.str"
)

settings.set_bomb_level(-2)
settings.set_warn_level(-1)

stream.charmm_script("set name aaa")
name = "aaa"

# Step 1: Read System
# -----------------------------------------------------------

read.sequence_string("ALA ALA ALA")

stream.charmm_script("GENERATE PEPT FIRST NTER LAST CNEU SETUP")
stream.charmm_script("ic param")
stream.charmm_script("ic seed 1 N 1 CA 1 C")
stream.charmm_script("ic build")


# read.psf_card('ala.psf')
# read.sequence_pdb('ala.pdb')

stream.charmm_script("print coor")
# Save system pdb file
write.psf_card("aaa.psf")
write.coor_pdb("aaa.pdb")
write.coor_card("aaa.crd")
# stream.charmm_script('skip cmap')

minimize.run_sd(**{"nstep": 2000, "tolenr": 1e-5, "tolgrd": 1e-5})


##########################
R = coor.get_positions()
ase_mol = ase.io.read("aaa.pdb")
Z = ase_mol.get_atomic_numbers()
Z = [_ if _ < 9 else 6 for _ in Z]
stream.charmm_script(f"echo {Z}")

atoms = ase.Atoms(Z, R)

from physnetjax.calc.helper_mlp import get_ase_calc

calculator = get_ase_calc(params, model, atoms)
atoms.set_calculator(calculator)
atoms1 = atoms.copy()

ml_selection = pycharmm.SelectAtoms(seg_id="PEPT")
print(ml_selection)

energy.show()
U = atoms.get_potential_energy() / (units.kcal / units.mol)
print(U)
stream.charmm_script(f"echo {U}")

charge = 1

Model = get_pyc(params, model, atoms)

print(dir(Model))

# Initialize PhysNet calculator
_ = pycharmm.MLpot(
    Model,
    Z,
    ml_selection,
    ml_fq=False,
)

with open("i_", "w") as f:
    print("...")
print(_)


energy.show()

ase_pept = io.read("aaa.pdb", format="proteindatabank")

energy.show()

# sys.exit()

# Step 3: MINI - CHARMM, CGenFF
# -----------------------------------------------------------

# Do minimization or read result from last run
if True:

    # stream.charmm_script('cons fix sele segid PEPT end')

    # Optimization with CGenFF
    minimize.run_sd(**{"nstep": 2000, "tolenr": 1e-5, "tolgrd": 1e-5})
    # minimize.run_abnr(**{
    #     'nstep': 2000,
    #     'tolenr': 1e-5,
    #     'tolgrd': 1e-5})
    # minimize.run_sd(**{
    #     'nstep': 2000,
    #     'tolenr': 1e-5,
    #     'tolgrd': 1e-5})
    # minimize.run_abnr(**{
    #     'nstep': 2000,
    #     'tolenr': 1e-5,
    #     'tolgrd': 1e-5})
    # stream.charmm_script('cons fix sele none end')

    # Write pdb file
    write.coor_pdb("mini.aaa.pdb", title="CGenFF Tripeptide in water - Minimized")
    write.coor_card("mini.aaa.cor", title="CGenFF Tripeptide in water - Minimized")

else:

    # Read minimized coordinates and check energy
    read.coor_card("mm.mini.cgenff.pept.aaa.cor")
    energy.show()

# Custom energy
energy.show()
# stream.charmm_script('cons fix sele segid PEPT end')
# stream.charmm_script('cons fix sele index 34 end')
# energy.show()
write.psf_card("mm.pept_nobonds.aaa.psf")
# sys.exit()

# -----------------------------------------------------------
# stream.charmm_script('cons hmcm force 1.0 refx 0.0 refy 0.0 refz 0.0 sele all end')
# stream.charmm_script('shake bonh  toler 1.0e-8 para')

if True:

    timestep = 0.001  # 0.5 fs
    tottime = 5.0  # 10 ps
    savetime = 0.10  # 100 fs
    temp = 100

    res_file = pycharmm.CharmmFile(
        file_name="mm.heat.aaa.res", file_unit=2, formatted=True, read_only=False
    )
    dcd_file = pycharmm.CharmmFile(
        file_name="mm.heat.aaa.dcd", file_unit=1, formatted=False, read_only=False
    )

    # Run some dynamics
    dynamics_dict = {
        "leap": False,
        "verlet": True,
        "cpt": False,
        "new": False,
        "langevin": False,
        "timestep": timestep,
        "start": True,
        "nstep": 30000,
        "nsavc": 10,
        "nsavv": 0,
        "inbfrq": 10,
        "ihbfrq": 0,
        "ilbfrq": 50,
        "imgfrq": 0,
        "ixtfrq": 0,
        "iunrea": -1,
        "iunwri": res_file.file_unit,
        "iuncrd": dcd_file.file_unit,
        "nsavl": 0,  # frequency for saving lambda values in lamda-dynamics
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "nprint": 100,  # Frequency to write to output
        "iprfrq": 1000,  # Frequency to calculate averages
        "isvfrq": 1000,  # Frequency to save restart file
        "ntrfrq": 0,
        "ihtfrq": 500,
        # 'ISEED' : '888277364 7132478 45762345 2343452',    # 200
        "ieqfrq": 100,
        "firstt": 10,
        "finalt": 300,
        "TEMINC": 10,
        "TWINDH": 10,
        "TWINDL": -10,
        "iasors": 1,
        "iasvel": 1,
        "ichecw": 0,
        "echeck": -1,
        # scaling factor for velocity scaling
    }

    dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_heat.run()
    write.coor_pdb("heat.aaa.pdb")
    write.coor_card("heat.aaa.cor")
    write.psf_card("heat.aaa.psf")
    res_file.close()
    dcd_file.close()

# Step 5: Equilibration - CHARMM, PhysNet
# -----------------------------------------------------------

if True:

    timestep = 0.001  # 0.2 fs
    tottime = 5.0  # 50 ps
    savetime = 0.01  # 10 fs

    str_file = pycharmm.CharmmFile(
        file_name="mm.heat.aaa.res", file_unit=3, formatted=True, read_only=False
    )
    res_file = pycharmm.CharmmFile(
        file_name="mm.equi.aaa.res", file_unit=2, formatted=True, read_only=False
    )
    dcd_file = pycharmm.CharmmFile(
        file_name="mm.equi.aaa.dcd", file_unit=1, formatted=False, read_only=False
    )

    # Run some dynamics
    dynamics_dict = {
        "leap": False,
        "verlet": True,
        "cpt": False,
        "new": False,
        "langevin": False,
        "timestep": timestep,
        "start": False,
        "restart": True,
        "nstep": 100000,
        "nsavc": 100,
        "nsavv": 0,
        "inbfrq": 10,
        "ihbfrq": 0,
        "ilbfrq": 0,
        "imgfrq": 10,
        "iunrea": str_file.file_unit,
        "iunwri": res_file.file_unit,
        "iuncrd": dcd_file.file_unit,
        "nsavl": 0,  # frequency for saving lambda values in lamda-dynamics
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "nprint": 10,  # Frequency to write to output
        "iprfrq": 100,  # Frequency to calculate averages
        "ieqfrq": 100,
        "firstt": 300,
        "finalt": 300,
        "TEMINC": 10,
        "TWINDH": 10,
        "TWINDL": -10,
        "iasors": 1,
        "iasvel": 1,
        "ichecw": 0,
    }

    dyn_equi = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_equi.run()

    str_file.close()
    res_file.close()
    dcd_file.close()
    write.coor_pdb("equi.aaa.pdb")
    write.coor_card("equi.aaa.cor")
    write.psf_card("equi.aaa.psf")

# Step 6: Production - CHARMM, PhysNet
# -----------------------------------------------------------

if True:

    timestep = 0.001  # 0.2 fs

    str_file = pycharmm.CharmmFile(
        file_name="mm.equi.aaa.res", file_unit=3, formatted=True, read_only=False
    )
    res_file = pycharmm.CharmmFile(
        file_name="mm.dyna.aaa.res", file_unit=2, formatted=True, read_only=False
    )
    dcd_file = pycharmm.CharmmFile(
        file_name="mm.dyna.aaa.dcd", file_unit=1, formatted=False, read_only=False
    )

    dynamics_dict = {
        "leap": False,
        "verlet": True,
        "cpt": False,
        "new": False,
        "langevin": False,
        "timestep": timestep,
        "start": False,
        "restart": True,
        "nstep": 1000000,
        "nsavc": 500,
        "nsavv": 0,
        "inbfrq": 10,
        "ihbfrq": 0,
        "ilbfrq": 0,
        "imgfrq": 10,
        "iunrea": str_file.file_unit,
        "iunwri": res_file.file_unit,
        "iuncrd": dcd_file.file_unit,
        "nsavl": 0,  # frequency for saving lambda values in lamda-dynamics
        "iunldm": -1,
        "ilap": -1,
        "ilaf": -1,
        "nprint": 10,  # Frequency to write to output
        "iprfrq": 100,  # Frequency to calculate averages
        # Frequency to save restart file
        "ieqfrq": 0,
        "firstt": 300,
        "finalt": 300,
        "TEMINC": 10,
        "TWINDH": 10,
        "TWINDL": -10,
        "iasors": 1,
        "iasvel": 1,
        "ichecw": 0,
    }

    dyn_prod = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_prod.run()
    str_file.close()
    res_file.close()
    dcd_file.close()
    write.coor_pdb("dyna.aaa.pdb")
    write.coor_card("dyna.aaa.cor")
    write.psf_card("dyna.aaa.psf")

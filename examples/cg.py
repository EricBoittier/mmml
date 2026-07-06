import mmml
import ase
import numpy as np

from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    CharmmNbondSettings,
    mm_system_energy_and_forces,
)

from pathlib import Path
from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
ensure_pycharmm_loaded()
from mmml.interfaces.pycharmmInterface.trialanine_water_box import build_trialanine_water_box_in_charmm
box = build_trialanine_water_box_in_charmm(n_waters=10, box_side_A=28.0, seed=11, workdir=Path('/tmp/tria_box'))
print(len(box.positions), box.psf_path)# ase.Atoms()
positions = box.positions
from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
prm = CGENFF_PRM

from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    charmm_bonded_forces_kcalmol_A,
    charmm_nonbonded_energy_components_kcalmol,
    run_charmm_nonbonded_ener_force,
    set_charmm_positions,
    setup_nonbonded_only_charmm,
)
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    load_nonbonded_system_from_charmm,
    nonbonded_energy_and_forces,
)


pos = np.asarray(positions, dtype=np.float64)
pos = np.random.uniform(-3.01, 3.01, pos.shape) + pos
set_charmm_positions(pos)

print("pos", pos)

setup_nonbonded_only_charmm()
print("setup_nonbonded_only_charmm")
# perform charmm minimization
from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_loud
run_charmm_script_loud("""
    MINI SD 10000
""")
print("run_charmm_script_loud")
from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
print("pos", pos)
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
z = get_Z_from_psf()
print("z", z)
atoms = ase.Atoms(z, pos)
print("atoms", atoms)

# save atoms to pdb file
atoms.write("atoms.pdb")

run_charmm_nonbonded_ener_force(silent=False)
charmm_terms = charmm_nonbonded_energy_components_kcalmol()
charmm_forces = charmm_bonded_forces_kcalmol_A()

# print(charmm_terms)
psf_path = box.psf_path
prm_path = CGENFF_PRM
cell = box.cell
from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import _nbond_settings_from_cutoffs
nb_settings = _nbond_settings_from_cutoffs(box.nbond_cutoffs)
nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)
jax_terms_raw, jax_forces = nonbonded_energy_and_forces(
    pos,
    nbond_data,
    cell,
    nb_settings,
)
jax_terms = {k: float(v) for k, v in jax_terms_raw.items()}
jax_forces_np = np.asarray(jax_forces, dtype=np.float64)

print(jax_terms)
# print(jax_forces_np)
print(charmm_terms)
# print(charmm_forces)



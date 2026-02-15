"""Mixed AB dimer simulation.

Creates a box with two different residue types (e.g. MEOH + ACET) using
packmol, then runs the hybrid ML/MM simulation via the general calculator
which supports heterogeneous monomer sizes and lambda scaling.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
config = {
    # Molecule A
    "RES_A": "MEOH",   # methanol
    "N_A": 10,          # number of A molecules
    # Molecule B
    "RES_B": "ACET",   # acetate (or any other CGenFF residue)
    "N_B": 10,          # number of B molecules
    # Box
    "L": 23.0,          # cubic box side length (Å)
    "skip_energy_show": False,
}

nb_dir = Path.cwd()
try:
    nb_dir = Path(get_ipython().ev("os.getcwd()"))  # noqa: F821
except Exception:
    pass

print(f"RES_A = {config['RES_A']} (N={config['N_A']}), "
      f"RES_B = {config['RES_B']} (N={config['N_B']}), "
      f"L = {config['L']}")

# ---------------------------------------------------------------------------
# 2. Generate residues (one PDB per type)
# ---------------------------------------------------------------------------
from mmml.cli import make_res  # noqa: E402

# --- Residue A ---
args_res_a = argparse.Namespace(
    res=config["RES_A"],
    skip_energy_show=config.get("skip_energy_show", False),
)
res_a = make_res.main_loop(args_res_a)
n_atoms_a = len(res_a)
pdb_a = Path("pdb") / f"{config['RES_A'].lower()}.pdb"
# make_res writes pdb/initial.pdb and copies to pdb/<res>.pdb
# Rename so both residues have distinct PDB files
shutil.copy("pdb/initial.pdb", pdb_a)
print(f"Residue A ({config['RES_A']}): {n_atoms_a} atoms -> {pdb_a}")

# --- Residue B ---
# PyCHARMM state must be reset between residue generations.
from mmml.pycharmmInterface.import_pycharmm import (  # noqa: E402
    reset_block,
    reset_block_no_internal,
    pycharmm,
    safe_energy_show,
)
from mmml.pycharmmInterface.pycharmmCommands import CLEAR_CHARMM  # noqa: E402

CLEAR_CHARMM()

args_res_b = argparse.Namespace(
    res=config["RES_B"],
    skip_energy_show=config.get("skip_energy_show", False),
)
res_b = make_res.main_loop(args_res_b)
n_atoms_b = len(res_b)
pdb_b = Path("pdb") / f"{config['RES_B'].lower()}.pdb"
shutil.copy("pdb/initial.pdb", pdb_b)
print(f"Residue B ({config['RES_B']}): {n_atoms_b} atoms -> {pdb_b}")

# ---------------------------------------------------------------------------
# 3. Pack mixed box with packmol
# ---------------------------------------------------------------------------
N_A = config["N_A"]
N_B = config["N_B"]
L = config["L"]
N_total = N_A + N_B

os.makedirs("pdb", exist_ok=True)
os.makedirs("packmol", exist_ok=True)

packmol_input = f"""
seed {np.random.randint(1_000_000)}
output pdb/init-packmol.pdb
filetype pdb
tolerance 2.0

structure {pdb_a}
  chain A
  resnumbers 2
  number {N_A}
  inside box 0.0 0.0 0.0 {L} {L} {L}
end structure

structure {pdb_b}
  chain A
  resnumbers 2
  number {N_B}
  inside box 0.0 0.0 0.0 {L} {L} {L}
end structure
"""

packmol_inp_path = Path("packmol") / "packmol_AB.inp"
packmol_inp_path.write_text(packmol_input)

PACKMOL_PATH = os.path.expanduser("~/mmml/mmml/packmol/packmol")
cmd = f"{PACKMOL_PATH} < {packmol_inp_path}"
print(f"Running: {cmd}")
ret = os.system(cmd)
if ret != 0:
    raise RuntimeError(f"packmol failed with exit code {ret}")
print("packmol done -> pdb/init-packmol.pdb")

# ---------------------------------------------------------------------------
# 4. Setup PyCHARMM box (reads PDB, generates PSF, etc.)
# ---------------------------------------------------------------------------
CLEAR_CHARMM()
from mmml.pycharmmInterface import setupBox  # noqa: E402

setupBox.setup_box_generic(
    "pdb/init-packmol.pdb",
    side_length=L,
    tag=f"{config['RES_A'].lower()}_{config['RES_B'].lower()}",
)

reset_block()
reset_block_no_internal()
reset_block()
nbonds_script = """
nbonds atom cutnb 14.0 ctofnb 12.0 ctonnb 10.0 -
vswitch NBXMOD 3 -
inbfrq -1 imgfrq -1
"""
pycharmm.lingo.charmm_script(nbonds_script)
safe_energy_show()
setupBox.minimize_box()
print("Box setup done.")

# ---------------------------------------------------------------------------
# 5. Build the heterogeneous atoms_per_monomer list
# ---------------------------------------------------------------------------
# Packmol places A molecules first, then B molecules.
# atoms_per_monomer = [n_atoms_a]*N_A + [n_atoms_b]*N_B
atoms_per_monomer = [n_atoms_a] * N_A + [n_atoms_b] * N_B
total_atoms = sum(atoms_per_monomer)
print(f"atoms_per_monomer = {atoms_per_monomer}")
print(f"total_atoms = {total_atoms}, N_total = {N_total}")

# ---------------------------------------------------------------------------
# 6. Configure and run simulation
# ---------------------------------------------------------------------------
sim_config = {
    "pdbfile": nb_dir / "pdb" / "init-packmol.pdb",
    "checkpoint": "/home/ericb/mmml/mmml/physnetjax/ckpts/progressive-stage2-9a3b53e8-80c5-4069-a1a9-e8a89899b016/",
    "n_monomers": N_total,
    # Pass the heterogeneous list -- run_sim.run() will forward this to setup_calculator
    "atoms_per_monomer": atoms_per_monomer,
    "cell": L,
    "temperature": 298.0,
    "timestep": 0.5,
    "nsteps_jaxmd": 10_000,
    "nsteps_ase": 100,
    "ensemble": "nvt",
    "output_prefix": f"AB_{config['RES_A']}_{config['RES_B']}",
    "energy_catch": 0.5,
    "ml_cutoff": 0.01,
    "mm_switch_on": 6.0,
    "mm_cutoff": 2.0,
    "heating_interval": 500,
    "write_interval": 50,
    "include_mm": True,
    "skip_ml_dimers": False,
    "validate": False,
    "debug": False,
}
print(f"pdbfile = {sim_config['pdbfile']}")
print(f"checkpoint = {sim_config['checkpoint']}")

from mmml.cli.run_sim import run  # noqa: E402

args = argparse.Namespace(
    pdbfile=sim_config["pdbfile"],
    checkpoint=sim_config["checkpoint"],
    validate=sim_config["validate"],
    energy_catch=sim_config["energy_catch"],
    cell=sim_config["cell"],
    n_monomers=sim_config["n_monomers"],
    # The key change: pass the list instead of a single int
    n_atoms_monomer=sim_config["atoms_per_monomer"],
    ml_cutoff=sim_config["ml_cutoff"],
    mm_switch_on=sim_config["mm_switch_on"],
    mm_cutoff=sim_config["mm_cutoff"],
    include_mm=sim_config["include_mm"],
    skip_ml_dimers=sim_config["skip_ml_dimers"],
    debug=sim_config["debug"],
    temperature=sim_config["temperature"],
    timestep=sim_config["timestep"],
    nsteps_jaxmd=sim_config["nsteps_jaxmd"],
    output_prefix=sim_config["output_prefix"],
    nsteps_ase=sim_config["nsteps_ase"],
    ensemble=sim_config["ensemble"],
    heating_interval=sim_config["heating_interval"],
    write_interval=sim_config["write_interval"],
)
run(args)
print("run_sim done.")

# Config (matches settings.source / 01_make.sh)
import argparse
from pathlib import Path

config = {
    "RES": "MEOH",   # residue name
    "N": 20,        # number of molecules in box
    "L": 23.0,      # box side length (Å)
    "skip_energy_show": False,  # set True on clusters/SLURM to avoid CHARMM segfault
}

# Optional: run from this dir (default = notebook’s dir)
nb_dir = Path("__file__").resolve().parent if "__file__" in dir() else Path.cwd()
try:
    nb_dir = Path(get_ipython().ev("os.getcwd()"))  # noqa
except Exception:
    pass
print("RES =", config["RES"], "N =", config["N"], "L =", config["L"])

# 1) make_res.py --res $RES [--skip-energy-show on cluster]
import sys
from mmml.cli import make_res

args_res = argparse.Namespace(
    res=config["RES"],
    skip_energy_show=config.get("skip_energy_show", False),
)
res = make_res.main_loop(args_res)
print(res)
print("make_res done.")

n_atoms_monomer = len(res)
print(f"{n_atoms_monomer} atoms per monomer ")
# 2) make_box.py --res $RES --n $N --side_length $L
from mmml.cli import make_box

args_box = argparse.Namespace(
    res=config["RES"],
    n=config["N"],
    side_length=config["L"],
    pdb=None,
    solvent=None,
    density=None,
)
atoms = make_box.main_loop(args_box)
print("make_box done.")

import argparse
from pathlib import Path

nb_dir = Path.cwd()
try:
    nb_dir = Path(get_ipython().ev("os.getcwd()"))  # noqa: F821
except Exception:
    pass

config = {
    "pdbfile": nb_dir / "pdb" / "init-packmol.pdb",
    "checkpoint": "/home/ericb/mmml/mmml/physnetjax/ckpts/DESdimers/", #nb_dir / "ACO-b4f39bb9-8ca7-485e-bf51-2e5236e51b56",
    "n_monomers": config["N"],
    "n_atoms_monomer": n_atoms_monomer,
    "cell": config["L"],  # cubic box side length (Å), or None
    "temperature": 298.0,
    "timestep": 0.5,
    "nsteps_jaxmd": 10_000,
    "nsteps_ase": 100,
    "ensemble": "nvt",
    "output_prefix": "test_run",
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
print("pdbfile =", config["pdbfile"], "| checkpoint =", config["checkpoint"])

# run_sim.py with args from config (same as: python -m mmml.cli.run_sim --pdbfile ... --checkpoint ...)
from mmml.cli.run_sim import run

args = argparse.Namespace(
    pdbfile=config["pdbfile"],
    checkpoint=config["checkpoint"],
    validate=config["validate"],
    energy_catch=config["energy_catch"],
    cell=config["cell"],
    n_monomers=config["n_monomers"],
    n_atoms_monomer=config["n_atoms_monomer"],
    ml_cutoff=config["ml_cutoff"],
    mm_switch_on=config["mm_switch_on"],
    mm_cutoff=config["mm_cutoff"],
    include_mm=config["include_mm"],
    skip_ml_dimers=config["skip_ml_dimers"],
    debug=config["debug"],
    temperature=config["temperature"],
    timestep=config["timestep"],
    nsteps_jaxmd=config["nsteps_jaxmd"],
    output_prefix=config["output_prefix"],
    nsteps_ase=config["nsteps_ase"],
    ensemble=config["ensemble"],
    heating_interval=config["heating_interval"],
    write_interval=config["write_interval"],
)
run(args)
print("run_sim done.")


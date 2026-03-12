# Config: 1 DCM molecule solvated with 10 water molecules
import argparse
from pathlib import Path

config = {
    "RES": "DCM",       # solute residue name
    "N_SOLVENT": 50,    # number of water molecules around the solute
    "L": 25.0,          # box side length (Å)
    "skip_energy_show": False,  # set True on clusters/SLURM to avoid CHARMM segfault
}

# Optional: run from this dir (default = notebook’s dir)
nb_dir = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
try:
    nb_dir = Path(get_ipython().ev("os.getcwd()"))  # noqa
except Exception:
    pass
print("RES =", config["RES"], "N_SOLVENT =", config["N_SOLVENT"], "L =", config["L"])

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
# 2) make_box.py: 1 DCM + N_SOLVENT water molecules
from mmml.cli import make_box

args_box = argparse.Namespace(
    res=config["RES"],
    n=config["N_SOLVENT"],
    side_length=config["L"],
    pdb=None,
    solvent="water",
    density=None,
)
atoms = make_box.main_loop(args_box)
print("make_box done.")


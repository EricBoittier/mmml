"""Mixed AB dimer simulation.

Creates a box with two different residue types (e.g. MEOH + ACET) using
``make_mixed_box``, then runs the hybrid ML/MM simulation via the general
calculator which supports heterogeneous monomer sizes and lambda scaling.
"""
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
config = {
    # Molecule types and counts
    "residues": ["ACO", "DCM"],
    "counts": [2, 2],
    # Box
    "L": 23.0,          # cubic box side length (Å)
    "skip_energy_show": False,
}

nb_dir = Path.cwd()
try:
    nb_dir = Path(get_ipython().ev("os.getcwd()"))  # noqa: F821
except Exception:
    pass

print(f"residues = {config['residues']}, counts = {config['counts']}, L = {config['L']}")
from mmml.cli.base import resolve_desdimers_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Build the mixed box (residue generation + packmol + CHARMM setup)
# ---------------------------------------------------------------------------
from mmml.cli.make.make_mixed_box import main_loop as make_mixed_box  # noqa: E402

box_args = argparse.Namespace(
    residues=config["residues"],
    counts=config["counts"],
    side_length=config["L"],
    skip_energy_show=config.get("skip_energy_show", False),
    tolerance=2.0,
)
box_result = make_mixed_box(box_args)

atoms_per_monomer = box_result["atoms_per_monomer"]
N_total = box_result["n_monomers"]
print(f"atoms_per_monomer = {atoms_per_monomer}")
print(f"N_total = {N_total}, total_atoms = {sum(atoms_per_monomer)}")

# ---------------------------------------------------------------------------
# 3. Configure and run simulation
# ---------------------------------------------------------------------------
sim_config = {
    "pdbfile": nb_dir / box_result["pdb_path"],
    "checkpoint": str(
        resolve_desdimers_checkpoint(__file__ if "__file__" in globals() else None)
    ),
    "n_monomers": N_total,
    "atoms_per_monomer": atoms_per_monomer,
    "cell": config["L"],
    "temperature": 298.0,
    "timestep": 0.5,
    "nsteps_jaxmd": 1000,
    "nsteps_ase": 100,
    "ensemble": "nvt",
    "output_prefix": f"AB_{'_'.join(config['residues'])}",
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

from mmml.cli.run.run_sim import run  # noqa: E402

args = argparse.Namespace(
    pdbfile=sim_config["pdbfile"],
    checkpoint=sim_config["checkpoint"],
    validate=sim_config["validate"],
    energy_catch=sim_config["energy_catch"],
    cell=sim_config["cell"],
    n_monomers=sim_config["n_monomers"],
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

#!/usr/bin/env python
# coding: utf-8

"""Example script demonstrating a hybrid ML/MM simulation workflow.

This script shows how to:
1. Build and minimize a system (Trialanine peptide in a water box) using PyCHARMM.
2. Initialize custom hybrid ASE calculators:
   - MonomerSumCalculator: computes intramolecular forces for each monomer independently.
   - JAXIntermolecularCalculator: computes intermolecular interactions using JAX.
3. Run structure optimization and molecular dynamics using ASE and PyCHARMM.
"""

import os
from pathlib import Path
import ase
import jax
import numpy as np
from ase.calculators.mixing import SumCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

import mmml
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    ensure_pycharmm_loaded,
    CGENFF_PRM,
    pycharmm_loud,
    coor,
)
from mmml.interfaces.pycharmmInterface.trialanine_water_box import build_trialanine_water_box_in_charmm
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    set_charmm_positions,
    setup_nonbonded_only_charmm,
)
from mmml.interfaces.pycharmmInterface.mm_system_energy import load_nonbonded_system_from_charmm
from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import _nbond_settings_from_cutoffs
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_loud
from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
from mmml.interfaces.calculators.hybrid import (
    MonomerSumCalculator,
    JAXIntermolecularCalculator,
)
from cg_common import (
    DualTrajectoryWriter,
    load_cg_checkpoint,
    load_cg_config,
    probe_charge_output,
)

# 1. Initialize JAX and PyCHARMM configuration
jax.config.update("jax_enable_x64", True)

# Runtime settings. Every key can be overridden through --config; common
# settings also have direct CLI flags.
_settings = load_cg_config(
    {
        "checkpoint": str(Path(__file__).parent / "params_aaa_long_2026-07-04_22-30-27.json"),
        "n_waters": 200,
        "box_size": 28.0,
        "seed": 11,
        "temperature": 300.0,
        "dt_fs": 0.5,
        "position_perturbation": 4.1,
        "fire_fmax": 0.5,
        "md_blocks": 5,
        "md_steps_per_block": 100,
        "workdir": "/tmp/tria_box",
        "output_dir": ".",
        "write_dcd": True,
    },
    description="Trialanine/water ASE hybrid example",
)
CKPT_PATH = str(_settings.checkpoint)
OUTPUT_DIR = Path(_settings.output_dir).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ensure_pycharmm_loaded()
pycharmm_loud()

# 2. Build the initial system in PyCHARMM
# Creates a water box containing the trialanine peptide and 200 water molecules.
print("--- Building Trialanine Water Box in CHARMM ---")
workdir = Path(_settings.workdir).expanduser()
box = build_trialanine_water_box_in_charmm(
    n_waters=int(_settings.n_waters),
    box_side_A=float(_settings.box_size),
    seed=int(_settings.seed),
    workdir=workdir,
)
print(f"System size: {len(box.positions)} atoms. PSF file path: {box.psf_path}")

# Perturb positions slightly to test minimization
pos = np.asarray(box.positions, dtype=np.float64)
perturbation = float(_settings.position_perturbation)
pos = np.random.uniform(-perturbation, perturbation, pos.shape) + pos
set_charmm_positions(pos)

# 3. Minimize the system coordinates using PyCHARMM
print("--- Minimizing System in CHARMM ---")
setup_nonbonded_only_charmm()

import pycharmm.lingo as lingo
# Apply constraint and run steepest descent minimization
lingo.charmm_script("CONStraint DROPlet FORC 0.01 EXPO 4")
run_charmm_script_loud("MINI SD 10000")
lingo.charmm_script("CONStraint DROPlet")
run_charmm_script_loud("MINI SD 10000")

# Retrieve the minimized positions and atomic numbers
pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
z = get_Z_from_psf()

# 4. Construct ASE Atoms object and save to PDB
atoms = ase.Atoms(z, pos)
atoms.set_cell(box.cell)
atoms.set_pbc(True)
atoms.write(OUTPUT_DIR / "atoms.pdb")
print(f"Initial Atoms structure saved to atoms.pdb: {atoms}")

# 5. Define monomer molecule grouping indices
# First 42 atoms correspond to the trialanine peptide monomer.
# Every subsequent 3 atoms correspond to a TIP3 water molecule.
n_trialanine = 42
z_array = np.asarray(atoms.numbers)

assert (len(z_array) - n_trialanine) % 3 == 0
assert np.all(z_array[n_trialanine:].reshape(-1, 3) == np.array([8, 1, 1]))

# Build a list of atom index arrays for each monomer (peptide + waters)
monomer_indices = [np.arange(n_trialanine)]
monomer_indices.extend(
    np.arange(i, i + 3)
    for i in range(n_trialanine, len(atoms), 3)
)

# Set up molecule IDs for intermolecular calculator masking (if needed)
molecule_id = np.empty(len(atoms), dtype=np.int32)
molecule_id[:n_trialanine] = 0
for mol_id, start in enumerate(range(n_trialanine, len(atoms), 3), start=1):
    molecule_id[start:start + 3] = mol_id

# 6. Initialize Hybrid ASE Calculators
print("--- Setting up Hybrid ML/MM Calculators ---")

# Load JAX nonbonded parameters and setup cutoffs
psf_path = box.psf_path
prm_path = CGENFF_PRM
nb_settings = _nbond_settings_from_cutoffs(box.nbond_cutoffs)
nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)

# (A) Intramolecular ML Potential Calculator
# Computes isolated gas-phase energy/forces for each monomer using the pretrained PhysNet.
checkpoint_calc, checkpoint_model, checkpoint_params = load_cg_checkpoint(CKPT_PATH)
probe_charge_output(
    checkpoint_model,
    checkpoint_params,
    z_array[:n_trialanine],
    np.asarray(atoms.positions[:n_trialanine]),
    charge=0.0,
    spin=1.0,
    label="peptide checkpoint",
)
probe_charge_output(
    checkpoint_model,
    checkpoint_params,
    z_array[n_trialanine:n_trialanine + 3],
    np.asarray(atoms.positions[n_trialanine:n_trialanine + 3]),
    charge=0.0,
    spin=1.0,
    label="water checkpoint",
)
physnet_monomers = MonomerSumCalculator(
    monomer_indices=monomer_indices,
    calculator_factory=lambda: create_calculator_from_checkpoint(CKPT_PATH),
)

# (B) Intermolecular Nonbonded Calculator
# Computes pairwise nonbonded interactions (Coulomb & LJ) between all monomers.
jax_inter = JAXIntermolecularCalculator(
    nbond_data=nbond_data,
    nb_settings=nb_settings,
    molecule_id=molecule_id,
)

# Combine intramolecular and intermolecular terms
atoms.calc = SumCalculator([
    physnet_monomers,
    jax_inter,
])

# Evaluate initial hybrid energy and forces
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
print(f"Initial Hybrid Potential Energy: {energy:.4f} eV")
print(f"Max Force Component: {np.abs(forces).max():.4f} eV/Å")

# 7. Perform structure minimization using ASE (FIRE)
print("--- Minimizing using ASE FIRE Optimizer ---")
opt = FIRE(atoms)
opt.run(fmax=float(_settings.fire_fmax))
print(f"Minimization complete. Minimized Energy: {atoms.get_potential_energy():.4f} eV")

# 8. Run Molecular Dynamics using ASE
print("--- Running ASE NVE/NVT VelocityVerlet Dynamics ---")
# Set momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=float(_settings.temperature))

def print_energy(atoms_obj: ase.Atoms) -> None:
    epot = atoms_obj.get_potential_energy() / len(atoms_obj)
    ekin = atoms_obj.get_kinetic_energy() / len(atoms_obj)
    temp = ekin / (1.5 * units.kB)
    print(f'Energy per atom: Epot = {epot:.3f} eV, Ekin = {ekin:.3f} eV (T = {temp:3.0f} K)')

# Initialize NVE Verlet dynamics with 0.5 fs time step
dyn = VelocityVerlet(
    atoms,
    float(_settings.dt_fs) * units.fs,
)
md_trajectory = DualTrajectoryWriter(
    OUTPUT_DIR / "md.traj",
    atoms,
    write_dcd=bool(_settings.write_dcd),
    dt_ps=float(_settings.dt_fs) * 0.001,
    steps_per_frame=1,
)
dyn.attach(md_trajectory.write, interval=1, atoms=atoms)

print_energy(atoms)
for i in range(int(_settings.md_blocks)):
    dyn.run(int(_settings.md_steps_per_block))
    print_energy(atoms)
    # Re-equilibrate temperature
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(_settings.temperature))
md_trajectory.close()

# 9. Perform dynamics in PyCHARMM (Optional / Alternative workflow)
print("--- Running CHARMM MD Script Workflow ---")
# Setup environment for CHARMM library
os.environ['CHARMM_LIB_DIR'] = '/Users/ericboittier/mmml/setup/charmm'

import pycharmm
import pycharmm.dynamics as charm_dyn

# Setup dynamics script
dyn_script = pycharmm.DynamicsScript(
    lang=True,
    restart=False,
    nstep=500,
    timest=0.005,
    firstt=298.0,
    finalt=298.0,
    tbath=298.0,
    tstruc=298.0,
    teminc=0.0,
    twindh=0.0,
    twindl=0.0,
    iasors=0,
    iasvel=1,
    ichecw=0,
    iscale=0,
    iscvel=0,
    echeck=-1.0,
    nsavc=10,
    nsavv=0,
    ntrfrq=1000,
    isvfrq=1000,
    iprfrq=50,
    nprint=10,
    ihtfrq=0,
    ieqfrq=1,
    ilbfrq=0
)
dyn_script.run()

# Retrieve final positions and save
pos = coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=float)
final_atoms = ase.Atoms(z, pos)
final_atoms.write(OUTPUT_DIR / "atoms_final.pdb")
print("Workflow complete! Saved final coordinates to atoms_final.pdb.")

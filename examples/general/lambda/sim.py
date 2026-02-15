"""Lambda dynamics / alchemical free energy example.

Demonstrates how to scale the potential of individual monomers using the
``lambda_monomer`` array in the general calculator.  This is useful for:

  - Free Energy Perturbation (FEP)
  - Thermodynamic Integration (TI)
  - Alchemical insertion / deletion of molecules

The workflow:
  1. Build a single-residue box (e.g. 20 MEOH molecules).
  2. For each lambda window, set ``lambda_monomer`` so that the *first*
     monomer has its interactions scaled by λ (from 1 → 0), while all
     other monomers remain fully coupled (λ = 1).
  3. Run a short equilibration + production at each window.
  4. Collect ⟨dU/dλ⟩ at each window for TI, or ΔU for FEP.

Usage::

    cd examples/general/lambda
    PYTHONPATH=/path/to/mmml:$PYTHONPATH python sim.py
"""

import argparse
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
config = {
    "RES": "MEOH",
    "N": 20,
    "L": 23.0,
    "skip_energy_show": False,
    # Lambda schedule: scale monomer 0 from fully coupled (1) to decoupled (0)
    "lambda_windows": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    # Which monomer to decouple (0-indexed)
    "decouple_monomer": 0,
    # Per-window simulation length
    "nsteps_equil": 500,    # equilibration steps per window
    "nsteps_prod": 1000,    # production steps per window
}

nb_dir = Path.cwd()
try:
    nb_dir = Path(get_ipython().ev("os.getcwd()"))  # noqa: F821
except Exception:
    pass

print(f"RES = {config['RES']}, N = {config['N']}, L = {config['L']}")
print(f"Lambda windows: {config['lambda_windows']}")

# ---------------------------------------------------------------------------
# 2. Build box (same as dimers example)
# ---------------------------------------------------------------------------
from mmml.cli import make_res, make_box  # noqa: E402

args_res = argparse.Namespace(
    res=config["RES"],
    skip_energy_show=config.get("skip_energy_show", False),
)
res = make_res.main_loop(args_res)
n_atoms_monomer = len(res)
print(f"{n_atoms_monomer} atoms per monomer")

args_box = argparse.Namespace(
    res=config["RES"],
    n=config["N"],
    side_length=config["L"],
    pdb=None,
    solvent=None,
    density=None,
)
make_box.main_loop(args_box)
print("Box setup done.")

# ---------------------------------------------------------------------------
# 3. Set up the simulation (single pass — reuse calculator across windows)
# ---------------------------------------------------------------------------
from mmml.cli.run_sim import run  # noqa: E402

# We import the general calculator's setup directly so we can access
# the calculator object and call set_lambda_monomer between windows.
from mmml.pycharmmInterface.mmml_calculator_general import (  # noqa: E402
    setup_calculator as setup_calculator_general,
)
from mmml.cli.base import (  # noqa: E402
    load_model_parameters,
    resolve_checkpoint_paths,
)

import ase.io  # noqa: E402
import jax.numpy as jnp  # noqa: E402

checkpoint = "/home/ericb/mmml/mmml/physnetjax/ckpts/progressive-stage2-9a3b53e8-80c5-4069-a1a9-e8a89899b016/"
pdbfile = nb_dir / "pdb" / "init-packmol.pdb"

base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(checkpoint)

# Read PDB
import pycharmm.psf as psf  # noqa: E402
import ase  # noqa: E402

pdb_ase_atoms = ase.io.read(str(pdbfile))

# Fix atomic numbers from PSF masses (same logic as run_sim)
psf_masses = psf.get_amass()
pdb_ase_atoms.set_masses(psf_masses)
psf_masses_arr = np.array(psf_masses)[:, np.newaxis]
correct_Z = np.argmin(
    np.abs(ase.data.atomic_masses_common[np.newaxis, :] - psf_masses_arr),
    axis=1,
)
pdb_ase_atoms.set_atomic_numbers(correct_Z)

N = config["N"]
L = config["L"]

from ase.cell import Cell  # noqa: E402

cell_obj = Cell.fromcellpar([L, L, L, 90.0, 90.0, 90.0])
pdb_ase_atoms.set_cell(cell_obj)
pdb_ase_atoms.set_pbc(True)

Z = pdb_ase_atoms.get_atomic_numbers()
R = pdb_ase_atoms.get_positions()
natoms = len(pdb_ase_atoms)

params, model = load_model_parameters(epoch_dir, natoms)
model.natoms = natoms

# Build the general calculator with lambda_monomer = all ones initially
atoms_per_monomer_list = [n_atoms_monomer] * N
initial_lambda = np.ones(N, dtype=np.float32)

calculator_factory = setup_calculator_general(
    ATOMS_PER_MONOMER=atoms_per_monomer_list,
    N_MONOMERS=N,
    ml_cutoff_distance=0.01,
    mm_switch_on=6.0,
    mm_cutoff=2.0,
    doML=True,
    doMM=True,
    doML_dimer=False,
    debug=False,
    model_restart_path=base_ckpt_dir,
    MAX_ATOMS_PER_SYSTEM=natoms,
    ml_energy_conversion_factor=1,
    ml_force_conversion_factor=1,
    cell=L,
    lambda_monomer=initial_lambda,
)

hybrid_calc, _ = calculator_factory(
    atomic_numbers=Z,
    atomic_positions=R,
    n_monomers=N,
    energy_conversion_factor=1.0,
    force_conversion_factor=1.0,
    do_pbc_map=True,
    pbc_map=getattr(calculator_factory, "pbc_map", None),
)

pdb_ase_atoms.calc = hybrid_calc

# ---------------------------------------------------------------------------
# 4. Lambda window loop
# ---------------------------------------------------------------------------
decouple_idx = config["decouple_monomer"]
lambda_windows = config["lambda_windows"]
n_equil = config["nsteps_equil"]
n_prod = config["nsteps_prod"]

results = []

print(f"\n{'='*60}")
print(f"Starting lambda dynamics: decoupling monomer {decouple_idx}")
print(f"{'='*60}\n")

for wi, lam in enumerate(lambda_windows):
    # Build lambda array: all ones, except the decoupled monomer
    lam_arr = np.ones(N, dtype=np.float32)
    lam_arr[decouple_idx] = lam
    hybrid_calc.set_lambda_monomer(lam_arr)

    print(f"\n--- Window {wi}/{len(lambda_windows)-1}: "
          f"lambda[{decouple_idx}] = {lam:.2f} ---")
    print(f"lambda_monomer = {hybrid_calc.lambda_monomer_values}")

    # Equilibration
    from ase.md.langevin import Langevin  # noqa: E402
    from ase import units  # noqa: E402

    dyn = Langevin(
        pdb_ase_atoms,
        timestep=0.5 * units.fs,
        temperature_K=298.0,
        friction=0.01 / units.fs,
    )

    print(f"  Equilibrating {n_equil} steps ...")
    dyn.run(n_equil)

    # Production: collect energies for TI / FEP
    energies_at_lambda = []

    def _collect_energy(atoms=pdb_ase_atoms):
        energies_at_lambda.append(float(atoms.get_potential_energy()))

    dyn_prod = Langevin(
        pdb_ase_atoms,
        timestep=0.5 * units.fs,
        temperature_K=298.0,
        friction=0.01 / units.fs,
    )
    dyn_prod.attach(_collect_energy, interval=10)

    print(f"  Production {n_prod} steps ...")
    dyn_prod.run(n_prod)

    mean_E = np.mean(energies_at_lambda)
    std_E = np.std(energies_at_lambda)
    results.append({
        "window": wi,
        "lambda": lam,
        "mean_E": mean_E,
        "std_E": std_E,
        "n_samples": len(energies_at_lambda),
    })
    print(f"  <E>(λ={lam:.2f}) = {mean_E:.4f} ± {std_E:.4f} eV "
          f"({len(energies_at_lambda)} samples)")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Lambda dynamics summary")
print(f"{'='*60}")
print(f"{'Window':>6s}  {'λ':>5s}  {'<E> (eV)':>12s}  {'σ (eV)':>10s}  {'N':>4s}")
for r in results:
    print(f"{r['window']:6d}  {r['lambda']:5.2f}  {r['mean_E']:12.4f}  {r['std_E']:10.4f}  {r['n_samples']:4d}")

# For TI: ∫₀¹ ⟨dU/dλ⟩ dλ  (numerical differentiation of <E> vs λ)
lambdas = np.array([r["lambda"] for r in results])
mean_Es = np.array([r["mean_E"] for r in results])

# Simple trapezoidal estimate of dE/dlambda
if len(lambdas) > 1:
    dEdl = np.gradient(mean_Es, lambdas)
    # TI: integrate <dU/dλ> from λ=0 to λ=1
    # Note: our windows go 1→0, so reverse for integration from 0→1
    sort_idx = np.argsort(lambdas)
    delta_F_TI = np.trapz(dEdl[sort_idx], lambdas[sort_idx])
    print(f"\nTI estimate ΔF = {delta_F_TI:.4f} eV "
          f"({delta_F_TI * 23.0609:.2f} kcal/mol)")

print("\nDone.")

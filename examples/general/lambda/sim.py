"""Lambda dynamics / alchemical free energy example.

Demonstrates how to scale the potential of individual monomers using the
``lambda_monomer`` array in the general calculator.  This is useful for:

  - Free Energy Perturbation (FEP)
  - Thermodynamic Integration (TI)
  - Alchemical insertion / deletion of molecules

The workflow:
  1. Build a single-residue box (e.g. 20 MEOH molecules).
  2. For each lambda window, set ``lambda_monomer`` so that the *first*
     monomer has its *inter-monomer* interactions scaled by λ (from 1 → 0),
     while all other monomers remain fully coupled (λ = 1).  Internal
     monomer energy is never decoupled.
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
    "RES": "DCM",
    "N": 100,
    "L": 20.0,
    "skip_energy_show": False,
    # Lambda schedule: scale monomer 0's inter-monomer interactions from coupled (1) to decoupled (0)
    "lambda_windows": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    # Which monomer to decouple (0-indexed)
    "decouple_monomer": 0,
    # Per-window simulation length
    "nsteps_min": 50,       # minimization steps before each equil
    "nsteps_equil": 1000,    # equilibration steps per window
    "nsteps_prod": 10000,    # production steps per window
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

from mmml.pycharmmInterface.import_pycharmm import (  # noqa: E402
    pycharmm,
    coor,
    safe_energy_show,
)

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
    setup_mmml_imports,
)

import ase.io  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402

_, _, _, get_ase_calc = setup_mmml_imports()

checkpoint = "/pchem-data/meuwly/boittier/home/mmml/mmml/physnetjax/ckpts/DESdimers/"
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

# Single-monomer PhysNet calculator for optimize_as_monomers
params_monomer, model_monomer = load_model_parameters(epoch_dir, n_atoms_monomer)
ase_monomer = pdb_ase_atoms[0:n_atoms_monomer].copy()
simple_physnet_calculator = get_ase_calc(params_monomer, model_monomer, ase_monomer)

# Monomer offsets for optimize_as_monomers
monomer_offsets = np.zeros(N + 1, dtype=int)
for _mi, _na in enumerate(atoms_per_monomer_list):
    monomer_offsets[_mi + 1] = monomer_offsets[_mi] + _na

calculator_factory = setup_calculator_general(
    ATOMS_PER_MONOMER=atoms_per_monomer_list,
    N_MONOMERS=N,
    ml_cutoff_distance=0.01,
    mm_switch_on=6.0,
    mm_cutoff=3.0,
    doML=True,
    doMM=True,
    doML_dimer=True,
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
n_min = config["nsteps_min"]
n_equil = config["nsteps_equil"]
n_prod = config["nsteps_prod"]

results = []

# Directory for per-window trajectories
traj_dir = Path("trajectories")
traj_dir.mkdir(exist_ok=True)

from ase.io.trajectory import Trajectory  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase import units  # noqa: E402
import ase.optimize as ase_opt  # noqa: E402


def wrap_positions_for_pbc(positions):
    """Apply PBC mapping to wrap positions into the cell (molecular wrapping)."""
    pbc_map_fn = getattr(hybrid_calc, "pbc_map", None)
    if pbc_map_fn is None or not getattr(hybrid_calc, "do_pbc_map", False):
        return positions
    R_mapped = pbc_map_fn(jnp.asarray(positions))
    return np.asarray(jax.device_get(R_mapped))


def optimize_as_monomers(atoms, run_index=0, nsteps=60, fmax=0.0006):
    """Optimize each monomer in isolation with PhysNet, then wrap into cell."""
    optimized_atoms_positions = np.zeros_like(atoms.get_positions())
    for i in range(N):
        off = int(monomer_offsets[i])
        n_i = atoms_per_monomer_list[i]
        monomer_atoms = atoms[off : off + n_i].copy()
        monomer_atoms.calc = simple_physnet_calculator
        _ = ase_opt.BFGS(monomer_atoms).run(fmax=fmax, steps=nsteps)
        optimized_atoms_positions[off : off + n_i] = monomer_atoms.get_positions()

    atoms.set_positions(optimized_atoms_positions)
    wrapped = wrap_positions_for_pbc(atoms.get_positions())
    atoms.set_positions(wrapped)
    xyz = pd.DataFrame(wrapped, columns=["x", "y", "z"])
    coor.set_positions(xyz)
    return atoms


def minimize_structure(atoms, run_index=0, nsteps=60, fmax=0.0006, charmm=False, output_prefix="lambda"):
    """Minimize structure: CHARMM ABNR + monomer optimization (if charmm=True), then BFGS with hybrid."""
    if charmm:
        pycharmm.minimize.run_abnr(nstep=10000, tolenr=1e-6, tolgrd=1e-6)
        pycharmm.lingo.charmm_script("ENER")
        safe_energy_show()
        atoms.set_positions(coor.get_positions())
        atoms = optimize_as_monomers(atoms, run_index=run_index, nsteps=100, fmax=0.0006)

    traj_path = traj_dir / f"bfgs_{run_index}_{output_prefix}_minimized.traj"
    traj = ase.io.Trajectory(str(traj_path), "w")
    print("Minimizing structure with hybrid calculator")
    print(f"Running BFGS for {nsteps} steps")
    print(f"Running BFGS with fmax: {fmax}")
    _ = ase_opt.BFGS(atoms, trajectory=traj).run(fmax=fmax, steps=nsteps)
    traj.close()
    # Sync with PyCHARMM
    xyz = pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"])
    coor.set_positions(xyz)
    return atoms



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

    # Trajectory files for this window
    traj_equil_path = traj_dir / f"window_{wi:02d}_lam{lam:.2f}_equil.traj"
    traj_prod_path = traj_dir / f"window_{wi:02d}_lam{lam:.2f}_prod.traj"

    if wi == 0:
        # Minimization before equil (CHARMM ABNR + monomer opt + BFGS)
        print(f"  Minimizing (charmm=True) {n_min} steps ...")
        minimize_structure(
            pdb_ase_atoms,
            run_index=wi,
            nsteps=n_min,
            fmax=0.01,
            charmm=True,
            output_prefix="lambda",
        )
    # Wrap positions into cell after BFGS (avoids unwrapped coords for PBC)
    pdb_ase_atoms.set_positions(
        pdb_ase_atoms.get_positions() - pdb_ase_atoms.get_positions().mean(axis=0)
    )
    wrapped = wrap_positions_for_pbc(pdb_ase_atoms.get_positions())
    pdb_ase_atoms.set_positions(wrapped)
    coor.set_positions(pd.DataFrame(wrapped, columns=["x", "y", "z"]))

    # Equilibration
    traj_equil = Trajectory(str(traj_equil_path), "w", pdb_ase_atoms)

    dyn = Langevin(
        pdb_ase_atoms,
        timestep=0.1 * units.fs,
        temperature_K=250.0,
        friction=0.01 / units.fs,
    )
    dyn.attach(traj_equil.write, interval=50)

    print(f"  Equilibrating {n_equil} steps ...")
    dyn.run(n_equil)
    traj_equil.close()
    print(f"  -> {traj_equil_path}")

    # Production: collect energies for TI / FEP and save trajectory
    energies_at_lambda = []

    def _collect_energy(atoms=pdb_ase_atoms):
        energies_at_lambda.append(float(atoms.get_potential_energy()))

    traj_prod = Trajectory(str(traj_prod_path), "w", pdb_ase_atoms)

    dyn_prod = Langevin(
        pdb_ase_atoms,
        timestep=0.1 * units.fs,
        temperature_K=250.0,
        friction=0.01 / units.fs,
    )
    dyn_prod.attach(_collect_energy, interval=10)
    dyn_prod.attach(traj_prod.write, interval=10)

    print(f"  Production {n_prod} steps ...")
    dyn_prod.run(n_prod)
    traj_prod.close()
    print(f"  -> {traj_prod_path}")

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
    delta_F_TI = np.trapezoid(dEdl[sort_idx], lambdas[sort_idx])
    print(f"\nTI estimate ΔF = {delta_F_TI:.4f} eV "
          f"({delta_F_TI * 23.0609:.2f} kcal/mol)")

print("\nDone.")

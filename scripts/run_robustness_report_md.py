#!/usr/bin/env python3
"""Run a short, REAL NVE MD trajectory with the actual charge-predicting
PhysNet checkpoint (`charged_electrostatic_best_forces`, natoms<=34,
charges=True, total_charge=0) on a small water cluster, and save
positions/forces/energies/per-atom charges/dipole for every frame.

This exists because the original large-scale NVE sweep
(workflows/mixed_calculator_sweep) had its raw trajectory.npz files pruned
from disk (only summary figures survive), and that sweep's checkpoint
(sppoky-epoch-0010) doesn't predict charges at all (model.charges=False) --
so neither "fluctuating charges/multipoles" nor a fresh from-scratch
conservation check could be built from it. This script produces a small but
completely real substitute: same model family, real trained charge head,
real Velocity Verlet integration, no synthetic data anywhere.

Output: artifacts/robustness_report/water_cluster_nve/trajectory.npz with
keys: positions (n_frames, n_atoms, 3) [A], Z (n_atoms,), time_fs
(n_frames,), energy_eV (n_frames,), kinetic_eV (n_frames,), forces_eV_A
(n_frames, n_atoms, 3), charges_e (n_frames, n_atoms), dipole_eA
(n_frames, 3).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from mmml.interfaces.calculators.checkpoint_loading import create_calculator_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT = REPO_ROOT / "mmml/models/physnetjax/defaults/hf_json/test-b4064dca-8cbd-471c-9871-08887107a1d8_epoch-550_portable.json"
OUT_DIR = REPO_ROOT / "artifacts" / "robustness_report" / "water_cluster_nve"

N_WATERS = 4
N_STEPS = 2000
DT_FS = 0.1
TEMPERATURE_K = 300.0
SEED = 0

# A second, deliberately-too-large timestep: unconstrained O-H bond stretch
# vibrates on a ~10 fs period, so a dt=0.5 fs Verlet step under-resolves it
# and leaks energy -- this is a well-known MD integration limit (fixed in
# production by bond constraints/SHAKE or a smaller dt), not a force/energy
# consistency bug. Run a short version at this dt too, for a direct
# side-by-side "why dt matters" panel in the report -- a robustness report
# should show the tool correctly *revealing* a bad integration setting, not
# just clean successes.
UNSTABLE_DT_FS = 0.5
UNSTABLE_N_STEPS = 400


def _water_cluster(n_waters: int, seed: int) -> Atoms:
    """A small tetrahedral-ish cluster of n_waters TIP3P-geometry water
    molecules, spaced ~2.9 A O-O (roughly hydrogen-bond distance) with
    random orientation per molecule -- NOT a relaxed/equilibrated
    structure, so the run script BFGS-relaxes it first."""
    rng = np.random.default_rng(seed)
    # Tetrahedral-ish O positions (subset of a cube's vertices), scaled to ~2.9 A.
    base_o = np.array([
        [0.0, 0.0, 0.0],
        [2.9, 2.9, 0.0],
        [2.9, 0.0, 2.9],
        [0.0, 2.9, 2.9],
        [2.9, 2.9, 2.9],
        [1.45, 1.45, 1.45],
    ])[:n_waters]

    numbers, positions = [], []
    for o_pos in base_o:
        # Random orientation for this molecule's H-O-H frame.
        axis1 = rng.normal(size=3)
        axis1 /= np.linalg.norm(axis1)
        axis2 = rng.normal(size=3)
        axis2 -= np.dot(axis2, axis1) * axis1
        axis2 /= np.linalg.norm(axis2)
        half_angle = np.deg2rad(104.5 / 2)
        h1 = o_pos + 0.9572 * (np.cos(half_angle) * axis1 + np.sin(half_angle) * axis2)
        h2 = o_pos + 0.9572 * (np.cos(half_angle) * axis1 - np.sin(half_angle) * axis2)
        numbers += [8, 1, 1]
        positions += [o_pos, h1, h2]
    return Atoms(numbers=numbers, positions=np.array(positions))


def _run_nve(atoms: Atoms, n_steps: int, dt_fs: float, label: str) -> dict[str, np.ndarray]:
    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    n_atoms = len(atoms)
    positions = np.zeros((n_steps, n_atoms, 3))
    forces = np.zeros((n_steps, n_atoms, 3))
    energy_ev = np.zeros(n_steps)
    kinetic_ev = np.zeros(n_steps)
    charges_e = np.zeros((n_steps, n_atoms))
    dipole_ea = np.zeros((n_steps, 3))
    time_fs = np.arange(n_steps) * dt_fs

    t0 = time.time()
    for i in range(n_steps):
        dyn.run(1)
        positions[i] = atoms.get_positions()
        forces[i] = atoms.get_forces()
        energy_ev[i] = atoms.get_potential_energy()
        kinetic_ev[i] = atoms.get_kinetic_energy()
        out = atoms.info["output"]
        charges_e[i] = np.asarray(out["charges"]).reshape(-1)
        dipole_ea[i] = np.asarray(out["dipoles"]).reshape(-1)[:3]
        if i % max(1, n_steps // 8) == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] step {i}/{n_steps}  E_tot={energy_ev[i] + kinetic_ev[i]:.5f} eV  "
                  f"({elapsed:.1f}s elapsed)")

    return dict(
        positions=positions, Z=atoms.get_atomic_numbers(), time_fs=time_fs,
        energy_eV=energy_ev, kinetic_eV=kinetic_ev, forces_eV_A=forces,
        charges_e=charges_e, dipole_eA=dipole_ea,
        checkpoint=str(CKPT.relative_to(REPO_ROOT)), dt_fs=dt_fs, temperature_K=TEMPERATURE_K,
    )


def _intramolecular_oh_bond_lengths(atoms: Atoms) -> np.ndarray:
    """O-H distances for each water molecule (atoms ordered O,H,H per
    molecule, per `_water_cluster`) -- used as a sanity check after
    relaxation, since this compact demo checkpoint's PES has spurious deep
    wells at unphysically short range (see the bond-scan caveat in the
    report) that BFGS can fall into for one molecule in a multi-molecule
    cluster without the total energy looking obviously wrong."""
    positions = atoms.get_positions()
    lengths = []
    for i in range(0, len(atoms), 3):
        o = positions[i]
        lengths.append(np.linalg.norm(positions[i + 1] - o))
        lengths.append(np.linalg.norm(positions[i + 2] - o))
    return np.asarray(lengths)


def _relax_to_physical_geometry(calc, n_waters: int, base_seed: int, max_attempts: int = 8) -> Atoms:
    """Build + BFGS-relax a water cluster, retrying with a new random
    initial orientation if relaxation collapses any O-H bond onto a
    spurious short-range artifact minimum (real failure mode observed once
    already -- see module docstring history) rather than silently shipping
    an unphysical starting geometry into the production MD run."""
    for attempt in range(max_attempts):
        atoms = _water_cluster(n_waters, base_seed + attempt)
        atoms.calc = calc
        BFGS(atoms, logfile=None).run(fmax=0.3, steps=60)
        oh = _intramolecular_oh_bond_lengths(atoms)
        if np.all((oh > 0.85) & (oh < 1.15)):
            print(f"  post-relax energy: {atoms.get_potential_energy():.4f} eV "
                  f"(seed {base_seed + attempt}, O-H range {oh.min():.3f}-{oh.max():.3f} A)")
            return atoms
        print(f"  seed {base_seed + attempt}: relaxation collapsed an O-H bond "
              f"(range {oh.min():.3f}-{oh.max():.3f} A) -- retrying with a new seed")
    raise RuntimeError(f"Could not reach a physical geometry in {max_attempts} attempts")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calc = create_calculator_from_checkpoint(str(CKPT))

    print(f"Relaxing a {3 * N_WATERS}-atom ({N_WATERS} waters) cluster before MD...")
    atoms = _relax_to_physical_geometry(calc, N_WATERS, SEED)
    relaxed_positions = atoms.get_positions().copy()

    # Stable production run: dt=0.1 fs correctly resolves O-H bond vibration.
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(SEED))
    Stationary(atoms)
    stable = _run_nve(atoms, N_STEPS, DT_FS, "stable dt=0.1fs")
    np.savez(OUT_DIR / "trajectory.npz", **stable)
    print(f"wrote {OUT_DIR / 'trajectory.npz'}")

    # Illustrative unstable run from the SAME relaxed geometry/velocity seed:
    # dt=0.5 fs under-resolves the same bond vibration and leaks energy --
    # included deliberately as a "the tool correctly reveals a bad
    # integration setting" contrast, not a bug in the model/forces.
    atoms.set_positions(relaxed_positions)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(SEED))
    Stationary(atoms)
    unstable = _run_nve(atoms, UNSTABLE_N_STEPS, UNSTABLE_DT_FS, "unstable dt=0.5fs")
    np.savez(OUT_DIR / "trajectory_unstable_dt.npz", **unstable)
    print(f"wrote {OUT_DIR / 'trajectory_unstable_dt.npz'}")


if __name__ == "__main__":
    main()

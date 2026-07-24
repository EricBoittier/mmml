#!/usr/bin/env python3
"""Vacuum NVE (ASE VelocityVerlet) for NH3–CH3Cl with the kl.json PhysNet ckpt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from _geometry import DEFAULT_NPZ, load_dimer_frame  # noqa: E402
from _io import attach_ase_trajectory, write_final_geometry, write_xyz_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "examples/m/kl.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--traj-interval",
        type=int,
        default=1,
        help="Save every N MD steps to .traj / .xyz (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/nh3_ch3cl/free_nve_ase",
    )
    args = parser.parse_args()

    from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    z, r = load_dimer_frame(args.data, index=args.frame, seed=0)
    ckpt = resolve_checkpoint(args.checkpoint)
    n_atoms = len(z)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    factory = setup_calculator(
        ATOMS_PER_MONOMER=n_atoms,
        N_MONOMERS=1,
        doML=True,
        doMM=False,
        model_restart_path=str(ckpt),
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        defer_xla_gpu_warmup=True,
        verbose=False,
    )
    calc, _sph, _ = unpack_factory_result(
        factory(atomic_numbers=z, atomic_positions=r, n_monomers=1)
    )
    atoms = Atoms(numbers=z, positions=r)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(args.temperature))
    Stationary(atoms)

    energies: list[float] = []
    xyz_frames: list[np.ndarray] = [np.asarray(atoms.get_positions(), dtype=np.float64)]
    e0 = float(atoms.get_potential_energy())
    energies.append(e0)
    dyn = VelocityVerlet(atoms, timestep=float(args.dt_fs) * units.fs)
    traj = attach_ase_trajectory(
        dyn, atoms, out / "md.traj", interval=int(args.traj_interval)
    )

    def _log(_atoms=atoms) -> None:
        energies.append(float(_atoms.get_potential_energy()))
        xyz_frames.append(np.asarray(_atoms.get_positions(), dtype=np.float64))

    dyn.attach(_log, interval=max(1, int(args.traj_interval)))
    dyn.run(int(args.n_steps))
    traj.close()
    e1 = float(atoms.get_potential_energy())
    energies.append(e1)
    xyz_frames.append(np.asarray(atoms.get_positions(), dtype=np.float64))

    geom = write_final_geometry(out, z, atoms.get_positions())
    write_xyz_frames(out / "md.xyz", z, xyz_frames)

    summary = {
        "backend": "ase",
        "ensemble": "nve",
        "checkpoint": str(ckpt),
        "n_atoms": n_atoms,
        "n_steps": int(args.n_steps),
        "dt_fs": float(args.dt_fs),
        "E0_kcal_mol": e0,
        "E1_kcal_mol": e1,
        "dE_kcal_mol": e1 - e0,
        "E_trace_kcal_mol": energies,
        "artifacts": {
            "traj": "md.traj",
            "xyz": "md.xyz",
            **geom,
        },
    }
    (out / "md_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"ASE NVE  E0={e0:.6f}  E1={e1:.6f}  dE={e1 - e0:.6f} kcal/mol  "
        f"steps={args.n_steps}"
    )
    print(f"Wrote {out / 'md.traj'}, {out / 'md.xyz'}, {out / 'final.xyz'}")
    if not np.isfinite(e1):
        print("FAIL: non-finite energy after NVE", file=sys.stderr)
        return 1
    print("PASS: ASE NVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

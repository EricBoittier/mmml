#!/usr/bin/env python3
"""
Run ASE molecular dynamics with a spooky PhysNetJAX checkpoint (Orbax ``final_params``).

Loads the same checkpoint layout as :mod:`train_spooky_h5` (``params`` + ``config``).
Structures with fewer atoms than ``model.natoms`` are **zero-padded** (extra sites
with Z=0), matching HDF5 training.

Prerequisites: jax, ase, e3x, orbax, numpy

Example::

  python examples/other/md_spooky_ase.py \\
    --checkpoint /path/to/ckpts_spooky_h5/final_params \\
    --structure start.xyz \\
    --temperature 300 --timestep 0.5 --nsteps 500 \\
    -o md_out/

  # Open-shell / charged system (broadcast to all atoms, same as training)
  python examples/other/md_spooky_ase.py \\
    --checkpoint ckpts_spooky_h5/final_params --structure mol.xyz \\
    --charge 0 --multiplicity 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
from mmml.models.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
from mmml.models.physnetjax.physnetjax.training.spooky_training import restart_params_only


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ASE MD with spooky PhysNetJAX Orbax checkpoint."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Orbax directory (e.g. .../final_params). Use 'latest' with --checkpoint-root.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default="ckpts_spooky_h5",
        help='If --checkpoint is "latest", load <checkpoint-root>/final_params.',
    )
    p.add_argument(
        "--structure",
        type=str,
        required=True,
        help="Initial geometry (xyz, extxyz, pdb, ...).",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="md_spooky_ase_out",
        help="Output directory for trajectory and final frame.",
    )
    p.add_argument("--temperature", type=float, default=300.0, help="K")
    p.add_argument("--timestep", type=float, default=0.5, help="fs")
    p.add_argument("--nsteps", type=int, default=200)
    p.add_argument("--printfreq", type=int, default=10, help="Trajectory write interval.")
    p.add_argument(
        "--friction",
        type=float,
        default=0.02,
        help="Langevin friction (1/fs), default 0.02.",
    )
    p.add_argument(
        "--charge",
        type=float,
        default=0.0,
        help="Total system charge (spooky conditioning, broadcast per atom).",
    )
    p.add_argument(
        "--multiplicity",
        type=float,
        default=1.0,
        help="Spin multiplicity (spooky conditioning).",
    )
    p.add_argument(
        "--no-center",
        action="store_true",
        help="Do not call atoms.center(vacuum=...) before dynamics.",
    )
    return p.parse_args()


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint == "latest":
        root = Path(args.checkpoint_root).resolve()
        p = root / "final_params"
        if not p.is_dir():
            raise FileNotFoundError(f"Expected {p} for --checkpoint latest.")
        return p
    p = Path(args.checkpoint)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _pad_atoms(numbers: np.ndarray, positions: np.ndarray, natoms: int):
    """Pad Z with 0 and R with zeros to length ``natoms`` (training convention)."""
    n = len(numbers)
    if n > natoms:
        raise ValueError(f"Structure has {n} atoms but model.natoms={natoms}.")
    if n == natoms:
        return numbers.astype(np.int32), positions.astype(np.float64)
    z = np.zeros(natoms, dtype=np.int32)
    z[:n] = numbers.astype(np.int32)
    r = np.zeros((natoms, 3), dtype=np.float64)
    r[:n] = positions.astype(np.float64)
    return z, r


def main() -> int:
    from ase import Atoms, units
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from ase.io import read as ase_read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
        ZeroRotation,
    )

    args = parse_args()
    ckpt = _resolve_checkpoint(args)
    if not ckpt.is_dir():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    struct_path = Path(args.structure)
    if not struct_path.is_file():
        print(f"Structure not found: {struct_path}", file=sys.stderr)
        return 1

    raw = ase_read(str(struct_path))

    checkpointer = ocp.PyTreeCheckpointer()
    params, cfg, _, _, _ = restart_params_only(ckpt, checkpointer)
    if cfg is None:
        print("Checkpoint has no 'config'; expected spooky train_spooky_h5 format.", file=sys.stderr)
        return 1

    def _to_native(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    model = SpookyEF(**{k: _to_native(v) for k, v in cfg.items()})
    natoms = int(model.natoms)
    z, r = _pad_atoms(raw.get_atomic_numbers(), raw.get_positions(), natoms=natoms)

    atoms = Atoms(numbers=z, positions=r)
    if not args.no_center:
        atoms.center(vacuum=5.0)

    calc = get_ase_calc(
        params,
        model,
        atoms,
        conversion={"energy": 1, "forces": 1, "dipole": 1},
        implemented_properties=["energy", "forces"],
        spooky_charge=float(args.charge),
        spooky_multiplicity=float(args.multiplicity),
    )
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        friction=args.friction,
    )
    traj_path = out_dir / "spooky_md.traj"
    tw = Trajectory(str(traj_path), "w", atoms)
    tw.write()
    dyn.attach(tw.write, interval=int(args.printfreq))

    print(
        f"Running Langevin: T={args.temperature} K, dt={args.timestep} fs, "
        f"nsteps={args.nsteps}, natoms={natoms} (padded), checkpoint={ckpt}"
    )
    dyn.run(int(args.nsteps))
    tw.close()

    final_xyz = out_dir / "spooky_md_final.xyz"
    write(final_xyz, atoms)
    print(f"Wrote {traj_path} and {final_xyz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

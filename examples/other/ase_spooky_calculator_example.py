#!/usr/bin/env python3
"""
Example: use Spooky model as a plain ASE calculator.

This script:
  1) Loads a molecular structure with ASE
  2) Loads a Spooky checkpoint (Orbax dir or JSON file)
  3) Attaches an ASE calculator that feeds charge/spin conditioning
  4) Evaluates energy and forces

Bonus mode:
  --demo-charge-spin-grid evaluates multiple (charge, multiplicity) pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import ase
import ase.calculators.calculator as ase_calc
from ase.build import molecule
from ase.build.molecule import extra
from ase.collections import g2
import ase.io
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from ase import units
from ase.io import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from mmml.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
from mmml.utils.model_checkpoint import load_model_checkpoint


def _to_native(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _load_spooky_checkpoint(checkpoint: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    is_json = (
        (checkpoint.is_file() and checkpoint.suffix == ".json")
        or (checkpoint.is_dir() and (checkpoint / "params.json").exists())
    )

    if is_json:
        loaded = load_model_checkpoint(
            checkpoint,
            use_orbax=False,
            load_params=True,
            load_config=True,
        )
        params = loaded.get("params")
        config = loaded.get("config", {})
    else:
        restored = ocp.PyTreeCheckpointer().restore(str(checkpoint))
        params = restored.get("params", restored) if isinstance(restored, dict) else restored
        config = {}
        if isinstance(restored, dict):
            config = restored.get("config") or restored.get("model_attributes") or {}

    if params is None:
        raise ValueError(f"No params found in checkpoint: {checkpoint}")
    if not config:
        # Fallback for params-only checkpoints/JSON: use Spooky defaults.
        # natoms is overwritten later from the loaded ASE structure.
        config = SpookyEF().return_attributes()
        print(
            "Warning: no `config`/`model_attributes` found in checkpoint. "
            "Falling back to default Spooky model attributes."
        )

    # Ensure Flax variables format.
    if isinstance(params, dict) and "params" not in params:
        params = {"params": params}

    config = {k: _to_native(v) for k, v in config.items()}
    return params, config


class SpookyASECalculator(ase_calc.Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, params: Dict[str, Any], model: SpookyEF, charge: float, multiplicity: float):
        super().__init__()
        self.params = params
        self.model = model
        self.charge = float(charge)
        self.multiplicity = float(multiplicity)

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: Iterable[str] = ("energy", "forces"),
        system_changes: Iterable[str] = ase.calculators.calculator.all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None

        z = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        r = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
        n_atoms = int(z.shape[0])
        self.model.natoms = n_atoms

        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        atom_mask = jnp.ones((n_atoms,), dtype=jnp.float32)
        batch_segments = jnp.zeros((n_atoms,), dtype=jnp.int32)
        batch_mask = jnp.ones_like(dst_idx, dtype=jnp.float32)
        q_atoms = jnp.full((n_atoms, 1), self.charge, dtype=jnp.float32)
        s_atoms = jnp.full((n_atoms, 1), self.multiplicity, dtype=jnp.float32)

        out = self.model.apply(
            self.params,
            atomic_numbers=z,
            charges=q_atoms,
            spins=s_atoms,
            positions=r,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )

        self.results["energy"] = float(np.asarray(out["energy"]).reshape(-1)[0])
        self.results["forces"] = np.asarray(out["forces"], dtype=np.float64)


def _parse_cases(case_str: str) -> list[Tuple[float, float]]:
    # Format: "0:1,1:2,-1:2"
    pairs: list[Tuple[float, float]] = []
    for block in case_str.split(","):
        q_str, s_str = block.strip().split(":")
        pairs.append((float(q_str), float(s_str)))
    return pairs


def _print_model_details(model: SpookyEF, params: Dict[str, Any], *, charge: float, multiplicity: float) -> None:
    cfg = model.return_attributes()
    try:
        leaves = jax.tree_util.tree_leaves(params)
        n_params = int(sum(int(np.asarray(x).size) for x in leaves))
    except Exception:
        n_params = -1

    print("Model details:")
    print(f"  class            : {type(model).__module__}.{type(model).__name__}")
    print(f"  parameter_count  : {n_params if n_params >= 0 else 'n/a'}")
    print(f"  features         : {cfg.get('features')}")
    print(f"  max_degree       : {cfg.get('max_degree')}")
    print(f"  num_iterations   : {cfg.get('num_iterations')}")
    print(f"  num_basis_funcs  : {cfg.get('num_basis_functions')}")
    print(f"  cutoff           : {cfg.get('cutoff')}")
    print(f"  max_atomic_num   : {cfg.get('max_atomic_number')}")
    print(f"  natoms(runtime)  : {getattr(model, 'natoms', 'n/a')}")
    print(f"  zbl/efa/charges  : {cfg.get('zbl')}/{cfg.get('efa')}/{cfg.get('charges')}")
    print(f"  conditioning Q,S : {charge}, {multiplicity}")


def _load_atoms(args: argparse.Namespace) -> ase.Atoms:
    if args.structure is not None:
        return ase.io.read(str(args.structure))
    if args.collection == "g2":
        return g2[args.molecule_name].copy()
    # extra collection (e.g., C60)
    try:
        # Newer ASE may accept custom collection through `data=...`.
        return molecule(args.molecule_name, data=extra)
    except TypeError:
        # Older ASE forwards unknown kwargs to Atoms(...), so construct directly.
        if args.molecule_name not in extra:
            raise KeyError(
                f"Molecule '{args.molecule_name}' not found in ase.build.molecule.extra"
            )
        return ase.Atoms(**extra[args.molecule_name])


def _run_nve(atoms: ase.Atoms, args: argparse.Namespace) -> None:
    if args.nve_steps <= 0:
        return

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.nve_temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = VelocityVerlet(atoms, timestep=args.nve_timestep_fs * units.fs)
    traj = Trajectory(args.nve_trajectory, "w", atoms)

    print(
        f"Running NVE: steps={args.nve_steps}, dt={args.nve_timestep_fs} fs, "
        f"T0={args.nve_temperature} K, traj={args.nve_trajectory}"
    )
    for step in range(1, args.nve_steps + 1):
        dyn.run(1)
        if (step % args.nve_write_interval == 0) or (step == args.nve_steps):
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            etot = epot + ekin
            print(
                f"  step={step:6d}  Epot={epot:12.6f} eV  "
                f"Ekin={ekin:12.6f} eV  Etot={etot:12.6f} eV"
            )
            traj.write(atoms)
    traj.close()


def _run_bfgs(atoms: ase.Atoms, args: argparse.Namespace, *, tag: str = "") -> None:
    if args.bfgs_steps <= 0:
        return
    traj_name = args.bfgs_trajectory
    if tag:
        stem, suffix = Path(traj_name).stem, Path(traj_name).suffix
        suffix = suffix or ".traj"
        traj_name = f"{stem}_{tag}{suffix}"
    print(
        f"Running BFGS: steps={args.bfgs_steps}, fmax={args.bfgs_fmax}, "
        f"traj={traj_name}"
    )
    opt = BFGS(atoms, trajectory=traj_name, logfile="-")
    opt.run(fmax=args.bfgs_fmax, steps=args.bfgs_steps)
    e = atoms.get_potential_energy()
    fmax = float(np.abs(atoms.get_forces()).max())
    print(f"BFGS done: E={e:.8f} eV, |F|_max={fmax:.6f} eV/A")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach Spooky model as ASE calculator.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Spooky checkpoint path (Orbax dir or JSON).")
    p.add_argument("--structure", type=Path, default=None, help="Molecule file readable by ASE (xyz/pdb/etc).")
    p.add_argument(
        "--collection",
        type=str,
        choices=["g2", "extra"],
        default="extra",
        help="ASE built-in collection when --structure is not provided.",
    )
    p.add_argument(
        "--molecule-name",
        type=str,
        default="C60",
        help="Molecule name from selected collection (default: C60 from extra).",
    )
    p.add_argument("--charge", type=float, default=0.0, help="Total system charge Q used as per-atom conditioning.")
    p.add_argument("--multiplicity", type=float, default=1.0, help="Spin multiplicity S used as per-atom conditioning.")
    p.add_argument(
        "--demo-charge-spin-grid",
        action="store_true",
        help="Evaluate multiple charge/spin pairs from --cases.",
    )
    p.add_argument(
        "--cases",
        type=str,
        default="0:1,1:2,-1:2",
        help="Comma-separated Q:S pairs for demo mode (default: '0:1,1:2,-1:2').",
    )
    p.add_argument("--nve-steps", type=int, default=0, help="Run short NVE if > 0.")
    p.add_argument("--nve-timestep-fs", type=float, default=0.5, help="NVE timestep in fs.")
    p.add_argument("--nve-temperature", type=float, default=300.0, help="Initial temperature in K for velocities.")
    p.add_argument(
        "--nve-write-interval",
        type=int,
        default=10,
        help="Print/write interval for NVE.",
    )
    p.add_argument(
        "--nve-trajectory",
        type=str,
        default="spooky_nve.traj",
        help="Output trajectory filename for NVE.",
    )
    p.add_argument("--bfgs-steps", type=int, default=0, help="Run BFGS optimization if > 0.")
    p.add_argument("--bfgs-fmax", type=float, default=0.05, help="BFGS force threshold in eV/A.")
    p.add_argument(
        "--bfgs-trajectory",
        type=str,
        default="spooky_bfgs.traj",
        help="Output trajectory filename for BFGS.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    atoms = _load_atoms(args)

    params, model_config = _load_spooky_checkpoint(args.checkpoint)
    model = SpookyEF(**model_config)

    structure_label = str(args.structure) if args.structure is not None else f"{args.collection}:{args.molecule_name}"
    print(f"Loaded structure: {structure_label} ({len(atoms)} atoms)")
    print(f"Loaded checkpoint: {args.checkpoint}")

    if not args.demo_charge_spin_grid:
        calc = SpookyASECalculator(
            params=params,
            model=model,
            charge=args.charge,
            multiplicity=args.multiplicity,
        )
        _print_model_details(model, params, charge=args.charge, multiplicity=args.multiplicity)
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        print(f"Q={args.charge:.3f}, S={args.multiplicity:.3f} -> E={e:.8f} eV, |F|_max={np.abs(f).max():.6f} eV/A")
        _run_bfgs(atoms, args)
        _run_nve(atoms, args)
        return 0

    print("Charge/spin sweep:")
    cases = _parse_cases(args.cases)
    _print_model_details(
        model,
        params,
        charge=float(cases[0][0]),
        multiplicity=float(cases[0][1]),
    )
    for q, s in cases:
        calc = SpookyASECalculator(params=params, model=model, charge=q, multiplicity=s)
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        print(f"  Q={q:6.2f}  S={s:6.2f}  E={e:14.8f} eV  |F|_max={np.abs(f).max():10.6f} eV/A")
        _run_bfgs(atoms, args, tag=f"Q{q:g}_S{s:g}")
    _run_nve(atoms, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""
Convert an Orbax checkpoint to portable JSON and run a 4-molecule cluster test.

This script is intended as a small, reusable smoke-test utility:
1) Restore an Orbax checkpoint directory (epoch-*).
2) Export parameters + model attributes to a cross-platform JSON file.
3) Build a same-species molecular cluster (default: 4x CO2).
4) Run one energy/force calculator call and save cluster geometry.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from orbax.checkpoint import PyTreeCheckpointer

from mmml.utils.model_checkpoint import orbax_to_json

try:
    _helper_mod = importlib.import_module("mmml.models.physnetjax.physnetjax.calc.helper_mlp")
    _model_mod = importlib.import_module("mmml.models.physnetjax.physnetjax.models.model")
except ModuleNotFoundError:
    _helper_mod = importlib.import_module("mmml.physnetjax.physnetjax.calc.helper_mlp")
    _model_mod = importlib.import_module("mmml.physnetjax.physnetjax.models.model")

get_ase_calc = _helper_mod.get_ase_calc
EF = _model_mod.EF


def _build_base_molecule(name: str) -> Atoms:
    key = name.strip().upper()
    if key == "CO2":
        return Atoms("CO2", positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
    if key == "H2O":
        return Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    if key == "NH3":
        return Atoms(
            "NH3",
            positions=[
                [0.0, 0.0, 0.0],
                [0.94, 0.0, 0.33],
                [-0.47, 0.82, 0.33],
                [-0.47, -0.82, 0.33],
            ],
        )
    raise ValueError(f"Unsupported molecule '{name}'. Choose one of: CO2, H2O, NH3")


def _build_cluster(base: Atoms, n_molecules: int, spacing: float) -> Atoms:
    if n_molecules < 1:
        raise ValueError("n_molecules must be >= 1")

    n_side = int(np.ceil(np.sqrt(n_molecules)))
    blocks: list[Atoms] = []
    for idx in range(n_molecules):
        ix = idx % n_side
        iy = idx // n_side
        shift = np.array([ix * spacing, iy * spacing, 0.0], dtype=float)
        mol = base.copy()
        mol.translate(shift)
        blocks.append(mol)
    return sum(blocks[1:], blocks[0].copy()) if len(blocks) > 1 else blocks[0]


def _latest_epoch_dir(ckpt_root: Path) -> Path:
    if (ckpt_root / "manifest.ocdbt").exists():
        return ckpt_root
    epoch_dirs = [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("epoch-")]
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch-* directories found in: {ckpt_root}")
    return max(epoch_dirs, key=lambda d: int(d.name.split("epoch-")[-1]))


def run(args: argparse.Namespace) -> int:
    ckpt_root = args.checkpoint.expanduser().resolve()
    epoch_dir = _latest_epoch_dir(ckpt_root)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    portable_json = out_dir / "params_portable.json"
    orbax_to_json(epoch_dir, portable_json)

    restored = PyTreeCheckpointer().restore(str(epoch_dir))
    model_attributes = dict(restored["model_attributes"])
    params = restored["params"]

    natoms = int(model_attributes.get("natoms", 0))
    if natoms <= 0:
        raise ValueError("Checkpoint model_attributes missing valid 'natoms'")

    base = _build_base_molecule(args.molecule)
    cluster = _build_cluster(base, args.n_molecules, args.spacing)
    if len(cluster) > natoms:
        raise ValueError(
            f"Cluster has {len(cluster)} atoms but checkpoint natoms={natoms}. "
            "Use fewer molecules or a smaller base molecule."
        )

    padded = Atoms(numbers=np.ones(natoms, dtype=int), positions=np.zeros((natoms, 3), dtype=float))
    padded.numbers[: len(cluster)] = cluster.numbers
    padded.positions[: len(cluster)] = cluster.positions

    model = EF(**model_attributes)
    model.natoms = natoms
    calc = get_ase_calc(params, model, padded)
    padded.calc = calc

    energy = float(padded.get_potential_energy())
    forces = padded.get_forces()[: len(cluster)]
    max_force = float(np.abs(forces).max())

    cluster.info["energy_eV"] = energy
    cluster.arrays["forces_eVA"] = np.asarray(forces, dtype=float)
    xyz_path = out_dir / "cluster_4mol.xyz"
    extxyz_path = out_dir / "cluster_4mol.extxyz"
    write(xyz_path, cluster)
    write(extxyz_path, cluster)

    print(f"Epoch used: {epoch_dir}")
    print(f"Portable checkpoint: {portable_json}")
    print(f"Cluster files: {xyz_path} | {extxyz_path}")
    print(f"Molecule: {args.molecule} x {args.n_molecules} (atoms={len(cluster)})")
    print(f"Energy (eV): {energy:.8f}")
    print(f"Max |force| (eV/A): {max_force:.8f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Orbax checkpoint to portable JSON and test a 4-molecule cluster."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Orbax checkpoint root or epoch-* directory")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoint_smoke"), help="Output directory")
    parser.add_argument("--molecule", type=str, default="CO2", choices=["CO2", "H2O", "NH3"], help="Base molecule")
    parser.add_argument("--n-molecules", type=int, default=4, help="Number of same molecules in cluster")
    parser.add_argument("--spacing", type=float, default=4.0, help="Grid spacing between molecule COMs in Angstrom")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

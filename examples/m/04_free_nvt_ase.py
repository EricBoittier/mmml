#!/usr/bin/env python3
"""Vacuum NVT (ASE Langevin) for NH3–CH3Cl with the kl.json PhysNet ckpt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from _geometry import DEFAULT_NPZ, load_dimer_frame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "examples/m/kl.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/nh3_ch3cl/free_nvt_ase",
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

    temps: list[float] = []
    energies: list[float] = []
    e0 = float(atoms.get_potential_energy())
    energies.append(e0)
    temps.append(float(atoms.get_temperature()))

    dyn = Langevin(
        atoms,
        timestep=float(args.dt_fs) * units.fs,
        temperature_K=float(args.temperature),
        friction=float(args.friction),
    )

    def _log(_atoms=atoms) -> None:
        energies.append(float(_atoms.get_potential_energy()))
        temps.append(float(_atoms.get_temperature()))

    dyn.attach(_log, interval=max(1, int(args.n_steps) // 20))
    dyn.run(int(args.n_steps))
    e1 = float(atoms.get_potential_energy())
    t1 = float(atoms.get_temperature())
    energies.append(e1)
    temps.append(t1)

    summary = {
        "backend": "ase",
        "ensemble": "nvt",
        "checkpoint": str(ckpt),
        "n_atoms": n_atoms,
        "n_steps": int(args.n_steps),
        "dt_fs": float(args.dt_fs),
        "temperature_K": float(args.temperature),
        "E0_kcal_mol": e0,
        "E1_kcal_mol": e1,
        "T_final_K": t1,
        "T_mean_K": float(np.mean(temps)),
        "E_trace_kcal_mol": energies,
        "T_trace_K": temps,
    }
    (out / "md_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out / "final.npz", Z=z, R=np.asarray(atoms.get_positions(), dtype=np.float64))

    print(
        f"ASE NVT  E0={e0:.6f}  E1={e1:.6f}  T_final={t1:.1f} K  "
        f"T_mean={np.mean(temps):.1f} K  steps={args.n_steps}"
    )
    print(f"Wrote {out / 'md_summary.json'}")
    if not np.isfinite(e1):
        print("FAIL: non-finite energy after NVT", file=sys.stderr)
        return 1
    print("PASS: ASE NVT")
    return 0


if __name__ == "__main__":
    sys.exit(main())

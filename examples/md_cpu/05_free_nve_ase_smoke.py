#!/usr/bin/env python3
"""Short vacuum NVE with ASE VelocityVerlet (ML-only, no PyCHARMM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import write as ase_write
from ase.md.verlet import VelocityVerlet

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.md_cpu._geometry import aco_dimer_cluster


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/md_cpu/free_nve_ase_smoke",
    )
    args = parser.parse_args()

    from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    z, r = aco_dimer_cluster(n_monomers=2, spacing=5.0)
    ckpt = resolve_checkpoint(args.checkpoint)
    n_atoms = len(z)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    factory = setup_calculator(
        ATOMS_PER_MONOMER=n_atoms // 2,
        N_MONOMERS=2,
        doML=True,
        doMM=False,
        model_restart_path=str(ckpt),
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        defer_xla_gpu_warmup=True,
        verbose=False,
    )
    calc, _sph, _ = unpack_factory_result(
        factory(atomic_numbers=z, atomic_positions=r, n_monomers=2)
    )
    atoms = Atoms(numbers=z, positions=r)
    atoms.calc = calc

    e0 = float(atoms.get_potential_energy())
    dyn = VelocityVerlet(atoms, timestep=float(args.dt_fs) * units.fs)
    dyn.run(int(args.n_steps))
    e1 = float(atoms.get_potential_energy())
    traj = out / "nve_ase_smoke.traj"
    ase_write(str(traj), atoms)

    print(f"E0={e0:.6f} kcal/mol  E1={e1:.6f} kcal/mol  steps={args.n_steps}")
    print(f"Wrote {traj}")
    if not np.isfinite(e1):
        print("FAIL: non-finite energy after NVE", file=sys.stderr)
        return 1
    print("PASS: ASE NVE smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""ML energy/forces on an ACO dimer via ASE (DESdimers JSON checkpoint)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ase
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.md_cpu._geometry import aco_dimer_cluster


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PhysNet JSON checkpoint (default: MMML_CKPT or examples/ckpts_json/DESdimers_params.json)",
    )
    parser.add_argument("--n-monomers", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=5.0)
    parser.add_argument(
        "--write-npz",
        type=Path,
        default=None,
        help="Write handoff-style geometry NPZ for --evaluate-npz (e.g. artifacts/md_cpu/aco_dimer.npz)",
    )
    args = parser.parse_args()

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint
    from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol, setup_calculator
    from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = aco_dimer_cluster(n_monomers=int(args.n_monomers), spacing=float(args.spacing))
    n_atoms = len(z)
    atoms_per = n_atoms // int(args.n_monomers)
    print(f"Cluster ACO×{args.n_monomers} ({n_atoms} atoms)  checkpoint={ckpt}")

    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per,
        N_MONOMERS=int(args.n_monomers),
        doML=True,
        doMM=False,
        model_restart_path=str(ckpt),
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        defer_xla_gpu_warmup=True,
        verbose=False,
    )
    calc, _sph, _ = unpack_factory_result(
        factory(atomic_numbers=z, atomic_positions=r, n_monomers=int(args.n_monomers))
    )
    atoms = ase.Atoms(numbers=z, positions=r)
    atoms.calc = calc

    energy_kcal = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    print(f"Energy (kcal/mol): {energy_kcal:.6f}")
    print(f"Forces max |F| (kcal/mol/Å): {float(np.abs(forces).max()):.6f}")

    if not np.all(np.isfinite(forces)):
        print("FAIL: non-finite forces", file=sys.stderr)
        return 1

    if args.write_npz is not None:
        out = Path(args.write_npz).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": "ase",
            "composition": f"ACO:{int(args.n_monomers)}",
            "checkpoint": str(ckpt),
        }
        np.savez_compressed(
            out,
            positions=np.asarray(r, dtype=np.float64),
            atomic_numbers=np.asarray(z, dtype=np.int32),
            pbc=np.array(False),
            metadata=json.dumps(meta),
            E=np.array([energy_kcal / ev2kcalmol], dtype=np.float64),
            F=forces.reshape(1, n_atoms, 3).astype(np.float64),
        )
        print(f"Wrote geometry NPZ: {out}")

    print("PASS: ASE ML energy/forces")
    return 0


if __name__ == "__main__":
    sys.exit(main())

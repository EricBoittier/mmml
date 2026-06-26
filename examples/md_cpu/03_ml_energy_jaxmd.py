#!/usr/bin/env python3
"""Compare ASE vs JAX-MD spherical ML energy on an ACO dimer (CPU, no CHARMM)."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n-monomers", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=5.0)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--force-atol", type=float, default=5.0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import e3x

    from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    ckpt = resolve_checkpoint(args.checkpoint)
    z, r = aco_dimer_cluster(n_monomers=int(args.n_monomers), spacing=float(args.spacing))
    n_atoms = len(z)
    n_monomers = int(args.n_monomers)
    atoms_per = n_atoms // n_monomers
    print(f"Cluster ACO×{n_monomers} ({n_atoms} atoms)  checkpoint={ckpt}")

    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per,
        N_MONOMERS=n_monomers,
        doML=True,
        doMM=False,
        model_restart_path=str(ckpt),
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        defer_xla_gpu_warmup=True,
        verbose=False,
    )
    calc, spherical_fn, _ = unpack_factory_result(
        factory(atomic_numbers=z, atomic_positions=r, n_monomers=n_monomers)
    )

    atoms = ase.Atoms(numbers=z, positions=r)
    atoms.calc = calc
    e_ase = float(atoms.get_potential_energy())
    f_ase = np.asarray(atoms.get_forces(), dtype=float)

    dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
    pair_idx = jnp.stack([dst, src], axis=1)
    pair_mask = jnp.ones(pair_idx.shape[0], dtype=jnp.float32)
    cutoff_params = CutoffParameters()

    result = spherical_fn(
        atomic_numbers=jnp.array(z),
        positions=jnp.array(r),
        n_monomers=n_monomers,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=False,
        doML_dimer=True,
        debug=False,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
    )
    e_jax = float(result.energy.reshape(-1)[0])

    @jax.jit
    def energy_fn(pos):
        out = spherical_fn(
            atomic_numbers=jnp.array(z),
            positions=jnp.array(pos),
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
            debug=False,
            mm_pair_idx=pair_idx,
            mm_pair_mask=pair_mask,
        )
        return out.energy.reshape(-1)[0]

    f_jax = np.asarray(-jax.grad(energy_fn)(jnp.array(r)), dtype=float)

    e_scale = max(abs(e_ase), abs(e_jax), 1e-6)
    f_max = float(np.abs(f_ase - f_jax).max())
    print(f"ASE energy (kcal/mol):  {e_ase:.8f}")
    print(f"JAX energy (kcal/mol):  {e_jax:.8f}")
    print(f"|dE|/scale:             {abs(e_ase - e_jax) / e_scale:.6e}")
    print(f"max |dF| (kcal/mol/Å):  {f_max:.8f}")

    if abs(e_ase - e_jax) / e_scale > float(args.rtol):
        print("FAIL: energy mismatch", file=sys.stderr)
        return 1
    if f_max > float(args.force_atol):
        print("FAIL: force mismatch", file=sys.stderr)
        return 1

    print("PASS: ASE vs JAX-MD ML energy/forces")
    return 0


if __name__ == "__main__":
    sys.exit(main())

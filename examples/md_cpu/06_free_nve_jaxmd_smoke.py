#!/usr/bin/env python3
"""Short vacuum NVE with JAX-MD (ML-only spherical calc, no PyCHARMM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import e3x
    from jax_md import simulate, space, units

    from examples.md_cpu._geometry import aco_dimer_cluster
    from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    z, r = aco_dimer_cluster(n_monomers=2, spacing=5.0)
    ckpt = resolve_checkpoint(args.checkpoint)
    n_atoms = len(z)
    n_monomers = 2

    factory = setup_calculator(
        ATOMS_PER_MONOMER=n_atoms // n_monomers,
        N_MONOMERS=n_monomers,
        doML=True,
        doMM=False,
        model_restart_path=str(ckpt),
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        defer_xla_gpu_warmup=True,
        verbose=False,
    )
    _calc, spherical_fn, _ = unpack_factory_result(
        factory(atomic_numbers=z, atomic_positions=r, n_monomers=n_monomers)
    )

    dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
    pair_idx = jnp.stack([dst, src], axis=1)
    pair_mask = jnp.ones(pair_idx.shape[0], dtype=jnp.float32)
    cutoff_params = CutoffParameters()

    def eval_fn(pos):
        return spherical_fn(
            atomic_numbers=jnp.array(z),
            positions=jnp.asarray(pos, dtype=jnp.float32),
            n_monomers=n_monomers,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=False,
            doML_dimer=True,
            debug=False,
            mm_pair_idx=pair_idx,
            mm_pair_mask=pair_mask,
        )

    @jax.jit
    def force_fn(pos):
        return jnp.asarray(eval_fn(pos).forces, dtype=jnp.float32)

    def energy_at(pos):
        return float(eval_fn(pos).energy.reshape(-1)[0])

    unit = units.metal_unit_system()
    displacement, shift = space.free()
    dt = float(args.dt_fs) * 0.001  # fs → ps (metal units)
    temperature_k = 300.0
    kT = temperature_k * unit["temperature"]
    init_fn, apply_fn = simulate.nve(force_fn, shift, dt)
    key = jax.random.PRNGKey(0)
    mass = jnp.ones((n_atoms,))
    pos0 = jnp.array(r, dtype=jnp.float32)
    state = init_fn(key, pos0, kT, mass=mass)
    e0 = energy_at(state.position)

    @jax.jit
    def step(state):
        return apply_fn(state)

    state = state
    for _ in range(int(args.n_steps)):
        state = step(state)
    e1 = energy_at(state.position)

    print(f"E0={e0:.6f} kcal/mol  E1={e1:.6f} kcal/mol  steps={args.n_steps}")
    if not np.isfinite(e1):
        print("FAIL: non-finite energy after JAX-MD NVE", file=sys.stderr)
        return 1
    print("PASS: JAX-MD NVE smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())

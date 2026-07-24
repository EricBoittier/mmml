#!/usr/bin/env python3
"""Vacuum NVT (JAX-MD Nose–Hoover) for NH3–CH3Cl with the kl.json PhysNet ckpt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.m._geometry import DEFAULT_NPZ, load_dimer_frame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "examples/m/kl.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/nh3_ch3cl/free_nvt_jaxmd",
    )
    args = parser.parse_args()

    import e3x
    import jax
    import jax.numpy as jnp
    from jax_md import quantity, simulate, space, units

    from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
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
    _calc, spherical_fn, _ = unpack_factory_result(
        factory(atomic_numbers=z, atomic_positions=r, n_monomers=1)
    )

    dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
    pair_idx = jnp.stack([dst, src], axis=1)
    pair_mask = jnp.ones(pair_idx.shape[0], dtype=jnp.float32)
    cutoff_params = CutoffParameters()
    z_j = jnp.array(z)

    def eval_fn(pos):
        return spherical_fn(
            atomic_numbers=z_j,
            positions=jnp.asarray(pos, dtype=jnp.float32),
            n_monomers=1,
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
    _displacement, shift = space.free()
    dt = float(args.dt_fs) * 0.001  # fs → ps
    kT = float(args.temperature) * unit["temperature"]
    init_fn, apply_fn = simulate.nvt_nose_hoover(force_fn, shift, dt, kT)
    key = jax.random.PRNGKey(0)
    mass = jnp.ones((n_atoms,))
    pos0 = jnp.array(r, dtype=jnp.float32)
    state = init_fn(key, pos0, mass=mass)
    e0 = energy_at(state.position)
    energies = [e0]
    temps = [float(quantity.temperature(momentum=state.momentum, mass=mass) / unit["temperature"])]

    @jax.jit
    def step(state):
        return apply_fn(state)

    log_every = max(1, int(args.n_steps) // 20)
    for i in range(int(args.n_steps)):
        state = step(state)
        if (i + 1) % log_every == 0:
            energies.append(energy_at(state.position))
            temps.append(
                float(
                    quantity.temperature(momentum=state.momentum, mass=mass)
                    / unit["temperature"]
                )
            )
    e1 = energy_at(state.position)
    t1 = float(quantity.temperature(momentum=state.momentum, mass=mass) / unit["temperature"])
    energies.append(e1)
    temps.append(t1)

    summary = {
        "backend": "jaxmd",
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
    np.savez(
        out / "final.npz",
        Z=z,
        R=np.asarray(state.position, dtype=np.float64),
    )

    print(
        f"JAX-MD NVT  E0={e0:.6f}  E1={e1:.6f}  T_final={t1:.1f} K  "
        f"T_mean={np.mean(temps):.1f} K  steps={args.n_steps}"
    )
    print(f"Wrote {out / 'md_summary.json'}")
    if not np.isfinite(e1):
        print("FAIL: non-finite energy after JAX-MD NVT", file=sys.stderr)
        return 1
    print("PASS: JAX-MD NVT")
    return 0


if __name__ == "__main__":
    sys.exit(main())

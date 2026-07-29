#!/usr/bin/env python3
"""Vacuum NVE (JAX-MD) for NH3–CH3Cl with the kl.json PhysNet ckpt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from _geometry import DEFAULT_NPZ, load_dimer_frame  # noqa: E402
from md_io import write_final_geometry, write_jaxmd_trajectory  # noqa: E402


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
        default=REPO_ROOT / "artifacts/nh3_ch3cl/free_nve_jaxmd",
    )
    args = parser.parse_args()

    import e3x
    import jax
    import jax.numpy as jnp
    from jax_md import simulate, space, units

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
    # jax-md metal units measure time in Å·sqrt(amu/eV) = 10.18 fs;
    # unit["time"] converts ps → those internal units. Passing bare ps
    # integrates ~98x too finely (stable, but 1% of the nominal duration).
    dt = float(args.dt_fs) * 0.001 * unit["time"]
    kT = float(args.temperature) * unit["temperature"]
    init_fn, apply_fn = simulate.nve(force_fn, shift, dt)
    key = jax.random.PRNGKey(0)
    mass = jnp.ones((n_atoms,))
    pos0 = jnp.array(r, dtype=jnp.float32)
    state = init_fn(key, pos0, kT, mass=mass)
    e0 = energy_at(state.position)
    energies = [e0]
    frames = [np.asarray(state.position, dtype=np.float64)]
    # Metal-unit velocities (same convention as ASE/JAX-MD handoff): p / m.
    mass_b = np.asarray(state.mass, dtype=np.float64).reshape(n_atoms, -1)
    forces_list = [np.asarray(force_fn(state.position), dtype=np.float64)]
    velocities_list = [
        np.asarray(state.momentum, dtype=np.float64) / mass_b
    ]

    @jax.jit
    def step(state):
        return apply_fn(state)

    def _sample(st):
        energies.append(energy_at(st.position))
        frames.append(np.asarray(st.position, dtype=np.float64))
        forces_list.append(np.asarray(force_fn(st.position), dtype=np.float64))
        m = np.asarray(st.mass, dtype=np.float64).reshape(n_atoms, -1)
        velocities_list.append(np.asarray(st.momentum, dtype=np.float64) / m)

    traj_every = max(1, int(args.traj_interval))
    for i in range(int(args.n_steps)):
        state = step(state)
        if (i + 1) % traj_every == 0:
            _sample(state)
    e1 = energy_at(state.position)
    f1 = np.asarray(force_fn(state.position), dtype=np.float64)
    m1 = np.asarray(state.mass, dtype=np.float64).reshape(n_atoms, -1)
    v1 = np.asarray(state.momentum, dtype=np.float64) / m1
    energies.append(e1)
    frames.append(np.asarray(state.position, dtype=np.float64))
    forces_list.append(f1)
    velocities_list.append(v1)

    traj_paths = write_jaxmd_trajectory(
        out,
        z,
        frames,
        energies=energies,
        forces=forces_list,
        velocities=velocities_list,
    )
    geom = write_final_geometry(
        out, z, state.position, energy=e1, forces=f1, velocities=v1
    )

    summary = {
        "backend": "jaxmd",
        "ensemble": "nve",
        "checkpoint": str(ckpt),
        "n_atoms": n_atoms,
        "n_steps": int(args.n_steps),
        "dt_fs": float(args.dt_fs),
        "E0_kcal_mol": e0,
        "E1_kcal_mol": e1,
        "dE_kcal_mol": e1 - e0,
        "E_trace_kcal_mol": energies,
        "artifacts": {**traj_paths, **geom},
    }
    (out / "md_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"JAX-MD NVE  E0={e0:.6f}  E1={e1:.6f}  dE={e1 - e0:.6f} kcal/mol  "
        f"steps={args.n_steps}"
    )
    print(f"Wrote {out / 'md.traj'}, {out / 'md.xyz'}, {out / 'final.xyz'}")
    if not np.isfinite(e1):
        print("FAIL: non-finite energy after JAX-MD NVE", file=sys.stderr)
        return 1
    print("PASS: JAX-MD NVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

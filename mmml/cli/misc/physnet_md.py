#!/usr/bin/env python
"""
CLI for PhysNet molecular dynamics sampling with ASE and JAX-MD.

Runs NVT Langevin (ASE) and NVT Nose-Hoover (JAX-MD) using a trained PhysNet
checkpoint as the energy/force calculator.

Usage:
    mmml physnet-md --checkpoint out/ckpts/cybz_physnet --structure molecule.xyz -o out/
    mmml physnet-md --checkpoint out/ckpts/cybz_physnet --data splits/energies_forces_dipoles_train.npz -o out/
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_structure_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load R, Z from NPZ (energies_forces_dipoles or similar)."""
    data = np.load(path, allow_pickle=True)
    R = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    if R.ndim == 3:
        R = R[0]
    if Z.ndim == 2:
        Z = Z[0]
    n_atoms = int(np.sum(Z > 0)) if np.any(Z > 0) else len(Z)
    return R[:n_atoms], Z[:n_atoms]


def _load_structure_from_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load R, Z from XYZ/PDB/etc via ASE."""
    from ase.io import read

    atoms = read(str(path))
    return atoms.get_positions(), atoms.get_atomic_numbers()


def main() -> int:
    """Run physnet-md CLI."""
    parser = argparse.ArgumentParser(
        description="PhysNet MD sampling with ASE and JAX-MD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PhysNet checkpoint directory (e.g. out/ckpts/cybz_physnet)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--structure",
        type=Path,
        help="Initial structure (XYZ, PDB, etc.)",
    )
    group.add_argument(
        "--data",
        type=Path,
        help="NPZ with R, Z (e.g. splits/energies_forces_dipoles_train.npz)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: .)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in K (default: 300)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Timestep in fs (default: 0.5)",
    )
    parser.add_argument(
        "--nsteps-ase",
        type=int,
        default=100,
        help="ASE Langevin steps (default: 100)",
    )
    parser.add_argument(
        "--nsteps-jaxmd",
        type=int,
        default=200,
        help="JAX-MD Nose-Hoover steps (default: 200)",
    )
    parser.add_argument(
        "--printfreq",
        type=int,
        default=25,
        help="Print/save interval (default: 25)",
    )
    parser.add_argument(
        "--skip-jaxmd",
        action="store_true",
        help="Skip JAX-MD (ASE only)",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    # Load initial geometry
    if args.structure is not None:
        if not args.structure.exists():
            print(f"Error: Structure file not found: {args.structure}", file=sys.stderr)
            return 1
        R, Z = _load_structure_from_file(args.structure)
    else:
        if not args.data.exists():
            print(f"Error: Data file not found: {args.data}", file=sys.stderr)
            return 1
        R, Z = _load_structure_from_npz(args.data)

    n_atoms = len(Z)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from mmml.physnetjax.physnetjax.restart.restart import get_last, get_params_model
    from mmml.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc
    import e3x
    import jax
    import jax.numpy as jnp

    restart = get_last(str(args.checkpoint))
    params, model = get_params_model(str(restart), natoms=n_atoms)

    from ase import Atoms, units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.io import write
    from ase.io.trajectory import Trajectory

    atoms = Atoms(numbers=Z, positions=R)
    atoms.center(vacuum=5.0)

    calc = get_ase_calc(
        params,
        model,
        atoms,
        conversion={"energy": 1, "forces": 1, "dipole": 1},
        implemented_properties=["energy", "forces"],
    )
    atoms.calc = calc

    # -------------------------------------------------------------------------
    # 1. ASE sampling (NVT Langevin)
    # -------------------------------------------------------------------------
    print("=== PhysNet MD sampling (ASE + JAX-MD) ===")
    print("\n--- ASE: NVT Langevin ---")

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        friction=0.01,
    )
    traj_ase = args.output_dir / "physnet_ase.traj"
    traj_writer = Trajectory(traj_ase, "w", atoms)
    traj_writer.write()
    dyn.attach(traj_writer.write, interval=args.printfreq)

    for i in range(args.nsteps_ase):
        dyn.run(1)
        if i % args.printfreq == 0:
            print(
                f"  step {i:4d}  E={atoms.get_potential_energy():.4f} eV  "
                f"T={atoms.get_temperature():.1f} K"
            )

    traj_writer.close()
    write(args.output_dir / "physnet_ase_final.xyz", atoms)
    print(f"  Saved ASE trajectory: {traj_ase}")

    # -------------------------------------------------------------------------
    # 2. JAX-MD sampling (NVT Nose-Hoover)
    # -------------------------------------------------------------------------
    if args.skip_jaxmd:
        print("\n--- JAX-MD: skipped (--skip-jaxmd) ---")
        return 0

    try:
        from jax_md import space, quantity, simulate
    except ImportError:
        print("\n--- JAX-MD: skipped (pip install jax-md) ---")
        return 0

    print("\n--- JAX-MD: NVT Nose-Hoover ---")
    R0 = np.array(atoms.get_positions())
    Z_jnp = jnp.array(Z, dtype=jnp.int32)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
    dst_idx = jnp.array(dst_idx, dtype=jnp.int32)
    src_idx = jnp.array(src_idx, dtype=jnp.int32)

    @jax.jit
    def model_apply(positions):
        return model.apply(
            params,
            atomic_numbers=Z_jnp,
            positions=positions[None, :, :],
            dst_idx=dst_idx,
            src_idx=src_idx,
        )

    def energy_fn(position, **kwargs):
        out = model_apply(position)
        return jnp.squeeze(out["energy"])

    _, shift = space.free()
    K_B = 8.617333e-5  # eV/K
    kT = K_B * args.temperature
    dt = args.timestep * 1e-3  # fs -> ps

    init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)
    apply_fn = jax.jit(apply_fn)

    from ase.data import atomic_masses

    masses = np.array([atomic_masses[z] for z in Z])
    key = jax.random.PRNGKey(42)
    state = init_fn(key, R0, mass=masses)

    positions_jaxmd = [R0]
    for i in range(args.nsteps_jaxmd):
        state = apply_fn(state)
        if (i + 1) % args.printfreq == 0:
            positions_jaxmd.append(np.array(state.position))
            T_curr = float(quantity.temperature(state.momentum, mass=masses) / K_B)
            E = float(energy_fn(state.position))
            print(f"  step {i+1:4d}  E={E:.4f} eV  T={T_curr:.1f} K")

    traj_jaxmd = args.output_dir / "physnet_jaxmd.xyz"
    for i, R_frame in enumerate(positions_jaxmd):
        at = Atoms(numbers=Z, positions=R_frame)
        write(traj_jaxmd, at, append=(i > 0))
    print(f"  Saved JAX-MD trajectory: {traj_jaxmd}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

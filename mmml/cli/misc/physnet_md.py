#!/usr/bin/env python
"""
CLI for PhysNet molecular dynamics sampling with ASE and JAX-MD.

Runs NVT Langevin (ASE) and NVT Nose-Hoover (JAX-MD) using a trained PhysNet
checkpoint as the energy/force calculator.

Usage:
    mmml physnet-md --checkpoint out/ckpts/cybz_physnet --structure molecule.xyz -o out/
    mmml physnet-md --checkpoint out/ckpts/cybz_physnet --data splits/energies_forces_dipoles_train.npz -o out/
    mmml physnet-md --checkpoint out/ckpts/cybz_physnet --data splits/train.npz -o out/ --n-replicas 4
"""

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _load_structure_from_npz(
    path: Path, index: int = 0, n_replicas: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Load R, Z from NPZ (energies_forces_dipoles or similar).
    If n_replicas > 1 and dataset has enough structures, returns (R_multi, Z)
    with R_multi shape (n_replicas, n_atoms, 3). Otherwise returns single (R, Z).
    """
    data = np.load(path, allow_pickle=True)
    R_all = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    if Z.ndim == 2:
        Z = Z[0]
    n_atoms = int(np.sum(Z > 0)) if np.any(Z > 0) else len(Z)
    Z = Z[:n_atoms]

    if R_all.ndim == 2:
        R_all = R_all[np.newaxis, ...]
    n_avail = R_all.shape[0]
    if n_replicas > 1 and n_avail >= n_replicas:
        R_multi = np.zeros((n_replicas, n_atoms, 3), dtype=np.float64)
        for k in range(n_replicas):
            rk = np.asarray(R_all[index + k])
            if rk.ndim == 3:
                rk = rk[0]
            R_multi[k] = rk[:n_atoms]
        return R_multi, Z
    R = np.asarray(R_all[index])
    if R.ndim == 3:
        R = R[0]
    return R[:n_atoms], Z


def _run_ase_replica(args: Dict[str, Any]) -> tuple[int, Path]:
    """Worker: run one ASE Langevin replica. Returns (replica_id, traj_path)."""
    from ase import Atoms, units
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    from mmml.physnetjax.physnetjax.restart.restart import get_last, get_params_model
    from mmml.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc

    replica_id = args["replica_id"]
    R = np.array(args["R"])
    Z = np.array(args["Z"])
    checkpoint = Path(args["checkpoint"])
    output_dir = Path(args["output_dir"])
    temperature = args["temperature"]
    timestep = args["timestep"]
    nsteps = args["nsteps"]
    printfreq = args["printfreq"]

    restart = get_last(str(checkpoint))
    params, model = get_params_model(str(restart), natoms=len(Z))

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

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        friction=0.01,
    )
    traj_path = output_dir / f"physnet_ase_replica{replica_id}.traj"
    traj_writer = Trajectory(traj_path, "w", atoms)
    traj_writer.write()
    dyn.attach(traj_writer.write, interval=printfreq)

    for i in range(nsteps):
        dyn.run(1)

    traj_writer.close()
    write(output_dir / f"physnet_ase_replica{replica_id}_final.xyz", atoms)
    return replica_id, traj_path


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
        default=0.1,
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
        default=100,
        help="JAX-MD Nose-Hoover steps (default: 200)",
    )
    parser.add_argument(
        "--printfreq",
        type=int,
        default=1,
        help="Print/save interval (default: 25)",
    )
    parser.add_argument(
        "--skip-jaxmd",
        action="store_true",
        help="Skip JAX-MD (ASE only)",
    )
    parser.add_argument(
        "--n-replicas",
        type=int,
        default=1,
        metavar="B",
        help="Number of independent replicas to run in parallel. "
        "ASE: ProcessPoolExecutor; JAX-MD: batched GPU. "
        "With --data, uses first B structures as initial geometries if available. (default: 1)",
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
        R, Z = _load_structure_from_npz(args.data, n_replicas=args.n_replicas)

    n_replicas = max(1, args.n_replicas)
    if R.ndim == 3:
        R_multi = R
        R = R[0]
    else:
        R_multi = None
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
    from ase.md.verlet import VelocityVerlet
    from ase.io import write
    from ase.io.trajectory import Trajectory

    atoms = Atoms(numbers=Z, positions=R)
    atoms.center(vacuum=5.0)
    R0_initial = np.array(atoms.get_positions(), dtype=np.float32)  # same start for ASE and JAX-MD

    calc = get_ase_calc(
        params,
        model,
        atoms,
        conversion={"energy": 1, "forces": 1, "dipole": 1},
        implemented_properties=["energy", "forces"],
    )
    atoms.calc = calc

    # -------------------------------------------------------------------------
    # 1. ASE sampling 
    # -------------------------------------------------------------------------
    print("=== PhysNet MD sampling  ===")

    if n_replicas > 1:
        # Parallel: run replicas in separate processes (like joint_trainer)
        print(f"\n--- ASE: ({n_replicas} replicas in parallel) ---")
        if R_multi is None:
            R_multi = np.tile(R[np.newaxis, :, :], (n_replicas, 1, 1))

        worker_args = [
            {
                "replica_id": k,
                "R": R_multi[k],
                "Z": Z,
                "checkpoint": args.checkpoint,
                "output_dir": args.output_dir,
                "temperature": args.temperature,
                "timestep": args.timestep,
                "nsteps": args.nsteps_ase,
                "printfreq": args.printfreq,
            }
            for k in range(n_replicas)
        ]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_replicas, mp_context=ctx) as executor:
            futures = {executor.submit(_run_ase_replica, a): a["replica_id"] for a in worker_args}
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    _, traj_path = future.result()
                    print(f"  Replica {rid} done: {traj_path}")
                except Exception as e:
                    print(f"  Replica {rid} failed: {e}", file=sys.stderr)
                    raise
    else:
        # Single replica
        print("\n--- ASE: NVT Langevin ---")
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
        Stationary(atoms)
        ZeroRotation(atoms)

        # dyn = Langevin(
        #     atoms,
        #     timestep=args.timestep * units.fs,
        #     temperature_K=args.temperature,
        #     friction=0.01,
        # )

        dyn = VelocityVerlet(atoms, timestep=args.timestep * units.fs)
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

    # Enable float64 for JAX-MD (avoids NaN from float32 truncation in Nose-Hoover)
    jax.config.update("jax_enable_x64", True)

    try:
        from jax_md import space, quantity, simulate
    except ImportError:
        print("\n--- JAX-MD: skipped (pip install jax-md) ---")
        return 0

    from ase.data import atomic_masses

    masses = np.array([atomic_masses[z] for z in Z])
    K_B = 8.617333e-5  # eV/K
    kT = K_B * args.temperature
    # dt in fs (jax_md with eV/Å/amu uses fs; jaxmd_dynamics passes timestep_fs directly)
    dt = args.timestep

    if n_replicas > 1:
        # Batched: all replicas in one JIT-compiled simulation (like jaxmd_dynamics, ase_md main_batched)
        print(f"\n--- JAX-MD: NVT Nose-Hoover ({n_replicas} replicas batched) ---")
        B = n_replicas
        dst_single, src_single = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_single = np.array(dst_single, dtype=np.int32)
        src_single = np.array(src_single, dtype=np.int32)
        offsets = np.arange(B, dtype=np.int32) * n_atoms
        dst_idx = np.concatenate([dst_single + off for off in offsets])
        src_idx = np.concatenate([src_single + off for off in offsets])
        dst_idx = jnp.array(dst_idx, dtype=jnp.int32)
        src_idx = jnp.array(src_idx, dtype=jnp.int32)

        batch_segments = jnp.repeat(jnp.arange(B, dtype=jnp.int32), n_atoms)
        Z_batched = jnp.tile(jnp.array(Z, dtype=jnp.int32), B)
        masses_batched = jnp.tile(jnp.array(masses, dtype=jnp.float64), B)
        batch_mask = jnp.ones(len(dst_idx), dtype=jnp.float32)
        atom_mask = jnp.ones(B * n_atoms, dtype=jnp.float32)

        if R_multi is not None:
            R0_packed = jnp.array(R_multi.reshape(B * n_atoms, 3), dtype=jnp.float64)
        else:
            R0_packed = jnp.tile(
                jnp.asarray(R0_initial, dtype=jnp.float64)[None, :, :], (B, 1, 1)
            ).reshape(B * n_atoms, 3)

        @jax.jit
        def model_apply_batched(positions):
            return model.apply(
                params,
                atomic_numbers=Z_batched,
                positions=positions,
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=batch_segments,
                batch_size=B,
                batch_mask=batch_mask,
                atom_mask=atom_mask,
            )

        def energy_sum_fn(position, **kwargs):
            out = model_apply_batched(position)
            return jnp.sum(out["energy"])

        _, shift = space.free()
        init_fn, apply_fn = simulate.nvt_nose_hoover(energy_sum_fn, shift, dt, kT)
        apply_fn = jax.jit(apply_fn)

        key = jax.random.PRNGKey(42)
        state = init_fn(key, R0_packed, mass=masses_batched)

        positions_jaxmd = [np.array(R0_packed).reshape(B, n_atoms, 3)]
        for i in range(args.nsteps_jaxmd):
            state = apply_fn(state)
            if (i + 1) % args.printfreq == 0:
                pos_batch = np.array(state.position).reshape(B, n_atoms, 3)
                positions_jaxmd.append(pos_batch)
                T_curr = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / K_B)
                E_total = float(energy_sum_fn(state.position))
                print(f"  step {i+1:4d}  E_total={E_total:.4f} eV  T={T_curr:.1f} K")

        for replica_id in range(B):
            traj_jaxmd = args.output_dir / f"physnet_jaxmd_replica{replica_id}.xyz"
            for frame_idx, pos_batch in enumerate(positions_jaxmd):
                at = Atoms(numbers=Z, positions=pos_batch[replica_id])
                write(traj_jaxmd, at, append=(frame_idx > 0))
            print(f"  Saved JAX-MD trajectory: {traj_jaxmd}")
    else:
        # Single replica
        print("\n--- JAX-MD: NVT Nose-Hoover ---")
        R0 = jnp.asarray(R0_initial, dtype=jnp.float64)
        Z_jnp = jnp.array(Z, dtype=jnp.int32)
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_idx = jnp.array(dst_idx, dtype=jnp.int32)
        src_idx = jnp.array(src_idx, dtype=jnp.int32)

        @jax.jit
        def model_apply(positions):
            return model.apply(
                params,
                atomic_numbers=Z_jnp,
                positions=positions,
                dst_idx=dst_idx,
                src_idx=src_idx,
            )

        def energy_fn(position, **kwargs):
            out = model_apply(position)
            return jnp.squeeze(out["energy"])

        # Sanity check: initial energy should be finite
        E0 = float(energy_fn(R0))
        if not np.isfinite(E0):
            print(f"  ERROR: Initial energy is {E0} (non-finite). Check geometry.", file=sys.stderr)
            return 1
        print(f"  Initial E={E0:.4f} eV")

        _, shift = space.free()
        init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)
        apply_fn = jax.jit(apply_fn)

        masses_jnp = jnp.asarray(masses, dtype=jnp.float64)
        key = jax.random.PRNGKey(42)
        state = init_fn(key, R0, mass=masses_jnp)

        positions_jaxmd = [R0]
        for i in range(args.nsteps_jaxmd):
            state = apply_fn(state)
            if (i + 1) % args.printfreq == 0:
                positions_jaxmd.append(np.array(state.position))
                T_curr = float(quantity.temperature(momentum=state.momentum, mass=state.mass) / K_B)
                E = float(energy_fn(state.position))
                print(f"  step {i+1:4d}  E={E:.4f} eV  T={T_curr:.1f} K")

        traj_jaxmd = args.output_dir / "physnet_jaxmd.xyz"
        for i, R_frame in enumerate(positions_jaxmd):
            at = Atoms(numbers=Z, positions=np.asarray(R_frame))
            write(traj_jaxmd, at, append=(i > 0))
        print(f"  Saved JAX-MD trajectory: {traj_jaxmd}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

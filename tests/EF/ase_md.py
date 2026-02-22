"""
ASE Molecular Dynamics with the Electric Field calculator.

Runs NVT (Langevin) or NVE (VelocityVerlet) MD using AseCalculatorEF.

Usage:
    python ase_md.py --params params.json --data data-full.npz --steps 1000 --dt 0.5
    python ase_md.py --params params.json --data data-full.npz --thermostat langevin --temperature 300
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

import argparse
import sys
from types import SimpleNamespace
import numpy as np
import ase
import ase.io as ase_io
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ase_calc_EF import AseCalculatorEF


def get_args(**kwargs):
    """
    Get configuration arguments. Works both from command line and notebooks.
    
    In notebooks, you can override defaults by passing keyword arguments:
        args = get_args(params="params.json", steps=1000, temperature=300)
    
    From command line, use argparse flags as before.
    """
    # Default values
    defaults = {
        "params": "params.json",
        "config": None,
        "data": "data-full.npz",
        "xyz": None,
        "index": 0,
        "electric_field": None,
        "thermostat": "nve",
        "temperature": 200.0,
        "friction": 0.01,
        "dt": 0.5,
        "steps": 10000,
        "log_interval": 100,
        "traj_interval": 1,
        "output": "md_trajectory.traj",
        "seed": 42,
        "save_charges": False,
        "optimize": False,
        "optimizer": "bfgs",
        "fmax": 0.05,
        "opt_steps": 2000,
        "maxstep": 0.04,
        "n_replicas": 1,
    }
    
    # Check if we're in a notebook/IPython environment
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    # If kwargs are provided, always use notebook mode
    if kwargs:
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)
    
    # Check if any command line arguments look like our flags (start with --)
    has_flag_args = any(arg.startswith('--') for arg in sys.argv[1:])
    
    # If command line arguments are provided AND we're not in a notebook, use argparse
    if has_flag_args and not in_notebook:
        parser = argparse.ArgumentParser(description="Run MD with AseCalculatorEF")
        parser.add_argument("--params", type=str, default=defaults["params"],
                           help="Path to parameters JSON file")
        parser.add_argument("--config", type=str, default=defaults["config"],
                           help="Path to config JSON file (auto-detected from params UUID)")
        parser.add_argument("--data", type=str, default=defaults["data"],
                           help="Path to dataset NPZ file (to extract initial geometry)")
        parser.add_argument("--xyz", type=str, default=defaults["xyz"],
                           help="Path to XYZ file for initial geometry (overrides --data)")
        parser.add_argument("--index", type=int, default=defaults["index"],
                           help="Index of structure in dataset to use as starting geometry")
        parser.add_argument("--electric-field", type=float, nargs=3, default=defaults["electric_field"],
                           help="Electric field vector (Ef_x, Ef_y, Ef_z). "
                                "If omitted, uses dataset value; pass '0 0 0' for zero field.")
        parser.add_argument("--thermostat", type=str, default=defaults["thermostat"],
                           choices=["langevin", "nve"],
                           help="Thermostat type: 'langevin' (NVT) or 'nve' (NVE)")
        parser.add_argument("--temperature", type=float, default=defaults["temperature"],
                           help="Temperature in Kelvin (for Langevin thermostat)")
        parser.add_argument("--friction", type=float, default=defaults["friction"],
                           help="Friction coefficient for Langevin thermostat (1/fs)")
        parser.add_argument("--dt", type=float, default=defaults["dt"],
                           help="Time step in femtoseconds")
        parser.add_argument("--steps", type=int, default=defaults["steps"],
                           help="Number of MD steps")
        parser.add_argument("--log-interval", type=int, default=defaults["log_interval"],
                           help="Log every N steps")
        parser.add_argument("--traj-interval", type=int, default=defaults["traj_interval"],
                           help="Save trajectory every N steps")
        parser.add_argument("--output", type=str, default=defaults["output"],
                           help="Output trajectory file (ASE .traj format)")
        parser.add_argument("--seed", type=int, default=defaults["seed"],
                           help="Random seed for initial velocities")
        parser.add_argument("--save-charges", action="store_true",
                           help="Save ML atomic charges per frame (slower, for VCD)")
        parser.add_argument("--optimize", action="store_true",
                           help="Geometry-optimise before starting MD")
        parser.add_argument("--optimizer", choices=["bfgs", "fire"], default=defaults["optimizer"],
                           help="Optimiser: 'bfgs' or 'fire' (default fire, more robust for ML)")
        parser.add_argument("--fmax", type=float, default=defaults["fmax"],
                           help="Convergence criterion: max force (eV/Å)")
        parser.add_argument("--opt-steps", type=int, default=defaults["opt_steps"],
                           help="Max optimisation steps")
        parser.add_argument("--maxstep", type=float, default=defaults["maxstep"],
                           help="Max step size in Å (default 0.04; ASE default 0.2)")
        parser.add_argument("--n-replicas", type=int, default=defaults["n_replicas"],
                           help="Run N independent replicas in a single GPU-batched "
                                "JIT-compiled loop (requires JAX). Default 1 = standard ASE MD.")
        
        args = parser.parse_args()
        return SimpleNamespace(
            params=args.params,
            config=args.config,
            data=args.data,
            xyz=args.xyz,
            index=args.index,
            electric_field=args.electric_field,
            thermostat=args.thermostat,
            temperature=args.temperature,
            friction=args.friction,
            dt=args.dt,
            steps=args.steps,
            log_interval=args.log_interval,
            traj_interval=args.traj_interval,
            output=args.output,
            seed=args.seed,
            save_charges=args.save_charges,
            optimize=args.optimize,
            optimizer=args.optimizer,
            fmax=args.fmax,
            opt_steps=args.opt_steps,
            maxstep=args.maxstep,
            n_replicas=args.n_replicas,
        )
    
    # Otherwise, use notebook mode (defaults only)
    return SimpleNamespace(**defaults)


def main_batched(args):
    """Run B independent MD replicas in one JIT-compiled GPU batch.

    All replicas share the same molecule and electric field but evolve
    from independent Maxwell-Boltzmann velocity draws.  Force evaluations
    for all replicas are fused into a single model forward pass, giving
    near-linear GPU throughput scaling with the number of replicas.

    Outputs one ASE .traj file per replica:
        <output>_replica0.traj, <output>_replica1.traj, ...
    """
    import jax
    import jax.numpy as jnp
    import e3x
    import time
    import functools
    from ase.data import atomic_masses as _ase_masses
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase_calc_EF import load_params, load_config
    from training import MessagePassingModel

    BOLTZMANN_EV = 8.617333262e-5
    AMU_TO_EV_FS2_ANG2 = 103.6427

    B = args.n_replicas

    print("=" * 70)
    print(f"  Batched MD — {B} non-interacting replicas (JIT-compiled)")
    print("=" * 70)
    print(f"  JAX devices : {jax.devices()}")
    print(f"  Backend     : {jax.default_backend()}")

    # ---- load initial geometry ------------------------------------------
    if args.xyz is not None:
        atoms_init = ase_io.read(args.xyz)
        Z = jnp.asarray(atoms_init.get_atomic_numbers(), dtype=jnp.int32)
        R_single = jnp.asarray(atoms_init.get_positions(), dtype=jnp.float32)
    else:
        dataset = np.load(args.data, allow_pickle=True)
        Z = jnp.asarray(dataset["Z"][args.index].astype(int), dtype=jnp.int32)
        R_single = jnp.asarray(dataset["R"][args.index].astype(float),
                                dtype=jnp.float32)
        if R_single.ndim == 3:
            R_single = R_single.squeeze(0)

    N = len(Z)
    print(f"  Atoms       : {N}")
    print(f"  Replicas    : {B}")

    # ---- electric field -------------------------------------------------
    if args.electric_field is not None:
        Ef = jnp.asarray(args.electric_field, dtype=jnp.float32)
    elif args.xyz is None:
        dataset = np.load(args.data, allow_pickle=True)
        Ef = (jnp.asarray(dataset["Ef"][args.index].astype(float),
                           dtype=jnp.float32)
              if "Ef" in dataset.files
              else jnp.zeros(3, dtype=jnp.float32))
    else:
        Ef = jnp.zeros(3, dtype=jnp.float32)
    print(f"  Ef          : {np.asarray(Ef)}")

    # ---- build model ----------------------------------------------------
    params_path = Path(args.params)
    params = load_params(params_path)

    config_path = args.config
    if config_path is None:
        if params_path.stem.startswith("params-") and len(params_path.stem) > 7:
            uuid_part = params_path.stem[7:]
            cand = params_path.parent / f"config-{uuid_part}.json"
            if cand.exists():
                config_path = str(cand)
            elif (params_path.parent / "config.json").exists():
                config_path = str(params_path.parent / "config.json")

    model_keys = {
        "features", "max_degree", "num_iterations", "num_basis_functions",
        "cutoff", "max_atomic_number", "include_pseudotensors",
        "dipole_field_coupling", "field_scale",
    }
    if config_path is not None:
        config = load_config(config_path)
        if "model" in config and isinstance(config["model"], dict):
            mc = {k: v for k, v in config["model"].items() if k in model_keys}
        elif "model_config" in config:
            mc = {k: v for k, v in config["model_config"].items()
                  if k in model_keys}
        else:
            mc = {k: v for k, v in config.items() if k in model_keys}
    else:
        mc = dict(features=64, max_degree=2, num_iterations=2,
                  num_basis_functions=64, cutoff=10.0,
                  max_atomic_number=55, include_pseudotensors=True)

    model = MessagePassingModel(**mc)
    print(f"  Model       : {mc}")

    # ---- batched graph (B non-interacting copies) -----------------------
    # Each replica gets its own atom indices; batch_segments assigns atoms
    # to replicas so the model's internal aggregations stay per-replica.
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(N)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

    batch_segments = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)
    offsets = jnp.arange(B, dtype=jnp.int32) * N
    dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
    src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

    Z_batched = jnp.tile(Z[None, :], (B, 1))       # (B, N)
    Ef_batched = jnp.tile(Ef[None, :], (B, 1))      # (B, 3)

    # ---- JIT-compiled batched force function ----------------------------
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def model_apply(params, atomic_numbers, positions, Ef_arg,
                    dst_idx_flat, src_idx_flat, batch_segments, batch_size,
                    dst_idx=None, src_idx=None):
        return model.apply(
            params, atomic_numbers, positions, Ef_arg,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx)

    @jax.jit
    def force_fn(positions):
        """(B, N, 3) -> energy (B,), forces (B, N, 3), dipole (B, 3)"""
        def energy_fn(pos):
            energy, dipole = model_apply(
                params, Z_batched, pos, Ef_batched,
                dst_idx_flat, src_idx_flat, batch_segments, B,
                dst_idx, src_idx)
            return -jnp.sum(energy), (energy, dipole)
        (_, (energy, dipole)), forces = jax.value_and_grad(
            energy_fn, has_aux=True)(positions)
        return energy, forces, dipole

    # ---- warm-up (JIT compile) ------------------------------------------
    R0 = jnp.tile(R_single[None, :, :], (B, 1, 1))  # (B, N, 3)
    print(f"\n  Warming up batched force function ...")
    t0 = time.perf_counter()
    E_w, F_w, mu_w = force_fn(R0)
    E_w.block_until_ready()
    print(f"  JIT compiled in {time.perf_counter() - t0:.2f} s")
    print(f"  E[0] = {float(E_w[0]):.6f} eV,  "
          f"max|F| = {float(jnp.max(jnp.abs(F_w))):.6f} eV/Å")

    # ---- optional geometry optimisation (single copy, then replicate) ---
    if args.optimize:
        from ase.optimize import BFGS, FIRE
        opt_cls = FIRE if args.optimizer == "fire" else BFGS
        print(f"\n  Geometry optimisation ({args.optimizer.upper()}, "
              f"fmax={args.fmax} eV/Å) ...")
        opt_atoms = ase.Atoms(numbers=np.asarray(Z),
                              positions=np.asarray(R_single))
        opt_atoms.info["electric_field"] = np.asarray(Ef)
        opt_calc = AseCalculatorEF(params_path=str(params_path),
                                   config_path=config_path)
        opt_atoms.calc = opt_calc
        opt_traj = str(Path(args.output).with_suffix(".opt.traj"))
        opt = opt_cls(opt_atoms, trajectory=opt_traj, logfile="-",
                      maxstep=args.maxstep)
        opt.run(fmax=args.fmax, steps=args.opt_steps)
        R_single = jnp.asarray(opt_atoms.get_positions(), dtype=jnp.float32)
        R0 = jnp.tile(R_single[None, :, :], (B, 1, 1))
        del opt_calc
        E_opt, _, _ = force_fn(R0)
        print(f"  Optimised E = {float(E_opt[0]):.6f} eV")

    # ---- initial velocities (independent per replica) -------------------
    masses_amu = jnp.asarray(_ase_masses[np.asarray(Z)], dtype=jnp.float32)
    inv_masses = 1.0 / (masses_amu[:, None] * AMU_TO_EV_FS2_ANG2)  # (N, 1)

    rng = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(rng, B + 1)
    rng, vel_keys = keys[0], keys[1:]

    kT = BOLTZMANN_EV * args.temperature
    sigma_v = jnp.sqrt(kT / (masses_amu[:, None] * AMU_TO_EV_FS2_ANG2))
    total_mass = jnp.sum(masses_amu)

    def _sample_v(key):
        v = sigma_v * jax.random.normal(key, shape=(N, 3))
        v_com = jnp.sum(masses_amu[:, None] * v, axis=0) / total_mass
        return v - v_com[None, :]

    V0 = jax.vmap(_sample_v)(vel_keys)  # (B, N, 3)

    Ekin0 = 0.5 * jnp.sum(
        masses_amu[None, :, None] * AMU_TO_EV_FS2_ANG2 * V0**2, axis=(1, 2))
    T0 = 2.0 * Ekin0 / (3.0 * N * BOLTZMANN_EV)
    print(f"\n  Initial T   : {float(T0.mean()):.1f} K mean  "
          f"[{float(T0.min()):.1f}, {float(T0.max()):.1f}]")

    # ---- integration parameters -----------------------------------------
    dt = args.dt
    si = args.traj_interval
    n_saved = args.steps // si + 1

    print(f"\n  Thermostat  : {args.thermostat}")
    print(f"  dt          : {dt} fs")
    print(f"  Steps       : {args.steps}")
    print(f"  Save every  : {si} steps -> {n_saved} frames")
    print(f"  Total time  : {args.steps * dt:.1f} fs "
          f"({args.steps * dt / 1000:.2f} ps)")
    print()

    # ---- initial forces -------------------------------------------------
    E0, F0, mu0 = force_fn(R0)

    # ---- save buffers ---------------------------------------------------
    R_buf  = jnp.zeros((n_saved, B, N, 3), dtype=jnp.float32).at[0].set(R0)
    V_buf  = jnp.zeros((n_saved, B, N, 3), dtype=jnp.float32).at[0].set(V0)
    E_buf  = jnp.zeros((n_saved, B),       dtype=jnp.float32).at[0].set(E0)
    mu_buf = jnp.zeros((n_saved, B, 3),    dtype=jnp.float32).at[0].set(mu0)

    def _save_cond(should, buf, idx, val):
        return jax.lax.cond(should,
                            lambda b: b.at[idx].set(val),
                            lambda b: b, buf)

    # ---- integration loop -----------------------------------------------
    if args.thermostat == "langevin":
        gamma = args.friction

        def body_fn(i, carry):
            (R, V, F), (sR, sV, sE, sM), step_rng = carry
            step_rng, key = jax.random.split(step_rng)
            # BAOAB Langevin splitting
            V = V + 0.5 * dt * F * inv_masses
            R = R + 0.5 * dt * V
            c1 = jnp.exp(-gamma * dt)
            ns = jnp.sqrt(kT * inv_masses * (1.0 - c1**2))
            V = c1 * V + ns * jax.random.normal(key, V.shape)
            R = R + 0.5 * dt * V
            E, Fn, mu = force_fn(R)
            V = V + 0.5 * dt * Fn * inv_masses
            fi = (i + 1) // si
            sv = ((i + 1) % si == 0)
            return ((R, V, Fn),
                    (_save_cond(sv, sR, fi, R), _save_cond(sv, sV, fi, V),
                     _save_cond(sv, sE, fi, E), _save_cond(sv, sM, fi, mu)),
                    step_rng)

        rng, md_key = jax.random.split(rng)
        carry0 = ((R0, V0, F0), (R_buf, V_buf, E_buf, mu_buf), md_key)
        label = "Langevin"
    else:
        def body_fn(i, carry):
            (R, V, F), (sR, sV, sE, sM) = carry
            # Velocity Verlet
            V = V + 0.5 * dt * F * inv_masses
            R = R + dt * V
            E, Fn, mu = force_fn(R)
            V = V + 0.5 * dt * Fn * inv_masses
            fi = (i + 1) // si
            sv = ((i + 1) % si == 0)
            return ((R, V, Fn),
                    (_save_cond(sv, sR, fi, R), _save_cond(sv, sV, fi, V),
                     _save_cond(sv, sE, fi, E), _save_cond(sv, sM, fi, mu)))

        carry0 = ((R0, V0, F0), (R_buf, V_buf, E_buf, mu_buf))
        label = "NVE"

    print(f"  JIT-compiling batched {label} "
          f"({args.steps} steps x {B} replicas) ...")
    t_start = time.perf_counter()
    final = jax.lax.fori_loop(0, args.steps, body_fn, carry0)
    if args.thermostat == "langevin":
        _, (R_traj, V_traj, E_traj, mu_traj), _ = final
    else:
        _, (R_traj, V_traj, E_traj, mu_traj) = final
    E_traj.block_until_ready()
    t_end = time.perf_counter()

    wall = t_end - t_start
    total_steps = args.steps * B
    ns_per_day = (args.steps * dt * 1e-6) / (wall / 86400.0)

    print(f"\n  Wall time        : {wall:.2f} s")
    print(f"  Steps (total)    : {total_steps} ({args.steps} x {B})")
    print(f"  Aggregate rate   : {total_steps / wall:.0f} steps/s")
    print(f"  Throughput       : {ns_per_day * B:.3f} ns/day (total)")
    print(f"                   : {ns_per_day:.3f} ns/day (per replica)")

    # ---- summary statistics ---------------------------------------------
    E_np = np.asarray(E_traj)           # (n_saved, B)
    V_np = np.asarray(V_traj)           # (n_saved, B, N, 3)
    masses_np = np.asarray(masses_amu)   # (N,)

    Ekin = 0.5 * np.sum(
        masses_np[None, None, :, None] * AMU_TO_EV_FS2_ANG2 * V_np**2,
        axis=(2, 3))                     # (n_saved, B)
    T_arr = 2.0 * Ekin / (3.0 * N * BOLTZMANN_EV)
    Etot = E_np + Ekin

    pi = max(1, n_saved // 20)
    print(f"\n  {'frame':>6s} {'time(fs)':>10s}", end="")
    for b in range(min(B, 4)):
        print(f" {'E_pot['+str(b)+'](eV)':>14s} {'T['+str(b)+'](K)':>8s}", end="")
    if B > 4:
        print("  ...", end="")
    print()
    print("  " + "-" * (18 + min(B, 4) * 24))
    for i in range(0, n_saved, pi):
        t_fs = i * si * dt
        print(f"  {i:6d} {t_fs:10.1f}", end="")
        for b in range(min(B, 4)):
            print(f" {E_np[i, b]:14.6f} {T_arr[i, b]:8.1f}", end="")
        print()
    if (n_saved - 1) % pi != 0:
        i = n_saved - 1
        t_fs = i * si * dt
        print(f"  {i:6d} {t_fs:10.1f}", end="")
        for b in range(min(B, 4)):
            print(f" {E_np[i, b]:14.6f} {T_arr[i, b]:8.1f}", end="")
        print()

    print(f"\n  T range (all)    : {T_arr.min():.1f} - {T_arr.max():.1f} K "
          f"(mean {T_arr.mean():.1f} K)")

    # ---- save per-replica trajectories ----------------------------------
    R_np = np.asarray(R_traj)           # (n_saved, B, N, 3)
    mu_np = np.asarray(mu_traj)          # (n_saved, B, 3)
    Z_np = np.asarray(Z)
    Ef_np = np.asarray(Ef)

    out = Path(args.output)
    stem, sfx = out.stem, (out.suffix or ".traj")
    print()
    for b in range(B):
        fpath = out.parent / f"{stem}_replica{b}{sfx}"
        traj = Trajectory(str(fpath), "w")
        for i in range(n_saved):
            a = ase.Atoms(numbers=Z_np, positions=R_np[i, b])
            v_ase = V_np[i, b] * np.sqrt(AMU_TO_EV_FS2_ANG2)
            a.set_momenta(masses_np[:, None] * v_ase)
            a.info["electric_field"] = Ef_np
            a.info["ml_dipole"] = mu_np[i, b]
            a.info["step"] = i * si
            a.info["time_fs"] = i * si * dt
            a.info["replica"] = b
            sp = SinglePointCalculator(a, energy=float(E_np[i, b]))
            sp.results["dipole"] = mu_np[i, b]
            a.calc = sp
            traj.write(a)
        traj.close()
        print(f"  Replica {b:3d} -> {fpath} ({n_saved} frames)")

    print(f"\n{'=' * 70}")
    print(f"  Batched MD complete — {B} trajectories x {n_saved} frames saved.")
    print(f"{'=' * 70}")


def main(args=None):
    if args is None:
        args = get_args()
    if getattr(args, "n_replicas", 1) > 1:
        return main_batched(args)
    """Run molecular dynamics simulation."""
    print("=" * 60)
    print("ASE Molecular Dynamics with Electric Field Model")
    print("=" * 60)

    # --- Load or build initial geometry ---
    if args.xyz is not None:
        print(f"\nLoading initial geometry from {args.xyz}...")
        atoms = ase_io.read(args.xyz)
    else:
        print(f"\nLoading initial geometry from {args.data} (index={args.index})...")
        dataset = np.load(args.data, allow_pickle=True)
        Z = dataset["Z"][args.index]
        R = dataset["R"][args.index]
        if R.ndim == 3 and R.shape[0] == 1:
            R = R.squeeze(0)
        atoms = ase.Atoms(numbers=Z, positions=R)

    # Set electric field
    if args.electric_field is not None:
        Ef = np.array(args.electric_field, dtype=np.float64)
        print(f"  Electric field (CLI): {Ef}")
    elif args.xyz is None:
        dataset = np.load(args.data, allow_pickle=True)
        if "Ef" in dataset.files:
            Ef = np.array(dataset["Ef"][args.index], dtype=np.float64)
            print(f"  Electric field (dataset): {Ef}")
        else:
            Ef = np.zeros(3, dtype=np.float64)
            print(f"  Electric field (default): {Ef}")
    else:
        Ef = np.zeros(3, dtype=np.float64)
        print(f"  Electric field (default): {Ef}")
    atoms.info['electric_field'] = Ef

    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Atomic numbers: {atoms.get_atomic_numbers()}")
    print(f"  Electric field: {Ef}")

    # --- Create calculator ---
    print(f"\nInitializing calculator from {args.params}...")
    calc = AseCalculatorEF(
        params_path=args.params,
        config_path=args.config,
        electric_field=Ef,
    )
    atoms.calc = calc

    # --- Initial energy/forces ---
    print("\nComputing initial properties...")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"  Initial energy: {energy:.6f} eV ({energy * 23.06035:.4f} kcal/mol)")
    print(f"  Max force: {np.max(np.abs(forces)):.6f} eV/A")

    # --- Optional geometry optimisation before MD ---
    if args.optimize:
        from ase.optimize import BFGS, FIRE
        opt_cls = FIRE if args.optimizer == 'fire' else BFGS
        print(f"\nOptimising geometry ({args.optimizer.upper()}, fmax={args.fmax} eV/Å, "
              f"max {args.opt_steps} steps, maxstep={args.maxstep} Å) ...")
        opt_traj = str(Path(args.output).with_suffix('.opt.traj'))
        opt = opt_cls(atoms, trajectory=opt_traj, logfile='-',
                      maxstep=args.maxstep)
        opt.run(fmax=args.fmax, steps=args.opt_steps)
        fmax_final = np.max(np.abs(atoms.get_forces()))
        if fmax_final <= args.fmax:
            print(f"  Converged in {opt.nsteps} steps.")
        else:
            print(f"  WARNING: not converged after {opt.nsteps} steps  "
                  f"(max |F| = {fmax_final:.6f} eV/Å)")
            print(f"  Consider increasing --opt-steps or relaxing --fmax.")
        energy = atoms.get_potential_energy()
        print(f"  Optimised energy: {energy:.6f} eV")
        print(f"  Optimised max|F|: {np.max(np.abs(atoms.get_forces())):.6f} eV/Å")
        print(f"  Opt trajectory  : {opt_traj}")

    # --- Initialize velocities ---
    print(f"\nInitializing velocities at T={args.temperature} K (seed={args.seed})...")
    np.random.seed(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    # --- Set up integrator ---
    dt = args.dt * units.fs

    if args.thermostat == "langevin":
        print(f"\nUsing Langevin thermostat: T={args.temperature} K, friction={args.friction} 1/fs, dt={args.dt} fs")
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=args.temperature,
            friction=args.friction / units.fs,
        )
    else:
        print(f"\nUsing NVE (VelocityVerlet): dt={args.dt} fs")
        dyn = VelocityVerlet(atoms, timestep=dt)

    # --- Prepare ASE trajectory output ---
    output_path = Path(args.output)
    traj = Trajectory(str(output_path), 'w', atoms)

    # --- MD loop ---
    print(f"\nRunning {args.steps} MD steps...")
    print(f"{'Step':>8s} {'Time(fs)':>10s} {'E_pot(eV)':>12s} {'E_kin(eV)':>12s} {'E_tot(eV)':>12s} {'T(K)':>8s} {'MaxF(eV/A)':>12s}")
    print("-" * 80)

    def print_status():
        """Print MD status at current step."""
        step = dyn.nsteps
        time_fs = step * args.dt
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        e_tot = e_pot + e_kin
        temp = e_kin / (1.5 * units.kB * len(atoms))
        max_force = np.max(np.abs(atoms.get_forces()))
        print(f"{step:8d} {time_fs:10.2f} {e_pot:12.6f} {e_kin:12.6f} {e_tot:12.6f} {temp:8.1f} {max_force:12.6f}")

    # --- Property-saving callback (dipole always, charges optional) ---
    def save_properties():
        """Copy model predictions into atoms.info so they persist in .traj."""
        results = getattr(atoms.calc, 'results', {})
        if 'dipole' in results:
            atoms.info['ml_dipole'] = np.array(results['dipole'])
        if args.save_charges:
            try:
                q, mu_at = calc.get_atomic_charges(atoms)
                atoms.arrays['ml_charges'] = q
                atoms.arrays['ml_atomic_dipoles'] = mu_at
            except Exception:
                pass

    # Log and save initial state
    print_status()
    save_properties()
    traj.write()

    # Attach callbacks — save_properties BEFORE traj.write
    dyn.attach(print_status, interval=args.log_interval)
    dyn.attach(save_properties, interval=args.traj_interval)
    dyn.attach(traj.write, interval=args.traj_interval)

    # Run MD
    dyn.run(args.steps)

    # Final status
    print("-" * 80)
    print_status()

    # Close trajectory file
    traj.close()
    n_frames = len(Trajectory(str(output_path)))
    print(f"\nTrajectory saved: {output_path} ({n_frames} frames)")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("MD Simulation Complete!")
    print(f"{'=' * 60}")
    print(f"  Total steps: {args.steps}")
    print(f"  Total time: {args.steps * args.dt:.1f} fs")
    print(f"  Frames saved: {n_frames}")
    print(f"  Output: {output_path}")

    final_energy = atoms.get_potential_energy()
    final_temp = atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))
    print(f"  Final energy: {final_energy:.6f} eV")
    print(f"  Final temperature: {final_temp:.1f} K")


if __name__ == "__main__":
    args = get_args()
    main(args)

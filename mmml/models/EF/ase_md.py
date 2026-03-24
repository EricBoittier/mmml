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
from pathlib import Path

from mmml.models.EF.ase_calc_EF import AseCalculatorEF


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
        "ramp_field_axis": None,
        "ramp_field_peak": None,
        "ramp_field_start": None,
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
        parser.add_argument("--ramp-field-axis", type=str, choices=["x", "y", "z"],
                           default=defaults["ramp_field_axis"],
                           help="Enable electric-field ramp mode on this axis.")
        parser.add_argument("--ramp-field-peak", type=float, default=defaults["ramp_field_peak"],
                           help="Peak electric-field value reached halfway through MD "
                                "for the selected ramp axis.")
        parser.add_argument("--ramp-field-start", type=float, default=defaults["ramp_field_start"],
                           help="Optional ramp start value on selected axis. "
                                "If omitted, uses initial field component.")
        
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
            ramp_field_axis=args.ramp_field_axis,
            ramp_field_peak=args.ramp_field_peak,
            ramp_field_start=args.ramp_field_start,
        )
    
    # Otherwise, use notebook mode (defaults only)
    return SimpleNamespace(**defaults)


def _ef_md_active_columns(zb: np.ndarray) -> int:
    """Columns [0, n) covering every Z>0 in each row (trailing pad stripped)."""
    zb = np.asarray(zb, dtype=np.int32)
    if zb.ndim == 1:
        pos = np.where(zb > 0)[0]
        return int(pos.max()) + 1 if len(pos) else 1
    ncols = 1
    for b in range(zb.shape[0]):
        pos = np.where(zb[b] > 0)[0]
        if len(pos):
            ncols = max(ncols, int(pos.max()) + 1)
    return max(ncols, 1)


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
    from mmml.models.EF.ase_calc_EF import load_params, load_config
    from mmml.models.EF.training import MessagePassingModel
    from mmml.models.EF.model_functions import energy_and_forces

    BOLTZMANN_EV = 8.617333262e-5
    AMU_TO_EV_FS2_ANG2 = 103.6427

    B = args.n_replicas

    print("=" * 70)
    print(f"  Batched MD — {B} non-interacting replicas (JIT-compiled)")
    print("=" * 70)
    print(f"  JAX devices : {jax.devices()}")
    print(f"  Backend     : {jax.default_backend()}")

    # ---- load initial geometry ------------------------------------------
    # XYZ → single geometry, tiled to all replicas.
    # Dataset with >= B structures → each replica gets a distinct geometry.
    # Dataset with < B structures → single geometry, tiled.
    R_multi = None  # (B, N, 3) when distinct geometries are available
    dataset = None
    if args.xyz is not None:
        atoms_init = ase_io.read(args.xyz)
        Z = jnp.asarray(atoms_init.get_atomic_numbers(), dtype=jnp.int32)
        R_single = jnp.asarray(atoms_init.get_positions(), dtype=jnp.float32)
        Z_batched = jnp.tile(Z[None, :], (B, 1))
        n_dataset = 0
    else:
        dataset = np.load(args.data, allow_pickle=True)
        n_dataset = len(dataset["R"])
        z_arr = np.asarray(dataset["Z"])
        if z_arr.ndim == 2 and n_dataset >= args.index + B:
            Z_batched_np = z_arr[args.index: args.index + B].astype(np.int32)
        elif z_arr.ndim == 1:
            z0 = z_arr.reshape(-1)
            Z_batched_np = np.tile(z0[np.newaxis, :], (B, 1))
        else:
            z0 = np.asarray(z_arr[args.index]).reshape(-1)
            Z_batched_np = np.tile(z0[np.newaxis, :], (B, 1))
        Z_batched = jnp.asarray(Z_batched_np, dtype=jnp.int32)
        Z = Z_batched[0]

        if n_dataset >= args.index + B:
            R_all = np.asarray(dataset["R"][args.index:args.index + B],
                               dtype=np.float64)
            R_multi = jnp.asarray(R_all, dtype=jnp.float32)
            if R_multi.ndim == 4:
                R_multi = R_multi.squeeze(1)
            R_single = R_multi[0]
        else:
            R_single = jnp.asarray(dataset["R"][args.index].astype(float),
                                    dtype=jnp.float32)
            if R_single.ndim == 3:
                R_single = R_single.squeeze(0)

    # Drop trailing padded atom slots (Z<=0) so the e3x graph matches
    # unpadded ASE / n_replicas=1 physics. Full-width padded batches
    # include ghost pairs in message passing and extra atomic energies,
    # which shifts the potential vs single-molecule MD.
    zb_np = np.asarray(Z_batched)
    n_active = _ef_md_active_columns(zb_np)
    n_full = zb_np.shape[1]
    if n_active < n_full:
        print(
            f"  Trim padding   : {n_full} → {n_active} atom slots "
            f"(trailing Z<=0 columns removed for graph / energy consistency)"
        )
    Z_batched = jnp.asarray(zb_np[:, :n_active], dtype=jnp.int32)
    Z = Z_batched[0]
    if R_multi is not None:
        R_multi = R_multi[:, :n_active, :]
        R_single = R_multi[0]
    else:
        R_single = R_single[:n_active]

    N = int(Z_batched.shape[1])
    print(f"  Atoms       : {N}")
    print(f"  Replicas    : {B}")
    if R_multi is not None:
        print(f"  Init geoms  : {B} distinct (dataset indices "
              f"{args.index}..{args.index + B - 1})")
    else:
        src = "XYZ file" if args.xyz else "dataset"
        print(f"  Init geoms  : single geometry from {src}, tiled")

    # ---- electric field -------------------------------------------------
    if args.electric_field is not None:
        Ef_batched = jnp.tile(
            jnp.asarray(args.electric_field, dtype=jnp.float32)[None, :], (B, 1)
        )
    elif args.xyz is None and dataset is not None and "Ef" in dataset.files:
        ef_arr = np.asarray(dataset["Ef"], dtype=np.float64)
        if ef_arr.ndim == 2 and len(ef_arr) >= args.index + B:
            Ef_batched = jnp.asarray(
                ef_arr[args.index: args.index + B], dtype=jnp.float32
            )
        else:
            ef0 = ef_arr[args.index] if ef_arr.ndim > 1 else ef_arr
            Ef_batched = jnp.tile(jnp.asarray(ef0, dtype=jnp.float32)[None, :], (B, 1))
    else:
        Ef_batched = jnp.zeros((B, 3), dtype=jnp.float32)
    print(f"  Ef (replica 0): {np.asarray(Ef_batched[0])}")

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

    # Z_batched / Ef_batched set above (per-replica when dataset has rows)

    # ---- JIT-compiled batched force function ----------------------------
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def model_apply(params, atomic_numbers, positions, Ef,
                    dst_idx_flat, src_idx_flat, batch_segments, batch_size,
                    dst_idx=None, src_idx=None):
        return model.apply(
            params, atomic_numbers, positions, Ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx)

    pad_mask = (Z_batched > 0).astype(jnp.float32)

    @jax.jit
    def force_fn(positions):
        """(B, N, 3) -> energy (B,), forces (B, N, 3), dipole (B, 3).

        Forces on padded sites (Z<=0) are zeroed so ghost atoms do not
        receive infinite accelerations from inv_mass blow-ups.
        """
        energy, forces, dipole = energy_and_forces(
            model_apply,
            params,
            atomic_numbers=Z_batched,
            positions=positions,
            Ef=Ef_batched,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=B,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        forces = forces * pad_mask[..., None]
        return energy, forces, dipole

    # ---- warm-up (JIT compile) ------------------------------------------
    if R_multi is not None:
        R0 = R_multi                                    # (B, N, 3)
    else:
        R0 = jnp.tile(R_single[None, :, :], (B, 1, 1)) # (B, N, 3)

    print(f"\n  Warming up batched force function ...")
    t0 = time.perf_counter()
    E_w, F_w, mu_w = force_fn(R0)
    E_w.block_until_ready()
    print(f"  JIT compiled in {time.perf_counter() - t0:.2f} s")
    print(f"  E[0] = {float(E_w[0]):.6f} eV,  "
          f"max|F| = {float(jnp.max(jnp.abs(F_w))):.6f} eV/Å")

    # ---- optional geometry optimisation (single copy, then replicate) ---
    if args.optimize:
        if R_multi is not None:
            print("\n  WARNING: --optimize with distinct starting geometries "
                  "only optimises the first structure and replicates it. "
                  "Pre-optimise your dataset if you need per-replica minima.")
        from ase.optimize import BFGS, FIRE
        opt_cls = FIRE if args.optimizer == "fire" else BFGS
        print(f"\n  Geometry optimisation ({args.optimizer.upper()}, "
              f"fmax={args.fmax} eV/Å) ...")
        opt_atoms = ase.Atoms(numbers=np.asarray(Z),
                              positions=np.asarray(R_single))
        opt_atoms.info["electric_field"] = np.asarray(Ef_batched[0])
        opt_calc = AseCalculatorEF(params_path=str(params_path),
                                   config_path=config_path)
        opt_atoms.calc = opt_calc
        opt_traj = str(Path(args.output).with_suffix(".opt.traj"))
        opt = opt_cls(opt_atoms, trajectory=opt_traj, logfile="-",
                      maxstep=args.maxstep)
        opt.run(fmax=args.fmax, steps=args.opt_steps)
        R_single = jnp.asarray(opt_atoms.get_positions(), dtype=jnp.float32)
        R0 = jnp.tile(R_single[None, :, :], (B, 1, 1))
        R_multi = None
        del opt_calc
        E_opt, _, _ = force_fn(R0)
        print(f"  Optimised E = {float(E_opt[0]):.6f} eV")

    # ---- initial velocities (independent per replica) -------------------
    # Per-replica (B, N) masses; padded atoms (Z<=0) get zero mass / inv_mass
    # so they never receive infinite accelerations from ASE mass[0]==0.
    ref_m_table = jnp.asarray(_ase_masses, dtype=jnp.float32)
    z_clip = jnp.clip(Z_batched, 0, ref_m_table.shape[0] - 1)
    masses_bn = jnp.where(Z_batched > 0, ref_m_table[z_clip], 0.0)
    inv_masses_bn = jnp.where(
        masses_bn[..., None] > 1e-9,
        1.0 / (masses_bn[..., None] * AMU_TO_EV_FS2_ANG2),
        0.0,
    )

    rng = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(rng, B + 1)
    rng, vel_keys = keys[0], keys[1:]

    kT = BOLTZMANN_EV * args.temperature

    def _sample_v(key, masses_row, z_row):
        sigma_v = jnp.sqrt(
            kT / (jnp.maximum(masses_row[:, None], 1e-12) * AMU_TO_EV_FS2_ANG2)
        )
        sigma_v = jnp.where(z_row[:, None] > 0, sigma_v, 0.0)
        v = sigma_v * jax.random.normal(key, shape=(N, 3))
        tm = jnp.sum(jnp.where(z_row > 0, masses_row, 0.0))
        v_com = jnp.sum(
            jnp.where(z_row[:, None] > 0, masses_row[:, None] * v, 0.0), axis=0
        )
        v_com = v_com / jnp.maximum(tm, 1e-12)
        v = jnp.where(z_row[:, None] > 0, v - v_com, 0.0)
        return v

    V0 = jax.vmap(_sample_v, in_axes=(0, 0, 0))(vel_keys, masses_bn, Z_batched)

    Ekin0 = 0.5 * jnp.sum(
        jnp.where(
            Z_batched[..., None] > 0,
            masses_bn[..., None] * AMU_TO_EV_FS2_ANG2 * V0**2,
            0.0,
        ),
        axis=(1, 2),
    )
    n_real = jnp.sum(Z_batched > 0, axis=1).astype(jnp.float32)
    T0 = 2.0 * Ekin0 / (3.0 * jnp.maximum(n_real, 1.0) * BOLTZMANN_EV)
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
            V = V + 0.5 * dt * F * inv_masses_bn
            R = R + 0.5 * dt * V
            c1 = jnp.exp(-gamma * dt)
            ns = jnp.sqrt(kT * inv_masses_bn * (1.0 - c1**2))
            V = c1 * V + ns * jax.random.normal(key, V.shape)
            R = R + 0.5 * dt * V
            E, Fn, mu = force_fn(R)
            V = V + 0.5 * dt * Fn * inv_masses_bn
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
            V = V + 0.5 * dt * F * inv_masses_bn
            R = R + dt * V
            E, Fn, mu = force_fn(R)
            V = V + 0.5 * dt * Fn * inv_masses_bn
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
    Z_np = np.asarray(Z_batched)        # (B, N)
    masses_np = np.asarray(masses_bn)   # (B, N)
    zm = (Z_np > 0).astype(np.float64)

    Ekin = 0.5 * np.sum(
        zm[None, :, :, None]
        * masses_np[None, :, :, None]
        * AMU_TO_EV_FS2_ANG2
        * V_np**2,
        axis=(2, 3),
    )  # (n_saved, B)
    n_real_np = np.maximum(np.sum(Z_np > 0, axis=1), 1).astype(np.float64)
    T_arr = 2.0 * Ekin / (3.0 * n_real_np[None, :] * BOLTZMANN_EV)
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

    # ---- compute atomic charges & dipoles (post-hoc) --------------------
    # Uses Flax's mutable='intermediates' to extract the sow'd values.
    # One batched call per saved frame (all B replicas at once).
    @jax.jit
    def get_charges_batch(positions):
        """(B, N, 3) -> charges (B, N), atomic_dipoles (B, N, 3)"""
        (_energy, _dipole), state = model.apply(
            params, Z_batched, positions, Ef_batched,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=B,
            dst_idx=dst_idx, src_idx=src_idx,
            mutable=['intermediates'])
        intermediates = state.get('intermediates', {})
        charges = intermediates.get('atomic_charges', (None,))[-1]
        at_dip = intermediates.get('atomic_dipoles', (None,))[-1]
        return charges, at_dip

    print(f"\n  Computing atomic charges for {n_saved} frames x {B} replicas ...")
    charges_all = np.zeros((n_saved, B, N), dtype=np.float32)
    at_dipoles_all = np.zeros((n_saved, B, N, 3), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n_saved):
        q, mu_at = get_charges_batch(R_traj[i])
        charges_all[i] = np.asarray(q)
        at_dipoles_all[i] = np.asarray(mu_at)
        if (i + 1) % max(1, n_saved // 10) == 0 or i == 0:
            print(f"    frame {i+1}/{n_saved}")
    print(f"  Done in {time.perf_counter() - t0:.2f} s")

    # ---- save all replicas to a single .npz ------------------------------
    R_np = np.asarray(R_traj)            # (n_saved, B, N, 3)
    mu_np = np.asarray(mu_traj)           # (n_saved, B, 3)
    Ef_np = np.asarray(Ef_batched)        # (B, 3) per-replica field when available

    out = Path(args.output)
    npz_path = out.parent / (out.stem + ".npz")

    t0 = time.perf_counter()
    np.savez(
        npz_path,
        Z=Z_np,                           # (B, N) atomic numbers (row = replica)
        R=R_np,                           # (n_saved, B, N, 3)  positions [Å]
        V=V_np,                           # (n_saved, B, N, 3)  velocities [Å/fs]
        E=E_np,                           # (n_saved, B)        potential energy [eV]
        D=mu_np,                          # (n_saved, B, 3)     dipole [a.u.]
        Q=charges_all,                    # (n_saved, B, N)     atomic charges [e]
        AD=at_dipoles_all,                # (n_saved, B, N, 3)  atomic dipoles [a.u.]
        Ef=Ef_np,                         # (B, 3) or (1, 3) if single field tiled
        masses=masses_np,                 # (B, N) amu
        dt_fs=np.float64(dt),
        save_interval=np.int32(si),
        n_replicas=np.int32(B),
        thermostat=np.array(args.thermostat),
        temperature=np.float64(args.temperature),
    )
    print(f"\n  Saved {npz_path}  ({n_saved} frames x {B} replicas, "
          f"{npz_path.stat().st_size / 1e6:.1f} MB, "
          f"{time.perf_counter() - t0:.2f} s)")

    print(f"\n{'=' * 70}")
    print(f"  Batched MD complete — {n_saved} frames x {B} replicas.")
    print(f"  Output: {npz_path}")
    print(f"{'=' * 70}")


def main(args=None):
    if args is None:
        args = get_args()
    ramp_requested = (
        getattr(args, "ramp_field_axis", None) is not None
        or getattr(args, "ramp_field_peak", None) is not None
        or getattr(args, "ramp_field_start", None) is not None
    )
    if getattr(args, "n_replicas", 1) > 1:
        if ramp_requested:
            raise NotImplementedError(
                "Electric-field ramp is currently supported only for n_replicas=1."
            )
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
    axis_map = {"x": 0, "y": 1, "z": 2}
    ramp_axis = getattr(args, "ramp_field_axis", None)
    ramp_peak = getattr(args, "ramp_field_peak", None)
    ramp_start_arg = getattr(args, "ramp_field_start", None)
    ramp_enabled = ramp_axis is not None or ramp_peak is not None or ramp_start_arg is not None

    if ramp_enabled and (ramp_axis is None or ramp_peak is None):
        raise ValueError(
            "Ramp mode requires both --ramp-field-axis and --ramp-field-peak."
        )

    if ramp_enabled:
        ramp_axis_idx = axis_map[ramp_axis]
        ramp_axis_start = float(Ef[ramp_axis_idx]) if ramp_start_arg is None else float(ramp_start_arg)
        ramp_axis_peak = float(ramp_peak)
        Ef[ramp_axis_idx] = ramp_axis_start
        atoms.info['electric_field'] = Ef.copy()

    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Atomic numbers: {atoms.get_atomic_numbers()}")
    print(f"  Electric field: {Ef}")
    if ramp_enabled:
        print(
            "  Field ramp     : enabled "
            f"(axis={ramp_axis}, start={ramp_axis_start:.6f}, peak={ramp_axis_peak:.6f})"
        )

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
        ef_now = np.array(atoms.info.get("electric_field", Ef), dtype=np.float64)
        ef_label = f" Ef[{ramp_axis}]={ef_now[ramp_axis_idx]: .6f}" if ramp_enabled else ""
        print(
            f"{step:8d} {time_fs:10.2f} {e_pot:12.6f} {e_kin:12.6f} "
            f"{e_tot:12.6f} {temp:8.1f} {max_force:12.6f}{ef_label}"
        )

    def update_ramped_electric_field():
        """Triangular ramp on one axis: start -> peak -> start over full trajectory."""
        if not ramp_enabled:
            return

        if args.steps <= 1:
            phase = 0.0
        else:
            phase = min(float(dyn.nsteps) / float(args.steps - 1), 1.0)

        if phase <= 0.5:
            frac = phase / 0.5
            axis_value = ramp_axis_start + frac * (ramp_axis_peak - ramp_axis_start)
        else:
            frac = (phase - 0.5) / 0.5
            axis_value = ramp_axis_peak + frac * (ramp_axis_start - ramp_axis_peak)

        ef_step = np.array(Ef, dtype=np.float64)
        ef_step[ramp_axis_idx] = axis_value
        atoms.info["electric_field"] = ef_step
        calc.set_electric_field(ef_step)

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
    dyn.attach(update_ramped_electric_field, interval=1)
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

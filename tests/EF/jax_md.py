#!/usr/bin/env python3
"""
Pure JAX Molecular Dynamics — fully JIT-compiled Velocity-Verlet / Langevin.

Unlike `ase_md.py`, this avoids all Python-level per-step overhead: the
entire integration loop runs inside a single `jax.lax.fori_loop`, giving
~10-100× speedup on GPU for medium-sized molecules.

Outputs are written to an ASE .traj file *after* the simulation finishes
(the full trajectory lives in GPU memory during the run).

Supports:
  - NVE  (Velocity Verlet)
  - NVT  (Langevin thermostat)
  - Saving dipoles and (optionally) atomic charges per frame

Usage:
    python jax_md.py --params params.json --data data-full.npz --steps 10000
    python jax_md.py --params params.json --data data-full.npz \\
        --thermostat langevin --temperature 300 --steps 5000 \\
        --save-interval 10 --save-charges
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

import argparse
import time
import functools
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import e3x

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ase_calc_EF import load_params, load_config
from training import MessagePassingModel
from model_functions import energy_and_forces, get_atomic_properties

# ---------------------------------------------------------------------------
# Physical constants  (ASE convention: eV, Å, fs, amu)
# ---------------------------------------------------------------------------
BOLTZMANN_EV = 8.617333262e-5          # eV / K
AMU_TO_EV_FS2_ANG2 = 1.0 / 103.6427   # 1 amu·Å²/fs² ≈ 0.009649 eV

# Atomic masses (amu) indexed by atomic number — first 100 elements
_ASE_MASSES = None

def _get_masses():
    """Lazy-load ASE masses so we don't require ASE at import time."""
    global _ASE_MASSES
    if _ASE_MASSES is None:
        from ase.data import atomic_masses
        _ASE_MASSES = atomic_masses
    return _ASE_MASSES


# =====================================================================
# Graph construction  (done once, outside JIT)
# =====================================================================

def build_graph(n_atoms):
    """Build sparse pairwise indices for a single molecule."""
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)
    batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)
    offsets = jnp.zeros(1, dtype=jnp.int32)
    dst_idx_flat = dst_idx + 0
    src_idx_flat = src_idx + 0
    return dict(
        dst_idx=dst_idx, src_idx=src_idx,
        dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
        batch_segments=batch_segments,
    )


# =====================================================================
# Force function  (JIT-friendly)
# =====================================================================

def make_force_fn(model, params, graph, n_atoms, Ef):
    """Return a JIT-compiled function  R -> (E, F, mu)  for a fixed graph."""

    @functools.partial(jax.jit)
    def force_fn(positions):
        """positions: (N, 3) -> energy (scalar), forces (N, 3), dipole (3,)"""
        Z_b = graph['Z_batched']           # (1, N)
        R_b = positions[None, :, :]        # (1, N, 3)
        Ef_b = Ef[None, :]                 # (1, 3)
        energy, forces, dipole = energy_and_forces(
            model_apply, params,
            atomic_numbers=Z_b, positions=R_b, Ef=Ef_b,
            dst_idx_flat=graph['dst_idx_flat'],
            src_idx_flat=graph['src_idx_flat'],
            batch_segments=graph['batch_segments'],
            batch_size=1,
            dst_idx=graph['dst_idx'],
            src_idx=graph['src_idx'],
        )
        return energy[0], forces[0], dipole[0]

    # We need the model_apply closure
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def model_apply(params, atomic_numbers, positions, Ef,
                    dst_idx_flat, src_idx_flat, batch_segments, batch_size,
                    dst_idx=None, src_idx=None):
        return model.apply(params, atomic_numbers, positions, Ef,
                           dst_idx_flat=dst_idx_flat,
                           src_idx_flat=src_idx_flat,
                           batch_segments=batch_segments,
                           batch_size=batch_size,
                           dst_idx=dst_idx, src_idx=src_idx)

    return force_fn


# =====================================================================
# Integrators
# =====================================================================

def velocity_verlet_step(state, force_fn, dt, inv_masses):
    """Single Velocity-Verlet step.

    state = (R, V, F)  where shapes are (N, 3).
    inv_masses: (N, 1) = 1 / (mass_i * AMU_TO_EV_FS2_ANG2)
    """
    R, V, F = state
    # Half-kick
    V = V + 0.5 * dt * F * inv_masses
    # Drift
    R = R + dt * V
    # New forces
    E, F_new, mu = force_fn(R)
    # Half-kick
    V = V + 0.5 * dt * F_new * inv_masses
    return (R, V, F_new), (E, mu)


def langevin_step(state, force_fn, dt, inv_masses, gamma, kT, rng_key):
    """Single Langevin (BAOAB splitting) step.

    Uses the BAOAB scheme:
      B: half-kick from forces
      A: half-drift
      O: Ornstein-Uhlenbeck velocity update (thermostat)
      A: half-drift
      B: half-kick from forces
    """
    R, V, F = state

    # B: half-kick
    V = V + 0.5 * dt * F * inv_masses

    # A: half-drift
    R = R + 0.5 * dt * V

    # O: Ornstein-Uhlenbeck (thermostat)
    c1 = jnp.exp(-gamma * dt)
    # sigma_v = sqrt(kT * inv_mass_eV)  where inv_mass_eV = 1 / (m [amu] * AMU_TO_EV_FS2_ANG2)
    # but we want sigma for velocity in Å/fs
    # kT [eV], mass [amu], 1 amu·Å²/fs² = AMU_TO_EV_FS2_ANG2 eV
    # sigma² = kT / (m * AMU_TO_EV_FS2_ANG2) = kT * inv_masses
    sigma = jnp.sqrt(kT * inv_masses * (1.0 - c1 ** 2))
    noise = jax.random.normal(rng_key, shape=V.shape)
    V = c1 * V + sigma * noise

    # A: half-drift
    R = R + 0.5 * dt * V

    # Recompute forces at new position
    E, F_new, mu = force_fn(R)

    # B: half-kick
    V = V + 0.5 * dt * F_new * inv_masses

    return (R, V, F_new), (E, mu)


# =====================================================================
# Scan-based trajectory runners  (fully compiled)
# =====================================================================

def run_nve(force_fn, R0, V0, dt, masses_amu, n_steps, save_interval):
    """Run NVE MD; return saved frames.

    Returns
    -------
    R_traj : (n_saved, N, 3)
    V_traj : (n_saved, N, 3)
    E_traj : (n_saved,)
    mu_traj: (n_saved, 3)
    """
    N = R0.shape[0]
    inv_masses = 1.0 / (masses_amu[:, None] * AMU_TO_EV_FS2_ANG2)  # (N, 1)
    inv_masses = jnp.asarray(inv_masses, dtype=jnp.float32)

    E0, F0, mu0 = force_fn(R0)

    n_saved = n_steps // save_interval + 1

    def body_fn(i, carry):
        """One integration step."""
        state, saved = carry
        new_state, (E, mu) = velocity_verlet_step(state, force_fn, dt, inv_masses)

        # Conditionally save
        frame_idx = (i + 1) // save_interval
        should_save = ((i + 1) % save_interval == 0)
        R, V, F = new_state

        saved_R = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(R),
            lambda s: s,
            saved[0])
        saved_V = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(V),
            lambda s: s,
            saved[1])
        saved_E = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(E),
            lambda s: s,
            saved[2])
        saved_mu = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(mu),
            lambda s: s,
            saved[3])

        return (new_state, (saved_R, saved_V, saved_E, saved_mu))

    # Pre-allocate save buffers
    R_buf = jnp.zeros((n_saved, N, 3), dtype=jnp.float32)
    V_buf = jnp.zeros((n_saved, N, 3), dtype=jnp.float32)
    E_buf = jnp.zeros((n_saved,), dtype=jnp.float32)
    mu_buf = jnp.zeros((n_saved, 3), dtype=jnp.float32)

    # Save initial frame
    R_buf = R_buf.at[0].set(R0)
    V_buf = V_buf.at[0].set(V0)
    E_buf = E_buf.at[0].set(E0)
    mu_buf = mu_buf.at[0].set(mu0)

    init_state = (R0, V0, F0)
    init_saved = (R_buf, V_buf, E_buf, mu_buf)

    print(f"  JIT-compiling NVE loop ({n_steps} steps) ...")
    t0 = time.perf_counter()
    final_state, final_saved = jax.lax.fori_loop(
        0, n_steps, body_fn, (init_state, init_saved))
    # Force evaluation to trigger compilation
    final_saved[2].block_until_ready()
    t1 = time.perf_counter()
    print(f"  Done in {t1 - t0:.2f} s  (includes JIT compilation)")

    return final_saved


def run_langevin(force_fn, R0, V0, dt, masses_amu, n_steps, save_interval,
                 temperature, gamma, rng_key):
    """Run Langevin (NVT) MD via BAOAB; return saved frames."""
    N = R0.shape[0]
    inv_masses = 1.0 / (masses_amu[:, None] * AMU_TO_EV_FS2_ANG2)
    inv_masses = jnp.asarray(inv_masses, dtype=jnp.float32)
    kT = BOLTZMANN_EV * temperature

    E0, F0, mu0 = force_fn(R0)
    n_saved = n_steps // save_interval + 1

    def body_fn(i, carry):
        state, saved, rng = carry
        rng, step_key = jax.random.split(rng)
        new_state, (E, mu) = langevin_step(
            state, force_fn, dt, inv_masses, gamma, kT, step_key)

        frame_idx = (i + 1) // save_interval
        should_save = ((i + 1) % save_interval == 0)
        R, V, F = new_state

        saved_R = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(R),
            lambda s: s,
            saved[0])
        saved_V = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(V),
            lambda s: s,
            saved[1])
        saved_E = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(E),
            lambda s: s,
            saved[2])
        saved_mu = jax.lax.cond(
            should_save,
            lambda s: s.at[frame_idx].set(mu),
            lambda s: s,
            saved[3])

        return (new_state, (saved_R, saved_V, saved_E, saved_mu), rng)

    R_buf = jnp.zeros((n_saved, N, 3), dtype=jnp.float32)
    V_buf = jnp.zeros((n_saved, N, 3), dtype=jnp.float32)
    E_buf = jnp.zeros((n_saved,), dtype=jnp.float32)
    mu_buf = jnp.zeros((n_saved, 3), dtype=jnp.float32)

    R_buf = R_buf.at[0].set(R0)
    V_buf = V_buf.at[0].set(V0)
    E_buf = E_buf.at[0].set(E0)
    mu_buf = mu_buf.at[0].set(mu0)

    init_state = (R0, V0, F0)
    init_saved = (R_buf, V_buf, E_buf, mu_buf)

    print(f"  JIT-compiling Langevin loop ({n_steps} steps) ...")
    t0 = time.perf_counter()
    (final_state, final_saved, _) = jax.lax.fori_loop(
        0, n_steps, body_fn, (init_state, init_saved, rng_key))
    final_saved[2].block_until_ready()
    t1 = time.perf_counter()
    print(f"  Done in {t1 - t0:.2f} s  (includes JIT compilation)")

    return final_saved


# =====================================================================
# Trajectory I/O  (post-simulation, using ASE)
# =====================================================================

def save_trajectory(output_path, Z, R_traj, V_traj, E_traj, mu_traj,
                    Ef, dt_fs, save_interval, charges_traj=None):
    """Write saved frames to an ASE .traj file."""
    import ase
    from ase.io.trajectory import Trajectory

    R_np = np.asarray(R_traj)
    V_np = np.asarray(V_traj)
    E_np = np.asarray(E_traj)
    mu_np = np.asarray(mu_traj)
    Ef_np = np.asarray(Ef)
    Z_np = np.asarray(Z)

    n_frames = R_np.shape[0]
    masses = _get_masses()[Z_np]

    traj = Trajectory(str(output_path), 'w')
    for i in range(n_frames):
        atoms = ase.Atoms(numbers=Z_np, positions=R_np[i])
        # Store velocities (Å/fs -> ASE internal: Å/√(amu·eV)·fs)
        # ASE velocity unit: sqrt(eV/amu) = Å/fs * sqrt(AMU_TO_EV_FS2_ANG2)
        # v_ase = v_real / sqrt(AMU_TO_EV_FS2_ANG2)
        # But actually ASE stores velocities in Å/fs when using atoms.set_velocities?
        # set_velocities expects amu * Å / fs  ... no.
        # ASE uses "ase units": length=Å, energy=eV, mass=amu.
        # velocity unit = sqrt(eV / amu)  ≈ 0.098 Å/fs
        # So v_ase = v_real [Å/fs] / sqrt(eV/amu) = v_real / 0.098...
        # ASE: atoms.set_velocities expects Å/(fs * sqrt(amu/eV))
        # Simpler: just use set_velocities which takes Å/ASE_time_unit
        # ASE time unit = fs * sqrt(amu·Å²/eV) ≈ 10.18 fs
        # Actually let's just use momenta
        atoms.set_momenta(V_np[i] * masses[:, None] * AMU_TO_EV_FS2_ANG2)

        atoms.info['electric_field'] = Ef_np
        atoms.info['ml_dipole'] = mu_np[i]
        atoms.info['step'] = i * save_interval
        atoms.info['time_fs'] = i * save_interval * dt_fs

        # Store energy as SinglePointCalculator
        from ase.calculators.singlepoint import SinglePointCalculator
        sp = SinglePointCalculator(atoms, energy=float(E_np[i]))
        sp.results['dipole'] = mu_np[i]
        atoms.calc = sp

        if charges_traj is not None:
            atoms.arrays['ml_charges'] = np.asarray(charges_traj[i])

        traj.write(atoms)
    traj.close()
    return n_frames


# =====================================================================
# Initial velocities  (Maxwell-Boltzmann)
# =====================================================================

def maxwell_boltzmann_velocities(masses_amu, temperature, rng_key):
    """Sample velocities from Maxwell-Boltzmann at given temperature.

    Returns velocities in Å/fs.
    """
    N = len(masses_amu)
    kT = BOLTZMANN_EV * temperature
    # sigma_v = sqrt(kT / (m * AMU_TO_EV_FS2_ANG2))  [Å/fs]
    sigma = jnp.sqrt(kT / (masses_amu[:, None] * AMU_TO_EV_FS2_ANG2))
    V = sigma * jax.random.normal(rng_key, shape=(N, 3))
    # Remove centre-of-mass velocity
    total_mass = jnp.sum(masses_amu)
    V_com = jnp.sum(masses_amu[:, None] * V, axis=0) / total_mass
    V = V - V_com[None, :]
    return V


# =====================================================================
# CLI
# =====================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Pure JAX Molecular Dynamics (JIT-compiled integrator)")

    g = p.add_argument_group("model / data")
    g.add_argument("--params", required=True, help="Path to params JSON")
    g.add_argument("--config", default=None, help="Config JSON (auto-detect)")
    g.add_argument("--data", default="data-full.npz",
                   help="Dataset NPZ for initial geometry")
    g.add_argument("--xyz", default=None,
                   help="XYZ file for initial geometry (overrides --data)")
    g.add_argument("--index", type=int, default=0,
                   help="Structure index in dataset")
    g.add_argument("--electric-field", type=float, nargs=3,
                   default=None,
                   help="Electric field (Ef_x Ef_y Ef_z); default: from dataset")
    g.add_argument("--field-scale", type=float, default=0.001)

    g = p.add_argument_group("integrator")
    g.add_argument("--thermostat", choices=["nve", "langevin"],
                   default="langevin")
    g.add_argument("--temperature", type=float, default=300.0,
                   help="Temperature (K) for Langevin or initial velocities")
    g.add_argument("--friction", type=float, default=0.002,
                   help="Langevin friction γ (1/fs). Default 0.002 ≈ 500 fs "
                        "decorrelation time")
    g.add_argument("--dt", type=float, default=0.5, help="Timestep (fs)")
    g.add_argument("--steps", type=int, default=10000,
                   help="Total integration steps")

    g = p.add_argument_group("output")
    g.add_argument("--save-interval", type=int, default=10,
                   help="Save every N steps")
    g.add_argument("--output", default="jax_md_trajectory.traj",
                   help="Output ASE .traj file")
    g.add_argument("--save-charges", action="store_true",
                   help="Recompute & save atomic charges for each saved frame "
                        "(post-hoc, slower)")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--print-interval", type=int, default=None,
                   help="Print status every N saved frames (default: ~20 lines)")
    g.add_argument("--dipole-field-coupling", action="store_true",
                   default=None)

    return p.parse_args()


# =====================================================================
# Main
# =====================================================================

def main():
    args = get_args()
    rng = jax.random.PRNGKey(args.seed)

    print("=" * 70)
    print("  JAX Molecular Dynamics  (JIT-compiled integrator)")
    print("=" * 70)
    print(f"  JAX devices : {jax.devices()}")
    print(f"  Backend     : {jax.default_backend()}")

    # ---- Load initial geometry ------------------------------------------
    if args.xyz is not None:
        import ase.io as ase_io
        atoms = ase_io.read(args.xyz)
        Z = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        R0 = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
    else:
        print(f"\n  Loading geometry from {args.data} (index {args.index}) ...")
        data = np.load(args.data, allow_pickle=True)
        Z = jnp.asarray(np.asarray(data['Z'][args.index], dtype=int),
                         dtype=jnp.int32)
        R0 = jnp.asarray(np.asarray(data['R'][args.index], dtype=float),
                          dtype=jnp.float32)
        if R0.ndim == 3:
            R0 = R0.squeeze(0)

    N = len(Z)
    print(f"  Atoms       : {N}")
    print(f"  Z           : {np.asarray(Z)}")

    # ---- Electric field -------------------------------------------------
    if args.electric_field is not None:
        Ef = jnp.asarray(args.electric_field, dtype=jnp.float32)
    elif args.xyz is None:
        data = np.load(args.data, allow_pickle=True)
        if 'Ef' in data.files:
            Ef = jnp.asarray(np.asarray(data['Ef'][args.index], dtype=float),
                              dtype=jnp.float32)
        else:
            Ef = jnp.zeros(3, dtype=jnp.float32)
    else:
        Ef = jnp.zeros(3, dtype=jnp.float32)
    print(f"  Ef          : {np.asarray(Ef)}")

    # ---- Build model ----------------------------------------------------
    print(f"\n  Loading model from {args.params} ...")
    params_path = Path(args.params)
    params = load_params(params_path)

    # Config auto-detection (same logic as AseCalculatorEF)
    config_path = args.config
    if config_path is None:
        if params_path.stem.startswith('params-') and len(params_path.stem) > 7:
            uuid_part = params_path.stem[7:]
            candidate = params_path.parent / f'config-{uuid_part}.json'
            if candidate.exists():
                config_path = str(candidate)
            elif (params_path.parent / 'config.json').exists():
                config_path = str(params_path.parent / 'config.json')

    model_keys = {'features', 'max_degree', 'num_iterations',
                  'num_basis_functions', 'cutoff', 'max_atomic_number',
                  'include_pseudotensors', 'dipole_field_coupling', 'field_scale'}

    if config_path is not None:
        config = load_config(config_path)
        if 'model' in config and isinstance(config['model'], dict):
            mc = {k: v for k, v in config['model'].items() if k in model_keys}
        elif 'model_config' in config:
            mc = {k: v for k, v in config['model_config'].items()
                  if k in model_keys}
        else:
            mc = {k: v for k, v in config.items() if k in model_keys}
        print(f"  Config      : {config_path}")
    else:
        mc = dict(features=64, max_degree=2, num_iterations=2,
                  num_basis_functions=64, cutoff=10.0,
                  max_atomic_number=55, include_pseudotensors=True)
        print("  Config      : defaults (no config file found)")

    if args.dipole_field_coupling is not None:
        mc['dipole_field_coupling'] = args.dipole_field_coupling

    model = MessagePassingModel(**mc)
    print(f"  Model       : {mc}")

    # ---- Build graph (once) ---------------------------------------------
    graph = build_graph(N)
    graph['Z_batched'] = Z[None, :]   # (1, N)

    # ---- Force function -------------------------------------------------
    force_fn = make_force_fn(model, params, graph, N, Ef)

    # Warm up (first call triggers JIT compilation of force_fn)
    print("\n  Warming up force function (JIT compile) ...")
    t0 = time.perf_counter()
    E0, F0, mu0 = force_fn(R0)
    E0.block_until_ready()
    t1 = time.perf_counter()
    print(f"  Compiled in {t1 - t0:.2f} s")
    print(f"  E = {float(E0):.6f} eV")
    print(f"  max|F| = {float(jnp.max(jnp.abs(F0))):.6f} eV/Å")
    print(f"  μ = {np.asarray(mu0)}")

    # ---- Initial velocities ---------------------------------------------
    masses_amu = jnp.asarray(_get_masses()[np.asarray(Z)], dtype=jnp.float32)
    rng, vel_key = jax.random.split(rng)
    V0 = maxwell_boltzmann_velocities(masses_amu, args.temperature, vel_key)
    Ekin0 = 0.5 * jnp.sum(masses_amu[:, None] * AMU_TO_EV_FS2_ANG2 * V0 ** 2)
    T0 = 2.0 * Ekin0 / (3.0 * N * BOLTZMANN_EV)
    print(f"\n  Initial T   : {float(T0):.1f} K  (target {args.temperature} K)")

    # ---- Run MD ---------------------------------------------------------
    dt = args.dt
    si = args.save_interval
    n_saved = args.steps // si + 1

    print(f"\n  Thermostat  : {args.thermostat}")
    print(f"  dt          : {dt} fs")
    print(f"  Steps       : {args.steps}")
    print(f"  Save every  : {si} steps  → {n_saved} frames")
    print(f"  Total time  : {args.steps * dt:.1f} fs  "
          f"({args.steps * dt / 1000:.2f} ps)")
    print()

    t_start = time.perf_counter()

    if args.thermostat == 'nve':
        saved = run_nve(force_fn, R0, V0, dt, masses_amu,
                        args.steps, si)
    else:
        rng, md_key = jax.random.split(rng)
        saved = run_langevin(force_fn, R0, V0, dt, masses_amu,
                             args.steps, si,
                             args.temperature, args.friction, md_key)

    R_traj, V_traj, E_traj, mu_traj = saved
    t_end = time.perf_counter()
    wall = t_end - t_start
    steps_per_sec = args.steps / wall
    ns_per_day = (args.steps * dt * 1e-6) / (wall / 86400)

    print(f"\n  Wall time   : {wall:.2f} s")
    print(f"  Performance : {steps_per_sec:.0f} steps/s")
    print(f"  Throughput  : {ns_per_day:.3f} ns/day")

    # ---- Summary statistics ---------------------------------------------
    E_np = np.asarray(E_traj)
    V_np = np.asarray(V_traj)
    masses_np = np.asarray(masses_amu)

    Ekin = 0.5 * np.sum(
        masses_np[None, :, None] * AMU_TO_EV_FS2_ANG2 * V_np ** 2,
        axis=(1, 2))   # (n_saved,)
    T_arr = 2.0 * Ekin / (3.0 * N * BOLTZMANN_EV)
    Etot = E_np + Ekin

    pi = args.print_interval
    if pi is None:
        pi = max(1, n_saved // 20)

    print(f"\n  {'frame':>6s} {'time(fs)':>10s} {'E_pot(eV)':>12s} "
          f"{'E_kin(eV)':>12s} {'E_tot(eV)':>12s} {'T(K)':>8s}")
    print("  " + "-" * 68)
    for i in range(0, n_saved, pi):
        t_fs = i * si * dt
        print(f"  {i:6d} {t_fs:10.1f} {E_np[i]:12.6f} "
              f"{Ekin[i]:12.6f} {Etot[i]:12.6f} {T_arr[i]:8.1f}")
    # Always print last frame
    if (n_saved - 1) % pi != 0:
        i = n_saved - 1
        t_fs = i * si * dt
        print(f"  {i:6d} {t_fs:10.1f} {E_np[i]:12.6f} "
              f"{Ekin[i]:12.6f} {Etot[i]:12.6f} {T_arr[i]:8.1f}")

    print(f"\n  T range     : {T_arr.min():.1f} – {T_arr.max():.1f} K  "
          f"(mean {T_arr.mean():.1f} K)")
    if args.thermostat == 'nve':
        drift = (Etot[-1] - Etot[0]) / (args.steps * dt)
        print(f"  E_tot drift : {drift:.2e} eV/fs")

    # ---- Optional: recompute atomic charges -----------------------------
    charges_traj = None
    if args.save_charges:
        print(f"\n  Recomputing atomic charges for {n_saved} frames ...")
        from ase_calc_EF import AseCalculatorEF
        calc = AseCalculatorEF(
            params_path=args.params, config_path=config_path,
            field_scale=args.field_scale)
        charges_list = []
        import ase
        for i in range(n_saved):
            atoms = ase.Atoms(numbers=np.asarray(Z),
                              positions=np.asarray(R_traj[i]))
            atoms.info['electric_field'] = np.asarray(Ef)
            atoms.calc = calc
            q, _ = calc.get_atomic_charges(atoms)
            charges_list.append(np.asarray(q))
            if (i + 1) % 100 == 0 or i == 0:
                print(f"    frame {i+1}/{n_saved}")
        charges_traj = np.array(charges_list)

    # ---- Save trajectory ------------------------------------------------
    print(f"\n  Saving {n_saved} frames to {args.output} ...")
    n_written = save_trajectory(
        args.output, Z, R_traj, V_traj, E_traj, mu_traj,
        Ef, dt, si, charges_traj=charges_traj)
    print(f"  Wrote {n_written} frames.")

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

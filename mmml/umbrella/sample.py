"""Batched JAX-MD NVT Nose-Hoover umbrella sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmml.umbrella.config import UmbrellaConfig
from mmml.umbrella.energy import (
    build_packed_graph,
    make_packed_energy_fn,
    packed_cv_distances,
)
from mmml.umbrella.io import (
    BIN_MINIMA_TRAJ,
    SNAPSHOTS_NPZ,
    SUMMARY_JSON,
    save_snapshots,
    write_summary,
)
from mmml.umbrella.structure import (
    load_structure,
    load_structure_frames,
    pack_window_seeds,
)


@dataclass(frozen=True)
class UmbrellaResult:
    """Artifacts from :func:`run_umbrella_nvt`."""

    output_dir: Path
    snapshots_path: Path
    summary_path: Path
    n_windows: int
    n_frames: int
    paths: dict[str, Path]


def center_com_positions(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Translate so the mass-weighted CoM is at the origin.

    ``positions`` may be ``(N, 3)`` or ``(..., N, 3)``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64).reshape(-1)
    if pos.ndim < 2 or pos.shape[-1] != 3:
        raise ValueError(f"positions must be (..., N, 3); got shape {pos.shape}")
    if pos.shape[-2] != m.shape[0]:
        raise ValueError(
            f"masses length {m.shape[0]} != n_atoms {pos.shape[-2]}"
        )
    total_mass = float(np.sum(m))
    if total_mass <= 0.0:
        raise ValueError("sum of masses must be positive")
    com = np.sum(pos * m.reshape((1,) * (pos.ndim - 2) + (-1, 1)), axis=-2) / total_mass
    return pos - com[..., None, :]


def select_lowest_energy_frames(
    positions: np.ndarray,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the lowest-``E_ML+W`` frame in each window.

    Returns ``(coords, frame_indices, energies)`` with shapes
    ``(K, N, 3)``, ``(K,)``, ``(K,)``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    ene = np.asarray(energies, dtype=np.float64)
    if pos.ndim != 4:
        raise ValueError(f"positions must be (K, T, N, 3); got {pos.shape}")
    if ene.shape != pos.shape[:2]:
        raise ValueError(
            f"energies shape {ene.shape} must match positions[:2] {pos.shape[:2]}"
        )
    idx = np.nanargmin(ene, axis=1).astype(np.int64)
    k = pos.shape[0]
    chosen = pos[np.arange(k), idx]
    return chosen, idx, ene[np.arange(k), idx]


def _per_window_temperatures_K(
    momenta,
    masses,
    *,
    n_windows: int,
    n_atoms: int,
    k_b: float,
) -> "np.ndarray":
    """Kinetic temperature of each packed window (K)."""
    import numpy as np

    p = np.asarray(momenta, dtype=np.float64).reshape(n_windows, n_atoms, 3)
    m = np.asarray(masses, dtype=np.float64).reshape(n_windows, n_atoms)
    ke = 0.5 * np.sum((p * p) / m[..., None], axis=(1, 2))
    dof = 3 * n_atoms
    return (2.0 * ke) / (dof * k_b)


def _schedule_targets_ks(sched):
    targets = [list(sched.xi0)]
    ks = [list(sched.k_x)]
    if sched.ndim == 2:
        assert sched.yi0 is not None and sched.k_y is not None
        targets.append(list(sched.yi0))
        ks.append(list(sched.k_y))
    return targets, ks


def run_umbrella_nvt(cfg: UmbrellaConfig) -> UmbrellaResult:
    """Run packed-batch NVT umbrella sampling with a PhysNet/SpookyNet checkpoint."""
    import jax
    import jax.numpy as jnp
    from ase.data import atomic_masses
    from ase.io import write
    from jax_md import quantity, simulate, space

    from mmml.umbrella.checkpoint import load_params_and_model

    jax.config.update("jax_enable_x64", True)

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not cfg.overwrite:
        raise FileExistsError(
            f"output_dir is not empty: {output_dir} (pass overwrite=True to proceed)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_path = Path(cfg.structure).expanduser().resolve()
    sched = cfg.resolve_schedule()
    k_windows = sched.n_windows
    targets_per_cv, k_per_cv = _schedule_targets_ks(sched)

    frames = None
    seed_mode = cfg.seed_mode
    if seed_mode == "frames":
        r_multi, z = load_structure_frames(
            structure_path,
            n_frames=k_windows,
            start_index=int(cfg.structure_index),
        )
        r0 = r_multi[0]
        frames = r_multi
    else:
        r0, z = load_structure(structure_path, index=int(cfg.structure_index))

    n_atoms = int(len(z))
    for i, j in sched.atom_pairs:
        if max(i, j) >= n_atoms:
            raise ValueError(
                f"atom indices ({i}, {j}) out of range for {n_atoms} atoms"
            )

    r0_cvs = [
        float(np.linalg.norm(r0[j] - r0[i])) for i, j in sched.atom_pairs
    ]
    move_groups: list[tuple[int, ...]] = [tuple(cfg.move_with)]
    if sched.ndim == 2:
        move_groups.append(tuple(cfg.move_with2))
    r_packed_np = pack_window_seeds(
        positions=r0,
        atom_pairs=sched.atom_pairs,
        targets_per_cv=targets_per_cv,
        seed_mode=seed_mode,
        frames=frames,
        move_groups=move_groups,
        invert_with=cfg.invert_with,
    )

    params, model = load_params_and_model(
        Path(cfg.checkpoint).expanduser().resolve(),
        natoms=n_atoms,
        prefer_ema=cfg.use_ema,
    )

    graph = build_packed_graph(n_atoms, k_windows)
    energy_sum_fn = make_packed_energy_fn(
        model_apply=model.apply,
        params=params,
        atomic_numbers=z,
        graph=graph,
        atom_pairs=sched.atom_pairs,
        targets_per_cv=targets_per_cv,
        k_per_cv=k_per_cv,
    )
    # Use explicit PhysNet forces + analytic bias forces. Autodiff of
    # energy_sum_fn nests AD through PhysNet's internal value_and_grad and
    # can yield NaN forces even when the energy is finite.
    force_fn = energy_sum_fn.force_fn  # type: ignore[attr-defined]
    per_window_energy_fn = energy_sum_fn.per_window_energy_fn  # type: ignore[attr-defined]
    energy_sum_fn = jax.jit(energy_sum_fn)
    force_fn = jax.jit(force_fn)
    per_window_energy_fn = jax.jit(per_window_energy_fn)

    masses = np.array([atomic_masses[int(zi)] for zi in z], dtype=np.float64)
    masses_batched = jnp.tile(jnp.asarray(masses), k_windows)
    r_packed = jnp.asarray(r_packed_np, dtype=jnp.float64)

    e0 = float(energy_sum_fn(r_packed))
    f0 = force_fn(r_packed)
    f0_max = float(jnp.max(jnp.abs(f0)))
    if not np.isfinite(e0) or not bool(jnp.all(jnp.isfinite(f0))):
        raise RuntimeError(
            f"initial umbrella E/F non-finite (E={e0}, max|F|={f0_max}). "
            "Check checkpoint, geometry, and --atoms / --atoms2 CV indices."
        )
    f_win = np.asarray(f0).reshape(k_windows, n_atoms, 3)
    fmax_k = np.max(np.abs(f_win), axis=(1, 2))
    hot = [
        (int(i), float(fmax_k[i]), float(targets_per_cv[0][i]))
        for i in range(k_windows)
        if float(fmax_k[i]) > float(cfg.max_seed_force)
    ]
    if hot:
        detail = ", ".join(
            f"k={i} max|F|={fm:.1f} ξ₀={xi:.3f}" for i, fm, xi in hot[:8]
        )
        more = f" (+{len(hot) - 8} more)" if len(hot) > 8 else ""
        raise RuntimeError(
            f"{len(hot)}/{k_windows} window seeds exceed --max-seed-force="
            f"{cfg.max_seed_force} eV/Å ({detail}{more}). "
            "For SN2-like 2D grids add --invert-with for CH3 H atoms "
            "(e.g. 6,7,8), use --move-with2 for NH3, soften --k/--ky, "
            "narrow the product grid, or --seed-mode frames from an NEB path."
        )

    k_b = 8.617333262145e-5  # eV/K
    kt = k_b * float(cfg.temperature_K)
    dt = float(cfg.timestep_fs)
    savefreq = cfg.effective_savefreq()
    if dt > 0.25:
        print(
            f"WARNING: timestep {dt} fs is large for PhysNet umbrella NVT with H atoms; "
            "prefer --timestep 0.1 (0.5 fs often NaNs by step ~100 even when seed forces look fine)."
        )

    _, shift = space.free()
    # Packed multi-window MD must not use a shared Nose-Hoover chain or global
    # COM velocity removal: one hot replica then drags every other window.
    if cfg.thermostat == "langevin":
        init_fn, apply_fn = simulate.nvt_langevin(
            force_fn,
            shift,
            dt,
            kt,
            gamma=float(cfg.langevin_gamma),
            center_velocity=False,
        )
    else:
        init_fn, apply_fn = simulate.nvt_nose_hoover(force_fn, shift, dt, kt)
    apply_fn = jax.jit(apply_fn)

    key = jax.random.PRNGKey(int(cfg.seed))
    state = init_fn(key, r_packed, mass=masses_batched)
    t_abort = (
        float(cfg.max_window_temp_K)
        if cfg.max_window_temp_K is not None
        else 5.0 * float(cfg.temperature_K)
    )

    def _cv_frame(pos):
        cols = [
            np.asarray(
                packed_cv_distances(pos, n_atoms, i, j, k_windows)
            )
            for i, j in sched.atom_pairs
        ]
        return np.stack(cols, axis=-1)  # (K, ndim)

    frames_traj: list[np.ndarray] = [
        np.asarray(state.position).reshape(k_windows, n_atoms, 3)
    ]
    cv_frames: list[np.ndarray] = [_cv_frame(state.position)]
    energy_frames: list[np.ndarray] = [
        np.asarray(per_window_energy_fn(state.position), dtype=np.float64)
    ]

    print(
        f"=== Umbrella NVT ({k_windows} windows, {sched.ndim}D, "
        f"T={cfg.temperature_K} K, dt={dt} fs, thermostat={cfg.thermostat}) ==="
    )
    print(
        f"  structure={structure_path.name}  seed_mode={seed_mode}  "
        f"pairs={sched.atom_pairs}  r0={r0_cvs}  grid={sched.grid_shape}"
    )
    if any(move_groups):
        print(f"  move_groups={move_groups}")
    if cfg.invert_with:
        print(f"  invert_with={tuple(cfg.invert_with)}")
    print(
        f"  per-window max|F|:"
        f" min={float(np.min(fmax_k)):.2f} median={float(np.median(fmax_k)):.2f}"
        f" max={float(np.max(fmax_k)):.2f} eV/Å"
    )
    print(f"  Initial E_total={e0:.4f} eV  max|F|={f0_max:.4f} eV/Å")
    rex_stats = None
    rex_phase = 0
    rex_rng = None
    if cfg.replica_exchange:
        from mmml.umbrella.rex import RexStats

        if k_windows < 2:
            raise ValueError("replica exchange requires at least 2 windows")
        rex_stats = RexStats()
        rex_rng = np.random.default_rng(int(cfg.seed) + 17)
        print(
            f"  replica_exchange=on  rex_freq={cfg.rex_freq}  "
            f"neighbor even/odd on grid={sched.grid_shape}"
        )
    for step in range(1, int(cfg.nsteps) + 1):
        state = apply_fn(state)
        if (
            cfg.replica_exchange
            and rex_stats is not None
            and rex_rng is not None
            and step % int(cfg.rex_freq) == 0
        ):
            import dataclasses

            from mmml.umbrella.rex import attempt_replica_exchanges

            cv_now = _cv_frame(state.position)
            force_arr = getattr(state, "force", None)
            pos_new, mom_new, frc_new, n_att, n_acc = attempt_replica_exchanges(
                positions_packed=np.asarray(state.position),
                momenta_packed=np.asarray(state.momentum),
                forces_packed=None if force_arr is None else np.asarray(force_arr),
                cv=cv_now,
                targets_per_cv=targets_per_cv,
                k_per_cv=k_per_cv,
                grid_shape=sched.grid_shape,
                phase=rex_phase,
                beta=1.0 / kt,
                rng=rex_rng,
                n_atoms=n_atoms,
                stats=rex_stats,
            )
            rex_phase += 1
            replace_kwargs = {
                "position": jnp.asarray(pos_new, dtype=state.position.dtype),
                "momentum": jnp.asarray(mom_new, dtype=state.momentum.dtype),
            }
            if frc_new is not None and force_arr is not None:
                replace_kwargs["force"] = jnp.asarray(frc_new, dtype=force_arr.dtype)
            state = dataclasses.replace(state, **replace_kwargs)
            if step % int(cfg.printfreq) == 0 or step == cfg.nsteps:
                print(
                    f"  rex step {step:6d}  accepted {n_acc}/{n_att}  "
                    f"cum_acc={rex_stats.acceptance:.3f}"
                )
        if step % savefreq == 0 or step == cfg.nsteps:
            pos_batch = np.asarray(state.position).reshape(k_windows, n_atoms, 3)
            frames_traj.append(pos_batch)
            cv_frames.append(_cv_frame(state.position))
            energy_frames.append(
                np.asarray(per_window_energy_fn(state.position), dtype=np.float64)
            )
        if step % int(cfg.printfreq) == 0 or step == cfg.nsteps:
            e_tot = float(energy_sum_fn(state.position))
            t_curr = float(
                quantity.temperature(momentum=state.momentum, mass=state.mass) / k_b
            )
            t_win = _per_window_temperatures_K(
                state.momentum,
                state.mass,
                n_windows=k_windows,
                n_atoms=n_atoms,
                k_b=k_b,
            )
            f_now = np.asarray(force_fn(state.position)).reshape(k_windows, n_atoms, 3)
            fmax_now = np.max(np.abs(f_now), axis=(1, 2))
            if not np.isfinite(e_tot) or not np.isfinite(t_curr) or not np.all(
                np.isfinite(t_win)
            ):
                hot_idx = int(np.nanargmax(t_win)) if np.any(np.isfinite(t_win)) else -1
                raise RuntimeError(
                    f"non-finite thermodynamics at step {step}: "
                    f"E={e_tot} T={t_curr} hottest_window={hot_idx}. "
                    "Try --thermostat langevin (default), smaller --timestep, "
                    "softer --k, or drop harsh 2D corners."
                )
            hot_t = [
                (int(i), float(t_win[i]), float(fmax_now[i]))
                for i in range(k_windows)
                if float(t_win[i]) > t_abort
            ]
            if hot_t:
                hot_t.sort(key=lambda x: -x[1])
                detail = ", ".join(
                    f"k={i} T={t:.0f}K max|F|={fm:.1f}" for i, t, fm in hot_t[:6]
                )
                xi = float(targets_per_cv[0][hot_t[0][0]])
                yi = (
                    float(targets_per_cv[1][hot_t[0][0]])
                    if len(targets_per_cv) > 1
                    else None
                )
                cv = f"ξ₀={xi:.3f}" + (f" η₀={yi:.3f}" if yi is not None else "")
                raise RuntimeError(
                    f"window temperature spike at step {step}: {detail} "
                    f"(limit {t_abort:.0f} K; hottest {cv}). "
                    "Packed Nose-Hoover couples replicas — prefer Langevin; "
                    "soften --k/--ky or remove that grid corner."
                )
            print(
                f"  step {step:6d}  E_total={e_tot:.4f} eV  "
                f"T={t_curr:.1f} K  "
                f"T_win[min/med/max]="
                f"{float(np.min(t_win)):.0f}/"
                f"{float(np.median(t_win)):.0f}/"
                f"{float(np.max(t_win)):.0f}"
            )

    print(
        f"=== MD finished ({cfg.nsteps} steps); writing snapshots "
        f"({k_windows} windows × {len(frames_traj)} frames) ==="
    )
    positions = np.stack(frames_traj, axis=1)
    cv_traj = np.stack(cv_frames, axis=1)  # (K, N_frames, ndim)
    energies = np.stack(energy_frames, axis=1)  # (K, N_frames)
    n_frames = int(positions.shape[1])
    minima_pos, minima_idx, minima_e = select_lowest_energy_frames(
        positions, energies
    )

    extra = {
        "ndim": np.int32(sched.ndim),
        "grid_shape": np.asarray(sched.grid_shape, dtype=np.int32),
        "energies_ev": np.asarray(energies, dtype=np.float64),
        "bin_minima_frame_idx": np.asarray(minima_idx, dtype=np.int64),
        "bin_minima_energy_ev": np.asarray(minima_e, dtype=np.float64),
    }
    if sched.ndim == 2:
        assert sched.yi0 is not None and sched.k_y is not None
        extra["yi0"] = np.asarray(sched.yi0, dtype=np.float64)
        extra["k_y_ev_A2"] = np.asarray(sched.k_y, dtype=np.float64)
        extra["atom_k"] = np.int32(sched.atom_pairs[1][0])
        extra["atom_l"] = np.int32(sched.atom_pairs[1][1])

    snapshots_path = output_dir / SNAPSHOTS_NPZ
    print(f"  writing {snapshots_path.name} …")
    save_snapshots(
        snapshots_path,
        positions=positions,
        Z=z,
        atom_i=sched.atom_pairs[0][0],
        atom_j=sched.atom_pairs[0][1],
        xi0=np.asarray(sched.xi0, dtype=np.float64),
        k_ev_A2=np.asarray(sched.k_x, dtype=np.float64),
        temperature_K=float(cfg.temperature_K),
        dt_fs=dt,
        cv_traj=cv_traj,
        checkpoint=str(Path(cfg.checkpoint).expanduser().resolve()),
        extra=extra,
    )
    print(f"  snapshots done → {snapshots_path}")

    traj_paths: dict[str, Path] = {}
    from ase import Atoms

    minima_centered = center_com_positions(minima_pos, masses)
    minima_path = output_dir / BIN_MINIMA_TRAJ
    minima_frames = [
        Atoms(
            numbers=z,
            positions=minima_centered[wid],
            masses=masses,
            info={
                "window": wid,
                "frame_idx": int(minima_idx[wid]),
                "energy_ev": float(minima_e[wid]),
            },
        )
        for wid in range(k_windows)
    ]
    write(minima_path, minima_frames)
    traj_paths["bin_minima"] = minima_path
    print(
        f"  wrote {minima_path.name} "
        f"({k_windows} CoM-centered lowest-E_ML+W frames)"
    )

    if cfg.write_window_xyz:
        print(
            f"  writing {k_windows} window XYZ trajectories "
            f"({n_frames} frames each, CoM→origin) …"
        )
        for wid in range(k_windows):
            traj_path = output_dir / f"umbrella_window{wid:03d}.xyz"
            centered = center_com_positions(positions[wid], masses)
            frames = [
                Atoms(numbers=z, positions=centered[frame_idx], masses=masses)
                for frame_idx in range(n_frames)
            ]
            write(traj_path, frames)
            traj_paths[f"window_{wid:03d}"] = traj_path
            if (wid + 1) % max(1, k_windows // 8) == 0 or wid + 1 == k_windows:
                print(f"    XYZ {wid + 1}/{k_windows}")
    else:
        print(
            "  skipping per-window XYZ (MBAR uses NPZ only); "
            "pass --write-window-xyz to export trajectories"
        )

    summary = {
        "args": cfg.to_dict(),
        "ndim": sched.ndim,
        "n_windows": k_windows,
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "atom_pairs": [list(p) for p in sched.atom_pairs],
        "xi0": list(sched.xi0),
        "yi0": list(sched.yi0) if sched.yi0 is not None else None,
        "k_ev_A2": list(sched.k_x),
        "k_y_ev_A2": list(sched.k_y) if sched.k_y is not None else None,
        "grid_shape": list(sched.grid_shape),
        "r0_cv_A": r0_cvs,
        "seed_mode": seed_mode,
        "replica_exchange": bool(cfg.replica_exchange),
        "rex_freq": int(cfg.rex_freq) if cfg.replica_exchange else None,
        "rex_attempted": None if rex_stats is None else rex_stats.attempted,
        "rex_accepted": None if rex_stats is None else rex_stats.accepted,
        "rex_acceptance": None if rex_stats is None else rex_stats.acceptance,
        "cv_mean": cv_traj.mean(axis=1).tolist(),
        "cv_std": cv_traj.std(axis=1).tolist(),
        "snapshots": str(snapshots_path),
        "bin_minima": str(minima_path),
        "bin_minima_frame_idx": minima_idx.tolist(),
        "bin_minima_energy_ev": minima_e.tolist(),
    }
    summary_path = write_summary(output_dir / SUMMARY_JSON, summary)

    paths: dict[str, Path] = {
        "snapshots": snapshots_path,
        "summary": summary_path,
        **traj_paths,
    }
    return UmbrellaResult(
        output_dir=output_dir,
        snapshots_path=snapshots_path,
        summary_path=summary_path,
        n_windows=k_windows,
        n_frames=n_frames,
        paths=paths,
    )

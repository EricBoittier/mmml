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
from mmml.umbrella.io import SNAPSHOTS_NPZ, SUMMARY_JSON, save_snapshots, write_summary
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
    energy_sum_fn = jax.jit(energy_sum_fn)
    force_fn = jax.jit(force_fn)

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
    init_fn, apply_fn = simulate.nvt_nose_hoover(force_fn, shift, dt, kt)
    apply_fn = jax.jit(apply_fn)

    key = jax.random.PRNGKey(int(cfg.seed))
    state = init_fn(key, r_packed, mass=masses_batched)

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

    print(
        f"=== Umbrella NVT ({k_windows} windows, {sched.ndim}D, "
        f"T={cfg.temperature_K} K, dt={dt} fs) ==="
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
    for step in range(1, int(cfg.nsteps) + 1):
        state = apply_fn(state)
        if step % savefreq == 0 or step == cfg.nsteps:
            pos_batch = np.asarray(state.position).reshape(k_windows, n_atoms, 3)
            frames_traj.append(pos_batch)
            cv_frames.append(_cv_frame(state.position))
        if step % int(cfg.printfreq) == 0 or step == cfg.nsteps:
            e_tot = float(energy_sum_fn(state.position))
            t_curr = float(
                quantity.temperature(momentum=state.momentum, mass=state.mass) / k_b
            )
            if not np.isfinite(e_tot) or not np.isfinite(t_curr):
                raise RuntimeError(
                    f"non-finite thermodynamics at step {step}: "
                    f"E={e_tot} T={t_curr}. Try smaller --timestep (default 0.1 fs), "
                    "softer --k, --move-with for rigid groups, or --seed-mode frames."
                )
            print(f"  step {step:6d}  E_total={e_tot:.4f} eV  T={t_curr:.1f} K")

    positions = np.stack(frames_traj, axis=1)
    cv_traj = np.stack(cv_frames, axis=1)  # (K, N_frames, ndim)
    n_frames = int(positions.shape[1])

    extra = {
        "ndim": np.int32(sched.ndim),
        "grid_shape": np.asarray(sched.grid_shape, dtype=np.int32),
    }
    if sched.ndim == 2:
        assert sched.yi0 is not None and sched.k_y is not None
        extra["yi0"] = np.asarray(sched.yi0, dtype=np.float64)
        extra["k_y_ev_A2"] = np.asarray(sched.k_y, dtype=np.float64)
        extra["atom_k"] = np.int32(sched.atom_pairs[1][0])
        extra["atom_l"] = np.int32(sched.atom_pairs[1][1])

    snapshots_path = output_dir / SNAPSHOTS_NPZ
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

    traj_paths: dict[str, Path] = {}
    for wid in range(k_windows):
        traj_path = output_dir / f"umbrella_window{wid:03d}.xyz"
        for frame_idx in range(n_frames):
            from ase import Atoms

            at = Atoms(numbers=z, positions=positions[wid, frame_idx])
            write(traj_path, at, append=(frame_idx > 0))
        traj_paths[f"window_{wid:03d}"] = traj_path

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
        "cv_mean": cv_traj.mean(axis=1).tolist(),
        "cv_std": cv_traj.std(axis=1).tolist(),
        "snapshots": str(snapshots_path),
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

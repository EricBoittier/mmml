"""Batched JAX-MD NVT Nose-Hoover umbrella sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmml.umbrella.config import UmbrellaConfig
from mmml.umbrella.energy import (
    build_packed_graph,
    make_packed_energy_fn,
    pack_positions,
    packed_cv_distances,
)
from mmml.umbrella.io import SNAPSHOTS_NPZ, SUMMARY_JSON, save_snapshots, write_summary


@dataclass(frozen=True)
class UmbrellaResult:
    """Artifacts from :func:`run_umbrella_nvt`."""

    output_dir: Path
    snapshots_path: Path
    summary_path: Path
    n_windows: int
    n_frames: int
    paths: dict[str, Path]


def _load_structure(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from ase.io import read

    atoms = read(str(path))
    if isinstance(atoms, list):
        atoms = atoms[0]
    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    r = np.asarray(atoms.get_positions(), dtype=np.float64)
    return r, z


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

    r0, z = _load_structure(Path(cfg.structure).expanduser().resolve())
    n_atoms = int(len(z))
    if max(cfg.atom_i, cfg.atom_j) >= n_atoms:
        raise ValueError(
            f"atom indices ({cfg.atom_i}, {cfg.atom_j}) out of range for "
            f"{n_atoms} atoms"
        )

    targets = cfg.resolve_targets()
    ks = cfg.resolve_force_constants()
    k_windows = len(targets)

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
        atom_i=cfg.atom_i,
        atom_j=cfg.atom_j,
        targets_A=targets,
        k_ev_A2=ks,
    )
    energy_sum_fn = jax.jit(energy_sum_fn)

    masses = np.array([atomic_masses[int(zi)] for zi in z], dtype=np.float64)
    masses_batched = jnp.tile(jnp.asarray(masses), k_windows)
    r_packed = jnp.asarray(pack_positions(r0, k_windows), dtype=jnp.float64)

    k_b = 8.617333262145e-5  # eV/K
    kt = k_b * float(cfg.temperature_K)
    dt = float(cfg.timestep_fs)
    savefreq = cfg.effective_savefreq()

    _, shift = space.free()
    init_fn, apply_fn = simulate.nvt_nose_hoover(energy_sum_fn, shift, dt, kt)
    apply_fn = jax.jit(apply_fn)

    key = jax.random.PRNGKey(int(cfg.seed))
    state = init_fn(key, r_packed, mass=masses_batched)

    frames: list[np.ndarray] = [
        np.asarray(state.position).reshape(k_windows, n_atoms, 3)
    ]
    cv_frames: list[np.ndarray] = [
        np.asarray(
            packed_cv_distances(
                state.position, n_atoms, cfg.atom_i, cfg.atom_j, k_windows
            )
        )
    ]

    print(
        f"=== Umbrella NVT ({k_windows} windows batched, "
        f"T={cfg.temperature_K} K, dt={dt} fs) ==="
    )
    for step in range(1, int(cfg.nsteps) + 1):
        state = apply_fn(state)
        if step % savefreq == 0 or step == cfg.nsteps:
            pos_batch = np.asarray(state.position).reshape(k_windows, n_atoms, 3)
            frames.append(pos_batch)
            cv_frames.append(
                np.asarray(
                    packed_cv_distances(
                        state.position, n_atoms, cfg.atom_i, cfg.atom_j, k_windows
                    )
                )
            )
        if step % int(cfg.printfreq) == 0 or step == cfg.nsteps:
            e_tot = float(energy_sum_fn(state.position))
            t_curr = float(
                quantity.temperature(momentum=state.momentum, mass=state.mass) / k_b
            )
            print(f"  step {step:6d}  E_total={e_tot:.4f} eV  T={t_curr:.1f} K")

    # frames: list of (K, N, 3) → (K, N_frames, N, 3)
    positions = np.stack(frames, axis=1)
    cv_traj = np.stack(cv_frames, axis=1)
    n_frames = int(positions.shape[1])

    snapshots_path = output_dir / SNAPSHOTS_NPZ
    save_snapshots(
        snapshots_path,
        positions=positions,
        Z=z,
        atom_i=cfg.atom_i,
        atom_j=cfg.atom_j,
        xi0=np.asarray(targets, dtype=np.float64),
        k_ev_A2=np.asarray(ks, dtype=np.float64),
        temperature_K=float(cfg.temperature_K),
        dt_fs=dt,
        cv_traj=cv_traj,
        checkpoint=str(Path(cfg.checkpoint).expanduser().resolve()),
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
        "n_windows": k_windows,
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "xi0": list(targets),
        "k_ev_A2": list(ks),
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

"""JAX-MD equilibration / production for lambda TI (shared with lambda_dynamics.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase.data import atomic_masses
from ase.io.trajectory import Trajectory
from jax import jit
from jax_md import simulate, space, units as jax_md_units

from mmml.cli.run.lambda_dynamics import (
    LambdaDynamicsConfig,
    LambdaMdSettings,
    ensure_jax_cuda_toolchain,
    is_lambda_prod_complete,
    lambda_min_traj_path,
    lambda_prod_traj_path,
    lambda_repeat_label,
    lambda_traj_frame_count,
    load_lambda_start_positions,
    load_lambda_traj_positions,
    minimize_lambda_structure,
    prepare_atoms_geometry,
    print_lambda_resume_plan,
    resolve_model_restart_path,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator

import pycharmm.param as param
import pycharmm.psf as psf


@dataclass
class LambdaJaxMdBundle:
    """JIT MMML spherical calculators for production (λ) and TI probes (λ=1, λ=0)."""

    wrapped_force_fn: Callable
    spherical_prod: Callable
    spherical_on: Callable
    spherical_off: Callable
    shift: Callable
    masses: jnp.ndarray
    atomic_numbers: jnp.ndarray
    n_monomers: int
    cutoff: CutoffParameters
    use_pbc: bool
    box_L: float | None
    get_update_fn: Callable | None
    pbc_state: dict[str, Any]


def _warm_mmml_spherical_cache(
    *,
    get_update_fn: Callable | None,
    positions: np.ndarray,
    cutoff: CutoffParameters,
) -> None:
    """Build MM energy/force caches outside JIT (mirrors ASE ``calculate`` / jaxmd_runner)."""
    if get_update_fn is None:
        raise RuntimeError(
            "MMML calculator returned no get_update_fn; cannot warm MM cache for JAX-MD."
        )
    get_update_fn(np.asarray(positions, dtype=float), cutoff)


def _ensemble_from_md_settings(md_settings: LambdaMdSettings) -> str:
    if md_settings.integrator == "nve":
        return "nve"
    if md_settings.integrator == "nvt_langevin":
        raise ValueError(
            "lambda_ti --backend jaxmd does not support Langevin NVT; "
            "use --nvt-integrator nhc or --lambda-md-mode free_nve|pbc_nve."
        )
    return "nvt"


def _zero_com_momentum(momentum: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    vel = momentum / masses[:, None]
    com_v = jnp.sum(masses[:, None] * vel, axis=0) / jnp.sum(masses)
    return momentum - masses[:, None] * com_v


def _center_positions(pos: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    com = jnp.sum(masses[:, None] * pos, axis=0) / jnp.sum(masses)
    return pos - com


def build_lambda_jaxmd_bundle(
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    model_restart_path: Path,
    atoms_per_monomer: list[int],
    n_monomers: int,
    couple_indices: list[int],
    lam_coupled: float,
    cutoff: CutoffParameters,
    ml_cutoff: float,
    mm_switch_on: float,
    mm_cutoff: float,
    at_codes: np.ndarray,
    ep_scale: np.ndarray,
    sig_scale: np.ndarray,
    cell_scalar: float | None,
    flat_bottom_radius: float | None,
    flat_bottom_k: float,
    md_settings: LambdaMdSettings,
) -> LambdaJaxMdBundle:
    """Build production + probe spherical calculators and a JAX-MD force function."""

    def _factory(lam_val: float):
        lam_arr = lambda_array(n_monomers, couple_indices, lam_val)
        max_atoms_per = int(max(atoms_per_monomer))
        cell_arg = float(cell_scalar) if cell_scalar is not None else None
        factory = setup_calculator(
            ATOMS_PER_MONOMER=atoms_per_monomer,
            N_MONOMERS=n_monomers,
            ml_cutoff_distance=ml_cutoff,
            mm_switch_on=mm_switch_on,
            mm_cutoff=mm_cutoff,
            doML=True,
            doMM=True,
            doML_dimer=True,
            debug=False,
            model_restart_path=model_restart_path,
            MAX_ATOMS_PER_SYSTEM=max_atoms_per * 2,
            cell=cell_arg,
            ep_scale=ep_scale,
            sig_scale=sig_scale,
            at_codes_override=at_codes,
            lambda_monomer=lam_arr,
            flat_bottom_radius=flat_bottom_radius,
            flat_bottom_force_const=flat_bottom_k,
        )
        return factory(
            atomic_numbers=atomic_numbers,
            atomic_positions=positions,
            n_monomers=n_monomers,
            cutoff_params=cutoff,
            doML=True,
            doMM=True,
            doML_dimer=True,
            backprop=False,
            debug=False,
            create_ase_calculator=False,
        )

    _, spherical_prod, get_update_fn = _factory(lam_coupled)
    _, spherical_on, get_update_on = _factory(1.0)
    _, spherical_off, get_update_off = _factory(0.0)

    pos_np = np.asarray(positions, dtype=float)
    for upd in (get_update_fn, get_update_on, get_update_off):
        _warm_mmml_spherical_cache(get_update_fn=upd, positions=pos_np, cutoff=cutoff)

    z_jnp = jnp.asarray(atomic_numbers, dtype=jnp.int32)
    use_pbc = md_settings.use_pbc
    box_L = float(md_settings.box_L) if md_settings.box_L is not None else None
    box_init = (
        jnp.array([box_L, box_L, box_L], dtype=jnp.float32) if use_pbc and box_L else None
    )
    pbc_state: dict[str, Any] = {"box": box_init, "pair_idx": None, "pair_mask": None}
    if use_pbc and get_update_fn is not None and box_L is not None:
        update_fn = get_update_fn(pos_np, cutoff)
        if update_fn is not None:
            box_nl = np.array([box_L, box_L, box_L], dtype=np.float64)
            pair_idx, pair_mask = update_fn(pos_np, box=box_nl)
            pbc_state["pair_idx"] = pair_idx
            pbc_state["pair_mask"] = pair_mask

    if use_pbc and box_L is not None:
        displacement, shift = space.periodic(box_L)
    else:
        displacement, shift = space.free()

    @jit
    def jax_md_force_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None):
        pos = jnp.asarray(position, dtype=jnp.float32)
        out = spherical_prod(
            pos,
            z_jnp,
            n_monomers,
            cutoff,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
        )
        return out.forces

    @jit
    def wrapped_force_fn(position, **_kwargs):
        return jax_md_force_fn(
            position,
            mm_pair_idx=pbc_state["pair_idx"],
            mm_pair_mask=pbc_state["pair_mask"],
            box=pbc_state["box"],
        )

    masses = jnp.asarray(atomic_masses[np.asarray(atomic_numbers, dtype=int)], dtype=jnp.float32)

    return LambdaJaxMdBundle(
        wrapped_force_fn=wrapped_force_fn,
        spherical_prod=spherical_prod,
        spherical_on=spherical_on,
        spherical_off=spherical_off,
        shift=shift,
        masses=masses,
        atomic_numbers=z_jnp,
        n_monomers=n_monomers,
        cutoff=cutoff,
        use_pbc=use_pbc,
        box_L=box_L,
        get_update_fn=get_update_fn,
        pbc_state=pbc_state,
    )


def _refresh_pbc_neighbors(bundle: LambdaJaxMdBundle, position: np.ndarray) -> None:
    if not bundle.use_pbc or bundle.get_update_fn is None or bundle.box_L is None:
        return
    pos = np.asarray(position, dtype=float)
    update_fn = bundle.get_update_fn(pos, bundle.cutoff)
    if update_fn is None:
        return
    box_nl = np.array([bundle.box_L, bundle.box_L, bundle.box_L], dtype=np.float64)
    pair_idx, pair_mask = update_fn(pos, box=box_nl)
    bundle.pbc_state["pair_idx"] = pair_idx
    bundle.pbc_state["pair_mask"] = pair_mask
    bundle.pbc_state["box"] = bundle.pbc_state["box"]


def _interaction_energy_eV(
    bundle: LambdaJaxMdBundle,
    spherical_fn: Callable,
    position: np.ndarray,
) -> float:
    """Evaluate inter-monomer ML+MM energy (eV) via the pre-JIT spherical calculator."""
    if bundle.use_pbc:
        _refresh_pbc_neighbors(bundle, np.asarray(position, dtype=float))
    pos = jnp.asarray(position, dtype=jnp.float32)
    box = bundle.pbc_state["box"]
    pair_idx = bundle.pbc_state["pair_idx"]
    pair_mask = bundle.pbc_state["pair_mask"]
    out = spherical_fn(
        pos,
        bundle.atomic_numbers,
        bundle.n_monomers,
        bundle.cutoff,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box,
    )
    e = jnp.sum(jnp.asarray(out.ml_2b_E)) + jnp.sum(jnp.asarray(out.mm_E))
    return float(jax.device_get(e))


def _dudl_at_position(bundle: LambdaJaxMdBundle, position: np.ndarray) -> float:
    e_on = _interaction_energy_eV(bundle, bundle.spherical_on, position)
    e_off = _interaction_energy_eV(bundle, bundle.spherical_off, position)
    return e_on - e_off


def _init_jaxmd_state(
    bundle: LambdaJaxMdBundle,
    positions: np.ndarray,
    *,
    ensemble: str,
    dt_fs: float,
    temperature_K: float,
    seed: int,
    remove_net_drift: bool,
    use_fix_com: bool,
):
    unit = jax_md_units.metal_unit_system()
    dt = float(dt_fs) * 0.001
    kT = float(temperature_K) * unit["temperature"]
    key = jax.random.PRNGKey(int(seed))

    R = jnp.asarray(positions, dtype=jnp.float32)
    if use_fix_com:
        R = _center_positions(R, bundle.masses)

    if bundle.use_pbc and bundle.box_L is not None:
        _refresh_pbc_neighbors(bundle, np.asarray(jax.device_get(R), dtype=float))

    if ensemble == "nvt":
        nhc_tau = 100.0
        init_fn, apply_fn = simulate.nvt_nose_hoover(
            bundle.wrapped_force_fn,
            bundle.shift,
            dt=dt,
            kT=kT,
            thermostat_kwargs={
                "chain_length": 3,
                "chain_steps": 2,
                "sy_steps": 3,
                "tau": jnp.array(nhc_tau * dt),
            },
        )
        state = init_fn(key, R, mass=bundle.masses)
    else:
        init_fn, apply_fn = simulate.nve(bundle.wrapped_force_fn, bundle.shift, dt)
        state = init_fn(key, R, kT, mass=bundle.masses)

    if remove_net_drift:
        state = state.set(momentum=_zero_com_momentum(state.momentum, bundle.masses))
    return jit(apply_fn), state, dt


def run_jaxmd_segment(
    bundle: LambdaJaxMdBundle,
    positions: np.ndarray,
    *,
    n_steps: int,
    interval: int,
    dt_fs: float,
    temperature_K: float,
    ensemble: str,
    seed: int,
    remove_net_drift: bool,
    use_fix_com: bool,
    traj_path: Path | None,
    neighbor_update_interval: int,
    z: np.ndarray,
    md_settings: LambdaMdSettings,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Run JAX-MD; return (dU/dλ samples, snapshot positions, traj frames)."""
    apply_fn, state, _dt = _init_jaxmd_state(
        bundle,
        positions,
        ensemble=ensemble,
        dt_fs=dt_fs,
        temperature_K=temperature_K,
        seed=seed,
        remove_net_drift=remove_net_drift,
        use_fix_com=use_fix_com,
    )

    samples: list[float] = []
    snapshots: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    traj: Trajectory | None = None
    if traj_path is not None:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        template = ase.Atoms(numbers=z, positions=positions)
        prepare_atoms_geometry(template, positions, md_settings)
        traj = Trajectory(str(traj_path), "w", template)

    interval = max(1, int(interval))
    nbr_every = max(1, int(neighbor_update_interval))

    for step in range(1, int(n_steps) + 1):
        if bundle.use_pbc and step % nbr_every == 0:
            _refresh_pbc_neighbors(
                bundle, np.asarray(jax.device_get(state.position), dtype=float)
            )
        state = apply_fn(state)
        if remove_net_drift and step % interval == 0:
            state = state.set(
                momentum=_zero_com_momentum(state.momentum, bundle.masses)
            )

        if step % interval == 0:
            pos_np = np.asarray(jax.device_get(state.position), dtype=float)
            samples.append(_dudl_at_position(bundle, pos_np))
            snapshots.append(pos_np.copy())
            frames.append(pos_np.copy())
            if traj is not None:
                frame = ase.Atoms(numbers=z, positions=pos_np)
                prepare_atoms_geometry(frame, pos_np, md_settings)
                traj.write(frame)

    if traj is not None:
        traj.close()
    final_pos = np.asarray(jax.device_get(state.position), dtype=float)
    return samples, snapshots, frames, final_pos


def run_lambda_dynamics_jaxmd(cfg: LambdaDynamicsConfig) -> dict[str, Any]:
    """λ-window TI with JAX-MD (CHARMM + ASE BFGS minimization unchanged)."""
    from mmml.cli.run.lambda_dynamics import (
        SNAPSHOTS_NPZ,
        SUMMARY_JSON,
        _print_run_banner,
        build_cluster_system,
        parse_couple_residue_list,
        plot_window_components,
        print_cluster_psf_monomer_diagnostics,
        repo_root_from_here,
        resolve_lambda_md_settings,
        save_snapshots_npz,
        snapshot_metadata_from_cluster,
    )

    ensure_jax_cuda_toolchain()

    if cfg.interval < 1:
        raise ValueError("--interval must be >= 1")

    repo_root = cfg.repo_root or repo_root_from_here()
    out_dir = cfg.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_run_banner(cfg, backend="jaxmd")

    cluster = build_cluster_system(cfg)
    z = cluster.z
    base_seed_positions = cluster.base_seed_positions
    couple_indices = parse_couple_residue_list(cfg.couple_residue_numbers, cluster.n_monomers)
    couple_residue_numbers_1b = [i + 1 for i in couple_indices]
    print_cluster_psf_monomer_diagnostics(cluster, monomer_index=couple_indices[0])

    model_restart_path = resolve_model_restart_path(cfg.checkpoint)
    md_settings = resolve_lambda_md_settings(cfg, base_seed_positions)
    ensemble = _ensemble_from_md_settings(md_settings)

    at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    ep_scale = np.ones(n_types, dtype=float)
    sig_scale = np.ones(n_types, dtype=float)
    cutoff = CutoffParameters(
        ml_cutoff=cfg.ml_cutoff,
        mm_switch_on=cfg.mm_switch_on,
        mm_cutoff=cfg.mm_cutoff,
    )
    cell_scalar = float(md_settings.box_L) if md_settings.use_pbc and md_settings.box_L else None

    use_fix_com = not cfg.no_fix_com
    remove_net_drift = not cfg.no_stationary
    lambda_windows = sorted(float(x) for x in cfg.lambda_windows)
    snap_meta = snapshot_metadata_from_cluster(cluster, cfg, couple_residue_numbers_1b)

    calc_common = dict(
        atomic_numbers=z,
        base_ckpt_dir=model_restart_path,
        atoms_per_monomer=cluster.atoms_per_monomer,
        n_monomers=cluster.n_monomers,
        couple_indices=couple_indices,
        cutoff=cutoff,
        ml_cutoff=cfg.ml_cutoff,
        mm_switch_on=cfg.mm_switch_on,
        mm_cutoff=cfg.mm_cutoff,
        at_codes=at_codes,
        ep_scale=ep_scale,
        sig_scale=sig_scale,
        cell_scalar=cell_scalar,
        flat_bottom_radius=cfg.flat_bottom_radius,
        flat_bottom_k=cfg.flat_bottom_k,
    )

    rows: list[dict] = []
    snapshots_per_window: list[list[np.ndarray]] = [[] for _ in range(len(lambda_windows))]
    traj_root = out_dir / "trajectories"
    traj_root.mkdir(exist_ok=True)

    if cfg.resume:
        print("=== lambda_ti resume ===", flush=True)
        print_lambda_resume_plan(
            traj_root,
            lambda_windows,
            repeats_per_window=cfg.repeats_per_window,
            n_prod=cfg.n_prod,
            interval=cfg.interval,
        )

    neighbor_update_interval = 10

    def _build_bundle(r_min: np.ndarray) -> LambdaJaxMdBundle:
        calc_kw = dict(calc_common)
        calc_kw["model_restart_path"] = calc_kw.pop("base_ckpt_dir")
        bundle = build_lambda_jaxmd_bundle(
            positions=r_min,
            lam_coupled=lam,
            md_settings=md_settings,
            **calc_kw,
        )
        _ = float(
            np.abs(
                jax.device_get(
                    bundle.wrapped_force_fn(jnp.asarray(r_min, dtype=jnp.float32))
                )
            ).max()
        )
        return bundle

    for wi, lam in enumerate(lambda_windows):
        samples: list[float] = []
        repeat_stats: list[dict[str, float | int | str]] = []
        for rep in range(cfg.repeats_per_window):
            label = lambda_repeat_label(wi, rep, lam)
            prod_path = lambda_prod_traj_path(traj_root, label)
            min_traj_path = lambda_min_traj_path(traj_root, label)

            if cfg.resume and is_lambda_prod_complete(prod_path, cfg.n_prod, cfg.interval):
                positions = load_lambda_traj_positions(prod_path)
                r_min = load_lambda_start_positions(traj_root, label, n_equil=cfg.n_equil)
                if r_min is None:
                    r_min = positions[0]
                bundle = _build_bundle(r_min)
                samples_rep = [_dudl_at_position(bundle, p) for p in positions]
                samples.extend(samples_rep)
                snapshots_per_window[wi].extend([p.copy() for p in positions])
                repeat_stats.append(
                    {
                        "repeat": rep,
                        "n_samples": len(samples_rep),
                        "mean_dUdlambda_eV": float(np.mean(samples_rep)) if samples_rep else float("nan"),
                        "std_dUdlambda_eV": float(np.std(samples_rep)) if len(samples_rep) > 1 else 0.0,
                        "traj": str(prod_path),
                        "equil_traj": None,
                        "minimization": {"resumed": True, "skipped": "complete_prod_traj"},
                    }
                )
                continue

            if cfg.resume and prod_path.is_file():
                prod_path.unlink()

            r_init = base_seed_positions.copy()
            atoms = ase.Atoms(numbers=z, positions=r_init)
            prepare_atoms_geometry(atoms, r_init, md_settings)

            skip_min = cfg.resume and lambda_traj_frame_count(min_traj_path) > 0
            if skip_min:
                r_min = load_lambda_start_positions(traj_root, label, n_equil=0)
                if r_min is None:
                    raise RuntimeError(f"resume: missing minimized structure for {label}")
                atoms.set_positions(r_min)
                min_summary = {"resumed": True, "from": str(min_traj_path)}
            else:
                min_summary = minimize_lambda_structure(
                    atoms,
                    cfg=cfg,
                    cluster=cluster,
                    model_restart_path=model_restart_path,
                    couple_indices=couple_indices,
                    lam_coupled=lam,
                    md_settings=md_settings,
                    label=label,
                    min_traj_path=min_traj_path,
                )
                r_min = np.asarray(atoms.get_positions(), dtype=float)

            bundle = _build_bundle(r_min)

            r_md = r_min
            equil_path: str | None = None
            if cfg.n_equil > 0:
                equil_path = str(traj_root / f"{label}_eq.traj") if cfg.save_equil_traj else None
                eq_interval = cfg.equil_traj_interval or cfg.interval
                _, _, _, r_md = run_jaxmd_segment(
                    bundle,
                    r_md,
                    n_steps=cfg.n_equil,
                    interval=max(1, int(eq_interval)),
                    dt_fs=cfg.timestep_fs,
                    temperature_K=cfg.temperature_K,
                    ensemble=ensemble,
                    seed=cfg.seed + wi * 1000 + rep,
                    remove_net_drift=remove_net_drift,
                    use_fix_com=use_fix_com,
                    traj_path=Path(equil_path) if equil_path else None,
                    neighbor_update_interval=neighbor_update_interval,
                    z=z,
                    md_settings=md_settings,
                )
                atoms.set_positions(r_md)

            samples_rep, snap_rep, _, _ = run_jaxmd_segment(
                bundle,
                r_md,
                n_steps=cfg.n_prod,
                interval=cfg.interval,
                dt_fs=cfg.timestep_fs,
                temperature_K=cfg.temperature_K,
                ensemble=ensemble,
                seed=cfg.seed + wi * 1000 + rep + 1,
                remove_net_drift=remove_net_drift,
                use_fix_com=use_fix_com,
                traj_path=prod_path,
                neighbor_update_interval=neighbor_update_interval,
                z=z,
                md_settings=md_settings,
            )
            samples.extend(samples_rep)
            snapshots_per_window[wi].extend(snap_rep)

            repeat_stats.append(
                {
                    "repeat": rep,
                    "n_samples": len(samples_rep),
                    "mean_dUdlambda_eV": float(np.mean(samples_rep)) if samples_rep else float("nan"),
                    "std_dUdlambda_eV": float(np.std(samples_rep)) if len(samples_rep) > 1 else 0.0,
                    "traj": str(prod_path),
                    "equil_traj": equil_path,
                    "minimization": min_summary,
                }
            )

        mean_b = float(np.mean(samples)) if samples else float("nan")
        std_b = float(np.std(samples)) if len(samples) > 1 else 0.0
        rows.append(
            {
                "window": wi,
                "lambda_coupled": lam,
                "couple_residue_numbers": couple_residue_numbers_1b,
                "repeats_per_window": cfg.repeats_per_window,
                "mean_dUdlambda_eV": mean_b,
                "std_dUdlambda_eV": std_b,
                "n_samples": len(samples),
                "repeat_stats": repeat_stats,
            }
        )

    from mmml.cli.run.lambda_dynamics import _EV_TO_KCAL, asdict

    lam_col = np.array([r["lambda_coupled"] for r in rows], dtype=float)
    mean_b = np.array([r["mean_dUdlambda_eV"] for r in rows], dtype=float)
    delta_f_ev = float(np.trapezoid(mean_b, lam_col)) if len(lam_col) > 1 else float("nan")
    delta_f_kcal = delta_f_ev * _EV_TO_KCAL

    snap_path = out_dir / SNAPSHOTS_NPZ
    save_snapshots_npz(
        snap_path,
        atomic_numbers=z,
        lambda_windows=lambda_windows,
        snapshots_per_window=snapshots_per_window,
        snapshot_meta=snap_meta,
    )

    coupled_labels = [f"{cluster.residue_labels[i]}#{i + 1}" for i in couple_indices]
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items() if k != "repo_root"}
    summary = {
        "system": {
            "composition": cluster.composition_summary,
            "composition_str": cluster.composition_str,
            "residue_labels": cluster.residue_labels,
            "n_molecules": cluster.n_monomers,
            "n_atoms": len(z),
            "spacing_A": float(cfg.spacing),
            "placement_seed": int(cfg.seed),
            "couple_residue_numbers": couple_residue_numbers_1b,
            "couple_residue_labels": coupled_labels,
            "backend": "jaxmd",
            "md_mode": cfg.md_mode,
            "integrator": md_settings.integrator,
            "jaxmd_ensemble": ensemble,
            "use_pbc": md_settings.use_pbc,
            "box_A": md_settings.box_L,
            "charmm_pre_minimize": cfg.charmm_pre_minimize,
            "calculator_pre_minimize": cfg.calculator_pre_minimize,
            "fix_com": use_fix_com,
            "no_fix_com": cfg.no_fix_com,
            "no_stationary": cfg.no_stationary,
        },
        "description": {
            "delta_F_couple_eV": "TI integral ∫ ⟨∂U/∂λ⟩ dλ (JAX-MD production)",
            "delta_F_diss_eV": "Negative of delta_F_couple",
            "lambda_definition": (
                f"Residues {couple_residue_numbers_1b} share λ; inter-monomer ML/MM scale as λ_i λ_j."
            ),
        },
        "delta_F_couple_eV": delta_f_ev,
        "delta_F_couple_kcal_mol": delta_f_kcal,
        "delta_F_diss_eV": -delta_f_ev,
        "delta_F_diss_kcal_mol": -delta_f_kcal,
        "mbar": None,
        "snapshots_npz": str(snap_path),
        "windows": rows,
        "args": cfg_dict,
    }
    plot_files = plot_window_components(out_dir, rows, None)
    summary["plots"] = plot_files

    out_json = out_dir / SUMMARY_JSON
    out_json.write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary["_summary_path"] = str(out_json)
    return summary

#!/usr/bin/env python3
"""Generalized MMML lambda dynamics (TI sampling) and MBAR post-processing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import ase
import numpy as np
import pandas as pd
from ase import units
from ase.constraints import FixCom
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.optimize.fire import FIRE

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.cli.base import resolve_checkpoint_paths
from mmml.cli.run.md_pbc_suite import ase as md_suite
from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster
from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator

import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings

pyci.read = read
pyci.settings = settings
pyci.psf = psf

_EV_TO_KCAL = 23.0609
SNAPSHOTS_NPZ = "lambda_ti_snapshots.npz"
SUMMARY_JSON = "lambda_ti_summary.json"


from mmml.utils.jax_gpu_warmup import ensure_jax_cuda_toolchain


def lambda_repeat_label(wi: int, rep: int, lam: float) -> str:
    return f"win_{wi:02d}_rep{rep:02d}_lam{lam:.2f}"


def lambda_expected_prod_frames(n_prod: int, interval: int) -> int:
    return int(n_prod) // max(1, int(interval))


def lambda_prod_traj_path(traj_root: Path, label: str) -> Path:
    return traj_root / f"{label}_prod.traj"


def lambda_min_traj_path(traj_root: Path, label: str) -> Path:
    return traj_root / f"{label}_min.traj"


def lambda_traj_frame_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with Trajectory(str(path), "r") as tr:
            return len(tr)
    except Exception:
        return 0


def is_lambda_prod_complete(path: Path, n_prod: int, interval: int) -> bool:
    expected = lambda_expected_prod_frames(n_prod, interval)
    if expected < 1:
        return path.is_file()
    return lambda_traj_frame_count(path) >= expected


def load_lambda_traj_positions(path: Path) -> list[np.ndarray]:
    with Trajectory(str(path), "r") as tr:
        return [np.asarray(at.get_positions(), dtype=float) for at in tr]


def load_lambda_start_positions(
    traj_root: Path,
    label: str,
    *,
    n_equil: int,
) -> np.ndarray | None:
    """Last frame after minimization (and equilibration, if any)."""
    if n_equil > 0:
        eq_path = traj_root / f"{label}_eq.traj"
        if lambda_traj_frame_count(eq_path) > 0:
            return load_lambda_traj_positions(eq_path)[-1]
    min_path = lambda_min_traj_path(traj_root, label)
    if lambda_traj_frame_count(min_path) > 0:
        return load_lambda_traj_positions(min_path)[-1]
    return None


def print_lambda_resume_plan(
    traj_root: Path,
    lambda_windows: list[float],
    *,
    repeats_per_window: int,
    n_prod: int,
    interval: int,
) -> None:
    expected = lambda_expected_prod_frames(n_prod, interval)
    n_skip = 0
    n_redo = 0
    n_run = 0
    for wi, lam in enumerate(lambda_windows):
        for rep in range(repeats_per_window):
            label = lambda_repeat_label(wi, rep, lam)
            prod_path = lambda_prod_traj_path(traj_root, label)
            if is_lambda_prod_complete(prod_path, n_prod, interval):
                n_skip += 1
                print(f"  resume: skip complete {label} ({lambda_traj_frame_count(prod_path)} frames)", flush=True)
            elif prod_path.is_file():
                n_redo += 1
                print(
                    f"  resume: redo incomplete {label} "
                    f"({lambda_traj_frame_count(prod_path)}/{expected} frames)",
                    flush=True,
                )
            else:
                n_run += 1
    print(
        f"  resume plan: skip={n_skip}, redo_incomplete={n_redo}, run_new={n_run} "
        f"(expected {expected} prod frames per repeat)",
        flush=True,
    )


@dataclass
class ClusterContext:
    z: np.ndarray
    base_seed_positions: np.ndarray
    atoms_per_monomer: list[int]
    residue_labels: list[str]
    monomer_offsets: np.ndarray
    n_monomers: int
    composition_summary: dict[str, int]
    composition_str: str | None


@dataclass
class LambdaDynamicsConfig:
    checkpoint: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("artifacts/lambda_ti"))
    composition: str | None = None
    n_molecules: int = 2
    residue: str = "MEOH"
    template_pdb: Path | None = None
    spacing: float = 5.0
    seed: int = 123
    min_com_start_distance: float = 2.0
    couple_residue_numbers: list[int] = field(default_factory=lambda: [1])
    lambda_windows: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    min_steps: int = 10
    min_fmax: float = 0.03
    n_equil: int = 500
    save_equil_traj: bool = False
    equil_traj_interval: int | None = None
    n_prod: int = 2000
    repeats_per_window: int = 1
    interval: int = 20
    timestep_fs: float = 0.5
    temperature_K: float = 100.0
    ml_cutoff: float = 1.0
    mm_switch_on: float = 5.0
    mm_cutoff: float = 5.0
    no_fix_com: bool = False
    no_stationary: bool = False
    md_mode: str = "free_nve"
    backend: str = "ase"
    box_size: float | None = None
    nvt_integrator: str = "auto"
    langevin_friction: float = 0.02
    charmm_pre_minimize: bool = True
    calculator_pre_minimize: bool = True
    pre_min_steps: int = 50
    pre_min_fmax: float = 0.1
    bfgs_maxstep: float = 0.05
    charmm_sd_steps: int = 25
    charmm_abnr_steps: int = 100
    charmm_tolenr: float = 1e-3
    charmm_tolgrd: float = 1e-3
    charmm_nbxmod: int = 5
    min_intermonomer_atom_distance: float = 0.1
    rescue_minimize: bool = True
    rescue_fire_steps: int = 300
    rescue_fire_fmax: float = 0.1
    rescue_fire_maxstep: float = 0.02
    max_fmax_after_min: float = 2.0
    flat_bottom_radius: float | None = None
    flat_bottom_k: float = 1.0
    flat_bottom_mode: str = "system"
    packmol_radius: float | None = None
    packmol_sphere: bool | None = None
    packmol_center: tuple[float, float, float] | None = None
    packmol_tolerance: float = 2.0
    skip_jit_warmup: bool = False
    resume: bool = False
    repo_root: Path | None = None


@dataclass
class MbarConfig:
    run_dir: Path
    checkpoint: Path | None = None
    temperature_K: float = 100.0
    couple_residue_numbers: list[int] | None = None
    ml_cutoff: float = 1.0
    mm_switch_on: float = 5.0
    mm_cutoff: float = 5.0
    mbar_verbose: bool = False
    write_plots: bool = True
    repo_root: Path | None = None


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_couple_residue_numbers(spec: str, n_monomers: int) -> list[int]:
    """Parse 1-based residue numbers (cluster order); return sorted unique 0-based indices."""
    spec = str(spec).strip()
    if not spec:
        raise ValueError("Empty --couple-residues")
    indices: list[int] = []
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        r = int(tok)
        if r < 1 or r > n_monomers:
            raise ValueError(
                f"Residue number {r} out of range 1..{n_monomers} "
                f"(cluster has {n_monomers} residues in composition order)"
            )
        indices.append(r - 1)
    if not indices:
        raise ValueError("No valid residue numbers in --couple-residues")
    return sorted(set(indices))


def parse_couple_residue_list(numbers: list[int], n_monomers: int) -> list[int]:
    """Validate 1-based residue list; return 0-based indices."""
    return parse_couple_residue_numbers(",".join(str(int(x)) for x in numbers), n_monomers)


def lambda_array(
    n_monomers: int,
    couple_indices: list[int],
    lam_coupled: float,
) -> np.ndarray:
    lam = np.ones(n_monomers, dtype=np.float32)
    for i in couple_indices:
        lam[i] = float(lam_coupled)
    return lam


def build_cluster_system(cfg: LambdaDynamicsConfig) -> ClusterContext:
    """Build CHARMM PSF and initial cluster geometry (same recipes as md-system)."""
    repo_root = cfg.repo_root or repo_root_from_here()
    md = md_suite

    if cfg.composition:
        from mmml.interfaces.pycharmmInterface.packmol_placement import (
            resolve_packmol_cube_side,
            resolve_packmol_placement_mode,
            resolve_packmol_sphere_radius,
            resolve_packmol_use,
        )

        composition = md._parse_composition(cfg.composition)
        use_packmol = resolve_packmol_use(
            composition=cfg.composition,
            packmol=getattr(cfg, "packmol", None),
        )
        if use_packmol:
            placement = resolve_packmol_placement_mode(
                packmol_placement=getattr(cfg, "packmol_placement", None),
                packmol_sphere=cfg.packmol_sphere,
            )
            center = cfg.packmol_center if cfg.packmol_center is not None else (0.0, 0.0, 0.0)
            cube_side: float | None = None
            radius: float | None = None
            if placement == "sphere":
                radius = resolve_packmol_sphere_radius(cfg.packmol_radius, cfg.flat_bottom_radius)
            else:
                cube_side = resolve_packmol_cube_side(
                    box_size=cfg.box_size,
                    packmol_radius=cfg.packmol_radius,
                    flat_bottom_radius=cfg.flat_bottom_radius,
                )
            z, r0, atoms_per_list, residue_labels = md._build_cluster_from_composition_packmol(
                composition=composition,
                placement=placement,
                center=center,
                cube_side=cube_side,
                radius=radius,
                tolerance=float(cfg.packmol_tolerance),
                seed=int(cfg.seed),
            )
        else:
            z, r0, atoms_per_list, residue_labels = md._build_cluster_from_composition(
                composition=composition,
                spacing=cfg.spacing,
            )
        composition_summary = {res: int(cnt) for res, cnt in composition}
        composition_str = cfg.composition.strip()
    else:
        residue = str(cfg.residue).upper()
        tmpl = cfg.template_pdb.expanduser().resolve() if cfg.template_pdb else None
        if tmpl is not None and not tmpl.is_file():
            tmpl = repo_root / tmpl
        z, r0 = _build_psf_ordered_cluster(
            residue,
            int(cfg.n_molecules),
            float(cfg.spacing),
            template_pdb=tmpl,
        )
        n_atoms = len(z)
        atoms_per_uniform = n_atoms // int(cfg.n_molecules)
        atoms_per_list = [int(atoms_per_uniform)] * int(cfg.n_molecules)
        residue_labels = [residue] * int(cfg.n_molecules)
        composition_summary = {residue: int(cfg.n_molecules)}
        composition_str = f"{residue}:{int(cfg.n_molecules)}"

    n_monomers = len(atoms_per_list)
    monomer_offsets = np.zeros(n_monomers + 1, dtype=int)
    monomer_offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))

    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_use

    if (
        not resolve_packmol_use(
            composition=cfg.composition,
            packmol=getattr(cfg, "packmol", None),
        )
        and cfg.composition
    ):
        r0 = md._randomize_monomer_com_positions(
            r0,
            monomer_offsets,
            spacing=float(cfg.spacing),
            min_com_distance=max(float(cfg.spacing), float(cfg.min_com_start_distance)),
            seed=int(cfg.seed),
        )
        r0 = md._enforce_min_com_separation(
            r0,
            monomer_offsets=monomer_offsets,
            min_com_distance=float(cfg.min_com_start_distance),
        )

    base_positions = np.asarray(r0, dtype=float).copy()
    return ClusterContext(
        z=z,
        base_seed_positions=base_positions,
        atoms_per_monomer=[int(x) for x in atoms_per_list],
        residue_labels=list(residue_labels),
        monomer_offsets=monomer_offsets,
        n_monomers=n_monomers,
        composition_summary=composition_summary,
        composition_str=composition_str,
    )


@dataclass
class LambdaMdSettings:
    use_pbc: bool
    integrator: str
    box_L: float | None


def resolve_lambda_md_settings(cfg: LambdaDynamicsConfig, positions: np.ndarray) -> LambdaMdSettings:
    mode = str(cfg.md_mode).lower()
    if mode not in {"free_nve", "free_nvt", "pbc_nve", "pbc_nvt"}:
        raise ValueError(
            f"md_mode must be free_nve, free_nvt, pbc_nve, or pbc_nvt; got {cfg.md_mode!r}"
        )
    use_pbc = mode.startswith("pbc_")
    if mode.endswith("_nve"):
        integrator = "nve"
    elif cfg.nvt_integrator == "langevin" or (
        cfg.nvt_integrator == "auto" and bool(cfg.composition)
    ):
        integrator = "nvt_langevin"
    else:
        integrator = "nvt_nhc"

    box_L: float | None = None
    if use_pbc:
        box_L = float(cfg.box_size) if cfg.box_size is not None else None
        if box_L is None:
            box_L = float(md_suite._cubic_box_length(positions, cfg.ml_cutoff))
    return LambdaMdSettings(use_pbc=use_pbc, integrator=integrator, box_L=box_L)


def prepare_atoms_geometry(
    atoms: ase.Atoms,
    positions: np.ndarray,
    md_settings: LambdaMdSettings,
) -> None:
    atoms.set_positions(positions)
    if md_settings.use_pbc:
        assert md_settings.box_L is not None
        L = float(md_settings.box_L)
        centered = positions - positions.mean(axis=0) + 0.5 * L
        atoms.set_positions(centered)
        atoms.set_cell([L, L, L])
        atoms.set_pbc(True)
    else:
        atoms.set_cell(None)
        atoms.set_pbc(False)


def make_md_integrator(
    atoms: ase.Atoms,
    *,
    integrator: str,
    dt_fs: float,
    temperature_K: float,
    seed: int,
    langevin_friction: float,
    initialize_velocities: bool = True,
    remove_net_drift: bool = True,
):
    dt = dt_fs * units.fs
    rng = np.random.default_rng(int(seed))
    if initialize_velocities:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
        if remove_net_drift:
            Stationary(atoms)
            ZeroRotation(atoms)
    if integrator == "nve":
        return VelocityVerlet(atoms, timestep=dt)
    if integrator == "nvt_nhc":
        tdamp = 100.0 * dt
        return NoseHooverChainNVT(
            atoms,
            timestep=dt,
            temperature_K=temperature_K,
            tdamp=tdamp,
            tchain=3,
            tloop=1,
        )
    if integrator == "nvt_langevin":
        return Langevin(
            atoms,
            timestep=dt,
            temperature_K=temperature_K,
            friction=langevin_friction,
            fixcm=False,
            rng=rng,
        )
    raise ValueError(f"Unknown integrator: {integrator!r}")


def minimize_lambda_structure(
    atoms: ase.Atoms,
    *,
    cfg: LambdaDynamicsConfig,
    cluster: ClusterContext,
    model_restart_path: Path,
    couple_indices: list[int],
    lam_coupled: float,
    md_settings: LambdaMdSettings,
    label: str,
    min_traj_path: Path | None,
) -> dict[str, float | int | str]:
    """CHARMM (MM) then MMML-calculator BFGS, matching ``md_10mer_mmml_pbc_suite``."""
    repo_root = cfg.repo_root or repo_root_from_here()
    md = md_suite
    timings: dict[str, float | int | str] = {}
    overlap_kw = dict(
        nstep_sd=cfg.charmm_sd_steps,
        nstep_abnr=cfg.charmm_abnr_steps,
        tolenr=cfg.charmm_tolenr,
        tolgrd=cfg.charmm_tolgrd,
        nbxmod=cfg.charmm_nbxmod,
        timings=timings,
    )

    md._check_or_charmm_overlap_rescue(
        atoms,
        cluster.monomer_offsets,
        min_distance=cfg.min_intermonomer_atom_distance,
        context=f"{label}: before minimization",
        **overlap_kw,
    )

    if cfg.charmm_pre_minimize:
        print(
            f"{label}: CHARMM minimization (SD={cfg.charmm_sd_steps}, "
            f"ABNR={cfg.charmm_abnr_steps})"
        )
        md._run_charmm_minimize(
            atoms,
            nstep_sd=cfg.charmm_sd_steps,
            nstep_abnr=cfg.charmm_abnr_steps,
            tolenr=cfg.charmm_tolenr,
            tolgrd=cfg.charmm_tolgrd,
            nbxmod=cfg.charmm_nbxmod,
            timings=timings,
            cubic_box_side_A=md_settings.box_L if md_settings.use_pbc else None,
        )
        md._check_or_charmm_overlap_rescue(
            atoms,
            cluster.monomer_offsets,
            min_distance=cfg.min_intermonomer_atom_distance,
            context=f"{label}: after CHARMM minimization",
            **overlap_kw,
        )

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
    calc = build_fixed_lambda_calculator(
        atomic_numbers=cluster.z,
        atomic_positions=atoms.get_positions(),
        base_ckpt_dir=model_restart_path,
        atoms_per_monomer=cluster.atoms_per_monomer,
        n_monomers=cluster.n_monomers,
        couple_indices=couple_indices,
        lam_coupled=lam_coupled,
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
        flat_bottom_mode=cfg.flat_bottom_mode,
    )
    atoms.calc = calc

    if not cfg.skip_jit_warmup:
        from mmml.utils.jax_gpu_warmup import warmup_ase_mmml_energy_forces

        warmup_ase_mmml_energy_forces(atoms, include_forces=True)

    if not cfg.calculator_pre_minimize:
        coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))
        return timings

    print(f"{label}: MMML BFGS (max {cfg.pre_min_steps} steps, fmax={cfg.pre_min_fmax})")
    bfgs_log = None
    opt = BFGS(
        atoms,
        logfile=bfgs_log,
        trajectory=str(min_traj_path) if min_traj_path else None,
        maxstep=cfg.bfgs_maxstep,
    )
    opt.attach(
        lambda: md._check_or_charmm_overlap_rescue(
            atoms,
            cluster.monomer_offsets,
            min_distance=cfg.min_intermonomer_atom_distance,
            context=f"{label}: ASE BFGS",
            **overlap_kw,
        ),
        interval=1,
    )
    opt.run(fmax=cfg.pre_min_fmax, steps=cfg.pre_min_steps)
    timings["bfgs_iterations"] = int(opt.get_number_of_steps())
    fmin = float(np.abs(atoms.get_forces()).max())
    timings["bfgs_fmax_eVA"] = fmin
    if min_traj_path is not None:
        timings["min_traj"] = str(min_traj_path)

    if fmin > cfg.pre_min_fmax and cfg.rescue_minimize:
        print(f"{label}: ASE FIRE rescue (fmax={fmin:.4f} > {cfg.pre_min_fmax})")
        fire = FIRE(atoms, logfile=bfgs_log, maxstep=cfg.rescue_fire_maxstep)
        fire.attach(
            lambda: md._check_or_charmm_overlap_rescue(
                atoms,
                cluster.monomer_offsets,
                min_distance=cfg.min_intermonomer_atom_distance,
                context=f"{label}: ASE FIRE",
                **overlap_kw,
            ),
            interval=1,
        )
        fire.run(fmax=cfg.rescue_fire_fmax, steps=cfg.rescue_fire_steps)
        fmin = float(np.abs(atoms.get_forces()).max())
        timings["fire_fmax_eVA"] = fmin

    if fmin > cfg.max_fmax_after_min:
        raise RuntimeError(
            f"{label}: post-minimization fmax={fmin:.4f} eV/Å exceeds "
            f"--max-fmax-after-min={cfg.max_fmax_after_min}"
        )

    coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))
    timings["final_fmax_eVA"] = fmin
    return timings


def ensure_psf_for_snapshots(meta: dict[str, Any], repo_root: Path) -> None:
    """Rebuild CHARMM PSF/topology so ``psf.get_iac()`` matches the sampled cluster."""
    md = md_suite
    spacing = float(meta.get("spacing", 5.0))
    composition_str = meta.get("composition_str")
    if composition_str:
        composition = md._parse_composition(str(composition_str))
        md._build_cluster_from_composition(composition=composition, spacing=spacing)
        return
    residue = str(meta.get("residue", "MEOH")).upper()
    n_mol = int(meta["n_monomers"])
    tmpl = meta.get("template_pdb")
    template_pdb = Path(tmpl) if tmpl else None
    if template_pdb is not None and not template_pdb.is_file():
        template_pdb = repo_root / template_pdb
    _build_psf_ordered_cluster(residue, n_mol, spacing, template_pdb=template_pdb)


def energy_at_positions(calc, atoms: ase.Atoms, r: np.ndarray) -> float:
    atoms.set_positions(r)
    calc.results.clear()
    return float(atoms.get_potential_energy())


def _to_float_sum(value) -> float:
    arr = np.asarray(value)
    return float(arr.sum()) if arr.size > 1 else float(arr.reshape(()))


def interaction_energy_from_results(calc) -> float:
    res = calc.results
    out = res.get("out")
    if out is not None:
        ml = getattr(out, "ml_2b_E", None)
        mm = getattr(out, "mm_E", None)
        if ml is not None and mm is not None:
            return _to_float_sum(ml) + _to_float_sum(mm)
        if hasattr(out, "internal_E") and hasattr(out, "energy"):
            return _to_float_sum(out.energy) - _to_float_sum(out.internal_E)
    ml_k = res.get("model_ml_2b_E")
    mm_k = res.get("model_mm_E")
    if ml_k is not None and mm_k is not None:
        return _to_float_sum(ml_k) + _to_float_sum(mm_k)
    internal = res.get("model_internal_E")
    if internal is not None and "energy" in res:
        return float(res["energy"]) - _to_float_sum(internal)
    raise KeyError(
        "Cannot read interaction energy from calculator results "
        "(expected results['out'].ml_2b_E/mm_E or model_ml_2b_E/model_mm_E)."
    )


def interaction_energy_at_positions(calc, atoms: ase.Atoms, r: np.ndarray) -> float:
    atoms.set_positions(r)
    calc.results.clear()
    atoms.get_potential_energy()
    return interaction_energy_from_results(calc)


def dUdlambda_at_R(
    calc_on: object,
    atoms_on: ase.Atoms,
    calc_off: object,
    atoms_off: ase.Atoms,
    r: np.ndarray,
) -> float:
    w_on = interaction_energy_at_positions(calc_on, atoms_on, r)
    w_off = interaction_energy_at_positions(calc_off, atoms_off, r)
    return w_on - w_off


def build_fixed_lambda_calculator(
    *,
    atomic_numbers: np.ndarray,
    atomic_positions: np.ndarray,
    base_ckpt_dir: Path,
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
    cell_scalar: float | None = None,
    flat_bottom_radius: float | None = None,
    flat_bottom_k: float = 1.0,
    flat_bottom_mode: str = "system",
) -> object:
    lam_arr = lambda_array(n_monomers, couple_indices, lam_coupled)
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
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=max_atoms_per * 2,
        cell=cell_arg,
        ep_scale=ep_scale,
        sig_scale=sig_scale,
        at_codes_override=at_codes,
        lambda_monomer=lam_arr,
        flat_bottom_radius=flat_bottom_radius,
        flat_bottom_force_const=flat_bottom_k,
        flat_bottom_mode=flat_bottom_mode,
    )
    calc, _, _ = factory(
        atomic_numbers=atomic_numbers,
        atomic_positions=atomic_positions,
        n_monomers=n_monomers,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        backprop=True,
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=False,
    )
    return calc


def snapshot_metadata_from_cluster(
    cluster: ClusterContext,
    cfg: LambdaDynamicsConfig,
    couple_residue_numbers_1based: list[int],
) -> dict[str, Any]:
    return {
        "composition_str": cluster.composition_str,
        "composition_summary": cluster.composition_summary,
        "residue_labels": cluster.residue_labels,
        "atoms_per_monomer": cluster.atoms_per_monomer,
        "n_monomers": cluster.n_monomers,
        "residue": str(cfg.residue).upper(),
        "spacing": float(cfg.spacing),
        "seed": int(cfg.seed),
        "template_pdb": str(cfg.template_pdb) if cfg.template_pdb else "",
        "couple_residue_numbers": [int(x) for x in couple_residue_numbers_1based],
    }


def save_snapshots_npz(
    path: Path,
    *,
    atomic_numbers: np.ndarray,
    lambda_windows: list[float],
    snapshots_per_window: list[list[np.ndarray]],
    snapshot_meta: dict[str, Any],
) -> None:
    K = len(lambda_windows)
    N_k = np.array([len(snapshots_per_window[k]) for k in range(K)], dtype=np.int64)
    N_max = int(N_k.max()) if K else 0
    natoms = len(atomic_numbers)
    positions = np.full((K, N_max, natoms, 3), np.nan, dtype=np.float64)
    for k in range(K):
        for n, r_snap in enumerate(snapshots_per_window[k]):
            positions[k, n] = np.asarray(r_snap, dtype=np.float64)
    np.savez_compressed(
        path,
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
        lambda_windows=np.asarray(lambda_windows, dtype=np.float64),
        N_k=N_k,
        positions=positions,
        couple_residue_numbers=np.asarray(snapshot_meta["couple_residue_numbers"], dtype=np.int32),
        atoms_per_monomer=np.asarray(snapshot_meta["atoms_per_monomer"], dtype=np.int32),
        spacing=np.float64(snapshot_meta["spacing"]),
        n_monomers=np.int64(snapshot_meta["n_monomers"]),
        composition_str=np.array(snapshot_meta.get("composition_str") or "", dtype=str),
        residue=np.array(snapshot_meta.get("residue", "MEOH"), dtype=str),
        template_pdb=np.array(snapshot_meta.get("template_pdb", ""), dtype=str),
        residue_labels=np.array(snapshot_meta["residue_labels"], dtype=str),
    )


def load_snapshots_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    K = int(data["lambda_windows"].shape[0])
    N_k = np.asarray(data["N_k"], dtype=np.int64)
    snapshots_per_window: list[list[np.ndarray]] = []
    positions = data["positions"]
    for k in range(K):
        snaps = [positions[k, n].copy() for n in range(int(N_k[k]))]
        snapshots_per_window.append(snaps)

    if "couple_residue_numbers" in data:
        couple_1b = [int(x) for x in np.asarray(data["couple_residue_numbers"]).reshape(-1)]
    elif "decouple_monomer_index" in data:
        couple_1b = [int(data["decouple_monomer_index"]) + 1]
    else:
        couple_1b = [1]

    comp_raw = str(np.asarray(data["composition_str"]).reshape(()))
    meta = {
        "atomic_numbers": np.asarray(data["atomic_numbers"]),
        "lambda_windows": [float(x) for x in data["lambda_windows"]],
        "snapshots_per_window": snapshots_per_window,
        "couple_residue_numbers": couple_1b,
        "atoms_per_monomer": [int(x) for x in np.asarray(data["atoms_per_monomer"]).reshape(-1)],
        "n_monomers": int(data["n_monomers"]),
        "spacing": float(data["spacing"]) if "spacing" in data else 5.0,
        "composition_str": comp_raw if comp_raw else None,
        "residue": str(np.asarray(data["residue"]).reshape(())) if "residue" in data else "MEOH",
        "template_pdb": str(np.asarray(data["template_pdb"]).reshape(())) if "template_pdb" in data else "",
        "residue_labels": [str(x) for x in np.asarray(data["residue_labels"]).reshape(-1)],
    }
    meta["couple_indices"] = parse_couple_residue_list(couple_1b, meta["n_monomers"])
    return meta


def plot_window_components(
    out_dir: Path,
    rows: list[dict],
    mbar_block: dict | None,
) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    if not rows:
        return written

    def _lam_val(row: dict) -> float:
        if "lambda_coupled" in row:
            return float(row["lambda_coupled"])
        return float(row.get("lambda_decoupled_monomer", np.nan))

    lam = np.array([_lam_val(r) for r in rows], dtype=float)
    mean = np.array([float(r["mean_dUdlambda_eV"]) for r in rows], dtype=float)
    std = np.array([float(r["std_dUdlambda_eV"]) for r in rows], dtype=float)
    n = np.array([max(1, int(r["n_samples"])) for r in rows], dtype=int)
    sem = std / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lam, mean, marker="o", linewidth=1.5, label="Mean dU/dλ")
    ax.errorbar(lam, mean, yerr=sem, fmt="none", ecolor="tab:blue", capsize=3, label="± SEM")
    ax.fill_between(lam, mean - std, mean + std, alpha=0.2, color="tab:blue", label="± 1σ")
    ax.set_xlabel("λ")
    ax.set_ylabel("dU/dλ (eV)")
    ax.set_title("TI components per window")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p1 = out_dir / "ti_components_per_window.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    written.append(str(p1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in enumerate(rows):
        rep = row.get("repeat_stats", [])
        if isinstance(rep, list) and rep:
            y = np.array([float(rr.get("mean_dUdlambda_eV", np.nan)) for rr in rep], dtype=float)
            x = np.full_like(y, lam[i], dtype=float)
            ax.scatter(x, y, s=25, alpha=0.7, color="tab:gray")
    ax.plot(lam, mean, marker="o", linewidth=1.5, color="tab:red", label="Window mean")
    ax.set_xlabel("λ")
    ax.set_ylabel("dU/dλ (eV)")
    ax.set_title("Per-repeat TI components")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p2 = out_dir / "ti_repeat_components_per_window.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    written.append(str(p2))

    if mbar_block and "error" not in mbar_block and "N_k" in mbar_block:
        n_k = np.array(mbar_block.get("N_k", []), dtype=float)
        n_eff = np.array(mbar_block.get("N_k_effective", []), dtype=float)
        g_k = np.array(mbar_block.get("g_k", []), dtype=float)
        if n_k.size == lam.size:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(lam, n_k, marker="o", label="N_k")
            if n_eff.size == lam.size:
                ax1.plot(lam, n_eff, marker="o", label="N_k effective")
            ax1.set_xlabel("λ")
            ax1.set_ylabel("Sample counts")
            ax1.grid(alpha=0.3)
            ax2 = ax1.twinx()
            if g_k.size == lam.size:
                ax2.plot(lam, g_k, marker="s", linestyle="--", color="tab:green", label="g_k")
                ax2.set_ylabel("Statistical inefficiency g")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
            ax1.set_title("MBAR per-window sampling diagnostics")
            fig.tight_layout()
            p3 = out_dir / "mbar_per_window_diagnostics.png"
            fig.savefig(p3, dpi=160)
            plt.close(fig)
            written.append(str(p3))

    return written


def resolve_model_restart_path(checkpoint: Path | None) -> Path:
    """Return a path ``setup_calculator`` accepts (portable .json, params.json dir, or Orbax epoch)."""
    base, epoch = resolve_checkpoint_paths(checkpoint)
    base = Path(base).resolve()
    epoch = Path(epoch).resolve()
    if base.is_file() and base.suffix.lower() == ".json":
        return base
    if base.is_dir() and (base / "params.json").exists():
        return base
    return epoch


def print_cluster_psf_monomer_diagnostics(
    cluster: ClusterContext,
    *,
    monomer_index: int = 0,
) -> dict[str, Any]:
    """Print CHARMM PSF atom types and MM charges for one monomer (post cluster build)."""
    mi = int(monomer_index)
    if mi < 0 or mi >= cluster.n_monomers:
        raise ValueError(
            f"monomer_index {mi} out of range 0..{cluster.n_monomers - 1}"
        )
    s = int(cluster.monomer_offsets[mi])
    e = int(cluster.monomer_offsets[mi + 1])
    n_atoms = e - s
    residue = cluster.residue_labels[mi]

    charges_all = np.asarray(psf.get_charges(), dtype=float)
    atypes_all = np.asarray(psf.get_atype(), dtype=str)
    iac_all = np.asarray(psf.get_iac(), dtype=int)
    masses_all = np.asarray(psf.get_amass(), dtype=float)
    z_all = np.asarray(cluster.z, dtype=int)

    charges = charges_all[s:e]
    atypes = atypes_all[s:e]
    iac = iac_all[s:e]
    masses = masses_all[s:e]
    z_loc = z_all[s:e]

    def _psf_slice(getter, dtype):
        try:
            arr = np.asarray(getter(), dtype=dtype)
        except Exception:
            return None
        if arr.size < e:
            return None
        sl = arr[s:e]
        return sl if sl.shape[0] == n_atoms else None

    segid = _psf_slice(psf.get_segid, str)
    resid = _psf_slice(psf.get_resid, int)
    resname = _psf_slice(psf.get_res, str)
    show_residue_cols = segid is not None and resid is not None and resname is not None

    atc_labels: list[str] = []
    try:
        atc = [str(x).strip() for x in param.get_atc()]
        atc_labels = [atc[int(c) - 1] if 0 < int(c) <= len(atc) else "?" for c in iac]
    except Exception:
        atc_labels = ["?"] * n_atoms

    lines = [
        "=== PSF / MM monomer diagnostic (CHARMM after cluster build) ===",
        f"  monomer {mi + 1} / {cluster.n_monomers}  residue_label={residue!r}  "
        f"atoms [{s}:{e})  n_atoms={n_atoms}",
        f"  composition: {cluster.composition_summary}",
        f"  charge_sum (e): {float(np.sum(charges)):.6f}",
        f"  total cluster charge (e): {float(np.sum(charges_all[: len(z_all)])):.6f}",
        "  index  Z   atype      iac  param_type     charge(e)    mass(amu)"
        + ("  segid  resid  resname" if show_residue_cols else ""),
    ]
    if not show_residue_cols:
        lines.append(
            "  (PSF segid/resid/resname not per-atom in this build; using residue_label above)"
        )
    for j in range(n_atoms):
        extra = ""
        if show_residue_cols:
            extra = (
                f"  {segid[j]!s:4s}  {int(resid[j]):5d}  {str(resname[j])!s}"
            )
        lines.append(
            f"  {s + j:4d}  {int(z_loc[j]):2d}  {atypes[j]!s:10s}  {int(iac[j]):3d}  "
            f"{atc_labels[j]!s:14s}  {float(charges[j]):+10.5f}  {float(masses[j]):8.4f}"
            + extra
        )

    same_type = [i for i, lab in enumerate(cluster.residue_labels) if lab == residue]
    if len(same_type) > 1:
        ref_charges = charges.copy()
        ref_atypes = list(atypes)
        mismatch: list[str] = []
        for other in same_type:
            if other == mi:
                continue
            os_ = int(cluster.monomer_offsets[other])
            oe = int(cluster.monomer_offsets[other + 1])
            oc = charges_all[os_:oe]
            oa = atypes_all[os_:oe]
            if list(oa) != ref_atypes:
                mismatch.append(f"monomer {other + 1}: atom types differ")
            elif not np.allclose(oc, ref_charges, atol=1e-8, rtol=0.0):
                mismatch.append(
                    f"monomer {other + 1}: charges differ "
                    f"(sum={float(np.sum(oc)):.6f} vs {float(np.sum(ref_charges)):.6f})"
                )
        if mismatch:
            lines.append("  WARNING: PSF differs across copies of same residue:")
            lines.extend(f"    - {m}" for m in mismatch)
        else:
            lines.append(
                f"  OK: all {len(same_type)} {residue!r} copies share identical atypes/charges."
            )

    for line in lines:
        print(line, flush=True)

    return {
        "monomer_index": mi,
        "residue_label": residue,
        "atom_slice": [s, e],
        "charge_sum_e": float(np.sum(charges)),
        "atom_types": [str(x) for x in atypes],
        "charges_e": [float(x) for x in charges],
        "iac": [int(x) for x in iac],
        "atomic_numbers": [int(x) for x in z_loc],
    }


def _print_run_banner(cfg: LambdaDynamicsConfig, *, backend: str) -> None:
    """Log key run options before CHARMM/cluster setup (stdout may be noisy afterward)."""
    use_fix_com = not cfg.no_fix_com
    remove_net_drift = not cfg.no_stationary
    comp = cfg.composition or f"{cfg.residue}:{cfg.n_molecules}"
    lines = [
        "=== lambda_ti run ===",
        f"  output_dir: {cfg.output_dir}",
        f"  composition: {comp}",
        f"  couple_residues (1-based): {cfg.couple_residue_numbers}",
        f"  backend: {backend}",
        f"  lambda_md_mode: {cfg.md_mode}",
        f"  COM handling: FixCom={'on' if use_fix_com else 'off'}, "
        f"Stationary/ZeroRotation on velocity init={'on' if remove_net_drift else 'off'}",
        f"  lambda_windows: {len(cfg.lambda_windows)} values, "
        f"repeats_per_window={cfg.repeats_per_window}",
        f"  n_equil={cfg.n_equil}, n_prod={cfg.n_prod}, interval={cfg.interval}",
        f"  resume: {'on' if cfg.resume else 'off'}",
        f"  cutoffs (Å): ml={cfg.ml_cutoff}, mm_switch_on={cfg.mm_switch_on}, mm_cutoff={cfg.mm_cutoff}",
    ]
    for line in lines:
        print(line, flush=True)


def run_lambda_dynamics(cfg: LambdaDynamicsConfig) -> dict[str, Any]:
    """Run λ-window MD/TI sampling; write summary JSON and snapshot NPZ (no MBAR)."""
    backend = str(cfg.backend).lower()
    if backend == "auto":
        backend = "ase"
    if backend == "jaxmd":
        from mmml.cli.run.lambda_jaxmd import run_lambda_dynamics_jaxmd

        return run_lambda_dynamics_jaxmd(cfg)
    if backend != "ase":
        raise ValueError(f"Unsupported backend for lambda_ti: {cfg.backend!r}")

    out_dir = cfg.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_run_banner(cfg, backend=backend)

    cluster = build_cluster_system(cfg)
    z = cluster.z
    base_seed_positions = cluster.base_seed_positions
    couple_indices = parse_couple_residue_list(cfg.couple_residue_numbers, cluster.n_monomers)
    couple_residue_numbers_1b = [i + 1 for i in couple_indices]
    print_cluster_psf_monomer_diagnostics(cluster, monomer_index=couple_indices[0])

    model_restart_path = resolve_model_restart_path(cfg.checkpoint)
    md_settings = resolve_lambda_md_settings(cfg, base_seed_positions)

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
        flat_bottom_mode=cfg.flat_bottom_mode,
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

    for wi, lam in enumerate(lambda_windows):
        samples: list[float] = []
        repeat_stats: list[dict[str, float | int | str]] = []
        for rep in range(cfg.repeats_per_window):
            label = lambda_repeat_label(wi, rep, lam)
            prod_path = lambda_prod_traj_path(traj_root, label)
            if cfg.resume and is_lambda_prod_complete(prod_path, cfg.n_prod, cfg.interval):
                positions = load_lambda_traj_positions(prod_path)
                atoms_probe_on = ase.Atoms(numbers=z, positions=positions[0].copy())
                atoms_probe_off = ase.Atoms(numbers=z, positions=positions[0].copy())
                calc_probe_on = build_fixed_lambda_calculator(
                    atomic_positions=atoms_probe_on.get_positions(),
                    lam_coupled=1.0,
                    **calc_common,
                )
                calc_probe_off = build_fixed_lambda_calculator(
                    atomic_positions=atoms_probe_off.get_positions(),
                    lam_coupled=0.0,
                    **calc_common,
                )
                atoms_probe_on.calc = calc_probe_on
                atoms_probe_off.calc = calc_probe_off
                samples_rep = [
                    dUdlambda_at_R(
                        calc_probe_on,
                        atoms_probe_on,
                        calc_probe_off,
                        atoms_probe_off,
                        p,
                    )
                    for p in positions
                ]
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
            if use_fix_com:
                atoms.set_constraint(FixCom())

            min_traj_path = lambda_min_traj_path(traj_root, label)
            skip_min = (
                cfg.resume
                and lambda_traj_frame_count(min_traj_path) > 0
            )
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

            equil_path: str | None = None
            traj_eq = None
            if cfg.save_equil_traj and cfg.n_equil > 0:
                equil_path = str(traj_root / f"{label}_eq.traj")
                traj_eq = Trajectory(equil_path, "w", atoms)
                eq_interval = cfg.equil_traj_interval or cfg.interval
                eq_interval = max(1, int(eq_interval))

            dyn_eq = make_md_integrator(
                atoms,
                integrator=md_settings.integrator,
                dt_fs=cfg.timestep_fs,
                temperature_K=cfg.temperature_K,
                seed=cfg.seed + wi * 1000 + rep,
                langevin_friction=cfg.langevin_friction,
                remove_net_drift=remove_net_drift,
            )
            if traj_eq is not None:
                dyn_eq.attach(traj_eq.write, interval=eq_interval)
            dyn_eq.run(cfg.n_equil)
            if traj_eq is not None:
                traj_eq.close()

            traj_prod = Trajectory(str(prod_path), "w", atoms)

            samples_rep: list[float] = []
            step_count = [0]
            atoms_probe_on = ase.Atoms(numbers=z, positions=atoms.get_positions().copy())
            atoms_probe_off = ase.Atoms(numbers=z, positions=atoms.get_positions().copy())
            calc_probe_on = build_fixed_lambda_calculator(
                atomic_positions=atoms_probe_on.get_positions(),
                lam_coupled=1.0,
                **calc_common,
            )
            calc_probe_off = build_fixed_lambda_calculator(
                atomic_positions=atoms_probe_off.get_positions(),
                lam_coupled=0.0,
                **calc_common,
            )
            atoms_probe_on.calc = calc_probe_on
            atoms_probe_off.calc = calc_probe_off
            atoms.calc = build_fixed_lambda_calculator(
                atomic_positions=atoms.get_positions(),
                lam_coupled=lam,
                **calc_common,
            )

            def _on_step(_atoms=atoms):
                step_count[0] += 1
                if step_count[0] % cfg.interval == 0:
                    d = dUdlambda_at_R(
                        calc_probe_on,
                        atoms_probe_on,
                        calc_probe_off,
                        atoms_probe_off,
                        _atoms.get_positions().copy(),
                    )
                    samples_rep.append(d)
                    samples.append(d)
                    snapshots_per_window[wi].append(_atoms.get_positions().copy())

            dyn_prod = make_md_integrator(
                atoms,
                integrator=md_settings.integrator,
                dt_fs=cfg.timestep_fs,
                temperature_K=cfg.temperature_K,
                seed=cfg.seed + wi * 1000 + rep + 1,
                langevin_friction=cfg.langevin_friction,
                initialize_velocities=False,
                remove_net_drift=False,
            )
            dyn_prod.attach(traj_prod.write, interval=max(1, cfg.interval))
            dyn_prod.attach(_on_step)
            dyn_prod.run(cfg.n_prod)
            traj_prod.close()

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

    coupled_labels = [
        f"{cluster.residue_labels[i]}#{i + 1}" for i in couple_indices
    ]
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
            "backend": backend,
            "md_mode": cfg.md_mode,
            "integrator": md_settings.integrator,
            "use_pbc": md_settings.use_pbc,
            "box_A": md_settings.box_L,
            "charmm_pre_minimize": cfg.charmm_pre_minimize,
            "calculator_pre_minimize": cfg.calculator_pre_minimize,
            "fix_com": use_fix_com,
            "no_fix_com": cfg.no_fix_com,
            "no_stationary": cfg.no_stationary,
        },
        "description": {
            "delta_F_couple_eV": "TI integral ∫ ⟨∂U/∂λ⟩ dλ (turn on intermolecular coupling of selected residues)",
            "delta_F_diss_eV": "Negative of delta_F_couple (decouple along same path)",
            "lambda_definition": (
                f"Residues {couple_residue_numbers_1b} share λ; all other residues λ=1. "
                "Inter-monomer ML/MM scale as λ_i λ_j."
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
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["_summary_path"] = str(out_json)
    return summary


def run_mbar_analysis(cfg: MbarConfig) -> dict[str, Any]:
    try:
        from pymbar import MBAR, timeseries
    except ImportError as exc:
        raise SystemExit(
            "pymbar is required for MBAR. Install with: "
            "uv sync --extra mbar   or   pip install 'pymbar>=4.0'"
        ) from exc

    repo_root = cfg.repo_root or repo_root_from_here()
    run_dir = cfg.run_dir.expanduser().resolve()
    snap_path = run_dir / SNAPSHOTS_NPZ
    if not snap_path.is_file():
        raise FileNotFoundError(f"Missing snapshots file: {snap_path} (run lambda dynamics first)")

    snap = load_snapshots_npz(snap_path)
    z = snap["atomic_numbers"]
    lambda_windows = snap["lambda_windows"]
    snapshots_per_window = snap["snapshots_per_window"]
    couple_indices = snap["couple_indices"]
    if cfg.couple_residue_numbers is not None:
        couple_indices = parse_couple_residue_list(cfg.couple_residue_numbers, snap["n_monomers"])

    ensure_psf_for_snapshots(
        {
            "composition_str": snap.get("composition_str"),
            "n_monomers": snap["n_monomers"],
            "residue": snap.get("residue", "MEOH"),
            "spacing": snap.get("spacing", 5.0),
            "template_pdb": snap.get("template_pdb") or None,
        },
        repo_root,
    )

    model_restart_path = resolve_model_restart_path(cfg.checkpoint)
    at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    ep_scale = np.ones(n_types, dtype=float)
    sig_scale = np.ones(n_types, dtype=float)
    cutoff = CutoffParameters(
        ml_cutoff=cfg.ml_cutoff,
        mm_switch_on=cfg.mm_switch_on,
        mm_cutoff=cfg.mm_cutoff,
    )
    calc_common = dict(
        atomic_numbers=z,
        base_ckpt_dir=model_restart_path,
        atoms_per_monomer=snap["atoms_per_monomer"],
        n_monomers=snap["n_monomers"],
        couple_indices=couple_indices,
        cutoff=cutoff,
        ml_cutoff=cfg.ml_cutoff,
        mm_switch_on=cfg.mm_switch_on,
        mm_cutoff=cfg.mm_cutoff,
        at_codes=at_codes,
        ep_scale=ep_scale,
        sig_scale=sig_scale,
    )

    K = len(lambda_windows)
    N_k = np.array([len(snapshots_per_window[k]) for k in range(K)], dtype=np.int64)
    if np.any(N_k == 0):
        return {
            "error": "MBAR skipped: at least one λ-window has no snapshots.",
            "N_k": N_k.tolist(),
        }

    N_max = int(N_k.max())
    beta = 1.0 / (float(units.kB) * cfg.temperature_K)
    u_kln = np.zeros((K, K, N_max), dtype=np.float64)
    mbar_atoms_bank: list[ase.Atoms] = []
    mbar_calc_bank: list[object] = []
    seed_pos = snapshots_per_window[0][0] if snapshots_per_window[0] else np.zeros((len(z), 3))
    for l in range(K):
        atoms_l = ase.Atoms(numbers=z, positions=seed_pos.copy())
        calc_l = build_fixed_lambda_calculator(
            atomic_positions=atoms_l.get_positions(),
            lam_coupled=lambda_windows[l],
            **calc_common,
        )
        atoms_l.calc = calc_l
        mbar_atoms_bank.append(atoms_l)
        mbar_calc_bank.append(calc_l)

    for k in range(K):
        for n in range(int(N_k[k])):
            r_snap = snapshots_per_window[k][n]
            for l in range(K):
                u_kln[k, l, n] = beta * energy_at_positions(
                    mbar_calc_bank[l],
                    mbar_atoms_bank[l],
                    r_snap,
                )

    g_k: list[float] = []
    selected_indices: list[np.ndarray] = []
    for k in range(K):
        u_self = u_kln[k, k, : int(N_k[k])]
        if u_self.size < 2:
            g_est = 1.0
            idx = np.arange(u_self.size, dtype=int)
        else:
            g_est = float(timeseries.statistical_inefficiency(u_self))
            g_est = max(1.0, g_est)
            idx = np.asarray(timeseries.subsample_correlated_data(u_self, g=g_est), dtype=int)
            if idx.size == 0:
                idx = np.array([u_self.size - 1], dtype=int)
        g_k.append(g_est)
        selected_indices.append(idx)

    N_k_eff = np.array([idx.size for idx in selected_indices], dtype=np.int64)
    N_max_eff = int(N_k_eff.max())
    u_kln_eff = np.zeros((K, K, N_max_eff), dtype=np.float64)
    for k in range(K):
        idx = selected_indices[k]
        for j, n_old in enumerate(idx):
            u_kln_eff[k, :, j] = u_kln[k, :, int(n_old)]

    mbar = MBAR(u_kln_eff, N_k_eff, verbose=cfg.mbar_verbose)
    fe = mbar.compute_free_energy_differences(compute_uncertainty=True)
    i0, i1 = 0, K - 1
    df_k = float(fe["Delta_f"][i0, i1])
    ddf_k = float(fe["dDelta_f"][i0, i1])
    kbt_ev = float(units.kB) * cfg.temperature_K
    df_ev = df_k * kbt_ev
    ddf_ev = ddf_k * kbt_ev
    return {
        "Delta_f_lambda1_minus_lambda0_kT": df_k,
        "dDelta_f_kT": ddf_k,
        "Delta_F_couple_eV": df_ev,
        "dDelta_F_couple_eV": ddf_ev,
        "Delta_F_couple_kcal_mol": df_ev * _EV_TO_KCAL,
        "dDelta_F_couple_kcal_mol": ddf_ev * _EV_TO_KCAL,
        "Delta_F_diss_eV": -df_ev,
        "Delta_F_diss_kcal_mol": -df_ev * _EV_TO_KCAL,
        "N_k": N_k.tolist(),
        "N_k_effective": N_k_eff.tolist(),
        "g_k": g_k,
        "couple_residue_numbers": [i + 1 for i in couple_indices],
        "note": "Coupling ΔF = F(λ=1) - F(λ=0) from MBAR; dissociation is the negative.",
    }


def merge_mbar_into_summary(run_dir: Path, mbar_block: dict[str, Any], write_plots: bool = True) -> Path:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / SUMMARY_JSON
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"windows": []}

    summary["mbar"] = mbar_block
    if write_plots and summary.get("windows"):
        extra = plot_window_components(run_dir, summary["windows"], mbar_block)
        plots = list(summary.get("plots") or [])
        for p in extra:
            if p not in plots:
                plots.append(p)
        summary["plots"] = plots

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def print_lambda_summary(summary: dict[str, Any]) -> None:
    delta_f_ev = summary.get("delta_F_couple_eV", float("nan"))
    delta_f_kcal = summary.get("delta_F_couple_kcal_mol", float("nan"))
    print(json.dumps(summary.get("description", {}), indent=2))
    sys = summary.get("system")
    if sys:
        print(
            f"Coupled residues: {sys.get('couple_residue_numbers')} "
            f"({sys.get('couple_residue_labels')})"
        )
    print(f"ΔF_couple (binding path, TI) = {delta_f_ev:.6f} eV = {delta_f_kcal:.4f} kcal/mol")
    print(f"ΔF_diss (decouple)          = {-delta_f_ev:.6f} eV = {-delta_f_kcal:.4f} kcal/mol")
    mbar_block = summary.get("mbar")
    if mbar_block and "error" not in mbar_block:
        print(
            f"ΔF_couple (MBAR)            = {mbar_block['Delta_F_couple_eV']:.6f} ± "
            f"{mbar_block['dDelta_F_couple_eV']:.6f} eV = "
            f"{mbar_block['Delta_F_couple_kcal_mol']:.4f} ± "
            f"{mbar_block['dDelta_F_couple_kcal_mol']:.4f} kcal/mol"
        )
    elif mbar_block and "error" in mbar_block:
        print(f"MBAR: {mbar_block['error']}")


def add_lambda_dynamics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lambda_ti"))
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Cluster composition RES:N,... (e.g. MEOH:2 or MEOH:1,TIP3:1). Overrides --n-molecules/--residue.",
    )
    parser.add_argument("--n-molecules", type=int, default=2, help="Homogeneous cluster size if --composition unset.")
    parser.add_argument("--residue", type=str, default="MEOH", help="Residue name for homogeneous clusters.")
    parser.add_argument(
        "--template-pdb",
        type=Path,
        default=None,
        help="Optional monomer PDB for homogeneous clusters (atom names must match PSF).",
    )
    parser.add_argument("--spacing", type=float, default=5.0, help="Minimum COM spacing when building the cluster (Å).")
    parser.add_argument("--seed", type=int, default=123, help="Random placement seed.")
    parser.add_argument(
        "--min-com-start-distance",
        type=float,
        default=2.0,
        help="Minimum inter-monomer COM distance after placement (Å).",
    )
    parser.add_argument(
        "--couple-residues",
        type=str,
        default="1",
        help="1-based residue numbers to couple with shared λ (comma-separated, cluster order).",
    )
    parser.add_argument(
        "--lambda-windows",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument(
        "--lambda-md-mode",
        choices=["free_nve", "free_nvt", "pbc_nve", "pbc_nvt"],
        default="free_nve",
        help="MD ensemble for equilibration/production (ASE): vacuum vs PBC, NVE vs NVT.",
    )
    parser.add_argument(
        "--backend",
        choices=["ase", "jaxmd", "auto"],
        default="ase",
        help="lambda TI MD engine: ase (default) or jaxmd (NVE/NHC-NVT; no Langevin).",
    )
    parser.add_argument("--box-size", type=float, default=None, help="PBC cubic box side (Å); auto if omitted.")
    parser.add_argument(
        "--nvt-integrator",
        choices=["auto", "nhc", "langevin"],
        default="auto",
        help="NVT thermostat when lambda-md-mode ends with _nvt.",
    )
    parser.add_argument("--pre-min-steps", type=int, default=50, help="MMML BFGS steps per λ window.")
    parser.add_argument("--pre-min-fmax", type=float, default=0.1, help="MMML BFGS fmax (eV/Å) per λ window.")
    parser.add_argument("--min-steps", type=int, default=None, help="Alias for --pre-min-steps.")
    parser.add_argument("--min-fmax", type=float, default=None, help="Alias for --pre-min-fmax.")
    parser.add_argument("--bfgs-maxstep", type=float, default=0.05)
    parser.add_argument(
        "--charmm-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CHARMM SD/ABNR before MMML BFGS each λ window.",
    )
    parser.add_argument(
        "--calculator-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ASE BFGS on the MMML calculator after CHARMM.",
    )
    parser.add_argument("--charmm-sd-steps", type=int, default=25)
    parser.add_argument("--charmm-abnr-steps", type=int, default=100)
    parser.add_argument("--charmm-tolenr", type=float, default=1e-3)
    parser.add_argument("--charmm-tolgrd", type=float, default=1e-3)
    parser.add_argument("--charmm-nbxmod", type=int, default=5)
    parser.add_argument(
        "--max-fmax-after-min",
        type=float,
        default=2.0,
        help="Abort if post-minimization fmax exceeds this (eV/Å).",
    )
    parser.add_argument(
        "--rescue-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ASE FIRE rescue if BFGS fmax remains above target.",
    )
    parser.add_argument("--rescue-fire-steps", type=int, default=300)
    parser.add_argument("--rescue-fire-fmax", type=float, default=0.1)
    parser.add_argument("--rescue-fire-maxstep", type=float, default=0.02)
    parser.add_argument("--skip-jit-warmup", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip λ repeats whose production trajectory is complete; redo partial prod.traj files.",
    )
    parser.add_argument("--n-equil", type=int, default=500)
    parser.add_argument(
        "--save-equil-traj",
        action="store_true",
        help="Write ASE trajectories during equilibration (…_eq.traj under trajectories/).",
    )
    parser.add_argument(
        "--equil-traj-interval",
        type=int,
        default=None,
        help="Frame interval for equil trajectories (default: same as --interval).",
    )
    parser.add_argument("--n-prod", type=int, default=2000)
    parser.add_argument("--repeats-per-window", type=int, default=1)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--temperature-K", type=float, default=100.0)
    parser.add_argument("--ml-cutoff", type=float, default=1.0)
    parser.add_argument("--mm-switch-on", type=float, default=5.0)
    parser.add_argument("--mm-cutoff", type=float, default=5.0)
    parser.add_argument(
        "--no-fix-com",
        action="store_true",
        help="Do not use ASE FixCom (allows system COM translation in vacuum NVE).",
    )
    parser.add_argument(
        "--no-stationary",
        action="store_true",
        help="Skip Stationary/ZeroRotation after velocity initialization (net COM drift possible).",
    )


def config_from_namespace(args: argparse.Namespace, repo_root: Path | None = None) -> LambdaDynamicsConfig:
    comp = getattr(args, "composition", None)
    n_mol = int(args.n_molecules)
    if comp:
        parsed = md_suite._parse_composition(comp)
        n_mol = sum(c for _, c in parsed)
    couple_1b = [i + 1 for i in parse_couple_residue_numbers(str(args.couple_residues), n_mol)]

    pre_min_steps = int(args.pre_min_steps)
    pre_min_fmax = float(args.pre_min_fmax)
    if getattr(args, "min_steps", None) is not None:
        pre_min_steps = int(args.min_steps)
    if getattr(args, "min_fmax", None) is not None:
        pre_min_fmax = float(args.min_fmax)

    backend = str(getattr(args, "backend", "ase")).lower()
    if backend == "auto":
        backend = "ase"

    return LambdaDynamicsConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        composition=comp,
        n_molecules=int(args.n_molecules),
        residue=str(args.residue),
        template_pdb=getattr(args, "template_pdb", None),
        spacing=float(args.spacing),
        seed=int(args.seed),
        min_com_start_distance=float(getattr(args, "min_com_start_distance", 2.0)),
        couple_residue_numbers=couple_1b,
        lambda_windows=list(args.lambda_windows),
        min_steps=pre_min_steps,
        min_fmax=pre_min_fmax,
        pre_min_steps=pre_min_steps,
        pre_min_fmax=pre_min_fmax,
        n_equil=args.n_equil,
        save_equil_traj=bool(getattr(args, "save_equil_traj", False)),
        equil_traj_interval=getattr(args, "equil_traj_interval", None),
        n_prod=args.n_prod,
        repeats_per_window=args.repeats_per_window,
        interval=args.interval,
        timestep_fs=float(getattr(args, "timestep_fs", getattr(args, "dt_fs", 0.5))),
        temperature_K=float(getattr(args, "temperature_K", getattr(args, "temperature", 100.0))),
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        no_fix_com=args.no_fix_com,
        no_stationary=bool(getattr(args, "no_stationary", False)),
        md_mode=str(getattr(args, "lambda_md_mode", "free_nve")),
        backend=backend,
        box_size=getattr(args, "box_size", None),
        nvt_integrator=str(getattr(args, "nvt_integrator", "auto")),
        langevin_friction=float(getattr(args, "langevin_friction", 0.02)),
        charmm_pre_minimize=bool(getattr(args, "charmm_pre_minimize", True)),
        calculator_pre_minimize=bool(getattr(args, "calculator_pre_minimize", True)),
        bfgs_maxstep=float(getattr(args, "bfgs_maxstep", 0.05)),
        charmm_sd_steps=int(getattr(args, "charmm_sd_steps", 25)),
        charmm_abnr_steps=int(getattr(args, "charmm_abnr_steps", 100)),
        charmm_tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
        charmm_tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
        charmm_nbxmod=int(getattr(args, "charmm_nbxmod", 5)),
        min_intermonomer_atom_distance=float(
            getattr(args, "min_intermonomer_atom_distance", 0.1)
        ),
        rescue_minimize=bool(getattr(args, "rescue_minimize", True)),
        rescue_fire_steps=int(getattr(args, "rescue_fire_steps", 300)),
        rescue_fire_fmax=float(getattr(args, "rescue_fire_fmax", 0.1)),
        rescue_fire_maxstep=float(getattr(args, "rescue_fire_maxstep", 0.02)),
        max_fmax_after_min=float(getattr(args, "max_fmax_after_min", 2.0)),
        flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
        flat_bottom_k=float(getattr(args, "flat_bottom_k", 1.0)),
        flat_bottom_mode=str(getattr(args, "flat_bottom_mode", "system")),
        packmol_radius=getattr(args, "packmol_radius", None),
        packmol_sphere=getattr(args, "packmol_sphere", None),
        packmol_center=(
            tuple(args.packmol_center)
            if getattr(args, "packmol_center", None) is not None
            else None
        ),
        packmol_tolerance=float(getattr(args, "packmol_tolerance", 2.0)),
        skip_jit_warmup=bool(getattr(args, "skip_jit_warmup", False)),
        resume=bool(getattr(args, "resume", False)),
        repo_root=repo_root,
    )


def main_lambda_dynamics(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MMML lambda dynamics / TI for arbitrary composition and coupled residues."
    )
    add_lambda_dynamics_args(parser)
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("--interval must be >= 1")
    if args.repeats_per_window < 1:
        parser.error("--repeats-per-window must be >= 1")
    summary = run_lambda_dynamics(config_from_namespace(args))
    print_lambda_summary(summary)
    print(f"Wrote {summary['_summary_path']}")
    print(f"Snapshots: {summary['snapshots_npz']}")
    print("Run MBAR: mmml lambda-mbar --run-dir", args.output_dir)
    return 0

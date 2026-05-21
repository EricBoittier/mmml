#!/usr/bin/env python3
"""
MMML lambda dynamics / thermodynamic integration for arbitrary clusters.

Build any composition (as ``mmml md-system``), select coupled residues by
1-based index in cluster order, and sample λ windows. MBAR is separate:
``mmml lambda-mbar`` or ``scripts/meoh_dimer_lambda_mbar.py``.

Example (methanol dimer, couple residue 1):

  mmml md-system --setup lambda_ti --composition MEOH:2 --couple-residues 1 \\
    --output-dir artifacts/meoh_dimer_lambda_ti --checkpoint PATH

Example (mixed cluster, couple residues 1 and 3):

  mmml md-system --setup lambda_ti --composition MEOH:2,TIP3:1 --couple-residues 1,3 \\
    --spacing 6 --n-prod 2000
"""

<<<<<<< HEAD
from mmml.cli.run.lambda_dynamics import main_lambda_dynamics
=======
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import ase
import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.cli.base import load_physnet_params_and_ef_model, resolve_checkpoint_paths
from mmml.interfaces.pycharmmInterface.import_pycharmm import (
    CGENFF_PRM,
    CGENFF_RTF,
    coor,
    pycharmm,
    reset_block,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator
from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from mmml.models.physnetjax.physnetjax.restart.restart import get_params_model

import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings

pyci.read = read
pyci.settings = settings
pyci.psf = psf

_EV_TO_KCAL = 23.0609


def _load_scan_module(repo_root: Path):
    path = repo_root / "scripts" / "scan_meoh_dimer_distance.py"
    spec = importlib.util.spec_from_file_location("scan_meoh_dimer_distance", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _energy_at_positions(calc, atoms: ase.Atoms, r: np.ndarray) -> float:
    """Total potential energy (eV) at positions ``r`` using a fixed-lambda calculator."""
    atoms.set_positions(r)
    calc.results.clear()
    return float(atoms.get_potential_energy())


def _to_float_sum(value) -> float:
    arr = np.asarray(value)
    return float(arr.sum()) if arr.size > 1 else float(arr.reshape(()))


def _interaction_energy_from_results(calc) -> float:
    """Inter-monomer energy (eV): ML 2-body + MM (MM pair list excludes intra-monomer)."""
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


def _interaction_energy_at_positions(calc, atoms: ase.Atoms, r: np.ndarray) -> float:
    atoms.set_positions(r)
    calc.results.clear()
    atoms.get_potential_energy()
    return _interaction_energy_from_results(calc)


def _dUdlambda_at_R(
    calc_on: object,
    atoms_on: ase.Atoms,
    calc_off: object,
    atoms_off: ase.Atoms,
    r: np.ndarray,
) -> float:
    """Explicit ∂U/∂λ at fixed R: W(λ=1) - W(λ=0) with W = inter-monomer energy from the calculator.

    Same as total-energy difference when internal monomer ML energy is λ-independent.
    """
    w_on = _interaction_energy_at_positions(calc_on, atoms_on, r)
    w_off = _interaction_energy_at_positions(calc_off, atoms_off, r)
    return w_on - w_off


def _meoh_dimer_com_distance_A(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    com0 = r[:6].mean(axis=0)
    com1 = r[6:12].mean(axis=0)
    return float(np.linalg.norm(com1 - com0))


def _meoh_dimer_set_com_separation(r: np.ndarray, sep_A: float) -> np.ndarray:
    """Monomer 0 COM at origin, monomer 1 COM at (+sep_A, 0, 0); preserves intramolecular shape."""
    r = np.asarray(r, dtype=float).copy()
    m0, m1 = r[:6], r[6:12]
    m0 = m0 - m0.mean(axis=0)
    m1 = m1 - m1.mean(axis=0) + np.array([float(sep_A), 0.0, 0.0], dtype=float)
    r[:6] = m0
    r[6:12] = m1
    return r


def _build_fixed_lambda_calculator(
    *,
    atomic_numbers: np.ndarray,
    atomic_positions: np.ndarray,
    base_ckpt_dir: Path,
    cutoff: CutoffParameters,
    ml_cutoff: float,
    mm_switch_on: float,
    mm_cutoff: float,
    at_codes: np.ndarray,
    ep_scale: np.ndarray,
    sig_scale: np.ndarray,
    dec_idx: int,
    lam_decoupled: float,
) -> object:
    """Create a fresh MMML ASE calculator with fixed lambda_monomer."""
    lam_arr = np.ones(2, dtype=np.float32)
    lam_arr[dec_idx] = float(lam_decoupled)
    factory = setup_calculator(
        ATOMS_PER_MONOMER=6,
        N_MONOMERS=2,
        ml_cutoff_distance=ml_cutoff,
        mm_switch_on=mm_switch_on,
        mm_cutoff=mm_cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=12,
        cell=None,
        ep_scale=ep_scale,
        sig_scale=sig_scale,
        at_codes_override=at_codes,
        lambda_monomer=lam_arr,
    )
    calc, _, _ = factory(
        atomic_numbers=atomic_numbers,
        atomic_positions=atomic_positions,
        n_monomers=2,
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


def _plot_window_components(
    out_dir: Path,
    rows: list[dict],
    mbar_block: dict | None,
) -> list[str]:
    """Write per-window component and uncertainty plots."""
    written: list[str] = []
    if not rows:
        return written

    lam = np.array([float(r["lambda_decoupled_monomer"]) for r in rows], dtype=float)
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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    scan = _load_scan_module(repo_root)

    parser = argparse.ArgumentParser(description="MEOH dimer TI with MMML lambda dynamics.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Orbax root, epoch-* dir, or portable .json (default: resolve_checkpoint_paths).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/meoh_dimer_lambda_ti"))
    parser.add_argument("--template-pdb", type=Path, default=Path("pdb/meoh.pdb"))
    parser.add_argument("--initial-sep", type=float, default=3.2, help="Initial COM separation (Å).")
    parser.add_argument("--decouple-monomer", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--lambda-windows",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="λ for the decoupled monomer (--decouple-monomer); partner stays at 1.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=10,
        help="BFGS minimization steps at each lambda window before equilibration.",
    )
    parser.add_argument(
        "--min-fmax",
        type=float,
        default=0.03,
        help="BFGS convergence threshold (eV/Å) for per-window minimization.",
    )
    parser.add_argument("--n-equil", type=int, default=500, help="Langevin steps per window (equil).")
    parser.add_argument("--n-prod", type=int, default=2000, help="Production steps per window.")
    parser.add_argument(
        "--repeats-per-window",
        type=int,
        default=1,
        help="Number of independent repeats per lambda window (all snapshots are pooled for MBAR).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Sample ⟨∂U/∂λ⟩ every N production steps (extra energy evals).",
    )
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--temperature-K", type=float, default=100.0, help="Unused in NVE mode; kept for output compatibility.")
    parser.add_argument("--friction", type=float, default=0.02, help="Unused in NVE mode; kept for output compatibility.")
    parser.add_argument("--ml-cutoff", type=float, default=1.0)
    parser.add_argument("--mm-switch-on", type=float, default=5.0)
    parser.add_argument("--mm-cutoff", type=float, default=5.0)
    parser.add_argument(
        "--no-mbar",
        action="store_true",
        help="Skip pymbar MBAR post-processing (no u_kln matrix or ΔF uncertainties).",
    )
    parser.add_argument(
        "--mbar-verbose",
        action="store_true",
        help="Verbose pymbar solver output.",
    )
    parser.add_argument(
        "--no-fix-com",
        action="store_true",
        help="Do not use FixCom (keeps fixcm=False; only use if COM constraint causes issues).",
    )
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be >= 1")
    if args.repeats_per_window < 1:
        parser.error("--repeats-per-window must be >= 1")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is None:
        ckpt_root, _ = resolve_checkpoint_paths(None)
    else:
        ckpt_root = args.checkpoint.expanduser().resolve()
    base_ckpt_dir, _ = resolve_checkpoint_paths(ckpt_root)

    tmpl = args.template_pdb.expanduser().resolve()
    if not tmpl.is_file():
        tmpl = repo_root / tmpl
    z, r0 = scan._setup_charmm_meoh_dimer(tmpl, initial_sep=args.initial_sep)
    base_seed_positions = _meoh_dimer_set_com_separation(r0, args.initial_sep)
    assert abs(_meoh_dimer_com_distance_A(base_seed_positions) - float(args.initial_sep)) < 1e-6

    if ckpt_root.is_file() and ckpt_root.suffix == ".json":
        phys_params, phys_model = load_physnet_params_and_ef_model(ckpt_root, natoms=len(z))
    else:
        epoch_dir = scan._latest_epoch_dir(ckpt_root)
        phys_params, phys_model = get_params_model(str(epoch_dir), natoms=len(z))
    phys_model.natoms = len(z)

    at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    ep_scale = np.ones(n_types, dtype=float)
    sig_scale = np.ones(n_types, dtype=float)

    cutoff = CutoffParameters(
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
    )

    use_fix_com = not args.no_fix_com

    lambda_windows = sorted(float(x) for x in args.lambda_windows)
    dec_idx = args.decouple_monomer
    other_idx = 1 - dec_idx

    rows: list[dict] = []
    snapshots_per_window: list[list[np.ndarray]] = [[] for _ in range(len(lambda_windows))]
    traj_root = out_dir / "trajectories"
    traj_root.mkdir(exist_ok=True)

    dt = args.timestep_fs * units.fs
    for wi, lam in enumerate(lambda_windows):
        samples: list[float] = []
        repeat_stats: list[dict[str, float | int | str]] = []
        for rep in range(args.repeats_per_window):
            r_init = _meoh_dimer_set_com_separation(base_seed_positions, args.initial_sep)
            atoms = ase.Atoms(numbers=z, positions=r_init)
            if use_fix_com:
                atoms.set_constraint(FixCom())
            calc_dyn = _build_fixed_lambda_calculator(
                atomic_numbers=z,
                atomic_positions=atoms.get_positions(),
                base_ckpt_dir=base_ckpt_dir,
                cutoff=cutoff,
                ml_cutoff=args.ml_cutoff,
                mm_switch_on=args.mm_switch_on,
                mm_cutoff=args.mm_cutoff,
                at_codes=at_codes,
                ep_scale=ep_scale,
                sig_scale=sig_scale,
                dec_idx=dec_idx,
                lam_decoupled=lam,
            )
            atoms.calc = calc_dyn
            # NVE setup: assign thermal velocities, then remove net COM drift.
            MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K)
            Stationary(atoms)
            coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))

            min_traj_path = traj_root / f"win_{wi:02d}_rep{rep:02d}_lam{lam:.2f}_min.traj"
            minimizer = BFGS(atoms, trajectory=str(min_traj_path), logfile=None)
            minimizer.run(fmax=args.min_fmax, steps=args.min_steps)
            coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))

            prod_path = traj_root / f"win_{wi:02d}_rep{rep:02d}_lam{lam:.2f}_prod.traj"
            traj_prod = Trajectory(str(prod_path), "w", atoms)

            dyn_eq = VelocityVerlet(atoms, timestep=dt)
            dyn_eq.run(args.n_equil)

            samples_rep: list[float] = []
            step_count = [0]
            atoms_probe_on = ase.Atoms(numbers=z, positions=atoms.get_positions().copy())
            atoms_probe_off = ase.Atoms(numbers=z, positions=atoms.get_positions().copy())
            calc_probe_on = _build_fixed_lambda_calculator(
                atomic_numbers=z,
                atomic_positions=atoms_probe_on.get_positions(),
                base_ckpt_dir=base_ckpt_dir,
                cutoff=cutoff,
                ml_cutoff=args.ml_cutoff,
                mm_switch_on=args.mm_switch_on,
                mm_cutoff=args.mm_cutoff,
                at_codes=at_codes,
                ep_scale=ep_scale,
                sig_scale=sig_scale,
                dec_idx=dec_idx,
                lam_decoupled=1.0,
            )
            calc_probe_off = _build_fixed_lambda_calculator(
                atomic_numbers=z,
                atomic_positions=atoms_probe_off.get_positions(),
                base_ckpt_dir=base_ckpt_dir,
                cutoff=cutoff,
                ml_cutoff=args.ml_cutoff,
                mm_switch_on=args.mm_switch_on,
                mm_cutoff=args.mm_cutoff,
                at_codes=at_codes,
                ep_scale=ep_scale,
                sig_scale=sig_scale,
                dec_idx=dec_idx,
                lam_decoupled=0.0,
            )
            atoms_probe_on.calc = calc_probe_on
            atoms_probe_off.calc = calc_probe_off

            def _on_step(_atoms=atoms):
                step_count[0] += 1
                if step_count[0] % args.interval == 0:
                    d = _dUdlambda_at_R(
                        calc_probe_on,
                        atoms_probe_on,
                        calc_probe_off,
                        atoms_probe_off,
                        _atoms.get_positions().copy(),
                    )
                    samples_rep.append(d)
                    samples.append(d)
                    snapshots_per_window[wi].append(_atoms.get_positions().copy())

            dyn_prod = VelocityVerlet(atoms, timestep=dt)
            dyn_prod.attach(traj_prod.write, interval=max(1, args.interval))
            dyn_prod.attach(_on_step)
            dyn_prod.run(args.n_prod)
            traj_prod.close()

            repeat_stats.append(
                {
                    "repeat": rep,
                    "n_samples": len(samples_rep),
                    "mean_dUdlambda_eV": float(np.mean(samples_rep)) if samples_rep else float("nan"),
                    "std_dUdlambda_eV": float(np.std(samples_rep)) if len(samples_rep) > 1 else 0.0,
                    "traj": str(prod_path),
                }
            )

        mean_b = float(np.mean(samples)) if samples else float("nan")
        std_b = float(np.std(samples)) if len(samples) > 1 else 0.0
        rows.append(
            {
                "window": wi,
                "lambda_decoupled_monomer": lam,
                "decouple_monomer_index": dec_idx,
                "repeats_per_window": args.repeats_per_window,
                "mean_dUdlambda_eV": mean_b,
                "std_dUdlambda_eV": std_b,
                "n_samples": len(samples),
                "repeat_stats": repeat_stats,
            }
        )

    lam_col = np.array([r["lambda_decoupled_monomer"] for r in rows], dtype=float)
    mean_b = np.array([r["mean_dUdlambda_eV"] for r in rows], dtype=float)

    delta_f_ev = float(np.trapezoid(mean_b, lam_col)) if len(lam_col) > 1 else float("nan")
    delta_f_kcal = delta_f_ev * _EV_TO_KCAL

    mbar_block: dict | None = None
    if not args.no_mbar:
        try:
            from pymbar import MBAR, timeseries
        except ImportError as exc:
            raise SystemExit(
                "pymbar is required unless --no-mbar is set. Install with: "
                "uv sync --extra mbar   or   pip install 'pymbar>=4.0'"
            ) from exc

        K = len(lambda_windows)
        N_k = np.array([len(snapshots_per_window[k]) for k in range(K)], dtype=np.int64)
        if np.any(N_k == 0):
            mbar_block = {
                "error": "MBAR skipped: at least one λ-window has no snapshots (increase --n-prod or decrease --interval).",
                "N_k": N_k.tolist(),
            }
        else:
            N_max = int(N_k.max())
            beta = 1.0 / (float(units.kB) * args.temperature_K)
            u_kln = np.zeros((K, K, N_max), dtype=np.float64)
            mbar_atoms_bank: list[ase.Atoms] = []
            mbar_calc_bank: list[object] = []
            for l in range(K):
                atoms_l = ase.Atoms(numbers=z, positions=base_seed_positions.copy())
                calc_l = _build_fixed_lambda_calculator(
                    atomic_numbers=z,
                    atomic_positions=atoms_l.get_positions(),
                    base_ckpt_dir=base_ckpt_dir,
                    cutoff=cutoff,
                    ml_cutoff=args.ml_cutoff,
                    mm_switch_on=args.mm_switch_on,
                    mm_cutoff=args.mm_cutoff,
                    at_codes=at_codes,
                    ep_scale=ep_scale,
                    sig_scale=sig_scale,
                    dec_idx=dec_idx,
                    lam_decoupled=lambda_windows[l],
                )
                atoms_l.calc = calc_l
                mbar_atoms_bank.append(atoms_l)
                mbar_calc_bank.append(calc_l)
            for k in range(K):
                for n in range(int(N_k[k])):
                    r_snap = snapshots_per_window[k][n]
                    for l in range(K):
                        u_kln[k, l, n] = beta * _energy_at_positions(
                            mbar_calc_bank[l],
                            mbar_atoms_bank[l],
                            r_snap,
                        )
            # Decorrelate per-window samples for MBAR (recommended by pymbar).
            # We estimate statistical inefficiency from the sampled state's own
            # reduced potential time series u_kln[k, k, :N_k[k]].
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

            mbar = MBAR(u_kln_eff, N_k_eff, verbose=args.mbar_verbose)
            fe = mbar.compute_free_energy_differences(compute_uncertainty=True)
            # Delta_f[i,j] ≈ (f_j - f_i) / kT; i=0 (λ=0), j=K-1 (λ=1) => coupling free energy
            i0, i1 = 0, K - 1
            df_k = float(fe["Delta_f"][i0, i1])
            ddf_k = float(fe["dDelta_f"][i0, i1])
            kbt_ev = float(units.kB) * args.temperature_K
            df_ev = df_k * kbt_ev
            ddf_ev = ddf_k * kbt_ev
            mbar_block = {
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
                "note": "Coupling ΔF = F(λ=1) - F(λ=0) from MBAR; dissociation is the negative.",
            }

    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "geometry": {
            "initial_com_separation_A": float(args.initial_sep),
            "seed_com_separation_A": _meoh_dimer_com_distance_A(base_seed_positions),
            "note": "Every λ window and repeat starts from the same seed with this COM separation (before minimization/MD).",
        },
        "description": {
            "delta_F_couple_eV": "TI integral ∫_0^1 ⟨∂U/∂λ⟩ dλ (turn on intermolecular coupling)",
            "delta_F_diss_eV": "Negative of delta_F_couple (decouple along same path)",
            "lambda_definition": f"lambda_monomer[{dec_idx}] varies; lambda_monomer[{other_idx}]=1",
        },
        "delta_F_couple_eV": delta_f_ev,
        "delta_F_couple_kcal_mol": delta_f_kcal,
        "delta_F_diss_eV": -delta_f_ev,
        "delta_F_diss_kcal_mol": -delta_f_kcal,
        "mbar": mbar_block,
        "windows": rows,
        "args": args_json,
    }
    plot_files = _plot_window_components(out_dir, rows, mbar_block)
    summary["plots"] = plot_files

    out_json = out_dir / "lambda_ti_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["description"], indent=2))
    print(f"ΔF_couple (binding path, TI) = {delta_f_ev:.6f} eV = {delta_f_kcal:.4f} kcal/mol")
    print(f"ΔF_diss (decouple)          = {-delta_f_ev:.6f} eV = {-delta_f_kcal:.4f} kcal/mol")
    if mbar_block and "error" not in mbar_block:
        print(
            f"ΔF_couple (MBAR)            = {mbar_block['Delta_F_couple_eV']:.6f} ± "
            f"{mbar_block['dDelta_F_couple_eV']:.6f} eV = "
            f"{mbar_block['Delta_F_couple_kcal_mol']:.4f} ± "
            f"{mbar_block['dDelta_F_couple_kcal_mol']:.4f} kcal/mol"
        )
    elif mbar_block and "error" in mbar_block:
        print(f"MBAR: {mbar_block['error']}")
    print(f"Wrote {out_json}")
    return 0

>>>>>>> a34cb3ef2 (artefacts)

if __name__ == "__main__":
    raise SystemExit(main_lambda_dynamics())

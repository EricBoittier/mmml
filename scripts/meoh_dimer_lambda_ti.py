#!/usr/bin/env python3
"""
Methanol dimer: MMML hybrid calculator + lambda dynamics (alchemical decoupling).

Uses per-monomer ``lambda_monomer`` scaling (see ``mmml_calculator.setup_calculator``):
inter-molecular ML/MM terms between monomers i,j scale as λ_i λ_j; intramolecular
terms are unchanged.

For a dimer with λ_0 = λ and λ_1 = 1, the potential is linear in λ:

    U(R, λ) = U_0(R) + λ B(R)

so ∂U/∂λ = B(R) = U(R; λ=[1,1]) - U(R; λ=[0,1]) at fixed R.

Thermodynamic integration (coupling the interaction, λ: 0 → 1):

    ΔF_bind_path = ∫_0^1 ⟨∂U/∂λ⟩_λ dλ

The free energy to decouple (dissociate along this path) is ΔF_diss = -ΔF_bind_path.

After sampling, **pymbar** (MBAR) recomputes potentials at every λ for each saved
snapshot and estimates ΔF between end states with uncertainties (optional extra
``mbar``; use ``--no-mbar`` to skip the post-processing pass).

By default Langevin uses ASE's ``fixcm=True`` (may emit a FutureWarning for
small systems). Pass ``--fix-com`` to use ``FixCom`` with ``fixcm=False`` per
ASE 3.28+ NVT guidance.

**Caveats:** gas-phase dimer; MBAR assumes uncorrelated snapshots (consider
subsampling with ``pymbar.timeseries`` for production); pymbar may enable JAX
64-bit mode in-process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

import ase
import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin

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


def _energy_at_lambda(calc, atoms: ase.Atoms, r: np.ndarray, dec_idx: int, lam_decoupled: float) -> float:
    """Total potential energy (eV) at positions ``r`` with λ_decoupled on ``dec_idx``."""
    lam_arr = np.ones(2, dtype=np.float32)
    lam_arr[dec_idx] = float(lam_decoupled)
    calc.set_lambda_monomer(lam_arr)
    atoms.set_positions(r)
    calc.results.clear()
    return float(atoms.get_potential_energy())


def _dUdlambda_at_R(calc, atoms: ase.Atoms, dec_idx: int, lam_restore: float) -> float:
    """∂U/∂λ at fixed R: U(λ_i=1) - U(λ_i=0) with the partner monomer at λ=1 (linear coupling)."""
    r = atoms.get_positions().copy()
    lam_full = np.ones(2, dtype=np.float32)
    lam_off = np.ones(2, dtype=np.float32)
    lam_off[dec_idx] = 0.0
    calc.set_lambda_monomer(lam_full)
    atoms.set_positions(r)
    # λ changed without moving atoms: ASE would reuse cached energy unless we invalidate.
    calc.results.clear()
    e_on = float(atoms.get_potential_energy())
    calc.set_lambda_monomer(lam_off)
    atoms.set_positions(r)
    calc.results.clear()
    e_off = float(atoms.get_potential_energy())
    lam_cur = np.ones(2, dtype=np.float32)
    lam_cur[dec_idx] = float(lam_restore)
    calc.set_lambda_monomer(lam_cur)
    atoms.set_positions(r)
    calc.results.clear()
    return e_on - e_off


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
    parser.add_argument("--template-pdb", type=Path, default=Path("mmml/generate/sample/pdb/meoh.pdb"))
    parser.add_argument("--initial-sep", type=float, default=3.2, help="Initial COM separation (Å).")
    parser.add_argument("--decouple-monomer", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--lambda-windows",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="λ for the decoupled monomer (--decouple-monomer); partner stays at 1.",
    )
    parser.add_argument("--n-equil", type=int, default=500, help="Langevin steps per window (equil).")
    parser.add_argument("--n-prod", type=int, default=2000, help="Production steps per window.")
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Sample ⟨∂U/∂λ⟩ every N production steps (extra energy evals).",
    )
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=0.02, help="Langevin friction (1/fs).")
    parser.add_argument("--ml-cutoff", type=float, default=5.0)
    parser.add_argument("--mm-switch-on", type=float, default=5.0)
    parser.add_argument("--mm-cutoff", type=float, default=3.0)
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
        "--fix-com",
        action="store_true",
        help="Constrain system center of mass (use with fixcm=False; ASE 3.28+ small-system NVT).",
    )
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be >= 1")

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

    initial_lambda = np.ones(2, dtype=np.float32)
    initial_lambda[args.decouple_monomer] = 1.0
    factory = setup_calculator(
        ATOMS_PER_MONOMER=6,
        N_MONOMERS=2,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
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
        lambda_monomer=initial_lambda,
    )
    cutoff = CutoffParameters(
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
    )

    atoms = ase.Atoms(numbers=z, positions=r0)
    hybrid_calc, _, _ = factory(
        atomic_numbers=z,
        atomic_positions=r0,
        n_monomers=2,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        backprop=False,
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=False,
    )
    atoms.calc = hybrid_calc
    if args.fix_com:
        atoms.set_constraint(FixCom())

    lambda_windows = sorted(float(x) for x in args.lambda_windows)
    dec_idx = args.decouple_monomer
    other_idx = 1 - dec_idx

    rows: list[dict] = []
    snapshots_per_window: list[list[np.ndarray]] = [[] for _ in range(len(lambda_windows))]
    traj_root = out_dir / "trajectories"
    traj_root.mkdir(exist_ok=True)

    dt = args.timestep_fs * units.fs
    friction = args.friction / units.fs

    for wi, lam in enumerate(lambda_windows):
        lam_arr = np.ones(2, dtype=np.float32)
        lam_arr[dec_idx] = lam
        hybrid_calc.set_lambda_monomer(lam_arr)
        coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))

        prod_path = traj_root / f"win_{wi:02d}_lam{lam:.2f}_prod.traj"
        traj_prod = Trajectory(str(prod_path), "w", atoms)

        dyn_eq = Langevin(
            atoms,
            timestep=dt,
            temperature_K=args.temperature_K,
            friction=friction,
            fixcm=not args.fix_com,
        )
        dyn_eq.run(args.n_equil)

        samples: list[float] = []
        step_count = [0]

        def _on_step(_atoms=atoms):
            step_count[0] += 1
            if step_count[0] % args.interval == 0:
                samples.append(_dUdlambda_at_R(hybrid_calc, _atoms, dec_idx, lam))
                snapshots_per_window[wi].append(_atoms.get_positions().copy())

        dyn_prod = Langevin(
            atoms,
            timestep=dt,
            temperature_K=args.temperature_K,
            friction=friction,
            fixcm=not args.fix_com,
        )
        dyn_prod.attach(traj_prod.write, interval=max(1, args.interval))
        dyn_prod.attach(_on_step)
        dyn_prod.run(args.n_prod)
        traj_prod.close()

        mean_b = float(np.mean(samples)) if samples else float("nan")
        std_b = float(np.std(samples)) if len(samples) > 1 else 0.0
        rows.append(
            {
                "window": wi,
                "lambda_decoupled_monomer": lam,
                "decouple_monomer_index": dec_idx,
                "mean_dUdlambda_eV": mean_b,
                "std_dUdlambda_eV": std_b,
                "n_samples": len(samples),
                "traj": str(prod_path),
            }
        )

    lam_col = np.array([r["lambda_decoupled_monomer"] for r in rows], dtype=float)
    mean_b = np.array([r["mean_dUdlambda_eV"] for r in rows], dtype=float)

    delta_f_ev = float(np.trapezoid(mean_b, lam_col)) if len(lam_col) > 1 else float("nan")
    delta_f_kcal = delta_f_ev * _EV_TO_KCAL

    mbar_block: dict | None = None
    if not args.no_mbar:
        try:
            from pymbar import MBAR
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
            for k in range(K):
                for n in range(int(N_k[k])):
                    r_snap = snapshots_per_window[k][n]
                    for l in range(K):
                        u_kln[k, l, n] = beta * _energy_at_lambda(
                            hybrid_calc, atoms, r_snap, dec_idx, lambda_windows[l]
                        )
            mbar = MBAR(u_kln, N_k, verbose=args.mbar_verbose)
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
                "note": "Coupling ΔF = F(λ=1) - F(λ=0) from MBAR; dissociation is the negative.",
            }

    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
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


if __name__ == "__main__":
    raise SystemExit(main())

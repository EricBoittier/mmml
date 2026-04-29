#!/usr/bin/env python3
"""
2D methanol-dimer scan to visualize cutoff behavior in MMML.

Scans COM distance (d) and alchemical lambda (λ), then reports:
  - Total energy and MMML components (internal, ML 2-body, MM)
  - Force forms along distance via -dE/dd for each component

Outputs CSV/JSON and plots in output_dir.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmml.cli.base import resolve_checkpoint_paths
from mmml.interfaces.pycharmmInterface.import_pycharmm import coor
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator

import pycharmm.param as param
import pycharmm.psf as psf


def _load_scan_utils(repo_root: Path):
    path = repo_root / "scripts" / "scan_meoh_dimer_distance.py"
    spec = importlib.util.spec_from_file_location("scan_meoh_dimer_distance", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _to_float_sum(value) -> float:
    arr = np.asarray(value)
    return float(arr.sum()) if arr.size > 1 else float(arr.reshape(()))


def _save_heatmap(
    out_path: Path,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    cbar_label: str,
    symmetric: bool = False,
    robust: bool = True,
) -> None:
    z_plot = np.asarray(z, dtype=float)
    if robust:
        lo = float(np.nanpercentile(z_plot, 2.0))
        hi = float(np.nanpercentile(z_plot, 98.0))
    else:
        lo = float(np.nanmin(z_plot))
        hi = float(np.nanmax(z_plot))
    if symmetric:
        lim = max(abs(lo), abs(hi))
        lim = 1e-12 if lim < 1e-12 else lim
        lo, hi = -lim, lim
    elif abs(hi - lo) < 1e-12:
        c = 0.5 * (hi + lo)
        lo, hi = c - 1e-12, c + 1e-12

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(
        z_plot,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        interpolation="nearest",
        vmin=lo,
        vmax=hi,
        cmap="RdBu_r" if symmetric else None,
    )
    ax.set_xlabel("Distance (A)")
    ax.set_ylabel("Lambda")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="2D dimer scan: distance x lambda with cutoff component analysis.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/meoh_dimer_2d_cutoff_scan"))
    parser.add_argument("--template-pdb", type=Path, default=Path("mmml/generate/sample/pdb/meoh.pdb"))
    parser.add_argument("--dmin", type=float, default=2.6)
    parser.add_argument("--dmax", type=float, default=8.0)
    parser.add_argument("--n-dist", type=int, default=36)
    parser.add_argument(
        "--lambda-windows",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--ml-cutoff", type=float, default=5.0)
    parser.add_argument("--mm-switch-on", type=float, default=5.0)
    parser.add_argument("--mm-cutoff", type=float, default=3.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    scan_utils = _load_scan_utils(repo_root)

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
    z, base_r = scan_utils._setup_charmm_meoh_dimer(tmpl, initial_sep=args.dmax)

    at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    ep_scale = np.ones(n_types, dtype=float)
    sig_scale = np.ones(n_types, dtype=float)

    cutoff = CutoffParameters(
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        complementary_handoff=True,
    )

    com0 = base_r[:6].mean(axis=0)
    com1 = base_r[6:].mean(axis=0)
    unit = com1 - com0
    norm = np.linalg.norm(unit)
    unit = np.array([1.0, 0.0, 0.0], dtype=float) if norm < 1e-10 else unit / norm

    distances = np.linspace(args.dmin, args.dmax, args.n_dist)
    lambdas = np.array(sorted(float(x) for x in args.lambda_windows), dtype=float)

    n_l = len(lambdas)
    n_d = len(distances)
    e_total = np.zeros((n_l, n_d), dtype=float)
    e_internal = np.zeros((n_l, n_d), dtype=float)
    e_ml2b = np.zeros((n_l, n_d), dtype=float)
    e_mm = np.zeros((n_l, n_d), dtype=float)

    rows: list[dict[str, float]] = []
    for il, lam in enumerate(lambdas):
        lam_arr = np.ones(2, dtype=np.float32)
        lam_arr[0] = float(lam)
        # Build calculator factory per lambda to avoid mutable lambda capture issues
        # in long grid scans.
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
            lambda_monomer=lam_arr,
            complementary_handoff=True,
        )
        for idd, d in enumerate(distances):
            r = base_r.copy()
            r[:6] -= r[:6].mean(axis=0)
            r[6:] -= r[6:].mean(axis=0)
            r[6:] += d * unit
            coor.set_positions(pd.DataFrame(r, columns=["x", "y", "z"]))

            atoms = ase.Atoms(numbers=z, positions=r)
            mmml_calc, _, _ = factory(
                atomic_numbers=z,
                atomic_positions=r,
                n_monomers=2,
                cutoff_params=cutoff,
                doML=True,
                doMM=True,
                doML_dimer=True,
                backprop=True,
                debug=False,
                energy_conversion_factor=1.0,
                force_conversion_factor=1.0,
                verbose=True,
            )
            atoms.calc = mmml_calc

            et = float(atoms.get_potential_energy())
            res = mmml_calc.results
            ei = _to_float_sum(res.get("model_internal_E", 0.0))
            eml = _to_float_sum(res.get("model_ml_2b_E", 0.0))
            emm = _to_float_sum(res.get("model_mm_E", 0.0))

            e_total[il, idd] = et
            e_internal[il, idd] = ei
            e_ml2b[il, idd] = eml
            e_mm[il, idd] = emm
            rows.append(
                {
                    "lambda": float(lam),
                    "distance_A": float(d),
                    "E_total_eV": et,
                    "E_internal_eV": ei,
                    "E_ml2b_eV": eml,
                    "E_mm_eV": emm,
                }
            )

    # Force forms along distance from component energies: F = -dE/dd
    e_interaction = e_total - e_internal
    f_total = -np.gradient(e_total, distances, axis=1)
    f_internal = -np.gradient(e_internal, distances, axis=1)
    f_ml2b = -np.gradient(e_ml2b, distances, axis=1)
    f_mm = -np.gradient(e_mm, distances, axis=1)
    dlam_total = e_total[-1] - e_total[0]
    dlam_internal = e_internal[-1] - e_internal[0]
    dlam_ml2b = e_ml2b[-1] - e_ml2b[0]
    dlam_mm = e_mm[-1] - e_mm[0]
    dlam_force_total = f_total[-1] - f_total[0]
    dlam_force_ml2b = f_ml2b[-1] - f_ml2b[0]
    dlam_force_mm = f_mm[-1] - f_mm[0]

    df = pd.DataFrame(rows)
    csv_path = out_dir / "scan_2d_components.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "args": {
            "dmin": args.dmin,
            "dmax": args.dmax,
            "n_dist": args.n_dist,
            "lambda_windows": lambdas.tolist(),
            "ml_cutoff": args.ml_cutoff,
            "mm_switch_on": args.mm_switch_on,
            "mm_cutoff": args.mm_cutoff,
            "cutoff_region_start_A": args.mm_switch_on - args.ml_cutoff,
            "cutoff_region_end_A": args.mm_switch_on,
        },
        "outputs": {
            "csv": str(csv_path),
        },
        "ranges": {
            "E_ml2b_minmax_eV": [float(np.min(e_ml2b)), float(np.max(e_ml2b))],
            "E_mm_minmax_eV": [float(np.min(e_mm)), float(np.max(e_mm))],
            "dlambda_ml2b_minmax_eV": [float(np.min(dlam_ml2b)), float(np.max(dlam_ml2b))],
            "dlambda_mm_minmax_eV": [float(np.min(dlam_mm)), float(np.max(dlam_mm))],
        },
    }

    _save_heatmap(out_dir / "E_total_2d.png", e_total, distances, lambdas, "Total energy E(d, lambda)", "eV")
    _save_heatmap(out_dir / "E_internal_2d.png", e_internal, distances, lambdas, "Internal component", "eV")
    _save_heatmap(out_dir / "E_interaction_2d.png", e_interaction, distances, lambdas, "Interaction energy E_total - E_internal", "eV")
    _save_heatmap(out_dir / "E_ml2b_2d.png", e_ml2b, distances, lambdas, "ML 2-body component", "eV")
    _save_heatmap(out_dir / "E_mm_2d.png", e_mm, distances, lambdas, "MM component", "eV")

    _save_heatmap(out_dir / "F_total_2d.png", f_total, distances, lambdas, "Force form -dE_total/dd", "eV/A")
    _save_heatmap(out_dir / "F_internal_2d.png", f_internal, distances, lambdas, "Force form -dE_internal/dd", "eV/A")
    _save_heatmap(out_dir / "F_ml2b_2d.png", f_ml2b, distances, lambdas, "Force form -dE_ml2b/dd", "eV/A")
    _save_heatmap(out_dir / "F_mm_2d.png", f_mm, distances, lambdas, "Force form -dE_mm/dd", "eV/A")

    # Explicit lambda-effect maps: these should reveal whether lambda is active.
    _save_heatmap(
        out_dir / "DeltaLambda_E_total.png",
        dlam_total[None, :],
        distances,
        np.array([0.0, 1.0]),
        "Δλ effect: E_total(λmax) - E_total(λmin)",
        "eV",
    )
    _save_heatmap(
        out_dir / "DeltaLambda_E_ml2b.png",
        dlam_ml2b[None, :],
        distances,
        np.array([0.0, 1.0]),
        "Δλ effect: E_ml2b(λmax) - E_ml2b(λmin)",
        "eV",
    )
    _save_heatmap(
        out_dir / "DeltaLambda_E_mm.png",
        dlam_mm[None, :],
        distances,
        np.array([0.0, 1.0]),
        "Δλ effect: E_mm(λmax) - E_mm(λmin)",
        "eV",
    )
    _save_heatmap(
        out_dir / "DeltaLambda_F_total.png",
        dlam_force_total[None, :],
        distances,
        np.array([0.0, 1.0]),
        "Δλ effect: F_total(λmax) - F_total(λmin)",
        "eV/A",
    )
    _save_heatmap(
        out_dir / "DeltaLambda_F_ml2b.png",
        dlam_force_ml2b[None, :],
        distances,
        np.array([0.0, 1.0]),
        "Δλ effect: F_ml2b(λmax) - F_ml2b(λmin)",
        "eV/A",
    )
    _save_heatmap(
        out_dir / "DeltaLambda_F_mm.png",
        dlam_force_mm[None, :],
        distances,
        np.array([0.0, 1.0]),
        "Δλ effect: F_mm(λmax) - F_mm(λmin)",
        "eV/A",
    )

    # Slice plots at representative lambdas with cutoff markers.
    picks = [0, len(lambdas) // 2, len(lambdas) - 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in picks:
        ax.plot(distances, e_total[i], label=f"Total λ={lambdas[i]:.2f}")
    ax.set_xlabel("Distance (A)")
    ax.set_ylabel("Total energy (eV)")
    ax.set_title("Total energy vs distance")
    yvals_total = e_total[picks].ravel()
    ylo_total, yhi_total = float(np.percentile(yvals_total, 2)), float(np.percentile(yvals_total, 98))
    if abs(yhi_total - ylo_total) < 1e-12:
        ymid_total = 0.5 * (yhi_total + ylo_total)
        ylo_total, yhi_total = ymid_total - 1e-6, ymid_total + 1e-6
    ax.set_ylim(ylo_total, yhi_total)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "total_energy_vs_distance.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in picks:
        ax.plot(distances, e_interaction[i], label=f"Interaction λ={lambdas[i]:.2f}")
    ax.set_xlabel("Distance (A)")
    ax.set_ylabel("Interaction energy (eV)")
    ax.set_title("Interaction energy vs distance")
    yvals_inter = e_interaction[picks].ravel()
    ylo_inter, yhi_inter = float(np.percentile(yvals_inter, 2)), float(np.percentile(yvals_inter, 98))
    if abs(yhi_inter - ylo_inter) < 1e-12:
        ymid_inter = 0.5 * (yhi_inter + ylo_inter)
        ylo_inter, yhi_inter = ymid_inter - 1e-6, ymid_inter + 1e-6
    ax.set_ylim(ylo_inter, yhi_inter)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_energy_vs_distance.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in picks:
        ax.plot(distances, e_ml2b[i], label=f"ML2B λ={lambdas[i]:.2f}")
        ax.plot(distances, e_mm[i], "--", label=f"MM λ={lambdas[i]:.2f}")
    ax.axvline(args.mm_switch_on - args.ml_cutoff, color="k", linestyle=":", label="switch start")
    ax.axvline(args.mm_switch_on, color="k", linestyle="--", label="switch end")
    ax.set_xlabel("Distance (A)")
    ax.set_ylabel("Energy component (eV)")
    ax.set_title("Cutoff region and component slices")
    yvals = np.concatenate([e_ml2b[picks].ravel(), e_mm[picks].ravel()])
    ylo, yhi = float(np.percentile(yvals, 2)), float(np.percentile(yvals, 98))
    if abs(yhi - ylo) < 1e-12:
        ymid = 0.5 * (yhi + ylo)
        ylo, yhi = ymid - 1e-6, ymid + 1e-6
    ax.set_ylim(ylo, yhi)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cutoff_component_slices.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in picks:
        ax.plot(distances, f_ml2b[i], label=f"-dE_ml2b/dd λ={lambdas[i]:.2f}")
        ax.plot(distances, f_mm[i], "--", label=f"-dE_mm/dd λ={lambdas[i]:.2f}")
    ax.axvline(args.mm_switch_on - args.ml_cutoff, color="k", linestyle=":", label="switch start")
    ax.axvline(args.mm_switch_on, color="k", linestyle="--", label="switch end")
    ax.set_xlabel("Distance (A)")
    ax.set_ylabel("Force form (eV/A)")
    ax.set_title("Cutoff region and force-form slices")
    yvals_f = np.concatenate([f_ml2b[picks].ravel(), f_mm[picks].ravel()])
    ylo_f, yhi_f = float(np.percentile(yvals_f, 2)), float(np.percentile(yvals_f, 98))
    if abs(yhi_f - ylo_f) < 1e-12:
        ymid_f = 0.5 * (yhi_f + ylo_f)
        ylo_f, yhi_f = ymid_f - 1e-6, ymid_f + 1e-6
    ax.set_ylim(ylo_f, yhi_f)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cutoff_forceform_slices.png", dpi=170)
    plt.close(fig)

    summary["outputs"]["plots"] = sorted(str(p) for p in out_dir.glob("*.png"))
    summary_path = out_dir / "scan_2d_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    for p in summary["outputs"]["plots"]:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


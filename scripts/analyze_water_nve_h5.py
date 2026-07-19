#!/usr/bin/env python3
"""Analyze a TIP3 NVE HDF5 from jaxmd (positions, charges, energies).

Produces:
  - IR spectrum from molecular dipole fluctuations (q·r)
  - Per-atom / per-element charge variance
  - O–H bond length and H–O–H angle distributions + charge vs geometry
  - Total / potential / kinetic energy time series and fluctuations

Example
-------
uv run python scripts/analyze_water_nve_h5.py \\
  --h5 scratch/spooky_muon3_nve/pbc_nve_jaxmd_nve.h5 \\
  --box-A 23 --output-dir scratch/spooky_muon3_nve/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mmml.spectra.spectra_md import dipole_fluctuation_ir_spectrum

EV_TO_KCAL_MOL = 23.06054783061903


def _mic(d: np.ndarray, box: float) -> np.ndarray:
    return d - box * np.round(d / box)


def molecular_dipoles(
    positions: np.ndarray,
    charges: np.ndarray,
    z: np.ndarray,
    box: float,
) -> np.ndarray:
    """Sum of per-molecule dipoles (e·Å) with intramolecular MIC about O."""
    n_frames, n_atoms, _ = positions.shape
    assert n_atoms % 3 == 0
    n_mol = n_atoms // 3
    mu = np.zeros((n_frames, 3), dtype=np.float64)
    for m in range(n_mol):
        i0 = 3 * m
        # Expected layout OH2, H1, H2
        if not (z[i0] == 8 and z[i0 + 1] == 1 and z[i0 + 2] == 1):
            raise ValueError(
                f"molecule {m}: expected O,H,H atomic numbers, got {z[i0:i0+3]}"
            )
        r_o = positions[:, i0]
        for k in (0, 1, 2):
            r = positions[:, i0 + k]
            dr = _mic(r - r_o, box)
            mu += charges[:, i0 + k, None] * dr
    return mu


def water_geometry(
    positions: np.ndarray,
    z: np.ndarray,
    box: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r_OH1, r_OH2, angle_HOH_deg) with shape (n_frames, n_mol)."""
    n_frames, n_atoms, _ = positions.shape
    n_mol = n_atoms // 3
    r1 = np.zeros((n_frames, n_mol), dtype=np.float64)
    r2 = np.zeros((n_frames, n_mol), dtype=np.float64)
    ang = np.zeros((n_frames, n_mol), dtype=np.float64)
    for m in range(n_mol):
        i0 = 3 * m
        ro = positions[:, i0]
        rh1 = positions[:, i0 + 1]
        rh2 = positions[:, i0 + 2]
        v1 = _mic(rh1 - ro, box)
        v2 = _mic(rh2 - ro, box)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        r1[:, m] = n1
        r2[:, m] = n2
        cos = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-12)
        ang[:, m] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return r1, r2, ang


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--box-A", type=float, default=None, help="Cubic box side (Å).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ir-max-cm", type=float, default=4500.0)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        pos = np.asarray(f["positions"], dtype=np.float64)
        q = np.asarray(f["charges"], dtype=np.float64)
        z = np.asarray(f.attrs["atomic_numbers"], dtype=np.int32)
        t_ps = np.asarray(f["time_ps"], dtype=np.float64)
        e_tot = np.asarray(f["total_energy"], dtype=np.float64)
        e_pot = np.asarray(f["potential_energy"], dtype=np.float64)
        e_kin = np.asarray(f["kinetic_energy"], dtype=np.float64)
        temp = np.asarray(f["temperature"], dtype=np.float64)
        dt_ps = float(f.attrs["dt_ps"])
        spr = int(f.attrs["steps_per_recording"])
        mm_mode = str(f.attrs.get("mm_charge_mode", "?"))
        box_attr = f.attrs.get("box_A", None)

    box = float(args.box_A if args.box_A is not None else (box_attr or 23.0))
    frame_dt_fs = dt_ps * 1000.0 * spr
    n_frames, n_atoms, _ = pos.shape
    n_mol = n_atoms // 3

    # --- energies ---
    e_tot_kcal = e_tot * EV_TO_KCAL_MOL
    e_pot_kcal = e_pot * EV_TO_KCAL_MOL
    e_kin_kcal = e_kin * EV_TO_KCAL_MOL
    drift = float(e_tot_kcal[-1] - e_tot_kcal[0])
    slope = float(np.polyfit(t_ps, e_tot_kcal, 1)[0])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t_ps, e_tot_kcal - e_tot_kcal[0], lw=0.8, color="#1f4e79")
    axes[0].set_ylabel(r"$\Delta E_\mathrm{tot}$ (kcal/mol)")
    axes[0].set_title(
        f"NVE energy (mm_charge_mode={mm_mode}, L={box:.1f} Å, "
        f"drift={drift:.3f} kcal/mol, slope={slope:.3f} kcal/mol/ps)"
    )
    axes[1].plot(t_ps, e_pot_kcal, lw=0.7, color="#c45c26", label="potential")
    axes[1].plot(t_ps, e_kin_kcal, lw=0.7, color="#2a6f3b", label="kinetic")
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("E (kcal/mol)")
    axes[2].plot(t_ps, temp, lw=0.7, color="#5a3d7a")
    axes[2].axhline(300.0, color="gray", ls="--", lw=0.8)
    axes[2].set_ylabel("T (K)")
    axes[2].set_xlabel("time (ps)")
    fig.tight_layout()
    fig.savefig(out / "energy_fluctuations.png", dpi=160)
    plt.close(fig)

    # --- charges ---
    q_mean = q.mean(axis=0)
    q_std = q.std(axis=0)
    q_var = q.var(axis=0)
    o_idx = np.where(z == 8)[0]
    h_idx = np.where(z == 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(q[:, o_idx].ravel(), bins=60, color="#c0392b", alpha=0.85, density=True)
    axes[0].set_xlabel("q_O (e)")
    axes[0].set_ylabel("density")
    axes[0].set_title(
        f"Oxygen charges  ⟨q⟩={q[:, o_idx].mean():.3f}  σ={q[:, o_idx].std():.4f}"
    )
    axes[1].hist(q[:, h_idx].ravel(), bins=60, color="#2980b9", alpha=0.85, density=True)
    axes[1].set_xlabel("q_H (e)")
    axes[1].set_title(
        f"Hydrogen charges  ⟨q⟩={q[:, h_idx].mean():.3f}  σ={q[:, h_idx].std():.4f}"
    )
    fig.suptitle("Per-element charge distributions (all molecules × frames)")
    fig.tight_layout()
    fig.savefig(out / "charge_distributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(np.arange(n_atoms), q_var, color=np.where(z == 8, "#c0392b", "#2980b9"), width=1.0)
    ax.set_xlabel("atom index")
    ax.set_ylabel(r"Var$(q)$ ($e^2$)")
    ax.set_title("Per-atom charge variance over the trajectory")
    fig.tight_layout()
    fig.savefig(out / "charge_variance_per_atom.png", dpi=160)
    plt.close(fig)

    # --- geometry ---
    r_oh1, r_oh2, ang = water_geometry(pos, z, box)
    r_oh = np.concatenate([r_oh1, r_oh2], axis=1)  # (n_frames, 2*n_mol)
    # charges on O and mean H for each molecule
    q_o = q[:, 0::3]
    q_h_mean = 0.5 * (q[:, 1::3] + q[:, 2::3])
    r_oh_mol = 0.5 * (r_oh1 + r_oh2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(r_oh.ravel(), bins=80, color="#16a085", density=True, alpha=0.9)
    axes[0].set_xlabel(r"$r_\mathrm{O–H}$ (Å)")
    axes[0].set_ylabel("density")
    axes[0].set_title(
        f"O–H bonds  ⟨r⟩={r_oh.mean():.4f} Å  σ={r_oh.std():.4f} Å"
    )
    axes[1].hist(ang.ravel(), bins=80, color="#8e44ad", density=True, alpha=0.9)
    axes[1].set_xlabel(r"∠H–O–H (deg)")
    axes[1].set_title(
        f"H–O–H angle  ⟨θ⟩={ang.mean():.2f}°  σ={ang.std():.2f}°"
    )
    fig.tight_layout()
    fig.savefig(out / "geometry_distributions.png", dpi=160)
    plt.close(fig)

    # subsample for scatter clarity
    rng = np.random.default_rng(0)
    flat_r = r_oh_mol.ravel()
    flat_qo = q_o.ravel()
    flat_qh = q_h_mean.ravel()
    flat_ang = ang.ravel()
    n_show = min(20000, flat_r.size)
    sel = rng.choice(flat_r.size, size=n_show, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(flat_r[sel], flat_qo[sel], s=2, alpha=0.25, c="#c0392b", rasterized=True)
    axes[0].set_xlabel(r"mean $r_\mathrm{O–H}$ (Å)")
    axes[0].set_ylabel(r"$q_\mathrm{O}$ (e)")
    axes[0].set_title("Oxygen charge vs mean O–H bond")
    axes[1].scatter(flat_r[sel], flat_qh[sel], s=2, alpha=0.25, c="#2980b9", rasterized=True)
    axes[1].set_xlabel(r"mean $r_\mathrm{O–H}$ (Å)")
    axes[1].set_ylabel(r"mean $q_\mathrm{H}$ (e)")
    axes[1].set_title("Hydrogen charge vs mean O–H bond")
    fig.tight_layout()
    fig.savefig(out / "charge_vs_bond.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(flat_ang[sel], flat_qo[sel], s=2, alpha=0.25, c="#c0392b", rasterized=True)
    axes[0].set_xlabel(r"∠H–O–H (deg)")
    axes[0].set_ylabel(r"$q_\mathrm{O}$ (e)")
    axes[0].set_title("Oxygen charge vs H–O–H angle")
    axes[1].scatter(flat_ang[sel], flat_qh[sel], s=2, alpha=0.25, c="#2980b9", rasterized=True)
    axes[1].set_xlabel(r"∠H–O–H (deg)")
    axes[1].set_ylabel(r"mean $q_\mathrm{H}$ (e)")
    axes[1].set_title("Hydrogen charge vs H–O–H angle")
    fig.tight_layout()
    fig.savefig(out / "charge_vs_angle.png", dpi=160)
    plt.close(fig)

    # correlations
    corr_qo_r = float(np.corrcoef(flat_r, flat_qo)[0, 1])
    corr_qh_r = float(np.corrcoef(flat_r, flat_qh)[0, 1])
    corr_qo_a = float(np.corrcoef(flat_ang, flat_qo)[0, 1])
    corr_qh_a = float(np.corrcoef(flat_ang, flat_qh)[0, 1])

    # --- IR ---
    mu = molecular_dipoles(pos, q, z, box)
    freq_cm, ir = dipole_fluctuation_ir_spectrum(mu, frame_dt_fs)
    mask = (freq_cm > 0) & (freq_cm <= args.ir_max_cm)
    # normalize peak to 1 for plotting
    ir_plot = ir[mask]
    ir_plot = ir_plot / max(float(ir_plot.max()), 1e-30)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freq_cm[mask], ir_plot, color="#1a1a1a", lw=1.0)
    ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
    ax.set_ylabel("IR intensity (arb., peak-normalized)")
    ax.set_title(
        f"IR from molecular dipole VACF  (frame Δt={frame_dt_fs:.2f} fs, "
        f"{n_frames} frames, {n_mol} TIP3)"
    )
    ax.set_xlim(0, args.ir_max_cm)
    fig.tight_layout()
    fig.savefig(out / "ir_spectrum.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(
        out / "ir_spectrum.npz",
        freq_cm=freq_cm,
        intensity=ir,
        frame_dt_fs=frame_dt_fs,
    )

    summary = {
        "h5": str(args.h5),
        "n_frames": int(n_frames),
        "n_molecules": int(n_mol),
        "box_A": box,
        "mm_charge_mode": mm_mode,
        "frame_dt_fs": frame_dt_fs,
        "duration_ps": float(t_ps[-1] - t_ps[0]),
        "energy": {
            "E_tot_drift_kcal_mol": drift,
            "E_tot_slope_kcal_mol_per_ps": slope,
            "E_tot_std_kcal_mol": float(np.std(e_tot_kcal)),
            "E_pot_std_kcal_mol": float(np.std(e_pot_kcal)),
            "E_kin_std_kcal_mol": float(np.std(e_kin_kcal)),
            "T_mean_K": float(np.mean(temp)),
            "T_std_K": float(np.std(temp)),
        },
        "charges": {
            "q_O_mean": float(q[:, o_idx].mean()),
            "q_O_std": float(q[:, o_idx].std()),
            "q_H_mean": float(q[:, h_idx].mean()),
            "q_H_std": float(q[:, h_idx].std()),
            "corr_qO_vs_rOH": corr_qo_r,
            "corr_qH_vs_rOH": corr_qh_r,
            "corr_qO_vs_angle": corr_qo_a,
            "corr_qH_vs_angle": corr_qh_a,
        },
        "geometry": {
            "r_OH_mean_A": float(r_oh.mean()),
            "r_OH_std_A": float(r_oh.std()),
            "angle_HOH_mean_deg": float(ang.mean()),
            "angle_HOH_std_deg": float(ang.std()),
        },
        "artifacts": [
            "energy_fluctuations.png",
            "charge_distributions.png",
            "charge_variance_per_atom.png",
            "geometry_distributions.png",
            "charge_vs_bond.png",
            "charge_vs_angle.png",
            "ir_spectrum.png",
            "ir_spectrum.npz",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

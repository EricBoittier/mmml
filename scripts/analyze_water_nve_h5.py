#!/usr/bin/env python3
"""Analyze a TIP3 NVE HDF5 from jaxmd (positions, charges, energies).

Produces:
  - IR from molecular dipole-velocity ACF with harmonic QM correction
    I(ω) ∝ ω (1 − e^{−βℏω}) C̃_μμ(ω), normalized on [0, 4500] cm⁻¹
  - Per-element charge variance
  - 2D fluctuation surfaces: Δq vs Δr_OH (per bond, not mean)
  - Twin-axis E_tot / E_kin (+ potential) fluctuation plots

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
from matplotlib.colors import LogNorm

from mmml.spectra.spectra_md import autocorrelation, correlation_to_spectrum

EV_TO_KCAL_MOL = 23.06054783061903
# hc/k_B in cm·K — converts ω[cm⁻¹] → βℏω at temperature T
HC_OVER_K_CM_K = 1.4387769


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
        v1 = _mic(positions[:, i0 + 1] - ro, box)
        v2 = _mic(positions[:, i0 + 2] - ro, box)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        r1[:, m] = n1
        r2[:, m] = n2
        cos = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-12)
        ang[:, m] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return r1, r2, ang


def _twin_equal_span(
    ax_left, ax_right, y_left: np.ndarray, y_right: np.ndarray
) -> None:
    """Match twin-axis *spans* so fluctuation amplitudes compare visually."""
    y_left = np.asarray(y_left, dtype=np.float64)
    y_right = np.asarray(y_right, dtype=np.float64)
    span_l = float(y_left.max() - y_left.min())
    span_r = float(y_right.max() - y_right.min())
    span = max(span_l, span_r, 1e-12) * 1.08
    mid_l = 0.5 * float(y_left.max() + y_left.min())
    mid_r = 0.5 * float(y_right.max() + y_right.min())
    ax_left.set_ylim(mid_l - 0.5 * span, mid_l + 0.5 * span)
    ax_right.set_ylim(mid_r - 0.5 * span, mid_r + 0.5 * span)


def ir_spectrum_qm_corrected(
    dipoles: np.ndarray,
    frame_dt_fs: float,
    temperature_K: float,
    *,
    zero_pad: int = 8,
    smooth_cm: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dipole-velocity ACF -> IR with harmonic / experimental QM correction.

    Uses J(t)=dmu/dt so librations do not swamp vibrational bands.  With
    C_JJ(w)=w^2 C_mm(w), absorption is

        I(w) ∝ [(1 − exp(−βℏω)) / ω] C_JJ(ω)
             = ω (1 − exp(−βℏω)) C_mm(ω)

    (physnetjax intensity_correction form; βℏω via hc/k_B T).
    Normalized so integral_0^4500 I(ω) dω = 1.

    Returns (freq_cm, I_raw_norm, I_smooth_norm).
    """
    mu = np.asarray(dipoles, dtype=np.float64)
    mu = mu - mu.mean(axis=0, keepdims=True)
    mu_dot = np.gradient(mu, frame_dt_fs, axis=0)
    acf = autocorrelation(mu_dot)
    freq_cm, c_jj = correlation_to_spectrum(
        acf, frame_dt_fs, window="blackman", zero_pad=zero_pad, qcf=None
    )
    c_jj = np.abs(c_jj)

    t_k = max(float(temperature_K), 1.0)
    beta_hbar_w = (HC_OVER_K_CM_K / t_k) * freq_cm
    exp_corr = 1.0 - np.exp(-np.clip(beta_hbar_w, 0.0, 100.0))
    intensity = np.zeros_like(c_jj)
    pos = freq_cm > 0.0
    intensity[pos] = (exp_corr[pos] / freq_cm[pos]) * c_jj[pos]

    band = (freq_cm > 0.0) & (freq_cm <= 4500.0)
    norm = float(np.trapezoid(intensity[band], freq_cm[band]))
    if norm > 0.0:
        intensity = intensity / norm

    if smooth_cm > 0.0 and freq_cm.size > 3:
        df = float(np.median(np.diff(freq_cm[freq_cm > 0])))
        sigma = max(smooth_cm / (2.0 * np.sqrt(2.0 * np.log(2.0))), df)
        half = int(max(3, np.ceil(4.0 * sigma / df)))
        x = np.arange(-half, half + 1) * df
        ker = np.exp(-0.5 * (x / sigma) ** 2)
        ker /= ker.sum()
        smooth = np.convolve(intensity, ker, mode="same")
        norm_s = float(np.trapezoid(smooth[band], freq_cm[band]))
        if norm_s > 0.0:
            smooth = smooth / norm_s
    else:
        smooth = intensity

    return freq_cm, intensity, smooth


def _hist2d_sym(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    bins: int = 80,
) -> None:
    """2D density on a symmetric (Δx, Δy) coordinate system."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    # Symmetric limits from the larger |percentile| so the origin is centered.
    lim_x = float(np.percentile(np.abs(dx), 99.5))
    lim_y = float(np.percentile(np.abs(dy), 99.5))
    lim_x = max(lim_x, 1e-6)
    lim_y = max(lim_y, 1e-6)
    h, xedges, yedges = np.histogram2d(
        dx,
        dy,
        bins=bins,
        range=[[-lim_x, lim_x], [-lim_y, lim_y]],
        density=True,
    )
    # Avoid log(0)
    h = np.maximum(h, h[h > 0].min() * 0.1 if np.any(h > 0) else 1e-12)
    pcm = ax.pcolormesh(
        xedges,
        yedges,
        h.T,
        shading="auto",
        norm=LogNorm(vmin=h.min(), vmax=h.max()),
        cmap="magma",
    )
    ax.axhline(0.0, color="white", lw=0.6, alpha=0.5)
    ax.axvline(0.0, color="white", lw=0.6, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("auto")
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="density")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--box-A", type=float, default=None, help="Cubic box side (Å).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ir-max-cm", type=float, default=4500.0)
    p.add_argument(
        "--ir-temperature-K",
        type=float,
        default=None,
        help="Temperature for βℏω QM correction (default: trajectory ⟨T⟩).",
    )
    p.add_argument(
        "--ir-smooth-cm",
        type=float,
        default=15.0,
        help="Gaussian HWHM (cm⁻¹) for display smoothing (0 disables).",
    )
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

    e_tot_kcal = e_tot * EV_TO_KCAL_MOL
    e_pot_kcal = e_pot * EV_TO_KCAL_MOL
    e_kin_kcal = e_kin * EV_TO_KCAL_MOL
    drift = float(e_tot_kcal[-1] - e_tot_kcal[0])
    slope = float(np.polyfit(t_ps, e_tot_kcal, 1)[0])
    t_mean = float(np.mean(temp))

    # --- energies: twin axes for E_tot and E_kin (each auto-scaled) ---
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)

    ax_l = axes[0]
    ax_r = ax_l.twinx()
    (ln1,) = ax_l.plot(
        t_ps, e_tot_kcal, color="#1f4e79", lw=0.85, label=r"$E_\mathrm{tot}$"
    )
    (ln2,) = ax_r.plot(
        t_ps, e_kin_kcal, color="#2a6f3b", lw=0.85, label=r"$E_\mathrm{kin}$"
    )
    ax_l.set_ylabel(r"$E_\mathrm{tot}$ (kcal/mol)", color="#1f4e79")
    ax_r.set_ylabel(r"$E_\mathrm{kin}$ (kcal/mol)", color="#2a6f3b")
    ax_l.tick_params(axis="y", colors="#1f4e79")
    ax_r.tick_params(axis="y", colors="#2a6f3b")
    ax_l.set_title(
        f"NVE energy (mm_charge_mode={mm_mode}, L={box:.1f} Å)  "
        f"drift={drift:.3f} kcal/mol, slope={slope:.3f} kcal/mol/ps"
    )
    ax_l.legend(handles=[ln1, ln2], loc="upper right", frameon=False)
    _twin_equal_span(ax_l, ax_r, e_tot_kcal, e_kin_kcal)

    ax_l2 = axes[1]
    ax_r2 = ax_l2.twinx()
    (ln3,) = ax_l2.plot(
        t_ps, e_pot_kcal, color="#c45c26", lw=0.85, label=r"$E_\mathrm{pot}$"
    )
    (ln4,) = ax_r2.plot(t_ps, temp, color="#5a3d7a", lw=0.85, label="T")
    ax_l2.set_ylabel(r"$E_\mathrm{pot}$ (kcal/mol)", color="#c45c26")
    ax_r2.set_ylabel("T (K)", color="#5a3d7a")
    ax_l2.tick_params(axis="y", colors="#c45c26")
    ax_r2.tick_params(axis="y", colors="#5a3d7a")
    ax_r2.axhline(300.0, color="gray", ls="--", lw=0.7)
    ax_l2.set_xlabel("time (ps)")
    ax_l2.legend(handles=[ln3, ln4], loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out / "energy_fluctuations.png", dpi=170)
    plt.close(fig)

    # --- charges (1D) ---
    o_idx = np.where(z == 8)[0]
    h_idx = np.where(z == 1)[0]
    q_var = q.var(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(q[:, o_idx].ravel(), bins=60, color="#c0392b", alpha=0.85, density=True)
    axes[0].set_xlabel(r"$q_\mathrm{O}$ (e)")
    axes[0].set_ylabel("density")
    axes[0].set_title(
        f"Oxygen  ⟨q⟩={q[:, o_idx].mean():.3f}  σ={q[:, o_idx].std():.4f}"
    )
    axes[1].hist(q[:, h_idx].ravel(), bins=60, color="#2980b9", alpha=0.85, density=True)
    axes[1].set_xlabel(r"$q_\mathrm{H}$ (e)")
    axes[1].set_title(
        f"Hydrogen  ⟨q⟩={q[:, h_idx].mean():.3f}  σ={q[:, h_idx].std():.4f}"
    )
    fig.suptitle("Per-element charge distributions")
    fig.tight_layout()
    fig.savefig(out / "charge_distributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(
        np.arange(n_atoms),
        q_var,
        color=np.where(z == 8, "#c0392b", "#2980b9"),
        width=1.0,
    )
    ax.set_xlabel("atom index")
    ax.set_ylabel(r"Var$(q)$ ($e^2$)")
    ax.set_title("Per-atom charge variance over the trajectory")
    fig.tight_layout()
    fig.savefig(out / "charge_variance_per_atom.png", dpi=160)
    plt.close(fig)

    # --- geometry + per-bond (not mean) charge surfaces ---
    r_oh1, r_oh2, ang = water_geometry(pos, box)
    r_oh_all = np.concatenate([r_oh1, r_oh2], axis=1)  # (F, 2M)
    q_o = q[:, 0::3]  # (F, M)
    q_h1 = q[:, 1::3]
    q_h2 = q[:, 2::3]
    # Pair each O–H bond with the O charge and that H's charge
    q_o_per_bond = np.concatenate([q_o, q_o], axis=1)
    q_h_per_bond = np.concatenate([q_h1, q_h2], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(r_oh_all.ravel(), bins=80, color="#16a085", density=True, alpha=0.9)
    axes[0].set_xlabel(r"$r_\mathrm{O–H}$ (Å)")
    axes[0].set_ylabel("density")
    axes[0].set_title(
        f"O–H bonds  ⟨r⟩={r_oh_all.mean():.4f} Å  σ={r_oh_all.std():.4f} Å"
    )
    axes[1].hist(ang.ravel(), bins=80, color="#8e44ad", density=True, alpha=0.9)
    axes[1].set_xlabel(r"∠H–O–H (deg)")
    axes[1].set_title(
        f"H–O–H angle  ⟨θ⟩={ang.mean():.2f}°  σ={ang.std():.2f}°"
    )
    fig.tight_layout()
    fig.savefig(out / "geometry_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _hist2d_sym(
        axes[0],
        r_oh_all,
        q_o_per_bond,
        xlabel=r"$\Delta r_\mathrm{O–H}$ (Å)",
        ylabel=r"$\Delta q_\mathrm{O}$ (e)",
        title=r"O charge vs O–H bond (per bond; centered)",
    )
    _hist2d_sym(
        axes[1],
        r_oh_all,
        q_h_per_bond,
        xlabel=r"$\Delta r_\mathrm{O–H}$ (Å)",
        ylabel=r"$\Delta q_\mathrm{H}$ (e)",
        title=r"H charge vs its O–H bond (per bond; centered)",
    )
    fig.tight_layout()
    fig.savefig(out / "charge_vs_bond_surface.png", dpi=170)
    plt.close(fig)

    # angle surfaces (still useful)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _hist2d_sym(
        axes[0],
        ang,
        q_o,
        xlabel=r"$\Delta\angle\mathrm{H–O–H}$ (deg)",
        ylabel=r"$\Delta q_\mathrm{O}$ (e)",
        title="O charge vs H–O–H angle (centered)",
    )
    q_h_mean_mol = 0.5 * (q_h1 + q_h2)
    _hist2d_sym(
        axes[1],
        ang,
        q_h_mean_mol,
        xlabel=r"$\Delta\angle\mathrm{H–O–H}$ (deg)",
        ylabel=r"$\Delta\langle q_\mathrm{H}\rangle$ (e)",
        title="⟨q_H⟩ vs H–O–H angle (centered)",
    )
    fig.tight_layout()
    fig.savefig(out / "charge_vs_angle_surface.png", dpi=170)
    plt.close(fig)

    corr_qo_r = float(np.corrcoef(r_oh_all.ravel(), q_o_per_bond.ravel())[0, 1])
    corr_qh_r = float(np.corrcoef(r_oh_all.ravel(), q_h_per_bond.ravel())[0, 1])
    corr_qo_a = float(np.corrcoef(ang.ravel(), q_o.ravel())[0, 1])
    corr_qh_a = float(np.corrcoef(ang.ravel(), q_h_mean_mol.ravel())[0, 1])

    # --- IR ---
    mu = molecular_dipoles(pos, q, z, box)
    t_ir = float(args.ir_temperature_K) if args.ir_temperature_K else t_mean
    # Prefer target 300 K for the QM factor when the run is cold — physical
    # correction should match the ensemble you intend to compare to experiment.
    if args.ir_temperature_K is None and t_mean < 250.0:
        t_ir = 300.0
    freq_cm, ir_raw, ir_smooth = ir_spectrum_qm_corrected(
        mu,
        frame_dt_fs,
        t_ir,
        smooth_cm=float(args.ir_smooth_cm),
    )
    mask = (freq_cm > 0) & (freq_cm <= args.ir_max_cm)
    # Vibrational zoom: hide the far-IR pedestal so bend/stretch set the scale.
    mask_vib = (freq_cm >= 400.0) & (freq_cm <= args.ir_max_cm)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=False)
    for ax, m, title_extra in (
        (axes[0], mask, "full band (0–4500 cm⁻¹)"),
        (axes[1], mask_vib, "vibrational window (≥400 cm⁻¹)"),
    ):
        ax.plot(
            freq_cm[m],
            ir_raw[m],
            color="#b0b0b0",
            lw=0.7,
            label="raw (QM-corrected)",
        )
        ax.plot(
            freq_cm[m],
            ir_smooth[m],
            color="#111111",
            lw=1.35,
            label=f"smoothed (HWHM={args.ir_smooth_cm:g} cm⁻¹)",
        )
        ax.set_ylabel(r"normalized absorbance")
        ax.set_title(title_extra)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_xlim(0, args.ir_max_cm)
    axes[1].set_xlim(400, args.ir_max_cm)
    axes[1].set_xlabel(r"wavenumber (cm$^{-1}$)")
    fig.suptitle(
        (
            f"IR from molecular dipole-velocity ACF · "
            f"QM corr. omega*(1-exp(-beta hbar omega)) at T={t_ir:.0f} K\n"
            f"frame dt={frame_dt_fs:.3f} fs · {n_frames} frames · {n_mol} TIP3 · "
            "integral 0-4500 = 1"
        ),
        y=1.02,
    )
    fig.savefig(out / "ir_spectrum.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        out / "ir_spectrum.npz",
        freq_cm=freq_cm,
        intensity=ir_raw,
        intensity_smooth=ir_smooth,
        frame_dt_fs=frame_dt_fs,
        temperature_K_qcf=t_ir,
        qm_correction="(1-exp(-beta*hbar*omega))/omega * C_JJ  [= omega*(1-exp)*C_mm]",
        method="dipole_velocity_acf",
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
            "T_mean_K": t_mean,
            "T_std_K": float(np.std(temp)),
        },
        "charges": {
            "q_O_mean": float(q[:, o_idx].mean()),
            "q_O_std": float(q[:, o_idx].std()),
            "q_H_mean": float(q[:, h_idx].mean()),
            "q_H_std": float(q[:, h_idx].std()),
            "corr_qO_vs_rOH_per_bond": corr_qo_r,
            "corr_qH_vs_rOH_per_bond": corr_qh_r,
            "corr_qO_vs_angle": corr_qo_a,
            "corr_qH_vs_angle": corr_qh_a,
        },
        "geometry": {
            "r_OH_mean_A": float(r_oh_all.mean()),
            "r_OH_std_A": float(r_oh_all.std()),
            "angle_HOH_mean_deg": float(ang.mean()),
            "angle_HOH_std_deg": float(ang.std()),
        },
        "ir": {
            "temperature_K_qcf": t_ir,
            "method": "dipole_velocity_acf",
            "qm_correction": "omega*(1-exp(-beta*hbar*omega))",
            "normalized_0_4500": True,
            "smooth_hwhm_cm": float(args.ir_smooth_cm),
        },
        "artifacts": [
            "energy_fluctuations.png",
            "charge_distributions.png",
            "charge_variance_per_atom.png",
            "geometry_distributions.png",
            "charge_vs_bond_surface.png",
            "charge_vs_angle_surface.png",
            "ir_spectrum.png",
            "ir_spectrum.npz",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

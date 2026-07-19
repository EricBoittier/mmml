#!/usr/bin/env python3
"""Analyze a TIP3 NVE HDF5 from jaxmd (positions, charges, energies).

Produces:
  - IR from atomic charge-current ACF with harmonic QM correction
    I(w) ~ w (1 - exp(-beta hbar w)) C_mm(w), normalized on [0, 4500] cm^-1
  - Per-element charge variance
  - 2D fluctuation surfaces: Delta q vs Delta r_OH (per bond, not mean)
  - Twin-axis E_tot / E_kin with matched axis spans

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
from mmml.spectra.spectra_md import autocorrelation, correlation_to_spectrum

EV_TO_KCAL_MOL = 23.06054783061903
# hc/k_B in cm*K — converts w[cm^-1] -> beta*hbar*w at temperature T
HC_OVER_K_CM_K = 1.4387769


def _mic(d: np.ndarray, box: float) -> np.ndarray:
    return d - box * np.round(d / box)


def molecular_dipoles(
    positions: np.ndarray,
    charges: np.ndarray,
    z: np.ndarray,
    box: float,
) -> np.ndarray:
    """Sum of per-molecule dipoles (e*A) with intramolecular MIC about O."""
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
    """Match twin-axis spans so fluctuation amplitudes compare visually."""
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
    current: np.ndarray,
    frame_dt_fs: float,
    temperature_K: float,
    *,
    zero_pad: int = 8,
    smooth_cm: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge-current ACF -> IR with harmonic / experimental QM correction.

    ``current`` is J(t) = sum_i q_i v_i or dmu/dt, shape (T, 3).  With
    C_JJ(w) = w^2 C_mm(w), absorption is

        I(w) ~ [(1 - exp(-beta hbar w)) / w] C_JJ(w)
             = w (1 - exp(-beta hbar w)) C_mm(w)

    (physnetjax intensity_correction form; beta hbar w via hc/k_B T).
    Normalized so integral_0^4500 I(w) dw = 1.

    Returns (freq_cm, I_raw_norm, I_smooth_norm).
    """
    J = np.asarray(current, dtype=np.float64)
    J = J - J.mean(axis=0, keepdims=True)
    acf = autocorrelation(J)
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


def _charge_geometry_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    *,
    cmap: str,
    xlabel: str,
    ylabel: str,
    clabel: str,
    title: str,
    s: float = 4.0,
    alpha: float = 0.35,
) -> None:
    """Scatter of geometry with charge (or variance) as color."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    sc = ax.scatter(
        x,
        y,
        c=c,
        s=s,
        alpha=alpha,
        cmap=cmap,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=clabel)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--box-A", type=float, default=None, help="Cubic box side (A).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ir-max-cm", type=float, default=4500.0)
    p.add_argument(
        "--ir-temperature-K",
        type=float,
        default=None,
        help="Temperature for beta-hbar-w QM correction (default: trajectory mean T).",
    )
    p.add_argument(
        "--ir-smooth-cm",
        type=float,
        default=15.0,
        help="Gaussian HWHM (cm^-1) for display smoothing (0 disables).",
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
        velocities = (
            np.asarray(f["velocities"], dtype=np.float64) if "velocities" in f else None
        )
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

    # --- energies: twin axes with matched spans ---
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
        f"NVE energy (mm_charge_mode={mm_mode}, L={box:.1f} A)  "
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
        f"Oxygen  <q>={q[:, o_idx].mean():.3f}  sigma={q[:, o_idx].std():.4f}"
    )
    axes[1].hist(q[:, h_idx].ravel(), bins=60, color="#2980b9", alpha=0.85, density=True)
    axes[1].set_xlabel(r"$q_\mathrm{H}$ (e)")
    axes[1].set_title(
        f"Hydrogen  <q>={q[:, h_idx].mean():.3f}  sigma={q[:, h_idx].std():.4f}"
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

    # --- geometry histograms + charge scatter in (r_sym, angle) ---
    r_oh1, r_oh2, ang = water_geometry(pos, box)
    r_oh_all = np.concatenate([r_oh1, r_oh2], axis=1)
    # Symmetric stretch: (r_HaO + r_HbO) / 2  (not a single-bond axis)
    r_sym = 0.5 * (r_oh1 + r_oh2)
    q_o = q[:, 0::3]
    q_h1 = q[:, 1::3]
    q_h2 = q[:, 2::3]
    q_h_mean_mol = 0.5 * (q_h1 + q_h2)
    # Per-molecule charge variance across the trajectory (broadcast to frames)
    var_q_o = np.broadcast_to(q_o.var(axis=0, keepdims=True), q_o.shape)
    var_q_h = np.broadcast_to(
        q_h_mean_mol.var(axis=0, keepdims=True), q_h_mean_mol.shape
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(r_oh_all.ravel(), bins=80, color="#16a085", density=True, alpha=0.9)
    axes[0].set_xlabel(r"$r_\mathrm{O-H}$ (A)")
    axes[0].set_ylabel("density")
    axes[0].set_title(
        f"O-H bonds  <r>={r_oh_all.mean():.4f} A  sigma={r_oh_all.std():.4f} A"
    )
    axes[1].hist(ang.ravel(), bins=80, color="#8e44ad", density=True, alpha=0.9)
    axes[1].set_xlabel(r"angle H-O-H (deg)")
    axes[1].set_title(
        f"H-O-H angle  <theta>={ang.mean():.2f} deg  sigma={ang.std():.2f} deg"
    )
    fig.tight_layout()
    fig.savefig(out / "geometry_distributions.png", dpi=160)
    plt.close(fig)

    # Charge colored by q (Reds / Blues)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    _charge_geometry_scatter(
        axes[0],
        r_sym,
        ang,
        q_o,
        cmap="Reds",
        xlabel=r"$(r_\mathrm{HaO}+r_\mathrm{HbO})/2$ (A)",
        ylabel=r"$\angle\mathrm{HOH}$ (deg)",
        clabel=r"$q_\mathrm{O}$ (e)",
        title=r"Oxygen charge",
    )
    _charge_geometry_scatter(
        axes[1],
        r_sym,
        ang,
        q_h_mean_mol,
        cmap="Blues",
        xlabel=r"$(r_\mathrm{HaO}+r_\mathrm{HbO})/2$ (A)",
        ylabel=r"$\angle\mathrm{HOH}$ (deg)",
        clabel=r"$\langle q_\mathrm{H}\rangle$ (e)",
        title=r"Hydrogen charge",
    )
    fig.suptitle("Charge vs symmetric stretch and HOH angle", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "charge_vs_geometry_scatter.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Variance colored (same axes; per-molecule Var_t(q))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    _charge_geometry_scatter(
        axes[0],
        r_sym,
        ang,
        var_q_o,
        cmap="Reds",
        xlabel=r"$(r_\mathrm{HaO}+r_\mathrm{HbO})/2$ (A)",
        ylabel=r"$\angle\mathrm{HOH}$ (deg)",
        clabel=r"$\mathrm{Var}_t(q_\mathrm{O})$ ($e^2$)",
        title=r"Oxygen charge variance",
    )
    _charge_geometry_scatter(
        axes[1],
        r_sym,
        ang,
        var_q_h,
        cmap="Blues",
        xlabel=r"$(r_\mathrm{HaO}+r_\mathrm{HbO})/2$ (A)",
        ylabel=r"$\angle\mathrm{HOH}$ (deg)",
        clabel=r"$\mathrm{Var}_t(\langle q_\mathrm{H}\rangle)$ ($e^2$)",
        title=r"Hydrogen charge variance",
    )
    fig.suptitle(
        "Per-molecule charge variance vs symmetric stretch and HOH angle", y=1.02
    )
    fig.tight_layout()
    fig.savefig(
        out / "charge_variance_vs_geometry_scatter.png", dpi=170, bbox_inches="tight"
    )
    plt.close(fig)

    corr_qo_r = float(np.corrcoef(r_sym.ravel(), q_o.ravel())[0, 1])
    corr_qh_r = float(np.corrcoef(r_sym.ravel(), q_h_mean_mol.ravel())[0, 1])
    corr_qo_a = float(np.corrcoef(ang.ravel(), q_o.ravel())[0, 1])
    corr_qh_a = float(np.corrcoef(ang.ravel(), q_h_mean_mol.ravel())[0, 1])

    # --- IR ---
    mu = molecular_dipoles(pos, q, z, box)
    if velocities is not None:
        J = np.sum(q[..., None] * velocities, axis=1)
        ir_method = "atomic_charge_current_acf"
    else:
        J = np.gradient(mu, frame_dt_fs, axis=0)
        ir_method = "dipole_velocity_acf"

    t_ir = float(args.ir_temperature_K) if args.ir_temperature_K else t_mean
    if args.ir_temperature_K is None and t_mean < 250.0:
        t_ir = 300.0

    freq_cm, ir_raw, ir_smooth = ir_spectrum_qm_corrected(
        J,
        frame_dt_fs,
        t_ir,
        smooth_cm=float(args.ir_smooth_cm),
    )
    mask = (freq_cm > 0) & (freq_cm <= args.ir_max_cm)
    mask_vib = (freq_cm >= 400.0) & (freq_cm <= args.ir_max_cm)

    # Renormalize vibrational window for a readable y-scale.
    ir_vib = ir_smooth.copy()
    vib_norm = float(np.trapezoid(ir_vib[mask_vib], freq_cm[mask_vib]))
    if vib_norm > 0.0:
        ir_vib = ir_vib / vib_norm
    ir_vib_raw = ir_raw.copy()
    vib_norm_r = float(np.trapezoid(ir_vib_raw[mask_vib], freq_cm[mask_vib]))
    if vib_norm_r > 0.0:
        ir_vib_raw = ir_vib_raw / vib_norm_r

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=False)
    axes[0].plot(freq_cm[mask], ir_raw[mask], color="#b0b0b0", lw=0.7, label="raw")
    axes[0].plot(
        freq_cm[mask],
        ir_smooth[mask],
        color="#111111",
        lw=1.35,
        label=f"smoothed (HWHM={args.ir_smooth_cm:g} cm$^{{-1}}$)",
    )
    axes[0].set_ylabel(r"I (norm. 0-4500)")
    axes[0].set_title("full band (integral 0-4500 = 1)")
    axes[0].set_xlim(0, args.ir_max_cm)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(
        freq_cm[mask_vib], ir_vib_raw[mask_vib], color="#b0b0b0", lw=0.7, label="raw"
    )
    axes[1].plot(
        freq_cm[mask_vib],
        ir_vib[mask_vib],
        color="#111111",
        lw=1.35,
        label=f"smoothed (HWHM={args.ir_smooth_cm:g} cm$^{{-1}}$)",
    )
    axes[1].set_ylabel(r"I (norm. 400-4500)")
    axes[1].set_title(r"vibrational window (renormalized on 400-4500 cm$^{-1}$)")
    axes[1].set_xlim(400, args.ir_max_cm)
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel(r"wavenumber (cm$^{-1}$)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        (
            f"IR from {ir_method} · QM corr. omega*(1-exp(-beta hbar omega)) "
            f"at T={t_ir:.0f} K\n"
            f"frame dt={frame_dt_fs:.3f} fs · {n_frames} frames · {n_mol} TIP3"
        ),
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "ir_spectrum.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        out / "ir_spectrum.npz",
        freq_cm=freq_cm,
        intensity=ir_raw,
        intensity_smooth=ir_smooth,
        intensity_vib_renorm=ir_vib,
        frame_dt_fs=frame_dt_fs,
        temperature_K_qcf=t_ir,
        qm_correction="(1-exp(-beta*hbar*omega))/omega * C_JJ",
        method=ir_method,
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
            "corr_qO_vs_r_sym": corr_qo_r,
            "corr_qH_vs_r_sym": corr_qh_r,
            "corr_qO_vs_angle": corr_qo_a,
            "corr_qH_vs_angle": corr_qh_a,
        },
        "geometry": {
            "r_OH_mean_A": float(r_oh_all.mean()),
            "r_OH_std_A": float(r_oh_all.std()),
            "r_sym_mean_A": float(r_sym.mean()),
            "r_sym_std_A": float(r_sym.std()),
            "angle_HOH_mean_deg": float(ang.mean()),
            "angle_HOH_std_deg": float(ang.std()),
        },
        "ir": {
            "temperature_K_qcf": t_ir,
            "method": ir_method,
            "qm_correction": "omega*(1-exp(-beta*hbar*omega))",
            "normalized_0_4500": True,
            "smooth_hwhm_cm": float(args.ir_smooth_cm),
            "note": (
                "OH stretch needs dense sampling; use --steps-per-recording 1 "
                f"(current frame_dt_fs={frame_dt_fs:.3f})."
            ),
        },
        "artifacts": [
            "energy_fluctuations.png",
            "charge_distributions.png",
            "charge_variance_per_atom.png",
            "geometry_distributions.png",
            "charge_vs_geometry_scatter.png",
            "charge_variance_vs_geometry_scatter.png",
            "ir_spectrum.png",
            "ir_spectrum.npz",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

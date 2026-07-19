#!/usr/bin/env python3
"""Analyze a TIP3 NVE HDF5 from jaxmd (positions, charges, energies).

Produces:
  - NVE validation dashboard (time series + rotated marginals + Fourier)
  - IR from atomic charge-current ACF with harmonic QM correction
    I(w) ~ w (1 - exp(-beta hbar w)) C_mm(w); cut <500 cm^-1 before smooth (arb. u.)
  - Charge vs (r_sym, angle) scatters (Reds/Blues)
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
from matplotlib.gridspec import GridSpec
from mmml.spectra.spectra_md import (
    FS_INV_TO_CM_INV,
    autocorrelation,
    correlation_to_spectrum,
)

EV_TO_KCAL_MOL = 23.06054783061903
# hc/k_B in cm*K — converts w[cm^-1] -> beta*hbar*w at temperature T
HC_OVER_K_CM_K = 1.4387769


def _periodogram_density(
    signal: np.ndarray,
    dt_fs: float,
    *,
    zero_pad: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density of a 1-D signal (Hann window).

    Returns ``(freq_cm, psd)`` with freq in cm^-1.  PSD is |FT|^2 normalized
    so Parseval holds approximately on the positive-frequency axis.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - x.mean()
    n = x.size
    w = np.hanning(n)
    n_fft = max(n * zero_pad, 8)
    ft = np.fft.rfft(x * w, n=n_fft)
    freq_cm = np.fft.rfftfreq(n_fft, d=dt_fs) * FS_INV_TO_CM_INV
    # Window power correction
    psd = (np.abs(ft) ** 2) / (np.sum(w**2))
    return freq_cm, psd


def _average_periodogram(
    signals: np.ndarray,
    dt_fs: float,
    *,
    zero_pad: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean PSD over columns of ``signals`` with shape (n_frames, n_series)."""
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, None]
    freq = None
    acc = None
    for j in range(signals.shape[1]):
        f, p = _periodogram_density(signals[:, j], dt_fs, zero_pad=zero_pad)
        if freq is None:
            freq = f
            acc = np.zeros_like(p)
        acc += p
    assert freq is not None and acc is not None
    return freq, acc / signals.shape[1]


def _smooth_spectrum(
    freq_cm: np.ndarray,
    intensity: np.ndarray,
    smooth_cm: float,
) -> np.ndarray:
    if smooth_cm <= 0.0 or freq_cm.size < 4:
        return intensity.copy()
    df = float(np.median(np.diff(freq_cm)))
    sigma = max(smooth_cm / (2.0 * np.sqrt(2.0 * np.log(2.0))), df)
    half = int(max(3, np.ceil(4.0 * sigma / df)))
    x = np.arange(-half, half + 1) * df
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    return np.convolve(intensity, ker, mode="same")


def write_geometry_power_spectra(
    path: Path,
    *,
    r_oh1: np.ndarray,
    r_oh2: np.ndarray,
    ang: np.ndarray,
    frame_dt_fs: float,
    min_cm: float = 500.0,
    max_cm: float = 4500.0,
    smooth_cm: float = 15.0,
) -> dict:
    """Power spectra of intramolecular coordinates (diagnostic for IR)."""
    r_oh = np.concatenate([r_oh1, r_oh2], axis=1)
    r_sym = 0.5 * (r_oh1 + r_oh2)
    r_asym = 0.5 * (r_oh1 - r_oh2)

    series = {
        r"$r_\mathrm{O-H}$ (all bonds)": r_oh,
        r"$r_\mathrm{sym}=(r_a+r_b)/2$": r_sym,
        r"$r_\mathrm{asym}=(r_a-r_b)/2$": r_asym,
        r"$\angle\mathrm{HOH}$": ang,
    }
    colors = {
        r"$r_\mathrm{O-H}$ (all bonds)": "#16a085",
        r"$r_\mathrm{sym}=(r_a+r_b)/2$": "#1f4e79",
        r"$r_\mathrm{asym}=(r_a-r_b)/2$": "#c45c26",
        r"$\angle\mathrm{HOH}$": "#8e44ad",
    }

    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, sig in series.items():
        spectra[name] = _average_periodogram(sig, frame_dt_fs, zero_pad=4)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=False)

    # Top: full band, log scale (shows far-IR dominance clearly)
    ax = axes[0]
    for name, (f, p) in spectra.items():
        m = (f > 0) & (f <= max_cm)
        ax.plot(f[m], p[m], color=colors[name], lw=1.0, label=name)
    ax.set_yscale("log")
    ax.set_xlim(0, max_cm)
    ax.set_ylabel("PSD (arb. u.)")
    ax.set_title(
        f"Intramolecular power spectra (log) · frame Δt={frame_dt_fs:.3f} fs"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)

    # Bottom: cut < min_cm before smooth; peak-norm each curve (shape compare)
    ax = axes[1]
    peaks: dict[str, float] = {}
    for name, (f, p) in spectra.items():
        m = (f >= min_cm) & (f <= max_cm)
        fc, pc = f[m], p[m]
        ps = _smooth_spectrum(fc, pc, smooth_cm)
        scale = float(np.max(ps)) if ps.size and np.max(ps) > 0 else 1.0
        ax.plot(fc, pc / scale, color=colors[name], lw=0.5, alpha=0.35)
        ax.plot(fc, ps / scale, color=colors[name], lw=1.35, label=name)
        if ps.size:
            peaks[name] = float(fc[int(np.argmax(ps))])
    ax.set_xlim(min_cm, max_cm)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
    ax.set_ylabel("PSD (peak-norm., arb.)")
    ax.set_title(
        f"cut <{min_cm:g} cm$^{{-1}}$ before smooth (HWHM={smooth_cm:g}) · "
        "each series peak-normalized"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.axvline(1600, color="0.7", ls=":", lw=0.8)
    ax.axvline(3400, color="0.7", ls=":", lw=0.8)
    fig.tight_layout()
    ymin, ymax = axes[1].get_ylim()
    if ymax > ymin:
        axes[1].text(
            1600,
            ymin + 0.92 * (ymax - ymin),
            "bend",
            ha="center",
            fontsize=7,
            color="0.45",
        )
        axes[1].text(
            3400,
            ymin + 0.92 * (ymax - ymin),
            "stretch",
            ha="center",
            fontsize=7,
            color="0.45",
        )
    fig.savefig(path, dpi=170)
    plt.close(fig)

    # Band power fractions for r_OH (diagnostic)
    f, p = spectra[r"$r_\mathrm{O-H}$ (all bonds)"]
    def _band(lo: float, hi: float) -> float:
        m = (f >= lo) & (f <= hi)
        return float(np.trapezoid(p[m], f[m])) if np.any(m) else 0.0

    tot = _band(0, max_cm) or 1.0
    return {
        "r_OH_psd_peak_ge500_cm": peaks.get(r"$r_\mathrm{O-H}$ (all bonds)"),
        "r_sym_psd_peak_ge500_cm": peaks.get(r"$r_\mathrm{sym}=(r_a+r_b)/2$"),
        "r_asym_psd_peak_ge500_cm": peaks.get(r"$r_\mathrm{asym}=(r_a-r_b)/2$"),
        "angle_psd_peak_ge500_cm": peaks.get(r"$\angle\mathrm{HOH}$"),
        "r_OH_power_frac_0_500": _band(0, 500) / tot,
        "r_OH_power_frac_1400_1800": _band(1400, 1800) / tot,
        "r_OH_power_frac_3000_3800": _band(3000, 3800) / tot,
        "artifact": str(path),
    }


def _running_linear_slope(
    t: np.ndarray, y: np.ndarray, *, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window linear slope of y(t). Returns mid-times and slopes."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if window < 8 or window >= len(t):
        window = max(8, len(t) // 10)
    half = window // 2
    mids, slopes = [], []
    for i in range(half, len(t) - half):
        sl = slice(i - half, i + half)
        slopes.append(float(np.polyfit(t[sl], y[sl], 1)[0]))
        mids.append(float(t[i]))
    return np.asarray(mids), np.asarray(slopes)


def _ts_with_rotated_hist(
    ax_ts,
    ax_hist,
    t: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    ylabel: str,
    bins: int = 48,
    lw: float = 0.8,
    label: str | None = None,
    href: float | None = None,
) -> None:
    """Time series on ``ax_ts`` with a matching rotated marginal on ``ax_hist``."""
    ax_ts.plot(t, y, color=color, lw=lw, label=label)
    if href is not None:
        ax_ts.axhline(href, color="0.55", ls="--", lw=0.7)
    ax_ts.set_ylabel(ylabel, color=color)
    ax_ts.tick_params(axis="y", colors=color)
    ax_ts.grid(True, axis="x", alpha=0.2, lw=0.5)

    counts, edges = np.histogram(y, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_hist.barh(
        centers,
        counts,
        height=np.diff(edges),
        color=color,
        alpha=0.75,
        align="center",
    )
    ax_hist.set_ylim(ax_ts.get_ylim())
    ax_hist.set_xlabel("dens.")
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(True, axis="x", alpha=0.2, lw=0.5)


def write_nve_validation_dashboard(
    path: Path,
    *,
    t_ps: np.ndarray,
    e_tot_kcal: np.ndarray,
    e_pot_kcal: np.ndarray,
    e_kin_kcal: np.ndarray,
    temp: np.ndarray,
    r_sym: np.ndarray,
    ang: np.ndarray,
    q_o: np.ndarray,
    q_h_mean: np.ndarray,
    freq_cm: np.ndarray,
    ir_smooth: np.ndarray,
    ir_vib: np.ndarray,
    frame_dt_fs: float,
    mm_mode: str,
    box: float,
    n_mol: int,
    ir_method: str,
    t_ir: float,
    ir_min_cm: float = 500.0,
) -> dict:
    """Composite NVE validation figure: TS + rotated marginals + Fourier."""
    drift = float(e_tot_kcal[-1] - e_tot_kcal[0])
    slope = float(np.polyfit(t_ps, e_tot_kcal, 1)[0])
    t_mean = float(np.mean(temp))
    corr_pk = float(np.corrcoef(e_pot_kcal, e_kin_kcal)[0, 1])
    de = e_tot_kcal - e_tot_kcal.mean()
    dk = e_kin_kcal - e_kin_kcal.mean()

    # Frequency content of conservation / kinetics (cm^-1)
    f_e, psd_e = _periodogram_density(de, frame_dt_fs)
    f_k, psd_k = _periodogram_density(dk, frame_dt_fs)
    f_t, psd_t = _periodogram_density(temp - t_mean, frame_dt_fs)
    # Bond symmetric-stretch DOS proxy (mean over molecules)
    r_sym_mean_t = r_sym.mean(axis=1)
    f_r, psd_r = _periodogram_density(r_sym_mean_t, frame_dt_fs)

    # Running drift of E_tot
    win = max(64, len(t_ps) // 20)
    t_run, slope_run = _running_linear_slope(t_ps, e_tot_kcal, window=win)

    # Molecule-averaged geometry / charge traces
    r_t = r_sym.mean(axis=1)
    a_t = ang.mean(axis=1)
    qo_t = q_o.mean(axis=1)
    qh_t = q_h_mean.mean(axis=1)

    fig = plt.figure(figsize=(14.5, 16.5), constrained_layout=False)
    gs = GridSpec(
        6,
        4,
        figure=fig,
        height_ratios=[0.55, 1.05, 1.05, 1.15, 1.0, 1.15],
        width_ratios=[1.0, 1.0, 1.0, 0.32],
        hspace=0.42,
        wspace=0.28,
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=0.04,
    )

    # --- metrics banner ---
    ax_m = fig.add_subplot(gs[0, :])
    ax_m.axis("off")
    banner = (
        f"NVE validation dashboard   ·   TIP3:{n_mol}   ·   "
        f"mm_charge_mode={mm_mode}   ·   L={box:.1f} A   ·   "
        f"frame Δt={frame_dt_fs:.3f} fs   ·   T={t_ps[-1] - t_ps[0]:.2f} ps\n"
        f"E_tot drift={drift:+.3f} kcal/mol   ·   "
        f"slope={slope:+.3f} kcal/mol/ps   ·   "
        f"σ(E_tot)={np.std(e_tot_kcal):.3f}   ·   "
        f"⟨T⟩={t_mean:.1f}±{np.std(temp):.1f} K   ·   "
        f"corr(E_pot,E_kin)={corr_pk:.3f}"
    )
    ax_m.text(
        0.0,
        0.55,
        banner,
        transform=ax_m.transAxes,
        va="center",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#cccccc"),
    )

    # --- Row 1: energies + twin + rotated ΔE marginal ---
    ax_e = fig.add_subplot(gs[1, :3])
    ax_e2 = ax_e.twinx()
    ax_eh = fig.add_subplot(gs[1, 3], sharey=None)
    (ln1,) = ax_e.plot(t_ps, e_tot_kcal, color="#1f4e79", lw=0.75, label=r"$E_\mathrm{tot}$")
    (ln2,) = ax_e2.plot(t_ps, e_kin_kcal, color="#2a6f3b", lw=0.75, label=r"$E_\mathrm{kin}$")
    ax_e.set_ylabel(r"$E_\mathrm{tot}$ (kcal/mol)", color="#1f4e79")
    ax_e2.set_ylabel(r"$E_\mathrm{kin}$ (kcal/mol)", color="#2a6f3b")
    ax_e.tick_params(axis="y", colors="#1f4e79")
    ax_e2.tick_params(axis="y", colors="#2a6f3b")
    ax_e.set_title("Energy conservation (twin axes, equal span)")
    ax_e.legend(handles=[ln1, ln2], loc="upper right", frameon=False, fontsize=8)
    _twin_equal_span(ax_e, ax_e2, e_tot_kcal, e_kin_kcal)
    ax_e.grid(True, axis="x", alpha=0.2, lw=0.5)
    # Marginal of demeaned energies (same kcal scale for visual comparison)
    c_de, e_de = np.histogram(de, bins=50, density=True)
    c_dk, e_dk = np.histogram(dk, bins=50, density=True)
    ax_eh.barh(
        0.5 * (e_de[:-1] + e_de[1:]),
        c_de,
        height=np.diff(e_de),
        color="#1f4e79",
        alpha=0.65,
        label=r"$\Delta E_\mathrm{tot}$",
    )
    ax_eh.barh(
        0.5 * (e_dk[:-1] + e_dk[1:]),
        c_dk,
        height=np.diff(e_dk),
        color="#2a6f3b",
        alpha=0.45,
        label=r"$\Delta E_\mathrm{kin}$",
    )
    ax_eh.set_xlabel("dens.")
    ax_eh.set_title(r"$\Delta E$", fontsize=9)
    ax_eh.legend(fontsize=7, frameon=False, loc="upper right")
    ax_eh.tick_params(axis="y", labelleft=False)

    # --- Row 2: temperature + marginal ---
    ax_t = fig.add_subplot(gs[2, :3], sharex=ax_e)
    ax_th = fig.add_subplot(gs[2, 3])
    _ts_with_rotated_hist(
        ax_t,
        ax_th,
        t_ps,
        temp,
        color="#5a3d7a",
        ylabel="T (K)",
        href=300.0,
        label="T",
    )
    ax_t.set_title("Kinetic temperature")
    ax_t.set_xlabel("time (ps)")

    # --- Row 3: Fourier / tendency ---
    ax_psd = fig.add_subplot(gs[3, 0:2])
    # Show low-frequency conservation band + kinetic content (log y)
    for f, p, c, lab in (
        (f_e, psd_e, "#1f4e79", r"$E_\mathrm{tot}$"),
        (f_k, psd_k, "#2a6f3b", r"$E_\mathrm{kin}$"),
        (f_t, psd_t, "#5a3d7a", "T"),
        (f_r, psd_r, "#c45c26", r"$\langle r_\mathrm{sym}\rangle$"),
    ):
        m = (f > 0) & (f <= 500)
        # Normalize each PSD to its max in-band for shape comparison
        pn = p[m] / max(float(p[m].max()), 1e-30)
        ax_psd.plot(f[m], pn, color=c, lw=1.0, label=lab)
    ax_psd.set_xlabel(r"wavenumber (cm$^{-1}$)")
    ax_psd.set_ylabel("PSD (peak-norm.)")
    ax_psd.set_title("Fluctuation spectra (0–500 cm$^{-1}$)")
    ax_psd.set_yscale("log")
    ax_psd.legend(fontsize=7, frameon=False, ncol=2)
    ax_psd.grid(True, alpha=0.25, which="both", lw=0.5)

    ax_run = fig.add_subplot(gs[3, 2])
    ax_run.plot(t_run, slope_run, color="#1f4e79", lw=0.9)
    ax_run.axhline(0.0, color="0.5", ls="--", lw=0.7)
    ax_run.axhline(slope, color="#c0392b", ls=":", lw=0.8, label="global slope")
    ax_run.set_xlabel("time (ps)")
    ax_run.set_ylabel("local dE/dt")
    ax_run.set_title(f"Running E_tot drift (win={win})")
    ax_run.legend(fontsize=7, frameon=False)

    ax_xy = fig.add_subplot(gs[3, 3])
    ax_xy.scatter(
        e_pot_kcal - e_pot_kcal.mean(),
        e_kin_kcal - e_kin_kcal.mean(),
        s=3,
        alpha=0.25,
        c="#444444",
        rasterized=True,
        linewidths=0,
    )
    ax_xy.set_xlabel(r"$\Delta E_\mathrm{pot}$")
    ax_xy.set_ylabel(r"$\Delta E_\mathrm{kin}$")
    ax_xy.set_title(f"exchange\ncorr={corr_pk:.2f}", fontsize=9)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.axhline(0, color="0.7", lw=0.5)
    ax_xy.axvline(0, color="0.7", lw=0.5)

    # --- Row 4: geometry ---
    ax_g = fig.add_subplot(gs[4, :3], sharex=ax_e)
    ax_g2 = ax_g.twinx()
    ax_gh = fig.add_subplot(gs[4, 3])
    (lg1,) = ax_g.plot(t_ps, r_t, color="#16a085", lw=0.8, label=r"$\langle r_\mathrm{sym}\rangle$")
    (lg2,) = ax_g2.plot(t_ps, a_t, color="#8e44ad", lw=0.8, label=r"$\langle\angle\rangle$")
    ax_g.set_ylabel(r"$\langle r_\mathrm{sym}\rangle$ (A)", color="#16a085")
    ax_g2.set_ylabel(r"$\langle\angle\mathrm{HOH}\rangle$ (deg)", color="#8e44ad")
    ax_g.tick_params(axis="y", colors="#16a085")
    ax_g2.tick_params(axis="y", colors="#8e44ad")
    ax_g.set_title("Mean intramolecular geometry")
    ax_g.legend(handles=[lg1, lg2], loc="upper right", frameon=False, fontsize=8)
    ax_g.set_xlabel("time (ps)")
    # Rotated hist for r_sym (all mols) — denser structural metric
    counts, edges = np.histogram(r_sym.ravel(), bins=40, density=True)
    ax_gh.barh(
        0.5 * (edges[:-1] + edges[1:]),
        counts,
        height=np.diff(edges),
        color="#16a085",
        alpha=0.75,
    )
    ax_gh.set_xlabel("dens.")
    ax_gh.set_title(r"$r_\mathrm{sym}$", fontsize=9)
    ax_gh.tick_params(axis="y", labelleft=False)

    # --- Row 5: charges + IR + geom-charge map ---
    # Split bottom into: charge TS+hist | IR | O scatter | H scatter — use nested gs
    gs5 = gs[5, :].subgridspec(1, 4, width_ratios=[1.35, 1.15, 1.0, 1.0], wspace=0.35)

    ax_q = fig.add_subplot(gs5[0, 0])
    ax_q2 = ax_q.twinx()
    (lq1,) = ax_q.plot(t_ps, qo_t, color="#c0392b", lw=0.8, label=r"$\langle q_\mathrm{O}\rangle$")
    (lq2,) = ax_q2.plot(t_ps, qh_t, color="#2980b9", lw=0.8, label=r"$\langle q_\mathrm{H}\rangle$")
    ax_q.set_ylabel(r"$\langle q_\mathrm{O}\rangle$ (e)", color="#c0392b")
    ax_q2.set_ylabel(r"$\langle q_\mathrm{H}\rangle$ (e)", color="#2980b9")
    ax_q.tick_params(axis="y", colors="#c0392b")
    ax_q2.tick_params(axis="y", colors="#2980b9")
    ax_q.set_title("Mean MM charges")
    ax_q.set_xlabel("time (ps)")
    ax_q.legend(handles=[lq1, lq2], loc="best", frameon=False, fontsize=7)

    ax_ir = fig.add_subplot(gs5[0, 1])
    ax_ir.plot(freq_cm, ir_smooth, color="#111111", lw=1.1)
    ax_ir.set_xlabel(r"cm$^{-1}$")
    ax_ir.set_ylabel("I (arb. u.)")
    ax_ir.set_title(
        f"IR (≥{ir_min_cm:g}) · T={t_ir:.0f}K", fontsize=9
    )
    if freq_cm.size:
        ax_ir.set_xlim(float(freq_cm[0]), float(freq_cm[-1]))
    ax_ir.set_ylim(bottom=0)

    ax_so = fig.add_subplot(gs5[0, 2])
    sc = ax_so.scatter(
        r_sym.ravel()[::2],
        ang.ravel()[::2],
        c=q_o.ravel()[::2],
        s=2,
        alpha=0.3,
        cmap="Reds",
        linewidths=0,
        rasterized=True,
    )
    ax_so.set_xlabel(r"$r_\mathrm{sym}$ (A)")
    ax_so.set_ylabel(r"$\angle$ (deg)")
    ax_so.set_title(r"$q_\mathrm{O}$", fontsize=9)
    plt.colorbar(sc, ax=ax_so, fraction=0.046, pad=0.04)

    ax_sh = fig.add_subplot(gs5[0, 3])
    sc2 = ax_sh.scatter(
        r_sym.ravel()[::2],
        ang.ravel()[::2],
        c=q_h_mean.ravel()[::2],
        s=2,
        alpha=0.3,
        cmap="Blues",
        linewidths=0,
        rasterized=True,
    )
    ax_sh.set_xlabel(r"$r_\mathrm{sym}$ (A)")
    ax_sh.set_ylabel(r"$\angle$ (deg)")
    ax_sh.set_title(r"$\langle q_\mathrm{H}\rangle$", fontsize=9)
    plt.colorbar(sc2, ax=ax_sh, fraction=0.046, pad=0.04)

    fig.suptitle("Hybrid ML/MM NVE validation", fontsize=13, fontweight="bold", y=0.985)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Peak of kinetic PSD in the far-IR band (diagnostic)
    m_far = (f_k > 5) & (f_k < 400)
    k_peak = float(f_k[m_far][np.argmax(psd_k[m_far])]) if np.any(m_far) else float("nan")

    return {
        "corr_Epot_Ekin": corr_pk,
        "E_tot_running_drift_window": int(win),
        "E_kin_psd_peak_cm": k_peak,
        "dashboard": str(path),
    }


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
    min_cm: float = 500.0,
    max_cm: float = 4500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge-current ACF -> IR with harmonic / experimental QM correction.

    ``current`` is J(t) = sum_i q_i v_i or dmu/dt, shape (T, 3).  With
    C_JJ(w) = w^2 C_mm(w), absorption is

        I(w) ~ [(1 - exp(-beta hbar w)) / w] C_JJ(w)
             = w (1 - exp(-beta hbar w)) C_mm(w)

    Frequencies below ``min_cm`` are discarded *before* smoothing.  Intensities
    are left in arbitrary units (no integral normalization).

    Returns ``(freq_cm, I_raw, I_smooth)`` restricted to [min_cm, max_cm].
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

    # Cut far-IR pedestal before any smooth / display scaling.
    band = (freq_cm >= float(min_cm)) & (freq_cm <= float(max_cm))
    freq_cm = freq_cm[band]
    intensity = intensity[band]

    if smooth_cm > 0.0 and freq_cm.size > 3:
        df = float(np.median(np.diff(freq_cm)))
        sigma = max(smooth_cm / (2.0 * np.sqrt(2.0 * np.log(2.0))), df)
        half = int(max(3, np.ceil(4.0 * sigma / df)))
        x = np.arange(-half, half + 1) * df
        ker = np.exp(-0.5 * (x / sigma) ** 2)
        ker /= ker.sum()
        smooth = np.convolve(intensity, ker, mode="same")
    else:
        smooth = intensity.copy()

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
    p.add_argument(
        "--ir-min-cm",
        type=float,
        default=500.0,
        help="Discard frequencies below this (cm^-1) before smooth/scale.",
    )
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

    # --- energies: twin axes + rotated marginals ---
    fig = plt.figure(figsize=(11.0, 7.2))
    gs_e = GridSpec(
        2, 2, figure=fig, width_ratios=[4.2, 1.0], wspace=0.08, hspace=0.28
    )
    ax_l = fig.add_subplot(gs_e[0, 0])
    ax_r = ax_l.twinx()
    ax_eh = fig.add_subplot(gs_e[0, 1])
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
    de = e_tot_kcal - e_tot_kcal.mean()
    dk = e_kin_kcal - e_kin_kcal.mean()
    c_de, e_de = np.histogram(de, bins=50, density=True)
    c_dk, e_dk = np.histogram(dk, bins=50, density=True)
    ax_eh.barh(
        0.5 * (e_de[:-1] + e_de[1:]),
        c_de,
        height=np.diff(e_de),
        color="#1f4e79",
        alpha=0.7,
        label=r"$\Delta E_\mathrm{tot}$",
    )
    ax_eh.barh(
        0.5 * (e_dk[:-1] + e_dk[1:]),
        c_dk,
        height=np.diff(e_dk),
        color="#2a6f3b",
        alpha=0.45,
        label=r"$\Delta E_\mathrm{kin}$",
    )
    ax_eh.set_xlabel("dens.")
    ax_eh.set_title(r"$\Delta E$", fontsize=9)
    ax_eh.legend(fontsize=7, frameon=False)
    ax_eh.tick_params(axis="y", labelleft=False)

    ax_l2 = fig.add_subplot(gs_e[1, 0], sharex=ax_l)
    ax_th = fig.add_subplot(gs_e[1, 1])
    _ts_with_rotated_hist(
        ax_l2,
        ax_th,
        t_ps,
        temp,
        color="#5a3d7a",
        ylabel="T (K)",
        href=300.0,
    )
    ax_l2.set_xlabel("time (ps)")
    ax_l2.set_title("Kinetic temperature")

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

    geom_psd = write_geometry_power_spectra(
        out / "geometry_power_spectra.png",
        r_oh1=r_oh1,
        r_oh2=r_oh2,
        ang=ang,
        frame_dt_fs=frame_dt_fs,
        min_cm=float(args.ir_min_cm),
        max_cm=float(args.ir_max_cm),
        smooth_cm=float(args.ir_smooth_cm),
    )

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
        min_cm=float(args.ir_min_cm),
        max_cm=float(args.ir_max_cm),
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(freq_cm, ir_raw, color="#b0b0b0", lw=0.7, label="raw")
    ax.plot(
        freq_cm,
        ir_smooth,
        color="#111111",
        lw=1.35,
        label=f"smoothed (HWHM={args.ir_smooth_cm:g} cm$^{{-1}}$)",
    )
    ax.set_xlabel(r"wavenumber (cm$^{-1}$)")
    ax.set_ylabel("intensity (arb. u.)")
    ax.set_title(
        f"IR from {ir_method} · QM corr. omega*(1-exp(-beta hbar omega)) "
        f"at T={t_ir:.0f} K\n"
        f"cut <{args.ir_min_cm:g} cm$^{{-1}}$ before smooth · "
        f"frame dt={frame_dt_fs:.3f} fs · {n_frames} frames · {n_mol} TIP3"
    )
    ax.set_xlim(float(args.ir_min_cm), float(args.ir_max_cm))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ir_spectrum.png", dpi=170)
    plt.close(fig)

    np.savez_compressed(
        out / "ir_spectrum.npz",
        freq_cm=freq_cm,
        intensity=ir_raw,
        intensity_smooth=ir_smooth,
        frame_dt_fs=frame_dt_fs,
        temperature_K_qcf=t_ir,
        ir_min_cm=float(args.ir_min_cm),
        ir_max_cm=float(args.ir_max_cm),
        qm_correction="(1-exp(-beta*hbar*omega))/omega * C_JJ",
        method=ir_method,
        units="arbitrary",
    )

    dash_meta = write_nve_validation_dashboard(
        out / "nve_validation_dashboard.png",
        t_ps=t_ps,
        e_tot_kcal=e_tot_kcal,
        e_pot_kcal=e_pot_kcal,
        e_kin_kcal=e_kin_kcal,
        temp=temp,
        r_sym=r_sym,
        ang=ang,
        q_o=q_o,
        q_h_mean=q_h_mean_mol,
        freq_cm=freq_cm,
        ir_smooth=ir_smooth,
        ir_vib=ir_smooth,
        frame_dt_fs=frame_dt_fs,
        mm_mode=mm_mode,
        box=box,
        n_mol=n_mol,
        ir_method=ir_method,
        t_ir=t_ir,
        ir_min_cm=float(args.ir_min_cm),
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
            "corr_Epot_Ekin": dash_meta["corr_Epot_Ekin"],
            "E_kin_psd_peak_cm": dash_meta["E_kin_psd_peak_cm"],
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
            "power_spectra": geom_psd,
        },
        "ir": {
            "temperature_K_qcf": t_ir,
            "method": ir_method,
            "qm_correction": "omega*(1-exp(-beta*hbar*omega))",
            "ir_min_cm": float(args.ir_min_cm),
            "ir_max_cm": float(args.ir_max_cm),
            "units": "arbitrary",
            "cut_before_smooth": True,
            "smooth_hwhm_cm": float(args.ir_smooth_cm),
            "note": (
                f"frame_dt_fs={frame_dt_fs:.3f}. "
                f"Frequencies <{args.ir_min_cm:g} cm^-1 removed before smooth."
            ),
        },
        "artifacts": [
            "nve_validation_dashboard.png",
            "energy_fluctuations.png",
            "charge_distributions.png",
            "charge_variance_per_atom.png",
            "geometry_distributions.png",
            "charge_vs_geometry_scatter.png",
            "charge_variance_vs_geometry_scatter.png",
            "geometry_power_spectra.png",
            "ir_spectrum.png",
            "ir_spectrum.npz",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

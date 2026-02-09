#!/usr/bin/env python3
"""
Compute IR and VCD spectra from MD trajectories.

Two complementary methods — use either or both:

  correlation  Dipole autocorrelation → FFT → spectrum
               ⟨μ(0)·μ(t)⟩  → IR,   ⟨μ(0)·m(t)⟩  → VCD
               Captures anharmonicity; broadening from dynamics.

  harmonic     Hessian + APT + AAT at selected snapshots
               Normal-mode analysis at each geometry.
               Gives stick spectra; explicit mode assignment.

For transient (time-resolved) spectroscopy, add --transient.
For 2D frequency–frequency correlation maps, add --spectra-2d.
For Noda generalised 2D correlation, add --noda (with --transient).

Usage:
  # Correlation IR + VCD from an MD trajectory
  python spectra_md.py --trajectory md.traj --params params.json

  # Transient VCD (e.g. after switching on E-field at t=0)
  python spectra_md.py --trajectory md.traj --params params.json \\
      --transient --window-size 500 --stride 100

  # 2D IR / VCD frequency-frequency correlation maps
  python spectra_md.py --trajectory md.traj --params params.json \\
      --spectra-2d --waiting-times 0 50 200

  # Noda 2D generalised correlation from transient spectrograms
  python spectra_md.py --trajectory md.traj --params params.json \\
      --transient --noda

  # Harmonic snapshots every 200 frames
  python spectra_md.py --trajectory md.traj --params params.json \\
      --method harmonic --snapshot-interval 200

  # Both methods + 2D + transient + Noda
  python spectra_md.py --trajectory md.traj --params params.json \\
      --method both --transient --spectra-2d --noda
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

import argparse
import sys
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ase
import ase.io
from ase.io.trajectory import Trajectory
from ase.data import atomic_masses as ASE_ATOMIC_MASSES

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ase_calc_EF import AseCalculatorEF

# =====================================================================
# Constants
# =====================================================================
C_AU = 137.035999084                   # speed of light (atomic units)
FS_INV_TO_CM_INV = 1e15 / 2.99792458e10  # 1/fs  →  cm⁻¹  (≈ 33 356.4)


# =====================================================================
# Property extraction from trajectory frames
# =====================================================================
def extract_properties(traj_frames, calc=None,
                       recompute_dipole=False,
                       recompute_charges=False):
    """Read / recompute  μ(t), q_s(t), r(t), v(t)  from trajectory.

    Dipoles are read from  atoms.info['ml_dipole']  if available;
    charges from  atoms.arrays['ml_charges'].
    If not present (or recompute=True), the calculator is used.

    Returns
    -------
    positions  : (T, N, 3)  Å
    velocities : (T, N, 3)  ASE internal units
    dipoles    : (T, 3)
    charges    : (T, N)
    """
    T = len(traj_frames)
    N = len(traj_frames[0])

    positions  = np.zeros((T, N, 3))
    velocities = np.zeros((T, N, 3))
    dipoles    = np.zeros((T, 3))
    charges    = np.zeros((T, N))
    has_vel = True

    for i, atoms in enumerate(traj_frames):
        positions[i] = atoms.get_positions()

        # Velocities
        try:
            v = atoms.get_velocities()
            if v is not None and np.any(v != 0):
                velocities[i] = v
            else:
                has_vel = False
        except Exception:
            has_vel = False

        # Dipole
        if not recompute_dipole and 'ml_dipole' in atoms.info:
            dipoles[i] = np.asarray(atoms.info['ml_dipole'])
        elif calc is not None:
            atoms.calc = calc
            atoms.get_potential_energy()          # triggers calculate()
            dipoles[i] = calc.results.get('dipole', np.zeros(3))
        # else: stays zero

        # Charges
        if not recompute_charges and 'ml_charges' in getattr(atoms, 'arrays', {}):
            charges[i] = atoms.arrays['ml_charges']
        elif calc is not None and recompute_charges:
            q, _ = calc.get_atomic_charges(atoms)
            charges[i] = q
        else:
            # fall back to nuclear charges (crude)
            charges[i] = atoms.get_atomic_numbers().astype(float)

        if (i + 1) % 200 == 0 or i == 0:
            print(f"      frame {i+1}/{T}")

    if not has_vel:
        print("    WARNING: trajectory has no velocities — "
              "VCD from correlation functions will be zero.")
    return positions, velocities, dipoles, charges


def compute_magnetic_dipoles(positions, velocities, charges):
    """m(t) = Σ_s  (q_s / 2c)  r_s × v_s   at every frame.

    Positions [Å], velocities [ASE internal], charges [e].
    Result in internally consistent (not SI) units —
    relative spectra are correct.
    """
    T = positions.shape[0]
    m = np.zeros((T, 3))
    for t in range(T):
        cross = np.cross(positions[t], velocities[t])   # (N, 3)
        m[t] = np.sum(charges[t, :, None] * cross, axis=0) / (2.0 * C_AU)
    return m


# =====================================================================
# FFT-based correlation functions
# =====================================================================
def _next_pow2(n):
    return 2 ** int(np.ceil(np.log2(n)))


def autocorrelation(signal):
    """⟨x(0)·x(τ)⟩  for a (T, 3) vector signal.  Returns (T,)."""
    T = len(signal)
    n_fft = _next_pow2(2 * T)
    acf = np.zeros(T)
    for a in range(signal.shape[1]):
        ft = np.fft.rfft(signal[:, a], n=n_fft)
        acf += np.fft.irfft(ft * np.conj(ft), n=n_fft)[:T]
    acf /= np.arange(T, 0, -1)          # normalise by overlap count
    return acf


def cross_correlation(a, b):
    """⟨a(0)·b(τ)⟩  for (T, 3) vector signals.  Returns (T,)."""
    T = len(a)
    n_fft = _next_pow2(2 * T)
    ccf = np.zeros(T)
    for alpha in range(a.shape[1]):
        fa = np.fft.rfft(a[:, alpha], n=n_fft)
        fb = np.fft.rfft(b[:, alpha], n=n_fft)
        ccf += np.fft.irfft(fa * np.conj(fb), n=n_fft)[:T]
    ccf /= np.arange(T, 0, -1)
    return ccf


# =====================================================================
# Correlation function  →  spectrum
# =====================================================================
def _make_window(n, kind):
    if kind == 'hann':
        return np.hanning(n)
    if kind == 'blackman':
        return np.blackman(n)
    if kind == 'gaussian':
        return np.exp(-0.5 * (np.arange(n) / (n / 6.0)) ** 2)
    return np.ones(n)


def correlation_to_spectrum(corr, dt_fs, window='hann',
                            zero_pad=4, qcf='harmonic'):
    """FT a correlation function to a power spectrum.

    Parameters
    ----------
    corr : (n,)
    dt_fs : timestep in fs
    window : 'hann' | 'blackman' | 'gaussian' | None
    zero_pad : multiply length before FFT
    qcf : quantum correction factor
        'harmonic' → I(ω) ∝ ω · C̃(ω)   (standard absorption)
        'classical' → I(ω) ∝ ω² · C̃(ω)
        None → raw C̃(ω)

    Returns
    -------
    freq_cm : (m,)  in cm⁻¹
    spectrum : (m,)
    """
    n = len(corr)
    w = _make_window(n, window) if window else np.ones(n)
    n_fft = n * zero_pad

    ft = np.fft.rfft(corr * w, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, d=dt_fs)      # 1/fs
    freq_cm = freq * FS_INV_TO_CM_INV            # cm⁻¹
    spec = np.real(ft)

    if qcf == 'harmonic':
        spec *= np.where(freq_cm > 0, freq_cm, 0)
    elif qcf == 'classical':
        spec *= np.where(freq_cm > 0, freq_cm ** 2, 0)
    return freq_cm, spec


# =====================================================================
# Windowed (transient) spectra
# =====================================================================
def transient_spectra(mu, m, dt_fs,
                      window_frames, stride_frames,
                      fft_window='hann', zero_pad=2):
    """Sliding-window IR + VCD spectrograms.

    Returns
    -------
    t_centres : (W,)     window centres in fs
    freq_cm   : (F,)     frequency axis in cm⁻¹
    ir_gram   : (W, F)
    vcd_gram  : (W, F)
    """
    T = len(mu)
<<<<<<< HEAD
    starts = list(range(0, T - window_frames + 1, stride_frames))
=======
    if window_frames > T:
        raise ValueError(
            f"--window-size ({window_frames}) exceeds trajectory length "
            f"({T} frames).  Use a smaller window or a longer trajectory.")
    starts = list(range(0, T - window_frames + 1, stride_frames))
    if len(starts) == 0:
        raise ValueError(
            f"No windows fit: window_frames={window_frames}, "
            f"stride={stride_frames}, trajectory={T} frames.")
>>>>>>> bdfda8dbbf49b1f2d64d87d88d2231ee12619451

    ir_rows, vcd_rows, t_centres = [], [], []
    for i, t0 in enumerate(starts):
        sl = slice(t0, t0 + window_frames)
        acf = autocorrelation(mu[sl])
        ccf = cross_correlation(mu[sl], m[sl])

        freq_cm, ir = correlation_to_spectrum(
            acf, dt_fs, window=fft_window, zero_pad=zero_pad)
        _, vcd = correlation_to_spectrum(
            ccf, dt_fs, window=fft_window, zero_pad=zero_pad)

        ir_rows.append(ir)
        vcd_rows.append(vcd)
        t_centres.append((t0 + window_frames / 2) * dt_fs)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"      window {i+1}/{len(starts)}  "
                  f"t_centre = {t_centres[-1]:.1f} fs")

    return (np.array(t_centres), freq_cm,
            np.array(ir_rows), np.array(vcd_rows))


# =====================================================================
# 2D spectra  (frequency–frequency correlation maps)
# =====================================================================

def stft_vector(signal, dt_fs, window_frames, stride_frames,
                window_fn='hann', zero_pad=2):
    """Short-time Fourier transform of a (T, 3) vector signal.

    Returns
    -------
    t_centres  : (W,)   window centres in fs
    freq_cm    : (F,)   frequency axis in cm⁻¹
    power      : (W, F) sum of |FT_α|² over α=x,y,z
    ft_complex : (W, F, 3) full complex STFT per component
    """
    T = len(signal)
    w = _make_window(window_frames, window_fn) if window_fn else np.ones(window_frames)
    starts = list(range(0, T - window_frames + 1, stride_frames))

    n_fft = window_frames * zero_pad
    freq = np.fft.rfftfreq(n_fft, d=dt_fs) * FS_INV_TO_CM_INV
    F = len(freq)

    ft_all = np.zeros((len(starts), F, 3), dtype=complex)
    t_centres = []
    for i, t0 in enumerate(starts):
        chunk = signal[t0:t0 + window_frames] * w[:, None]
        for a in range(3):
            ft_all[i, :, a] = np.fft.rfft(chunk[:, a], n=n_fft)
        t_centres.append((t0 + window_frames / 2) * dt_fs)

    power = np.sum(np.abs(ft_all) ** 2, axis=-1)   # (W, F)
    return np.array(t_centres), freq, power, ft_all


def spectra_2d_correlation(ft_mu, ft_m, freq_cm,
                           t_centres, waiting_time_fs):
    """2D IR and 2D VCD at a given waiting time T.

    For each pair of windows separated by T:
      2D-IR  (ω₁, ω₃)  =  ⟨ P_μ(ω₁, t) · P_μ(ω₃, t+T) ⟩_t
      2D-VCD (ω₁, ω₃)  =  ⟨ Σ_α Re[ μ̃*_α(ω₁,t) · m̃_α(ω₃,t+T) ] ⟩_t

    Parameters
    ----------
    ft_mu, ft_m : (W, F, 3) complex STFT of dipole / magnetic dipole
    freq_cm     : (F,)
    t_centres   : (W,)
    waiting_time_fs : float

    Returns
    -------
    ir_2d  : (F, F)
    vcd_2d : (F, F)
    """
    W = len(t_centres)
    dt_win = t_centres[1] - t_centres[0] if W > 1 else 1.0
    T_idx = max(1, int(round(waiting_time_fs / dt_win)))
    if T_idx >= W:
        raise ValueError(f"Waiting time {waiting_time_fs:.0f} fs "
                         f"exceeds usable range ({(W-1)*dt_win:.0f} fs)")

    n_pairs = W - T_idx
    F = len(freq_cm)

    # Power spectra (IR)
    P_mu = np.sum(np.abs(ft_mu) ** 2, axis=-1)    # (W, F)
    P_mean = P_mu.mean(axis=0)
    dP = P_mu - P_mean[None, :]

    ir_2d = np.zeros((F, F))
    for t in range(n_pairs):
        ir_2d += np.outer(dP[t], dP[t + T_idx])
    ir_2d /= n_pairs

    # Cross-spectrum (VCD) — Re[ μ̃*(ω₁) · m̃(ω₃) ] summed over α
    vcd_2d = np.zeros((F, F))
    for t in range(n_pairs):
        cross_1 = np.sum(np.abs(ft_mu[t]) ** 2, axis=-1)          # proxy pump
        cross_3 = np.sum(
            (ft_mu[t + T_idx].conj() * ft_m[t + T_idx]).real,
            axis=-1)                                               # probe
        vcd_2d += np.outer(cross_1, cross_3)
    vcd_2d /= n_pairs

    return ir_2d, vcd_2d


def noda_2d(spectrogram):
    """Generalised 2D correlation spectroscopy  (Noda, 1993).

    Parameters
    ----------
    spectrogram : (n_times, n_freq)

    Returns
    -------
    synchronous  : (n_freq, n_freq)  Φ — covariance of spectral changes
    asynchronous : (n_freq, n_freq)  Ψ — sequential spectral changes
    """
    n_t, n_f = spectrogram.shape
    mean = spectrogram.mean(axis=0)
    dyn = spectrogram - mean[None, :]

    # Synchronous = covariance
    sync = dyn.T @ dyn / max(n_t - 1, 1)

    # Hilbert–Noda matrix  H_{jk} = 1/(π(k−j))  for j≠k, 0 for j==k
    j = np.arange(n_t)
    diff = j[None, :] - j[:, None]                 # k − j
    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.where(diff != 0, 1.0 / (np.pi * diff), 0.0)

    async_ = dyn.T @ H @ dyn / max(n_t - 1, 1)
    return sync, async_


# =====================================================================
# 2D plotting helpers
# =====================================================================
def _plot_2d_map(ax, freq, data, title, cmap='RdBu_r', symmetric=True):
    """Plot a 2D frequency–frequency map."""
    if symmetric:
        vmax = np.percentile(np.abs(data), 98) or 1.0
        im = ax.pcolormesh(freq, freq, data,
                           shading='auto', cmap=cmap,
                           vmin=-vmax, vmax=vmax)
    else:
        im = ax.pcolormesh(freq, freq, data,
                           shading='auto', cmap=cmap)
    ax.set_xlabel('ω₃  (cm⁻¹)')
    ax.set_ylabel('ω₁  (cm⁻¹)')
    ax.set_title(title)
    ax.set_aspect('equal')
    # diagonal guide
    lims = [freq.min(), freq.max()]
    ax.plot(lims, lims, 'k--', lw=0.5, alpha=0.4)
    return im


def plot_2d_spectra(freq, ir_2d, vcd_2d, T_fs, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = _plot_2d_map(axes[0], freq, ir_2d,
                       f'2D IR  (T = {T_fs:.0f} fs)', cmap='hot',
                       symmetric=False)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = _plot_2d_map(axes[1], freq, vcd_2d,
                       f'2D VCD  (T = {T_fs:.0f} fs)')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_noda(freq, sync, async_, kind, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = _plot_2d_map(axes[0], freq, sync,
                       f'Synchronous Φ  ({kind})')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = _plot_2d_map(axes[1], freq, async_,
                       f'Asynchronous Ψ  ({kind})')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# CLI
# =====================================================================
def get_args(**kwargs):
    """
    Get configuration arguments. Works both from command line and notebooks.
    
    In notebooks, you can override defaults by passing keyword arguments:
        args = get_args(trajectory="md.traj", method="both", transient=True)
    
    From command line, use argparse flags as before.
    """
    # Default values
    defaults = {
        "trajectory": None,  # Required, but None for notebook mode
        "params": "params.json",
        "config": None,
        "field_scale": 0.001,
        "method": "correlation",
        "recompute_dipole": False,
        "recompute_charges": False,
        "dt": None,
        "window_fn": "hann",
        "zero_pad": 4,
        "transient": False,
        "window_size": 500,
        "stride": 100,
        "spectra_2d": False,
        "waiting_times": [0, 50, 200],
        "stft_window": 256,
        "stft_stride": 32,
        "noda": False,
        "snapshot_interval": 200,
        "hessian_delta": 0.001,
        "broadening": 10.0,
        "freq_min": 0.0,
        "freq_max": 4000.0,
        "output_dir": "spectra_md",
    }
    
    # Check if we're in a notebook/IPython environment
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    # If kwargs are provided, always use notebook mode
    if kwargs:
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)
    
    # Check if any command line arguments look like our flags (start with --)
    has_flag_args = any(arg.startswith('--') for arg in sys.argv[1:])
    
    # If command line arguments are provided AND we're not in a notebook, use argparse
    if has_flag_args and not in_notebook:
        p = argparse.ArgumentParser(
            description="IR / VCD spectra from MD trajectories "
                        "(correlation functions and/or harmonic snapshots)")

        g = p.add_argument_group("input")
        g.add_argument("--trajectory", required=True,
                       help=".traj or multi-frame .xyz")
        g.add_argument("--params",  default=defaults["params"])
        g.add_argument("--config",  default=defaults["config"])
        g.add_argument("--field-scale", type=float, default=defaults["field_scale"])

        g = p.add_argument_group("method")
        g.add_argument("--method",
                       choices=["correlation", "harmonic", "both"],
                       default=defaults["method"])

        g = p.add_argument_group("property extraction")
        g.add_argument("--recompute-dipole", action="store_true",
                       help="Recompute dipoles even if stored in trajectory")
        g.add_argument("--recompute-charges", action="store_true",
                       help="Recompute charges even if stored in trajectory")

        g = p.add_argument_group("correlation parameters")
        g.add_argument("--dt", type=float, default=defaults["dt"],
                       help="MD timestep in fs (auto-detect or default 0.5)")
        g.add_argument("--window-fn",
                       choices=["hann", "blackman", "gaussian", "none"],
                       default=defaults["window_fn"])
        g.add_argument("--zero-pad", type=int, default=defaults["zero_pad"])

        g = p.add_argument_group("transient (sliding-window)")
        g.add_argument("--transient", action="store_true")
        g.add_argument("--window-size", type=int, default=defaults["window_size"],
                       help="Sliding window in frames")
        g.add_argument("--stride", type=int, default=defaults["stride"],
                       help="Stride in frames")

        g = p.add_argument_group("2D spectra")
        g.add_argument("--spectra-2d", action="store_true",
                       help="Compute 2D IR / 2D VCD frequency-frequency maps")
        g.add_argument("--waiting-times", type=float, nargs='+',
                       default=defaults["waiting_times"],
                       help="Waiting times T (fs) for 2D spectra")
        g.add_argument("--stft-window", type=int, default=defaults["stft_window"],
                       help="STFT window size in frames for 2D spectra")
        g.add_argument("--stft-stride", type=int, default=defaults["stft_stride"],
                       help="STFT stride in frames for 2D spectra")
        g.add_argument("--noda", action="store_true",
                       help="Compute Noda 2D correlation from transient spectra "
                            "(requires --transient)")

        g = p.add_argument_group("harmonic snapshots")
        g.add_argument("--snapshot-interval", type=int, default=defaults["snapshot_interval"],
                       help="Compute harmonic spectrum every N frames")
        g.add_argument("--hessian-delta", type=float, default=defaults["hessian_delta"])
        g.add_argument("--broadening", type=float, default=defaults["broadening"],
                       help="Lorentzian HWHM for harmonic spectra (cm⁻¹)")

        g = p.add_argument_group("output")
        g.add_argument("--freq-min", type=float, default=defaults["freq_min"])
        g.add_argument("--freq-max", type=float, default=defaults["freq_max"])
        g.add_argument("--output-dir", default=defaults["output_dir"])

        args = p.parse_args()
        # Convert argparse namespace to match our naming (hyphens to underscores)
        return SimpleNamespace(
            trajectory=args.trajectory,
            params=args.params,
            config=args.config,
            field_scale=args.field_scale,
            method=args.method,
            recompute_dipole=args.recompute_dipole,
            recompute_charges=args.recompute_charges,
            dt=args.dt,
            window_fn=args.window_fn,
            zero_pad=args.zero_pad,
            transient=args.transient,
            window_size=args.window_size,
            stride=args.stride,
            spectra_2d=args.spectra_2d,
            waiting_times=args.waiting_times,
            stft_window=args.stft_window,
            stft_stride=args.stft_stride,
            noda=args.noda,
            snapshot_interval=args.snapshot_interval,
            hessian_delta=args.hessian_delta,
            broadening=args.broadening,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            output_dir=args.output_dir,
        )
    
    # Otherwise, use notebook mode (defaults only)
    return SimpleNamespace(**defaults)


# =====================================================================
# Plotting helpers
# =====================================================================
def _freq_mask(freq, fmin, fmax):
    return (freq >= fmin) & (freq <= fmax)


def plot_correlation(freq, ir, vcd, n_frames, total_fs, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(freq, ir, 'b-', lw=1)
    ax1.fill_between(freq, ir, alpha=0.25)
    ax1.set_ylabel('IR intensity (arb. u.)')
    ax1.set_title(f'IR — dipole autocorrelation '
                  f'({n_frames} frames, {total_fs:.0f} fs)')
    ax1.invert_xaxis()

    ax2.plot(freq, vcd, 'k-', lw=1)
    ax2.fill_between(freq, 0, vcd,
                     where=(vcd >= 0), color='red', alpha=0.3)
    ax2.fill_between(freq, 0, vcd,
                     where=(vcd < 0), color='blue', alpha=0.3)
    ax2.axhline(0, color='grey', lw=0.5)
    ax2.set_ylabel('VCD rot. strength (arb. u.)')
    ax2.set_title('VCD — μ / m cross-correlation')
    ax2.set_xlabel('Frequency (cm⁻¹)')
    ax2.invert_xaxis()

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_transient(t_centres, freq, ir_gram, vcd_gram, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    im1 = ax1.pcolormesh(freq, t_centres, ir_gram,
                         shading='auto', cmap='hot')
    ax1.set_ylabel('Time (fs)')
    ax1.set_title('Transient IR')
    ax1.invert_xaxis()
    plt.colorbar(im1, ax=ax1, label='Intensity')

    vmax = np.percentile(np.abs(vcd_gram), 98) or 1.0
    im2 = ax2.pcolormesh(freq, t_centres, vcd_gram,
                         shading='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    ax2.set_xlabel('Frequency (cm⁻¹)')
    ax2.set_ylabel('Time (fs)')
    ax2.set_title('Transient VCD')
    ax2.invert_xaxis()
    plt.colorbar(im2, ax=ax2, label='Rot. strength')

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_harmonic_snapshots(t_arr, freq_ax, ir_gram, vcd_gram, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    im1 = ax1.pcolormesh(freq_ax, t_arr, ir_gram,
                         shading='auto', cmap='hot')
    ax1.set_ylabel('Time (fs)')
    ax1.set_title('IR (harmonic snapshots)')
    ax1.invert_xaxis()
    plt.colorbar(im1, ax=ax1, label='IR intensity')

    vmax = np.percentile(np.abs(vcd_gram), 98) or 1.0
    im2 = ax2.pcolormesh(freq_ax, t_arr, vcd_gram,
                         shading='auto', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    ax2.set_xlabel('Frequency (cm⁻¹)')
    ax2.set_ylabel('Time (fs)')
    ax2.set_title('VCD (harmonic snapshots)')
    ax2.invert_xaxis()
    plt.colorbar(im2, ax=ax2, label='Rot. strength')

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main(args=None):
    if args is None:
        args = get_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Spectra from MD Trajectory")
    print("=" * 70)

    # ---- load trajectory -----------------------------------------------
    print(f"\n  Trajectory : {args.trajectory}")
    traj_path = Path(args.trajectory)
    if traj_path.suffix == '.traj':
        traj_frames = list(Trajectory(str(traj_path)))
    else:
        traj_frames = ase.io.read(str(traj_path), index=':')

    T = len(traj_frames)
    N = len(traj_frames[0])
    print(f"  Frames     : {T}")
    print(f"  Atoms      : {N}")

    # ---- timestep ------------------------------------------------------
    dt_fs = args.dt
    if dt_fs is None:
        dt_fs = 0.5
        print(f"  Timestep   : {dt_fs} fs  (default — use --dt to override)")
    else:
        print(f"  Timestep   : {dt_fs} fs")
    total_fs = (T - 1) * dt_fs
    print(f"  Total time : {total_fs:.1f} fs  ({total_fs / 1000:.2f} ps)")

    # ---- calculator ----------------------------------------------------
    print(f"\n  Loading calculator from {args.params} ...")
    calc = AseCalculatorEF(
        params_path=args.params, config_path=args.config,
        field_scale=args.field_scale,
    )

    # ================================================================
    # 1.  Extract  μ(t), q(t), r(t), v(t)
    # ================================================================
    need_calc = (args.recompute_dipole or args.recompute_charges
                 or args.method in ('correlation', 'both'))
    print(f"\n[1] Extracting trajectory properties ...")
    positions, velocities, dipoles, charges = extract_properties(
        traj_frames, calc=calc,
        recompute_dipole=args.recompute_dipole,
        recompute_charges=args.recompute_charges,
    )

    # ================================================================
    # 2.  Magnetic dipoles  m(t)
    # ================================================================
    print(f"\n[2] Computing magnetic dipoles  m(t) = Σ (q/2c) r×v ...")
    mag_dipoles = compute_magnetic_dipoles(positions, velocities, charges)
    mu_norm = np.linalg.norm(dipoles, axis=1)
    m_norm  = np.linalg.norm(mag_dipoles, axis=1)
    print(f"    |μ| : {mu_norm.min():.4f} – {mu_norm.max():.4f}")
    print(f"    |m| : {m_norm.min():.6f} – {m_norm.max():.6f}")

    # ================================================================
    # 3.  Correlation spectra
    # ================================================================
    if args.method in ('correlation', 'both'):
        print(f"\n[3] Correlation spectra  (window={args.window_fn}, "
              f"zero-pad ×{args.zero_pad}) ...")

        acf = autocorrelation(dipoles)
        ccf = cross_correlation(dipoles, mag_dipoles)

        win = args.window_fn if args.window_fn != 'none' else None
        freq_cm, ir_spec = correlation_to_spectrum(
            acf, dt_fs, window=win, zero_pad=args.zero_pad)
        _, vcd_spec = correlation_to_spectrum(
            ccf, dt_fs, window=win, zero_pad=args.zero_pad)

        fm = _freq_mask(freq_cm, args.freq_min, args.freq_max)
        freq_p, ir_p, vcd_p = freq_cm[fm], ir_spec[fm], vcd_spec[fm]

        res = freq_cm[1] - freq_cm[0]
        print(f"    Resolution : {res:.2f} cm⁻¹")

        np.savez(out / "correlation_spectra.npz",
                 freq_cm=freq_p, ir=ir_p, vcd=vcd_p,
                 acf=acf, ccf=ccf)
        print(f"    Data → {out / 'correlation_spectra.npz'}")

        plot_correlation(freq_p, ir_p, vcd_p, T, total_fs,
                         out / "correlation_spectra.png")
        print(f"    Plot → {out / 'correlation_spectra.png'}")

    # ================================================================
    # 4.  Transient (windowed) spectra
    # ================================================================
    if args.transient:
<<<<<<< HEAD
        print(f"\n[4] Transient spectra  (window={args.window_size} frames, "
=======
        ws = args.window_size
        if ws > T:
            print(f"\n  ⚠  --window-size {ws} > trajectory length {T}. "
                  f"Clamping to {T}.")
            ws = T
        print(f"\n[4] Transient spectra  (window={ws} frames, "
>>>>>>> bdfda8dbbf49b1f2d64d87d88d2231ee12619451
              f"stride={args.stride}) ...")

        t_cen, freq_t, ir_gram, vcd_gram = transient_spectra(
            dipoles, mag_dipoles, dt_fs,
<<<<<<< HEAD
            window_frames=args.window_size,
=======
            window_frames=ws,
>>>>>>> bdfda8dbbf49b1f2d64d87d88d2231ee12619451
            stride_frames=args.stride,
            fft_window=args.window_fn if args.window_fn != 'none' else None,
            zero_pad=min(args.zero_pad, 2),   # keep spectrograms compact
        )

        fm = _freq_mask(freq_t, args.freq_min, args.freq_max)
        freq_tp = freq_t[fm]
        ir_gp   = ir_gram[:, fm]
        vcd_gp  = vcd_gram[:, fm]

        np.savez(out / "transient_spectra.npz",
                 time_fs=t_cen, freq_cm=freq_tp,
                 ir_spectrogram=ir_gp, vcd_spectrogram=vcd_gp)
        print(f"    Data → {out / 'transient_spectra.npz'}")

        plot_transient(t_cen, freq_tp, ir_gp, vcd_gp,
                       out / "transient_spectra.png")
        print(f"    Plot → {out / 'transient_spectra.png'}")

    # ================================================================
    # 4b.  Noda 2D generalised correlation  (from transient spectrograms)
    # ================================================================
    if args.noda:
        if not args.transient:
            print("\n  ⚠  --noda requires --transient  (skipping)")
        else:
            print(f"\n[4b] Noda 2D correlation spectroscopy ...")
            # IR synchronous / asynchronous
            sync_ir, async_ir = noda_2d(ir_gp)
            plot_noda(freq_tp, sync_ir, async_ir, "IR",
                      out / "noda_2d_ir.png")
            np.savez(out / "noda_2d_ir.npz",
                     freq_cm=freq_tp, synchronous=sync_ir,
                     asynchronous=async_ir)
            print(f"    IR   → {out / 'noda_2d_ir.png'}")

            # VCD synchronous / asynchronous
            sync_vcd, async_vcd = noda_2d(vcd_gp)
            plot_noda(freq_tp, sync_vcd, async_vcd, "VCD",
                      out / "noda_2d_vcd.png")
            np.savez(out / "noda_2d_vcd.npz",
                     freq_cm=freq_tp, synchronous=sync_vcd,
                     asynchronous=async_vcd)
            print(f"    VCD  → {out / 'noda_2d_vcd.png'}")

    # ================================================================
    # 4c.  2D IR / VCD from STFT frequency–frequency correlation
    # ================================================================
    if args.spectra_2d:
        print(f"\n[4c] 2D spectra  (STFT window={args.stft_window} frames, "
              f"stride={args.stft_stride}) ...")

        t_stft, freq_stft, _, ft_mu = stft_vector(
            dipoles, dt_fs,
            window_frames=args.stft_window,
            stride_frames=args.stft_stride,
            window_fn=args.window_fn if args.window_fn != 'none' else None,
            zero_pad=args.zero_pad,
        )
        _, _, _, ft_m = stft_vector(
            mag_dipoles, dt_fs,
            window_frames=args.stft_window,
            stride_frames=args.stft_stride,
            window_fn=args.window_fn if args.window_fn != 'none' else None,
            zero_pad=args.zero_pad,
        )

        fm2d = _freq_mask(freq_stft, args.freq_min, args.freq_max)
        freq_2d = freq_stft[fm2d]
        ft_mu_c = ft_mu[:, fm2d, :]
        ft_m_c  = ft_m[:, fm2d, :]

        n_win = len(t_stft)
        dt_win = t_stft[1] - t_stft[0] if n_win > 1 else dt_fs
        max_T = (n_win - 2) * dt_win
        print(f"    STFT windows : {n_win}")
        print(f"    Freq points  : {len(freq_2d)}")
        print(f"    Max waiting  : {max_T:.0f} fs")

        for T_fs in args.waiting_times:
            if T_fs > max_T:
                print(f"    T={T_fs:.0f} fs exceeds max ({max_T:.0f} fs), "
                      f"skipping")
                continue
            print(f"    T = {T_fs:.0f} fs  ...", end=" ")
            ir_2d, vcd_2d = spectra_2d_correlation(
                ft_mu_c, ft_m_c, freq_2d, t_stft, T_fs)

            tag = f"T{int(T_fs)}"
            plot_2d_spectra(freq_2d, ir_2d, vcd_2d, T_fs,
                            out / f"2d_spectra_{tag}.png")
            np.savez(out / f"2d_spectra_{tag}.npz",
                     freq_cm=freq_2d, ir_2d=ir_2d, vcd_2d=vcd_2d,
                     waiting_time_fs=T_fs)
            print(f"→ {out / f'2d_spectra_{tag}.png'}")

    # ================================================================
    # 5.  Harmonic snapshots
    # ================================================================
    if args.method in ('harmonic', 'both'):
        from calc_spectra import (compute_normal_modes, compute_ir,
                                  compute_vcd, hessian_fd, broaden)

        intv = args.snapshot_interval
        snap_idx = list(range(0, T, intv))
        print(f"\n[5] Harmonic snapshots  ({len(snap_idx)} geometries, "
              f"every {intv} frames) ...")

        all_freqs, all_ir, all_vcd, all_t = [], [], [], []

        for si, idx in enumerate(snap_idx):
            t_fs = idx * dt_fs
            atoms = traj_frames[idx].copy()
            if 'electric_field' not in atoms.info:
                atoms.info['electric_field'] = \
                    traj_frames[0].info.get('electric_field', [0, 0, 0])
            atoms.calc = calc

            Z = atoms.get_atomic_numbers()
            masses = ASE_ATOMIC_MASSES[Z]

            print(f"    [{si+1}/{len(snap_idx)}] frame {idx}  "
                  f"t = {t_fs:.1f} fs ...")

            hess = hessian_fd(calc, atoms, delta=args.hessian_delta)
            freqs, _, evec_cart = compute_normal_modes(hess, masses)
            apt = calc.get_atomic_polar_tensor(atoms)
            ir_int, _ = compute_ir(apt, evec_cart)

            try:
                aat, _ = calc.get_aat_born(atoms)
                rot, _, _ = compute_vcd(apt, aat, evec_cart)
            except Exception:
                rot = np.zeros_like(ir_int)

            all_freqs.append(freqs)
            all_ir.append(ir_int)
            all_vcd.append(rot)
            all_t.append(t_fs)

        all_freqs = np.array(all_freqs)
        all_ir    = np.array(all_ir)
        all_vcd   = np.array(all_vcd)
        all_t     = np.array(all_t)

        np.savez(out / "harmonic_snapshots.npz",
                 times_fs=all_t, frequencies=all_freqs,
                 ir_intensities=all_ir,
                 vcd_rotational_strengths=all_vcd)
        print(f"    Data → {out / 'harmonic_snapshots.npz'}")

        # Build broadened spectrograms
        freq_ax = np.linspace(args.freq_min, args.freq_max, 2000)
        gamma   = args.broadening
        ir_sgram  = np.zeros((len(all_t), len(freq_ax)))
        vcd_sgram = np.zeros((len(all_t), len(freq_ax)))
        for i in range(len(all_t)):
            ir_sgram[i]  = broaden(freq_ax, all_freqs[i], all_ir[i], gamma)
            vcd_sgram[i] = broaden(freq_ax, all_freqs[i], all_vcd[i], gamma)

        plot_harmonic_snapshots(all_t, freq_ax, ir_sgram, vcd_sgram,
                                out / "harmonic_snapshots.png")
        print(f"    Plot → {out / 'harmonic_snapshots.png'}")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

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

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

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
# HDF5 trajectory loading
# =====================================================================
def load_hdf5_trajectory(path):
    """Load positions, velocities, and metadata from an HDF5 trajectory.

    Reads files written by ``mmml.utils.hdf5_reporter.HDF5Reporter``.

    Returns
    -------
    positions  : (T, N, 3)
    velocities : (T, N, 3) or None
    dt_fs      : float or None  (timestep in fs, if stored)
    metadata   : dict           (all scalar time-series and HDF5 attrs)
    """
    if not _HAS_H5PY:
        raise ImportError("h5py is required to load HDF5 trajectories")

    with h5py.File(str(path), "r") as f:
        positions = f["positions"][:]              # (T, N, 3)

        velocities = None
        if "velocities" in f:
            velocities = f["velocities"][:]        # (T, N, 3)

        metadata = {}
        for k in f.keys():
            if k not in ("positions", "velocities"):
                metadata[k] = f[k][:]
        for k, v in f.attrs.items():
            metadata[f"attr_{k}"] = v

        dt_fs = None
        if "attr_dt_ps" in metadata:
            dt_fs = float(metadata["attr_dt_ps"]) * 1000.0
        elif "time_ps" in metadata and len(metadata["time_ps"]) > 1:
            dt_fs = float(metadata["time_ps"][1] - metadata["time_ps"][0]) * 1000.0

        n_steps_per_rec = metadata.get("attr_steps_per_recording", None)
        if dt_fs is not None and n_steps_per_rec is not None:
            dt_fs = dt_fs / float(n_steps_per_rec)

    return positions, velocities, dt_fs, metadata


def extract_properties_hdf5(positions, velocities, calc, atomic_numbers,
                            recompute_dipole=False):
    """Compute dipoles and charges from HDF5-loaded positions/velocities.

    Parameters
    ----------
    positions      : (T, N, 3)
    velocities     : (T, N, 3) or None
    calc           : ASE calculator with dipole/charge support
    atomic_numbers : (N,) int array
    recompute_dipole : bool

    Returns
    -------
    positions, velocities, dipoles, charges  (same convention as extract_properties)
    """
    T, N, _ = positions.shape

    if velocities is None:
        velocities = np.zeros_like(positions)
        print("    WARNING: HDF5 has no velocities — "
              "VCD from correlation functions will be zero.")

    dipoles = np.zeros((T, 3))
    charges = np.zeros((T, N))

    dummy = ase.Atoms(numbers=atomic_numbers,
                      positions=positions[0])
    dummy.calc = calc

    for i in range(T):
        dummy.set_positions(positions[i])
        dummy.get_potential_energy()
        dipoles[i] = calc.results.get("dipole", np.zeros(3))

        if hasattr(calc, "get_atomic_charges"):
            q, _ = calc.get_atomic_charges(dummy)
            charges[i] = q
        else:
            charges[i] = atomic_numbers.astype(float)

        if (i + 1) % 200 == 0 or i == 0:
            print(f"      frame {i+1}/{T}")

    return positions, velocities, dipoles, charges


# =====================================================================
# GPU-batched polarizability  α(t) = dμ/dEf  (for Raman)
# =====================================================================
def compute_polarizability_batched(positions, atomic_numbers, Ef,
                                    model, params, chunk_size=32,
                                    field_scale=0.001):
    """Compute polarizability α(t) for every frame, GPU-batched.

    Processes `chunk_size` frames in parallel through the model,
    computing the Jacobian dμ/dEf via a shared-field trick that
    avoids redundant cross-replica derivatives.

    Parameters
    ----------
    positions      : (T, N, 3) float
    atomic_numbers : (N,) int
    Ef             : (3,) float — electric field in model input units
    model          : MessagePassingModel
    params         : model parameters
    chunk_size     : int — frames per GPU batch
    field_scale    : float — Ef_physical = Ef_input * field_scale

    Returns
    -------
    alpha : (T, 3, 3) polarizability in atomic units (Bohr³)
    """
    import jax
    import jax.numpy as jnp
    import e3x
    import functools
    import time

    T, N, _ = positions.shape
    Z = jnp.asarray(atomic_numbers, dtype=jnp.int32)
    Ef_jax = jnp.asarray(Ef, dtype=jnp.float32).reshape(3)

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(N)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

    B = chunk_size
    batch_segments = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)
    offsets = jnp.arange(B, dtype=jnp.int32) * N
    dst_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
    src_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
    Z_batch = jnp.tile(Z[None, :], (B, 1))

    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def _model_apply(params, atomic_numbers, positions, Ef,
                     dst_idx_flat, src_idx_flat, batch_segments, batch_size,
                     dst_idx=None, src_idx=None):
        return model.apply(
            params, atomic_numbers, positions, Ef,
            dst_idx_flat=dst_idx_flat, src_idx_flat=src_idx_flat,
            batch_segments=batch_segments, batch_size=batch_size,
            dst_idx=dst_idx, src_idx=src_idx)

    @jax.jit
    def alpha_chunk(pos_batch):
        """(B, N, 3) -> (B, 3, 3) polarizability."""
        def dipole_fn(ef_shared):
            ef_b = jnp.tile(ef_shared[None, :], (B, 1))
            _, dipole = _model_apply(
                params, Z_batch, pos_batch, ef_b,
                dst_flat, src_flat, batch_segments, B,
                dst_idx, src_idx)
            return dipole  # (B, 3)
        return jax.jacrev(dipole_fn)(Ef_jax)  # (B, 3, 3)

    alpha_all = np.zeros((T, 3, 3), dtype=np.float32)
    n_chunks = (T + B - 1) // B

    t0 = time.perf_counter()
    for ci in range(n_chunks):
        start = ci * B
        end = min(start + B, T)
        actual = end - start

        pos_pad = np.zeros((B, N, 3), dtype=np.float32)
        pos_pad[:actual] = positions[start:end]

        jac = alpha_chunk(jnp.asarray(pos_pad))
        alpha_all[start:end] = np.asarray(jac)[:actual] / field_scale

        if (ci + 1) % max(1, n_chunks // 10) == 0 or ci == 0:
            print(f"      chunk {ci+1}/{n_chunks}")

    print(f"      Done in {time.perf_counter() - t0:.2f} s")
    return alpha_all


def extract_dipoles_batched(positions, atomic_numbers, Ef,
                             model, params, chunk_size=32):
    """GPU-batched extraction of dipoles and charges from positions.

    Much faster than per-frame ASE calculator calls when properties
    are not already stored in the trajectory.

    Returns
    -------
    dipoles        : (T, 3)
    charges        : (T, N)
    atomic_dipoles : (T, N, 3)
    """
    import jax
    import jax.numpy as jnp
    import e3x
    import time

    T, N, _ = positions.shape
    Z = jnp.asarray(atomic_numbers, dtype=jnp.int32)
    Ef_jax = jnp.asarray(Ef, dtype=jnp.float32).reshape(3)

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(N)
    dst_idx = jnp.asarray(dst_idx, dtype=jnp.int32)
    src_idx = jnp.asarray(src_idx, dtype=jnp.int32)

    B = chunk_size

    dipoles = np.zeros((T, 3), dtype=np.float32)
    charges = np.zeros((T, N), dtype=np.float32)
    at_dipoles = np.zeros((T, N, 3), dtype=np.float32)

    n_chunks = (T + B - 1) // B
    t0 = time.perf_counter()

    for ci in range(n_chunks):
        start = ci * B
        end = min(start + B, T)
        actual = end - start

        pos_pad = np.zeros((B, N, 3), dtype=np.float32)
        pos_pad[:actual] = positions[start:end]

        batch_segments = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)
        offsets = jnp.arange(B, dtype=jnp.int32) * N
        dst_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
        src_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)
        Z_batch = jnp.tile(Z[None, :], (B, 1))
        Ef_batch = jnp.tile(Ef_jax[None, :], (B, 1))

        (energy, dipole), state = model.apply(
            params,
            Z_batch, jnp.asarray(pos_pad), Ef_batch,
            dst_idx_flat=dst_flat, src_idx_flat=src_flat,
            batch_segments=batch_segments, batch_size=B,
            dst_idx=dst_idx, src_idx=src_idx,
            mutable=['intermediates'])

        intermediates = state.get('intermediates', {})
        q = intermediates.get('atomic_charges', (None,))[-1]
        mu_at = intermediates.get('atomic_dipoles', (None,))[-1]

        dipoles[start:end] = np.asarray(dipole)[:actual]
        if q is not None:
            charges[start:end] = np.asarray(q)[:actual]
        if mu_at is not None:
            at_dipoles[start:end] = np.asarray(mu_at)[:actual]

        if (ci + 1) % max(1, n_chunks // 10) == 0 or ci == 0:
            print(f"      chunk {ci+1}/{n_chunks}")

    print(f"      Done in {time.perf_counter() - t0:.2f} s")
    return dipoles, charges, at_dipoles


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
# Raman from polarizability autocorrelation
# =====================================================================
def polarizability_autocorrelation(alpha_traj):
    """Isotropic and anisotropic ACFs of the polarizability tensor.

    Parameters
    ----------
    alpha_traj : (T, 3, 3) polarizability in atomic units

    Returns
    -------
    acf_iso   : (T,) — ⟨ᾱ(0)·ᾱ(τ)⟩   where ᾱ = (1/3)Tr(α)
    acf_aniso : (T,) — ⟨β(0):β(τ)⟩    where β = α - ᾱ·I  (traceless part)
    """
    T = len(alpha_traj)
    n_fft = _next_pow2(2 * T)
    norm = np.arange(T, 0, -1, dtype=float)

    alpha_iso = np.trace(alpha_traj, axis1=1, axis2=2) / 3.0  # (T,)
    ft = np.fft.rfft(alpha_iso, n=n_fft)
    acf_iso = np.fft.irfft(ft * np.conj(ft), n=n_fft)[:T] / norm

    beta = alpha_traj - alpha_iso[:, None, None] * np.eye(3)[None, :, :]
    beta_flat = beta.reshape(T, 9)
    acf_aniso = np.zeros(T)
    for k in range(9):
        ft = np.fft.rfft(beta_flat[:, k], n=n_fft)
        acf_aniso += np.fft.irfft(ft * np.conj(ft), n=n_fft)[:T]
    acf_aniso /= norm

    return acf_iso, acf_aniso


def raman_to_spectrum(acf_iso, acf_aniso, dt_fs,
                      window='hann', zero_pad=4):
    """Raman spectrum from isotropic/anisotropic polarizability ACFs.

    Returns
    -------
    freq_cm     : (F,)
    raman_par   : (F,) — I_∥  ∝ 45 ᾱ² + 4 γ²
    raman_perp  : (F,) — I_⊥  ∝ 3 γ²
    raman_total : (F,) — I_∥ + I_⊥
    """
    freq_cm, spec_iso = correlation_to_spectrum(
        acf_iso, dt_fs, window=window, zero_pad=zero_pad, qcf=None)
    _, spec_aniso = correlation_to_spectrum(
        acf_aniso, dt_fs, window=window, zero_pad=zero_pad, qcf=None)

    omega = np.where(freq_cm > 0, freq_cm, 0.0)
    raman_par   = (45.0 * spec_iso + 4.0 * spec_aniso) * omega
    raman_perp  = 3.0 * spec_aniso * omega
    raman_total = raman_par + raman_perp
    return freq_cm, raman_par, raman_perp, raman_total


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
    if window_frames > T:
        raise ValueError(
            f"--window-size ({window_frames}) exceeds trajectory length "
            f"({T} frames).  Use a smaller window or a longer trajectory.")
    starts = list(range(0, T - window_frames + 1, stride_frames))
    if len(starts) == 0:
        raise ValueError(
            f"No windows fit: window_frames={window_frames}, "
            f"stride={stride_frames}, trajectory={T} frames.")

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
        "window_size": 20,
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
        "raman": False,
        "trajectories": None,
        "batch_size": 32,
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
        g.add_argument("--trajectories", default=defaults["trajectories"],
                       help="Glob pattern for replica trajectories to average "
                            "(e.g. 'md_replica*.traj'). Spectra are averaged "
                            "over all matching files.")
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

        g = p.add_argument_group("Raman")
        g.add_argument("--raman", action="store_true",
                       help="Compute Raman spectrum from polarizability ACF "
                            "(GPU-batched dmu/dEf)")
        g.add_argument("--batch-size", type=int, default=defaults["batch_size"],
                       help="Frames per GPU batch for Raman polarizability "
                            "and batched dipole extraction")

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
            raman=args.raman,
            trajectories=args.trajectories,
            batch_size=args.batch_size,
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


def plot_raman(freq, raman_par, raman_perp, raman_total,
               n_frames, total_fs, path, n_traj=1):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    label = (f'{n_frames} frames, {total_fs:.0f} fs'
             + (f', {n_traj} replicas' if n_traj > 1 else ''))

    axes[0].plot(freq, raman_total, 'b-', lw=1)
    axes[0].fill_between(freq, raman_total, alpha=0.25)
    axes[0].set_ylabel('Total Raman (arb. u.)')
    axes[0].set_title(f'Raman — polarizability ACF  ({label})')

    axes[1].plot(freq, raman_par, 'r-', lw=1, label='I_∥')
    axes[1].fill_between(freq, raman_par, alpha=0.2, color='red')
    axes[1].set_ylabel('I_∥  (arb. u.)')
    axes[1].legend(frameon=False)

    axes[2].plot(freq, raman_perp, 'g-', lw=1, label='I_⊥')
    axes[2].fill_between(freq, raman_perp, alpha=0.2, color='green')
    axes[2].set_ylabel('I_⊥  (arb. u.)')
    axes[2].set_xlabel('Frequency (cm⁻¹)')
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_correlation_averaged(freq, ir, vcd, n_frames, total_fs,
                               path, n_traj=1):
    """Same as plot_correlation but with replica count in title."""
    label = (f'{n_frames} frames, {total_fs:.0f} fs'
             + (f', {n_traj} replicas' if n_traj > 1 else ''))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(freq, ir, 'b-', lw=1)
    ax1.fill_between(freq, ir, alpha=0.25)
    ax1.set_ylabel('IR intensity (arb. u.)')
    ax1.set_title(f'IR — dipole autocorrelation  ({label})')
    ax1.invert_xaxis()

    ax2.plot(freq, vcd, 'k-', lw=1)
    ax2.fill_between(freq, 0, vcd,
                     where=(vcd >= 0), color='red', alpha=0.3)
    ax2.fill_between(freq, 0, vcd,
                     where=(vcd < 0), color='blue', alpha=0.3)
    ax2.axhline(0, color='grey', lw=0.5)
    ax2.set_ylabel('VCD rot. strength (arb. u.)')
    ax2.set_title('VCD — dipole / magnetic-dipole cross-correlation')
    ax2.set_xlabel('Frequency (cm⁻¹)')
    ax2.invert_xaxis()

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

    # ---- Build trajectory list -----------------------------------------
    import glob as glob_mod

    traj_paths = [args.trajectory]
    if getattr(args, 'trajectories', None):
        extra = sorted(glob_mod.glob(args.trajectories))
        if extra:
            traj_paths = extra
    n_traj = len(traj_paths)

    print(f"\n  Trajectories : {n_traj}")
    for tp in traj_paths:
        print(f"    - {tp}")

    # ---- timestep (from first trajectory or CLI) -----------------------
    dt_fs = args.dt
    first_path = Path(traj_paths[0])
    is_hdf5 = first_path.suffix in ('.h5', '.hdf5')

    if dt_fs is None and is_hdf5:
        _, _, hdf5_dt, _ = load_hdf5_trajectory(first_path)
        if hdf5_dt is not None:
            dt_fs = hdf5_dt
    if dt_fs is None:
        dt_fs = 0.5
        print(f"  Timestep     : {dt_fs} fs  (default — use --dt to override)")
    else:
        print(f"  Timestep     : {dt_fs} fs")

    # ---- calculator ----------------------------------------------------
    print(f"\n  Loading calculator from {args.params} ...")
    calc = AseCalculatorEF(
        params_path=args.params, config_path=args.config,
        field_scale=args.field_scale,
    )

    # For Raman, we also need the raw model & params
    raman_model = raman_params = None
    if getattr(args, 'raman', False):
        from ase_calc_EF import load_params, load_config
        from training import MessagePassingModel

        params_path = Path(args.params)
        raman_params = load_params(params_path)
        config_path = args.config
        if config_path is None:
            if params_path.stem.startswith("params-") and len(params_path.stem) > 7:
                uuid_part = params_path.stem[7:]
                cand = params_path.parent / f"config-{uuid_part}.json"
                if cand.exists():
                    config_path = str(cand)
                elif (params_path.parent / "config.json").exists():
                    config_path = str(params_path.parent / "config.json")
        model_keys = {
            "features", "max_degree", "num_iterations",
            "num_basis_functions", "cutoff", "max_atomic_number",
            "include_pseudotensors", "dipole_field_coupling", "field_scale",
        }
        if config_path is not None:
            config = load_config(config_path)
            if "model" in config and isinstance(config["model"], dict):
                mc = {k: v for k, v in config["model"].items()
                      if k in model_keys}
            elif "model_config" in config:
                mc = {k: v for k, v in config["model_config"].items()
                      if k in model_keys}
            else:
                mc = {k: v for k, v in config.items() if k in model_keys}
        else:
            mc = dict(features=64, max_degree=2, num_iterations=2,
                      num_basis_functions=64, cutoff=10.0,
                      max_atomic_number=55, include_pseudotensors=True)
        raman_model = MessagePassingModel(**mc)

    # ================================================================
    # 1–2.  Extract properties + magnetic dipoles from each trajectory
    # ================================================================
    all_acfs, all_ccfs = [], []
    all_raman_iso, all_raman_aniso = [], []

    # Keep first trajectory's data for transient / 2D / harmonic
    first_dipoles = first_mag = None
    first_traj_frames = None
    T = N = 0

    for ti, tp in enumerate(traj_paths):
        tag = f"[{ti+1}/{n_traj}]" if n_traj > 1 else ""
        traj_path = Path(tp)
        is_hdf5_i = traj_path.suffix in ('.h5', '.hdf5')

        print(f"\n{tag} Loading {traj_path} ...")

        if is_hdf5_i:
            hdf5_pos, hdf5_vel, _, hdf5_meta = load_hdf5_trajectory(traj_path)
            Ti, N = hdf5_pos.shape[:2]
            traj_frames_i = None
        else:
            if traj_path.suffix == '.traj':
                traj_frames_i = list(Trajectory(str(traj_path)))
            else:
                traj_frames_i = ase.io.read(str(traj_path), index=':')
            Ti = len(traj_frames_i)
            N = len(traj_frames_i[0])
            hdf5_pos = hdf5_vel = hdf5_meta = None

        print(f"    Frames : {Ti},  Atoms : {N}")

        # ---- extract μ, q, r, v ----------------------------------------
        if is_hdf5_i:
            atomic_numbers = hdf5_meta.get("attr_atomic_numbers", None)
            if atomic_numbers is None:
                raise ValueError(
                    "HDF5 file has no 'atomic_numbers' attribute.")
            positions, velocities, dipoles, charges = extract_properties_hdf5(
                hdf5_pos, hdf5_vel, calc,
                atomic_numbers=np.asarray(atomic_numbers, dtype=int),
                recompute_dipole=args.recompute_dipole,
            )
        else:
            positions, velocities, dipoles, charges = extract_properties(
                traj_frames_i, calc=calc,
                recompute_dipole=args.recompute_dipole,
                recompute_charges=args.recompute_charges,
            )

        # ---- magnetic dipoles -------------------------------------------
        mag_dipoles = compute_magnetic_dipoles(positions, velocities, charges)
        mu_norm = np.linalg.norm(dipoles, axis=1)
        m_norm  = np.linalg.norm(mag_dipoles, axis=1)
        print(f"    |μ| : {mu_norm.min():.4f} – {mu_norm.max():.4f}")
        print(f"    |m| : {m_norm.min():.6f} – {m_norm.max():.6f}")

        # ---- IR / VCD correlation functions -----------------------------
        if args.method in ('correlation', 'both'):
            acf = autocorrelation(dipoles)
            ccf = cross_correlation(dipoles, mag_dipoles)
            all_acfs.append(acf)
            all_ccfs.append(ccf)

        # ---- Raman polarizability ACF -----------------------------------
        if getattr(args, 'raman', False) and raman_model is not None:
            Z_np = (np.asarray(atomic_numbers, dtype=int) if is_hdf5_i
                    else traj_frames_i[0].get_atomic_numbers())
            Ef_np = np.zeros(3, dtype=np.float32)
            if not is_hdf5_i and 'electric_field' in traj_frames_i[0].info:
                Ef_np = np.asarray(
                    traj_frames_i[0].info['electric_field'], dtype=np.float32)

            print(f"    Computing polarizability α(t) "
                  f"(batch_size={args.batch_size}) ...")
            alpha = compute_polarizability_batched(
                positions, Z_np, Ef_np,
                raman_model, raman_params,
                chunk_size=args.batch_size,
                field_scale=args.field_scale)

            acf_iso, acf_aniso = polarizability_autocorrelation(alpha)
            all_raman_iso.append(acf_iso)
            all_raman_aniso.append(acf_aniso)

        # ---- store first trajectory's data ------------------------------
        if ti == 0:
            T = Ti
            first_dipoles = dipoles
            first_mag = mag_dipoles
            first_traj_frames = traj_frames_i
            first_is_hdf5 = is_hdf5_i
            if is_hdf5_i:
                first_hdf5_pos = hdf5_pos
                first_hdf5_meta = hdf5_meta

    total_fs = (T - 1) * dt_fs
    print(f"\n  Frames per trajectory : {T}")
    print(f"  Total time            : {total_fs:.1f} fs  "
          f"({total_fs / 1000:.2f} ps)")

    # ================================================================
    # 3.  Averaged correlation spectra  (IR + VCD)
    # ================================================================
    if args.method in ('correlation', 'both') and all_acfs:
        min_len = min(len(a) for a in all_acfs)
        avg_acf = np.mean([a[:min_len] for a in all_acfs], axis=0)
        avg_ccf = np.mean([c[:min_len] for c in all_ccfs], axis=0)

        win = args.window_fn if args.window_fn != 'none' else None
        print(f"\n[3] Correlation spectra  (window={args.window_fn}, "
              f"zero-pad x{args.zero_pad}, {n_traj} trajectory(s)) ...")

        freq_cm, ir_spec = correlation_to_spectrum(
            avg_acf, dt_fs, window=win, zero_pad=args.zero_pad)
        _, vcd_spec = correlation_to_spectrum(
            avg_ccf, dt_fs, window=win, zero_pad=args.zero_pad)

        fm = _freq_mask(freq_cm, args.freq_min, args.freq_max)
        freq_p, ir_p, vcd_p = freq_cm[fm], ir_spec[fm], vcd_spec[fm]

        res = freq_cm[1] - freq_cm[0]
        print(f"    Resolution : {res:.2f} cm-1")

        np.savez(out / "correlation_spectra.npz",
                 freq_cm=freq_p, ir=ir_p, vcd=vcd_p,
                 acf=avg_acf, ccf=avg_ccf, n_trajectories=n_traj)
        print(f"    Data -> {out / 'correlation_spectra.npz'}")

        plot_correlation_averaged(freq_p, ir_p, vcd_p, T, total_fs,
                                  out / "correlation_spectra.png",
                                  n_traj=n_traj)
        print(f"    Plot -> {out / 'correlation_spectra.png'}")

    # ================================================================
    # 3b.  Averaged Raman spectrum
    # ================================================================
    if getattr(args, 'raman', False) and all_raman_iso:
        min_len = min(len(a) for a in all_raman_iso)
        avg_iso = np.mean([a[:min_len] for a in all_raman_iso], axis=0)
        avg_aniso = np.mean([a[:min_len] for a in all_raman_aniso], axis=0)

        win = args.window_fn if args.window_fn != 'none' else None
        print(f"\n[3b] Raman spectrum  ({n_traj} trajectory(s)) ...")

        freq_cm, raman_par, raman_perp, raman_total = raman_to_spectrum(
            avg_iso, avg_aniso, dt_fs,
            window=win, zero_pad=args.zero_pad)

        fm = _freq_mask(freq_cm, args.freq_min, args.freq_max)
        freq_r = freq_cm[fm]
        rp, rpp, rt = raman_par[fm], raman_perp[fm], raman_total[fm]

        np.savez(out / "raman_spectrum.npz",
                 freq_cm=freq_r, raman_parallel=rp,
                 raman_perpendicular=rpp, raman_total=rt,
                 acf_iso=avg_iso, acf_aniso=avg_aniso,
                 n_trajectories=n_traj)
        print(f"    Data -> {out / 'raman_spectrum.npz'}")

        plot_raman(freq_r, rp, rpp, rt, T, total_fs,
                   out / "raman_spectrum.png", n_traj=n_traj)
        print(f"    Plot -> {out / 'raman_spectrum.png'}")

    # ================================================================
    # 4.  Transient (windowed) spectra  — from first trajectory
    # ================================================================
    dipoles = first_dipoles
    mag_dipoles = first_mag

    if args.transient and dipoles is not None:
        ws = args.window_size
        if ws > T:
            print(f"\n  WARNING: --window-size {ws} > trajectory length {T}. "
                  f"Clamping to {T}.")
            ws = T
        print(f"\n[4] Transient spectra  (window={ws} frames, "
              f"stride={args.stride}) ...")

        t_cen, freq_t, ir_gram, vcd_gram = transient_spectra(
            dipoles, mag_dipoles, dt_fs,
            window_frames=ws,
            stride_frames=args.stride,
            fft_window=args.window_fn if args.window_fn != 'none' else None,
            zero_pad=min(args.zero_pad, 2),
        )

        fm = _freq_mask(freq_t, args.freq_min, args.freq_max)
        freq_tp = freq_t[fm]
        ir_gp   = ir_gram[:, fm]
        vcd_gp  = vcd_gram[:, fm]

        np.savez(out / "transient_spectra.npz",
                 time_fs=t_cen, freq_cm=freq_tp,
                 ir_spectrogram=ir_gp, vcd_spectrogram=vcd_gp)
        print(f"    Data -> {out / 'transient_spectra.npz'}")

        plot_transient(t_cen, freq_tp, ir_gp, vcd_gp,
                       out / "transient_spectra.png")
        print(f"    Plot -> {out / 'transient_spectra.png'}")

    # ================================================================
    # 4b.  Noda 2D generalised correlation  (from transient spectrograms)
    # ================================================================
    if args.noda:
        if not args.transient:
            print("\n  WARNING: --noda requires --transient  (skipping)")
        else:
            print(f"\n[4b] Noda 2D correlation spectroscopy ...")
            sync_ir, async_ir = noda_2d(ir_gp)
            plot_noda(freq_tp, sync_ir, async_ir, "IR",
                      out / "noda_2d_ir.png")
            np.savez(out / "noda_2d_ir.npz",
                     freq_cm=freq_tp, synchronous=sync_ir,
                     asynchronous=async_ir)
            print(f"    IR   -> {out / 'noda_2d_ir.png'}")

            sync_vcd, async_vcd = noda_2d(vcd_gp)
            plot_noda(freq_tp, sync_vcd, async_vcd, "VCD",
                      out / "noda_2d_vcd.png")
            np.savez(out / "noda_2d_vcd.npz",
                     freq_cm=freq_tp, synchronous=sync_vcd,
                     asynchronous=async_vcd)
            print(f"    VCD  -> {out / 'noda_2d_vcd.png'}")

    # ================================================================
    # 4c.  2D IR / VCD from STFT frequency–frequency correlation
    # ================================================================
    if args.spectra_2d and dipoles is not None:
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
            print(f"-> {out / f'2d_spectra_{tag}.png'}")

    # ================================================================
    # 5.  Harmonic snapshots  — from first trajectory
    # ================================================================
    traj_frames = first_traj_frames
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
            if first_is_hdf5:
                atoms = ase.Atoms(
                    numbers=np.asarray(
                        first_hdf5_meta["attr_atomic_numbers"], dtype=int),
                    positions=first_hdf5_pos[idx])
                atoms.info['electric_field'] = [0, 0, 0]
            else:
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
        print(f"    Data -> {out / 'harmonic_snapshots.npz'}")

        freq_ax = np.linspace(args.freq_min, args.freq_max, 2000)
        gamma   = args.broadening
        ir_sgram  = np.zeros((len(all_t), len(freq_ax)))
        vcd_sgram = np.zeros((len(all_t), len(freq_ax)))
        for i in range(len(all_t)):
            ir_sgram[i]  = broaden(freq_ax, all_freqs[i], all_ir[i], gamma)
            vcd_sgram[i] = broaden(freq_ax, all_freqs[i], all_vcd[i], gamma)

        plot_harmonic_snapshots(all_t, freq_ax, ir_sgram, vcd_sgram,
                                out / "harmonic_snapshots.png")
        print(f"    Plot -> {out / 'harmonic_snapshots.png'}")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

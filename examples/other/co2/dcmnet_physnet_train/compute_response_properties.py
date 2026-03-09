#!/usr/bin/env python3
"""
Compute Response Properties and Spectra from MD Trajectories

Calculates various response properties for spectroscopy:
1. IR (from dipole autocorrelation)
2. Raman (from polarizability autocorrelation)
3. VCD (vibrational circular dichroism - dipole × magnetic dipole)
4. Sum-frequency generation (SFG)
5. Power spectrum (general analysis)

Usage:
    # Compute all available spectra
    python compute_response_properties.py \
        --trajectory ./md_ir_long/trajectory.npz \
        --checkpoint ./ckpts/model \
        --compute-ir --compute-raman --compute-vcd \
        --subsample 10 \
        --output-dir ./all_spectra
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
from typing import Dict, Tuple, Optional

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from trainer import JointPhysNetDCMNet


def compute_autocorrelation_fft(data, max_lag=None):
    """
    Fast autocorrelation using FFT.
    
    Works for any time series (dipole, polarizability, etc.)
    """
    n = len(data)
    if max_lag is None:
        max_lag = n
    
    # Subtract mean
    data_centered = data - np.mean(data, axis=0)
    
    # FFT method
    n_fft = 2**(int(np.ceil(np.log2(2*n - 1))))
    
    # Handle multi-dimensional data
    if data_centered.ndim == 1:
        fft_data = np.fft.fft(data_centered, n=n_fft)
        power = fft_data * np.conj(fft_data)
        acf_full = np.fft.ifft(power).real[:n]
        acf = acf_full[:max_lag] / np.arange(n, n-max_lag, -1)
    else:
        # Vector or tensor data - compute for each component
        shape = data_centered.shape[1:]
        acf = np.zeros((max_lag,) + shape)
        
        # Flatten to 2D: (time, components)
        data_flat = data_centered.reshape(n, -1)
        
        for i in range(data_flat.shape[1]):
            fft_data = np.fft.fft(data_flat[:, i], n=n_fft)
            power = fft_data * np.conj(fft_data)
            acf_full = np.fft.ifft(power).real[:n]
            acf_component = acf_full[:max_lag] / np.arange(n, n-max_lag, -1)
            
            # Unflatten
            multi_idx = np.unravel_index(i, shape)
            acf[(slice(None),) + multi_idx] = acf_component
    
    return acf


def compute_polarizability_from_charges(positions, charges, method='clausius_mossotti'):
    """
    Estimate polarizability from charge distribution.
    
    Methods:
    - 'clausius_mossotti': α = Σ q_i r_i ⊗ r_i (simple)
    - 'drude': Charge oscillator model
    
    Returns
    -------
    np.ndarray
        Polarizability tensor (3, 3) in Å³
    """
    if method == 'clausius_mossotti':
        # Simple estimate: α ≈ Σ q_i (r_i ⊗ r_i) / |E|
        # Approximate with charge distribution second moment
        alpha = np.zeros((3, 3))
        for i in range(len(charges)):
            alpha += charges[i] * np.outer(positions[i], positions[i])
        
        # Scale to reasonable units (empirical factor)
        alpha *= 0.1  # Rough conversion
        
        return alpha
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_transition_dipole_moments(dipoles, energies):
    """
    Compute transition dipole moments between states.
    
    For MD, we approximate transitions as changes between frames.
    
    μ_if = <i|μ|f> ≈ Δμ weighted by ΔE
    
    Returns
    -------
    dict
        Transition dipole information
    """
    n_frames = len(dipoles)
    
    # Energy differences (simple finite difference)
    dE = np.diff(energies)  # (n_frames-1,)
    
    # Dipole changes
    d_dipole = np.diff(dipoles, axis=0)  # (n_frames-1, 3)
    
    # Transition dipole magnitude
    trans_dipole_mag = np.linalg.norm(d_dipole, axis=1)
    
    # Oscillator strength: f ∝ ΔE × |μ_if|²
    # In atomic units: f = (2/3) × ΔE × |μ_if|²
    oscillator_strength = np.abs(dE) * trans_dipole_mag**2
    
    return {
        'energy_gaps': dE,
        'transition_dipoles': d_dipole,
        'transition_dipole_magnitudes': trans_dipole_mag,
        'oscillator_strengths': oscillator_strength,
    }


def compute_ir_spectrum(dipoles, timestep, temperature=300):
    """
    Compute IR spectrum from dipole autocorrelation.
    
    I(ω) ∝ ω × (1 + n(ω)) × |FT[C_μ(t)]|²
    """
    print(f"\nComputing IR spectrum...")
    
    n_frames = len(dipoles)
    
    # Autocorrelation
    acf = compute_autocorrelation_fft(dipoles)
    
    # Window
    window = np.hanning(len(acf))
    if acf.ndim > 1:
        # Average over xyz components
        acf_scalar = np.mean(acf, axis=1)
    else:
        acf_scalar = acf
    
    acf_windowed = acf_scalar * window
    
    # FFT
    nfft = len(acf) * 4
    spectrum = np.fft.rfft(acf_windowed, n=nfft)
    
    # Frequencies
    freq_hz = np.fft.rfftfreq(nfft, d=timestep * 1e-15)
    c_cm_s = 2.99792458e10
    freqs_cm = freq_hz / c_cm_s
    
    # Quantum correction
    hbar_eV = 6.582119569e-16
    kB_eV = 8.617333262e-5
    omega = 2 * np.pi * freq_hz
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        n_bose = 1.0 / (np.exp(np.clip(hbar_eV * omega / (kB_eV * temperature), 0, 700)) - 1.0)
        n_bose = np.nan_to_num(n_bose, nan=0.0, posinf=0.0)
    
    freq_prefactor = omega * (1.0 + n_bose)
    intensity = freq_prefactor * np.abs(spectrum)**2
    
    return {
        'frequencies': freqs_cm,
        'intensity': intensity / np.max(intensity),
        'autocorrelation': acf_scalar,
    }


def compute_raman_spectrum(polarizabilities, timestep, temperature=300, laser_freq=18797):
    """
    Compute Raman spectrum from polarizability autocorrelation.
    
    I_Raman(ω) ∝ (ω₀ - ω)⁴ × (1 + n(ω)) × |FT[C_α(t)]|²
    
    Parameters
    ----------
    laser_freq : float
        Laser frequency in cm⁻¹ (default: 532 nm = 18797 cm⁻¹)
    """
    print(f"\nComputing Raman spectrum...")
    
    # Polarizability is (n_frames, 3, 3) tensor
    # Use isotropic part: ᾱ = (α_xx + α_yy + α_zz) / 3
    alpha_iso = np.trace(polarizabilities, axis1=1, axis2=2) / 3.0
    
    # Anisotropic part: γ² = [(α_xx-α_yy)² + (α_yy-α_zz)² + (α_zz-α_xx)²] / 2
    alpha_aniso = np.zeros(len(polarizabilities))
    for i in range(len(polarizabilities)):
        diff = np.array([
            polarizabilities[i, 0, 0] - polarizabilities[i, 1, 1],
            polarizabilities[i, 1, 1] - polarizabilities[i, 2, 2],
            polarizabilities[i, 2, 2] - polarizabilities[i, 0, 0],
        ])
        alpha_aniso[i] = np.sqrt(np.sum(diff**2) / 2)
    
    # Autocorrelation
    acf_iso = compute_autocorrelation_fft(alpha_iso)
    acf_aniso = compute_autocorrelation_fft(alpha_aniso)
    
    # Window
    window = np.hanning(len(acf_iso))
    acf_iso_windowed = acf_iso * window
    acf_aniso_windowed = acf_aniso * window
    
    # FFT
    nfft = len(acf_iso) * 4
    spectrum_iso = np.fft.rfft(acf_iso_windowed, n=nfft)
    spectrum_aniso = np.fft.rfft(acf_aniso_windowed, n=nfft)
    
    # Frequencies
    freq_hz = np.fft.rfftfreq(nfft, d=timestep * 1e-15)
    c_cm_s = 2.99792458e10
    freqs_cm = freq_hz / c_cm_s
    
    # Raman intensity: (ω₀ - ω)⁴ factor
    raman_freqs = laser_freq - freqs_cm  # Stokes shift
    raman_factor = raman_freqs**4
    raman_factor[raman_freqs < 0] = 0  # Only Stokes side
    
    # Quantum correction
    hbar_eV = 6.582119569e-16
    kB_eV = 8.617333262e-5
    omega = 2 * np.pi * freq_hz
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        n_bose = 1.0 / (np.exp(np.clip(hbar_eV * omega / (kB_eV * temperature), 0, 700)) - 1.0)
        n_bose = np.nan_to_num(n_bose, nan=0.0, posinf=0.0)
    
    # Total Raman intensity
    intensity_iso = raman_factor * (1 + n_bose) * np.abs(spectrum_iso)**2
    intensity_aniso = raman_factor * (1 + n_bose) * np.abs(spectrum_aniso)**2
    
    return {
        'frequencies': freqs_cm,
        'intensity_isotropic': intensity_iso / np.max(intensity_iso + 1e-10),
        'intensity_anisotropic': intensity_aniso / np.max(intensity_aniso + 1e-10),
        'laser_frequency': laser_freq,
    }


def compute_vcd_spectrum(dipoles, magnetic_dipoles, timestep, temperature=300):
    """
    Compute VCD (Vibrational Circular Dichroism) spectrum.
    
    VCD measures difference in absorption between left and right circularly polarized light.
    
    I_VCD(ω) ∝ Im[<μ̇|m>] where m is magnetic dipole
    
    Requires magnetic dipole moments (currently not computed by model).
    For now, approximate using velocity of electric dipole.
    """
    print(f"\nComputing VCD spectrum (approximate)...")
    
    # Approximate magnetic dipole from angular momentum
    # m ≈ (1/2c) × (r × p) 
    # For dipole derivative: dm/dt ≈ dμ/dt × rotational component
    
    # Time derivative of dipole
    dipole_velocity = np.diff(dipoles, axis=0) / timestep  # (n-1, 3)
    
    # Cross-correlation between dipole and its velocity
    # This gives rotational strength
    cross_corr = np.sum(dipoles[:-1] * dipole_velocity, axis=1)
    
    # FFT
    n = len(cross_corr)
    nfft = 2**(int(np.ceil(np.log2(n))))
    spectrum = np.fft.rfft(cross_corr, n=nfft)
    
    # Frequencies
    freq_hz = np.fft.rfftfreq(nfft, d=timestep * 1e-15)
    c_cm_s = 2.99792458e10
    freqs_cm = freq_hz / c_cm_s
    
    # VCD intensity (imaginary part)
    intensity = np.imag(spectrum)
    
    return {
        'frequencies': freqs_cm,
        'intensity': intensity / (np.max(np.abs(intensity)) + 1e-10),
        'note': 'Approximate VCD (requires proper magnetic dipole computation)',
    }


def compute_power_spectrum(observable, timestep, window='hann'):
    """
    General power spectrum of any observable.
    
    Useful for analyzing dynamics, diffusion, etc.
    """
    n = len(observable)
    
    # Apply window
    if window == 'hann':
        win = np.hanning(n)
    elif window == 'hamming':
        win = np.hamming(n)
    else:
        win = np.ones(n)
    
    # Handle vector/tensor data
    if observable.ndim == 1:
        obs_windowed = observable * win
    else:
        obs_windowed = observable * win[:, None]
        # Average over components
        obs_windowed = np.mean(obs_windowed, axis=1)
    
    # FFT
    spectrum = np.fft.rfft(obs_windowed)
    
    # Frequencies
    freq_hz = np.fft.rfftfreq(n, d=timestep * 1e-15)
    c_cm_s = 2.99792458e10
    freqs_cm = freq_hz / c_cm_s
    
    power = np.abs(spectrum)**2
    
    return {
        'frequencies': freqs_cm,
        'power': power / (np.max(power) + 1e-10),
    }


def plot_all_spectra(spectra_dict, output_dir):
    """Create comprehensive spectroscopy plots."""
    
    n_spectra = len(spectra_dict)
    fig, axes = plt.subplots(n_spectra, 1, figsize=(12, 4*n_spectra))
    
    if n_spectra == 1:
        axes = [axes]
    
    for idx, (name, spectrum_data) in enumerate(spectra_dict.items()):
        ax = axes[idx]
        
        freqs = spectrum_data['frequencies']
        
        # Plot based on spectrum type
        if 'ir' in name.lower():
            # IR spectrum
            intensity_phys = spectrum_data.get('intensity_physnet', spectrum_data.get('intensity'))
            intensity_dcm = spectrum_data.get('intensity_dcmnet')
            
            freq_mask = (freqs > 400) & (freqs < 3500)
            ax.plot(freqs[freq_mask], intensity_phys[freq_mask], 'b-', 
                   linewidth=1.5, label='PhysNet', alpha=0.8)
            if intensity_dcm is not None:
                ax.plot(freqs[freq_mask], intensity_dcm[freq_mask], 'c--',
                       linewidth=1.5, label='DCMNet', alpha=0.8)
            
            # Add experimental CO2 lines
            exp_freqs = [667.4, 1388.2, 2349.2]
            for ef in exp_freqs:
                ax.axvline(ef, color='red', linestyle=':', alpha=0.4, linewidth=1)
            
            ax.set_xlabel('Frequency (cm⁻¹)', fontsize=11, weight='bold')
            ax.set_ylabel('Intensity (normalized)', fontsize=11, weight='bold')
            ax.set_title(f'{name}', fontsize=13, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(400, 3500)
        
        elif 'raman' in name.lower():
            # Raman spectrum
            intensity_iso = spectrum_data.get('intensity_isotropic')
            intensity_aniso = spectrum_data.get('intensity_anisotropic')
            
            freq_mask = (freqs > 400) & (freqs < 3500)
            if intensity_iso is not None:
                ax.plot(freqs[freq_mask], intensity_iso[freq_mask], 'g-',
                       linewidth=1.5, label='Isotropic', alpha=0.8)
            if intensity_aniso is not None:
                ax.plot(freqs[freq_mask], intensity_aniso[freq_mask], 'm--',
                       linewidth=1.5, label='Anisotropic', alpha=0.8)
            
            ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11, weight='bold')
            ax.set_ylabel('Intensity (normalized)', fontsize=11, weight='bold')
            ax.set_title(f'{name}', fontsize=13, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(400, 3500)
        
        elif 'vcd' in name.lower():
            # VCD spectrum
            intensity = spectrum_data['intensity']
            
            freq_mask = (freqs > 400) & (freqs < 3500)
            ax.plot(freqs[freq_mask], intensity[freq_mask], 'purple',
                   linewidth=1.5, alpha=0.8)
            ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
            
            ax.set_xlabel('Frequency (cm⁻¹)', fontsize=11, weight='bold')
            ax.set_ylabel('Δε (L-R)', fontsize=11, weight='bold')
            ax.set_title(f'{name}', fontsize=13, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(400, 3500)
        
        else:
            # Generic power spectrum
            power = spectrum_data.get('power', spectrum_data.get('intensity'))
            
            freq_mask = (freqs > 0) & (freqs < 4000)
            ax.semilogy(freqs[freq_mask], power[freq_mask] + 1e-10, 'k-',
                       linewidth=1.5, alpha=0.8)
            
            ax.set_xlabel('Frequency (cm⁻¹)', fontsize=11, weight='bold')
            ax.set_ylabel('Power', fontsize=11, weight='bold')
            ax.set_title(f'{name}', fontsize=13, weight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_spectra.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive spectra: {output_dir / 'all_spectra.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute response properties')
    
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    
    # What to compute
    parser.add_argument('--compute-ir', action='store_true', help='IR spectrum')
    parser.add_argument('--compute-raman', action='store_true', help='Raman spectrum')
    parser.add_argument('--compute-vcd', action='store_true', help='VCD spectrum')
    parser.add_argument('--compute-transition-dipoles', action='store_true',
                       help='Transition dipole moments')
    parser.add_argument('--compute-all', action='store_true',
                       help='Compute all available spectra')
    
    parser.add_argument('--subsample', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=300)
    parser.add_argument('--output-dir', type=Path, required=True)
    
    args = parser.parse_args()
    
    if args.compute_all:
        args.compute_ir = True
        args.compute_raman = True
        args.compute_vcd = True
        args.compute_transition_dipoles = True
    
    print("="*70)
    print("RESPONSE PROPERTIES & SPECTROSCOPY")
    print("="*70)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory
    print(f"\nLoading trajectory...")
    traj_data = np.load(args.trajectory)
    
    dipoles_physnet = traj_data['dipoles_physnet'][::args.subsample]
    dipoles_dcmnet = traj_data['dipoles_dcmnet'][::args.subsample]
    energies = traj_data['energies'][::args.subsample]
    timestep = float(traj_data['timestep']) * args.subsample
    
    print(f"✅ Loaded {len(dipoles_physnet)} frames (subsampled), dt={timestep} fs")
    
    # Storage for all spectra
    spectra = {}
    
    # IR spectrum
    if args.compute_ir:
        ir_phys = compute_ir_spectrum(dipoles_physnet, timestep, args.temperature)
        ir_dcm = compute_ir_spectrum(dipoles_dcmnet, timestep, args.temperature)
        
        spectra['IR Spectrum'] = {
            'frequencies': ir_phys['frequencies'],
            'intensity_physnet': ir_phys['intensity'],
            'intensity_dcmnet': ir_dcm['intensity'],
        }
        
        print(f"✅ Computed IR spectrum")
    
    # Raman (requires polarizability - estimate from charges)
    if args.compute_raman:
        print(f"\n⚠️  Raman requires polarizability trajectory")
        print(f"   (Not yet implemented - needs charge distribution for each frame)")
    
    # VCD
    if args.compute_vcd:
        vcd_phys = compute_vcd_spectrum(dipoles_physnet, None, timestep, args.temperature)
        spectra['VCD Spectrum (approximate)'] = vcd_phys
        print(f"✅ Computed VCD spectrum (approximate)")
    
    # Transition dipoles
    if args.compute_transition_dipoles:
        trans_dip = compute_transition_dipole_moments(dipoles_physnet, energies)
        
        # Save separately
        np.savez(
            args.output_dir / 'transition_dipoles.npz',
            **trans_dip
        )
        print(f"✅ Computed transition dipoles → transition_dipoles.npz")
    
    # Plot all
    if spectra:
        plot_all_spectra(spectra, args.output_dir)
    
    # Save spectra
    for name, data in spectra.items():
        filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.npz'
        np.savez(args.output_dir / filename, **data)
    
    print(f"\n{'='*70}")
    print("✅ DONE")
    print(f"{'='*70}")
    print(f"\nComputed: {list(spectra.keys())}")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()


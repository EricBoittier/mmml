#!/usr/bin/env python3
"""
Comprehensive IR and Raman Spectrum Analysis Script

Analyzes MD-based IR and Raman spectra and compares with experimental data.
Performs automated fitting with frequency shifts, intensity scaling, and Gaussian broadening.

Usage:
    python analyze_ir_spectrum.py --md-spectrum md_ir_spectrum.npz \
                                   --exp-spectrum experimental_co2_ir.npz \
                                   --output-dir ./ir_analysis \
                                   --smooth-window 5 \
                                   --include-raman
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import argparse
from pathlib import Path


# Okabe-Ito colorblind-friendly palette
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9', 
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'black': '#000000'
}

# Experimental CO2 frequencies and activities
# IR active: ν₂ (bend), ν₃ (antisym stretch), combinations
# Raman active: ν₁ (sym stretch), ν₂ (bend - weak), combinations
EXP_CO2_PEAKS = [
    ('ν₂ Bending', 667.4, 'both'),           # IR + Raman (weak)
    ('ν₁ Symmetric Stretch', 1388.2, 'raman'),  # Raman only (strong)
    ('ν₃ Asymmetric Stretch', 2349.2, 'ir'),    # IR only (strong)
    ('2ν₂ + ν₃ Combination', 3715.0, 'both')    # Both
]

# For IR-only analysis
EXP_CO2_IR_PEAKS = [(name, freq) for name, freq, activity in EXP_CO2_PEAKS]

# For Raman-only analysis (Raman-active modes)
EXP_CO2_RAMAN_PEAKS = [
    ('ν₁ Symmetric Stretch', 1388.2),  # Strong
    ('ν₂ Bending', 667.4),             # Weak
]


def rolling_average(data, window_size=10):
    """Apply rolling average smoothing"""
    if window_size <= 1:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def apply_gaussian_broadening(frequencies, intensities, sigma=20.0, freq_grid=None):
    """
    Broaden stick spectrum with Gaussian lineshapes.
    
    Parameters
    ----------
    frequencies : array
        Peak frequencies (cm⁻¹)
    intensities : array
        Peak intensities
    sigma : float
        Gaussian width (FWHM = 2.355 * sigma) in cm⁻¹
    freq_grid : array, optional
        Output frequency grid
    
    Returns
    -------
    freq_grid, broadened : arrays
    """
    if freq_grid is None:
        freq_grid = np.linspace(frequencies.min() - 100, frequencies.max() + 100, 2000)
    
    broadened = np.zeros_like(freq_grid)
    
    for freq, intensity in zip(frequencies, intensities):
        broadened += intensity * np.exp(-0.5 * ((freq_grid - freq) / sigma)**2)
    
    return freq_grid, broadened


def fit_individual_peak(calc_freq, calc_int, exp_freq, exp_int, 
                        peak_center, window=150, smooth_window=5):
    """
    Fit a single peak region by optimizing shift and Gaussian width.
    """
    # Define region
    freq_min = peak_center - window
    freq_max = peak_center + window
    
    # Mask data
    exp_mask = (exp_freq >= freq_min) & (exp_freq <= freq_max)
    calc_mask = (calc_freq >= freq_min) & (calc_freq <= freq_max)
    
    exp_f = exp_freq[exp_mask]
    exp_i = exp_int[exp_mask]
    calc_f = calc_freq[calc_mask]
    calc_i_raw = calc_int[calc_mask]
    
    # Apply rolling average to smooth calculated data
    if smooth_window > 1 and len(calc_i_raw) >= smooth_window:
        calc_i = np.convolve(calc_i_raw, np.ones(smooth_window)/smooth_window, mode='same')
    else:
        calc_i = calc_i_raw
    
    if len(calc_f) == 0 or len(exp_f) == 0:
        return None
    
    # Interpolate experimental
    exp_interp = interp1d(exp_f, exp_i, bounds_error=False, fill_value=0.0)
    
    def objective(params):
        shift, sigma = params
        
        # Apply shift and broaden
        shifted = calc_f + shift
        grid = np.linspace(freq_min, freq_max, 500)
        _, broadened = apply_gaussian_broadening(shifted, calc_i, sigma, grid)
        
        # Scale to match peak height
        exp_on_grid = exp_interp(grid)
        if broadened.max() > 0 and exp_on_grid.max() > 0:
            scale = exp_on_grid.max() / broadened.max()
            broadened_scaled = broadened * scale
            mse = np.mean((broadened_scaled - exp_on_grid)**2)
        else:
            mse = 1e10
        
        return mse
    
    # Optimize
    x0 = [0.0, 10.0]  # Start with narrower peaks
    bounds = [(-100.0, 100.0), (2.0, 30.0)]  # Reduced broadening
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    shift, sigma = result.x
    
    # Generate fitted spectrum
    shifted = calc_f + shift
    grid = np.linspace(freq_min, freq_max, 500)
    _, broadened = apply_gaussian_broadening(shifted, calc_i, sigma, grid)
    
    # Scale to match peak height
    exp_on_grid = exp_interp(grid)
    if broadened.max() > 0 and exp_on_grid.max() > 0:
        scale = exp_on_grid.max() / broadened.max()
    else:
        scale = 1.0
    
    broadened_scaled = broadened * scale
    
    return {
        'shift': shift,
        'sigma': sigma,
        'fwhm': 2.355 * sigma,
        'scale': scale,
        'mse': result.fun,
        'grid': grid,
        'broadened': broadened_scaled,
        'exp_grid': exp_on_grid,
        'peak_center': peak_center,
        'window': window
    }


def plot_overview(freqs, int_phys, int_dcm, exp_freq, exp_abs, exp_peaks, output_dir):
    """Create overview figure with experimental and calculated spectra"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('MD-Based IR Spectra', fontsize=16, weight='bold', y=0.995)
    
    # Plot 1: Full range
    ax = axes[0]
    mask_full = (freqs > 300) & (freqs < 4000)
    ax.fill_between(freqs[mask_full], 0, int_phys[mask_full]/int_phys[mask_full].max(), 
                    color=COLORS['blue'], alpha=0.3, label='PhysNet')
    ax.plot(freqs[mask_full], int_phys[mask_full]/int_phys[mask_full].max(), 
            color=COLORS['blue'], lw=1.5, alpha=0.9)
    ax.fill_between(freqs[mask_full], 0, int_dcm[mask_full]/int_dcm[mask_full].max(), 
                    color=COLORS['orange'], alpha=0.2, label='DCMNet')
    ax.plot(freqs[mask_full], int_dcm[mask_full]/int_dcm[mask_full].max(), 
            color=COLORS['orange'], lw=1.5, ls='--', alpha=0.9)
    
    for name, freq in exp_peaks:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.7)
        ax.text(freq, ax.get_ylim()[1]*0.92, name.split()[0], rotation=90, 
                va='bottom', fontsize=9, color=COLORS['vermillion'], weight='bold')
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (arb. units)', fontsize=12, weight='bold')
    ax.set_title('Full Spectrum', fontsize=13, weight='bold', loc='left')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(300, 4000)
    
    # Plot 2: With experimental comparison
    ax = axes[1]
    exp_mask = (exp_freq >= 300) & (exp_freq <= 4000)
    ax.fill_between(exp_freq[exp_mask], 0, exp_abs[exp_mask]/exp_abs[exp_mask].max(),
                    color=COLORS['black'], alpha=0.15, label='Experimental', zorder=3)
    ax.plot(exp_freq[exp_mask], exp_abs[exp_mask]/exp_abs[exp_mask].max(),
            color=COLORS['black'], lw=2, alpha=0.8, zorder=3)
    ax.fill_between(freqs[mask_full], 0, int_phys[mask_full]/int_phys[mask_full].max(),
                    color=COLORS['blue'], alpha=0.25, label='PhysNet (MD)', zorder=2)
    ax.plot(freqs[mask_full], int_phys[mask_full]/int_phys[mask_full].max(),
            color=COLORS['blue'], lw=1.5, alpha=0.9, zorder=2)
    ax.fill_between(freqs[mask_full], 0, int_dcm[mask_full]/int_dcm[mask_full].max(),
                    color=COLORS['orange'], alpha=0.2, label='DCMNet (MD)', zorder=1)
    ax.plot(freqs[mask_full], int_dcm[mask_full]/int_dcm[mask_full].max(),
            color=COLORS['orange'], lw=1.5, ls='--', alpha=0.9, zorder=1)
    
    for name, freq in exp_peaks:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('Comparison with Experimental', fontsize=13, weight='bold', loc='left')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(300, 4000)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overview_spectra.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'overview_spectra.png'}")
    plt.close()


def plot_individual_peaks(peak_fits_physnet, peak_fits_dcmnet, output_dir):
    """Plot individual peak fits in a grid"""
    n_peaks = len(peak_fits_physnet)
    fig, axes = plt.subplots(n_peaks, 2, figsize=(16, 5*n_peaks))
    fig.suptitle('Individual Peak Line Shape Comparison', 
                 fontsize=16, weight='bold', y=0.998)
    
    if n_peaks == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ((name_p, fit_p), (name_d, fit_d)) in enumerate(zip(peak_fits_physnet, peak_fits_dcmnet)):
        # PhysNet
        ax = axes[idx, 0]
        ax.fill_between(fit_p['grid'], 0, fit_p['exp_grid'], 
                        color=COLORS['black'], alpha=0.15, label='Experimental')
        ax.plot(fit_p['grid'], fit_p['exp_grid'], 
                color=COLORS['black'], lw=2, alpha=0.8)
        ax.fill_between(fit_p['grid'], 0, fit_p['broadened'],
                        color=COLORS['blue'], alpha=0.3, label='PhysNet (fitted)')
        ax.plot(fit_p['grid'], fit_p['broadened'],
                color=COLORS['blue'], lw=1.5, alpha=0.9)
        ax.axvline(fit_p['peak_center'], color=COLORS['vermillion'], 
                   ls=':', lw=1.5, alpha=0.7, label='Expected')
        ax.axvline(fit_p['peak_center'] + fit_p['shift'], 
                   color=COLORS['blue'], ls='--', lw=1.5, alpha=0.5, label='Fitted')
        
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, weight='bold')
        ax.set_ylabel('Intensity (abs. units)', fontsize=12, weight='bold')
        ax.set_title(f'{name_p} - PhysNet\nshift={fit_p["shift"]:+.1f} cm⁻¹, FWHM={fit_p["fwhm"]:.1f} cm⁻¹',
                     fontsize=12, weight='bold', loc='left')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, ls='--')
        ax.set_xlim(fit_p['peak_center'] - fit_p['window'], 
                    fit_p['peak_center'] + fit_p['window'])
        
        # DCMNet
        ax = axes[idx, 1]
        ax.fill_between(fit_d['grid'], 0, fit_d['exp_grid'],
                        color=COLORS['black'], alpha=0.15, label='Experimental')
        ax.plot(fit_d['grid'], fit_d['exp_grid'],
                color=COLORS['black'], lw=2, alpha=0.8)
        ax.fill_between(fit_d['grid'], 0, fit_d['broadened'],
                        color=COLORS['orange'], alpha=0.25, label='DCMNet (fitted)')
        ax.plot(fit_d['grid'], fit_d['broadened'],
                color=COLORS['orange'], lw=1.5, ls='--', alpha=0.9)
        ax.axvline(fit_d['peak_center'], color=COLORS['vermillion'],
                   ls=':', lw=4.5, alpha=0.7, label='Expected')
        ax.axvline(fit_d['peak_center'] + fit_d['shift'],
                   color=COLORS['orange'], ls='--', lw=1.5, alpha=0.5, label='Fitted')
        
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, weight='bold')
        ax.set_ylabel('Intensity (abs. units)', fontsize=12, weight='bold')
        ax.set_title(f'{name_d} - DCMNet\nshift={fit_d["shift"]:+.1f} cm⁻¹, FWHM={fit_d["fwhm"]:.1f} cm⁻¹',
                     fontsize=12, weight='bold', loc='left')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, ls='--')
        ax.set_xlim(fit_d['peak_center'] - fit_d['window'],
                    fit_d['peak_center'] + fit_d['window'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_peaks.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'individual_peaks.png'}")
    plt.close()


def plot_zoomed_grid(exp_co2, peak_fits_physnet, peak_fits_dcmnet, freqs, int_phys, int_dcm, output_dir):
    """Plot zoomed view of all peaks in a 2x2 grid"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Zoomed Peak Comparisons with Frequency Markers', 
                 fontsize=16, weight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, ((name, exp_freq), (name_p, fit_p), (name_d, fit_d)) in enumerate(zip(exp_co2, peak_fits_physnet, peak_fits_dcmnet)):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Plot data
        ax.fill_between(fit_p['grid'], 0, fit_p['exp_grid'],
                        color=COLORS['black'], alpha=0.15, zorder=1)
        ax.plot(fit_p['grid'], fit_p['exp_grid'],
                color=COLORS['black'], lw=2, label='Experimental', alpha=0.8, zorder=4)
        ax.fill_between(fit_p['grid'], 0, fit_p['broadened'],
                        color=COLORS['blue'], alpha=0.3, zorder=2)
        ax.plot(fit_p['grid'], fit_p['broadened'],
                color=COLORS['blue'], lw=1.5, label='PhysNet', alpha=0.9, zorder=3)
        ax.fill_between(fit_d['grid'], 0, fit_d['broadened'],
                        color=COLORS['orange'], alpha=0.25, zorder=1)
        ax.plot(fit_d['grid'], fit_d['broadened'],
                color=COLORS['orange'], lw=1.5, ls='--', label='DCMNet', alpha=0.9, zorder=2)
        
        # Mark frequencies
        ax.axvline(exp_freq, color=COLORS['vermillion'], ls=':', lw=2, 
                   alpha=0.7, label=f'Exp: {exp_freq:.1f}', zorder=4)
        
        # Find original and final positions
        peak_center = fit_p['peak_center']
        window = fit_p['window']
        mask = (freqs >= peak_center - window) & (freqs <= peak_center + window)
        if mask.sum() > 0:
            orig_peak_p = freqs[mask][np.argmax(int_phys[mask])]
            orig_peak_d = freqs[mask][np.argmax(int_dcm[mask])]
            
            final_p = orig_peak_p + fit_p['shift']
            final_d = orig_peak_d + fit_d['shift']
            
            ax.axvline(final_p, color=COLORS['blue'], ls='--', lw=1.5, alpha=0.4, zorder=0)
            ax.axvline(final_d, color=COLORS['orange'], ls='--', lw=1.5, alpha=0.4, zorder=0)
            
            # Add error text
            error_p = final_p - exp_freq
            error_d = final_d - exp_freq
            ax.text(0.02, 0.98, f'PhysNet: {error_p:+.1f} cm⁻¹\nDCMNet: {error_d:+.1f} cm⁻¹',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11, weight='bold')
        ax.set_ylabel('Intensity', fontsize=11, weight='bold')
        ax.set_title(f'{name}\nExp: {exp_freq:.1f} cm⁻¹', 
                    fontsize=12, weight='bold', loc='left')
        ax.legend(fontsize=9, framealpha=0.9, loc='best')
        ax.grid(True, alpha=0.3, ls='--')
        ax.set_xlim(peak_center - window, peak_center + window)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'zoomed_peaks_grid.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'zoomed_peaks_grid.png'}")
    plt.close()


def plot_combined_spectrum(peak_fits_physnet, peak_fits_dcmnet, exp_freq, exp_abs, exp_co2, output_dir):
    """Plot final combined spectrum"""
    # Build full fitted spectrum
    freq_grid_full = np.linspace(400, 4000, 3000)
    spectrum_physnet_full = np.zeros_like(freq_grid_full)
    spectrum_dcmnet_full = np.zeros_like(freq_grid_full)
    
    for name, fit_p in peak_fits_physnet:
        peak_interp = interp1d(fit_p['grid'], fit_p['broadened'], 
                              bounds_error=False, fill_value=0.0)
        spectrum_physnet_full += peak_interp(freq_grid_full)
    
    for name, fit_d in peak_fits_dcmnet:
        peak_interp = interp1d(fit_d['grid'], fit_d['broadened'],
                              bounds_error=False, fill_value=0.0)
        spectrum_dcmnet_full += peak_interp(freq_grid_full)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Final Combined Spectrum: All Peaks Fitted', 
                 fontsize=16, weight='bold', y=0.995)
    
    # Full range
    ax = axes[0]
    exp_mask = (exp_freq >= 400) & (exp_freq <= 4000)
    ax.fill_between(exp_freq[exp_mask], 0, exp_abs[exp_mask] / exp_abs[exp_mask].max(),
                    color=COLORS['black'], alpha=0.15, zorder=1)
    ax.plot(exp_freq[exp_mask], exp_abs[exp_mask] / exp_abs[exp_mask].max(),
            color=COLORS['black'], lw=2, label='Experimental', alpha=0.8, zorder=4)
    ax.fill_between(freq_grid_full, 0, spectrum_physnet_full / spectrum_physnet_full.max(),
                    color=COLORS['blue'], alpha=0.3, zorder=2)
    ax.plot(freq_grid_full, spectrum_physnet_full / spectrum_physnet_full.max(),
            color=COLORS['blue'], lw=1.5, label='PhysNet (all peaks fitted)', alpha=0.9, zorder=3)
    ax.fill_between(freq_grid_full, 0, spectrum_dcmnet_full / spectrum_dcmnet_full.max(),
                    color=COLORS['orange'], alpha=0.25, zorder=1)
    ax.plot(freq_grid_full, spectrum_dcmnet_full / spectrum_dcmnet_full.max(),
            color=COLORS['orange'], lw=1.5, ls='--', label='DCMNet (all peaks fitted)', alpha=0.9, zorder=2)
    
    for name, freq in exp_co2:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
        ax.text(freq, ax.get_ylim()[1]*0.95, name.split()[0], 
               rotation=90, va='bottom', ha='right',
               fontsize=8, color=COLORS['vermillion'], weight='bold')
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('Full Spectrum (400-4000 cm⁻¹)', fontsize=13, weight='bold', loc='left')
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(400, 4000)
    ax.set_ylim(0, 1.05)
    
    # Zoomed to fundamentals
    ax = axes[1]
    zoom_mask = (exp_freq >= 400) & (exp_freq <= 2600)
    grid_mask = (freq_grid_full >= 400) & (freq_grid_full <= 2600)
    
    ax.fill_between(exp_freq[zoom_mask], 0, exp_abs[zoom_mask] / exp_abs[zoom_mask].max(),
                    color=COLORS['black'], alpha=0.15, zorder=1)
    ax.plot(exp_freq[zoom_mask], exp_abs[zoom_mask] / exp_abs[zoom_mask].max(),
            color=COLORS['black'], lw=2, label='Experimental', alpha=0.8, zorder=4)
    ax.fill_between(freq_grid_full[grid_mask], 0,
                    spectrum_physnet_full[grid_mask] / spectrum_physnet_full[grid_mask].max(),
                    color=COLORS['blue'], alpha=0.3, zorder=2)
    ax.plot(freq_grid_full[grid_mask], 
            spectrum_physnet_full[grid_mask] / spectrum_physnet_full[grid_mask].max(),
            color=COLORS['blue'], lw=1.5, label='PhysNet (fitted)', alpha=0.9, zorder=3)
    ax.fill_between(freq_grid_full[grid_mask], 0,
                    spectrum_dcmnet_full[grid_mask] / spectrum_dcmnet_full[grid_mask].max(),
                    color=COLORS['orange'], alpha=0.25, zorder=1)
    ax.plot(freq_grid_full[grid_mask],
            spectrum_dcmnet_full[grid_mask] / spectrum_dcmnet_full[grid_mask].max(),
            color=COLORS['orange'], lw=1.5, ls='--', label='DCMNet (fitted)', alpha=0.9, zorder=2)
    
    for name, freq in exp_co2[:3]:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('Fundamental Peaks (400-2600 cm⁻¹)', fontsize=13, weight='bold', loc='left')
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(400, 2600)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_spectrum.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'combined_spectrum.png'}")
    plt.close()


def plot_raman_overview(freqs, raman_phys, raman_dcm, exp_co2, output_dir):
    """Create overview figure for Raman spectra"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('MD-Based Raman Spectra', fontsize=16, weight='bold', y=0.995)
    
    # Plot 1: PhysNet Raman
    ax = axes[0]
    mask_full = (freqs > 300) & (freqs < 4000)
    ax.fill_between(freqs[mask_full], 0, raman_phys[mask_full]/raman_phys[mask_full].max(), 
                    color=COLORS['blue'], alpha=0.3)
    ax.plot(freqs[mask_full], raman_phys[mask_full]/raman_phys[mask_full].max(), 
            color=COLORS['blue'], lw=1.5, alpha=0.9)
    
    for name, freq in exp_co2:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
        ax.text(freq, ax.get_ylim()[1]*0.92, name.split()[0], rotation=90, 
                va='bottom', fontsize=9, color=COLORS['vermillion'], weight='bold')
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (arb. units)', fontsize=12, weight='bold')
    ax.set_title('PhysNet Raman', fontsize=13, weight='bold', loc='left')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(300, 4000)
    
    # Plot 2: DCMNet Raman
    ax = axes[1]
    ax.fill_between(freqs[mask_full], 0, raman_dcm[mask_full]/raman_dcm[mask_full].max(), 
                    color=COLORS['orange'], alpha=0.3)
    ax.plot(freqs[mask_full], raman_dcm[mask_full]/raman_dcm[mask_full].max(), 
            color=COLORS['orange'], lw=1.5, ls='--', alpha=0.9)
    
    for name, freq in exp_co2:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (arb. units)', fontsize=12, weight='bold')
    ax.set_title('DCMNet Raman', fontsize=13, weight='bold', loc='left')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(300, 4000)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'raman_spectra.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'raman_spectra.png'}")
    plt.close()


def plot_raman_comparison(freqs, raman_phys, raman_dcm, exp_co2, output_dir):
    """Create comparison figure for Raman spectra"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle('Raman Spectrum Comparison', fontsize=16, weight='bold', y=0.995)
    
    mask_full = (freqs > 300) & (freqs < 4000)
    
    # PhysNet
    ax.fill_between(freqs[mask_full], 0, raman_phys[mask_full]/raman_phys[mask_full].max(), 
                    color=COLORS['blue'], alpha=0.3, label='PhysNet')
    ax.plot(freqs[mask_full], raman_phys[mask_full]/raman_phys[mask_full].max(), 
            color=COLORS['blue'], lw=1.5, alpha=0.9)
    
    # DCMNet
    ax.fill_between(freqs[mask_full], 0, raman_dcm[mask_full]/raman_dcm[mask_full].max(), 
                    color=COLORS['orange'], alpha=0.25, label='DCMNet')
    ax.plot(freqs[mask_full], raman_dcm[mask_full]/raman_dcm[mask_full].max(), 
            color=COLORS['orange'], lw=1.5, ls='--', alpha=0.9)
    
    # Mark expected peaks
    for name, freq in exp_co2:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
        ax.text(freq, ax.get_ylim()[1]*0.95, name.split()[0], 
               rotation=90, va='bottom', ha='right',
               fontsize=8, color=COLORS['vermillion'], weight='bold')
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (normalized)', fontsize=12, weight='bold')
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(300, 4000)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'raman_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'raman_comparison.png'}")
    plt.close()


def plot_ir_raman_combined(freqs, int_phys, int_dcm, raman_phys, raman_dcm, exp_co2, output_dir):
    """Create combined IR and Raman comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('IR and Raman Spectra Comparison', fontsize=16, weight='bold', y=0.995)
    
    mask_full = (freqs > 300) & (freqs < 4000)
    
    # IR subplot
    ax = axes[0]
    ax.fill_between(freqs[mask_full], 0, int_phys[mask_full]/int_phys[mask_full].max(), 
                    color=COLORS['blue'], alpha=0.3, label='PhysNet (IR)')
    ax.plot(freqs[mask_full], int_phys[mask_full]/int_phys[mask_full].max(), 
            color=COLORS['blue'], lw=1.5, alpha=0.9)
    ax.fill_between(freqs[mask_full], 0, int_dcm[mask_full]/int_dcm[mask_full].max(), 
                    color=COLORS['orange'], alpha=0.25, label='DCMNet (IR)')
    ax.plot(freqs[mask_full], int_dcm[mask_full]/int_dcm[mask_full].max(), 
            color=COLORS['orange'], lw=1.5, ls='--', alpha=0.9)
    
    for name, freq in exp_co2:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('IR Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('Infrared (IR) Spectrum', fontsize=13, weight='bold', loc='left')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(300, 4000)
    ax.set_ylim(0, 1.05)
    
    # Raman subplot
    ax = axes[1]
    ax.fill_between(freqs[mask_full], 0, raman_phys[mask_full]/raman_phys[mask_full].max(), 
                    color=COLORS['blue'], alpha=0.3, label='PhysNet (Raman)')
    ax.plot(freqs[mask_full], raman_phys[mask_full]/raman_phys[mask_full].max(), 
            color=COLORS['blue'], lw=1.5, alpha=0.9)
    ax.fill_between(freqs[mask_full], 0, raman_dcm[mask_full]/raman_dcm[mask_full].max(), 
                    color=COLORS['orange'], alpha=0.25, label='DCMNet (Raman)')
    ax.plot(freqs[mask_full], raman_dcm[mask_full]/raman_dcm[mask_full].max(), 
            color=COLORS['orange'], lw=1.5, ls='--', alpha=0.9)
    
    for name, freq in exp_co2:
        ax.axvline(freq, color=COLORS['vermillion'], ls=':', lw=1.5, alpha=0.5)
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('Raman Spectrum', fontsize=13, weight='bold', loc='left')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(300, 4000)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ir_raman_combined.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'ir_raman_combined.png'}")
    plt.close()


def print_summary_tables(exp_co2, peak_fits_physnet, peak_fits_dcmnet, freqs, int_phys, int_dcm, output_file):
    """Print comprehensive summary tables"""
    with open(output_file, 'w') as f:
        # Frequency comparison table
        f.write("="*90 + "\n")
        f.write("FREQUENCY COMPARISON TABLE\n")
        f.write("="*90 + "\n\n")
        f.write(f"{'Peak':<30} {'Model':<10} {'Exp.':<10} {'Calc.':<10} {'Shift':<10} {'Final':<10} {'Error':<10}\n")
        f.write(f"{'':30} {'':10} {'(cm⁻¹)':<10} {'(cm⁻¹)':<10} {'(cm⁻¹)':<10} {'(cm⁻¹)':<10} {'(cm⁻¹)':<10}\n")
        f.write("-"*90 + "\n")
        
        errors_p = []
        errors_d = []
        
        for (name, exp_freq), (name_p, fit_p), (name_d, fit_d) in zip(exp_co2, peak_fits_physnet, peak_fits_dcmnet):
            peak_center = fit_p['peak_center']
            window = fit_p['window']
            mask = (freqs >= peak_center - window) & (freqs <= peak_center + window)
            
            if mask.sum() > 0:
                orig_peak_p = freqs[mask][np.argmax(int_phys[mask])]
                orig_peak_d = freqs[mask][np.argmax(int_dcm[mask])]
                final_freq_p = orig_peak_p + fit_p['shift']
                final_freq_d = orig_peak_d + fit_d['shift']
                error_p = final_freq_p - exp_freq
                error_d = final_freq_d - exp_freq
                
                errors_p.append(error_p)
                errors_d.append(error_d)
                
                f.write(f"{name:<30} {'PhysNet':<10} {exp_freq:8.1f}  {orig_peak_p:8.1f}  "
                       f"{fit_p['shift']:+8.1f}  {final_freq_p:8.1f}  {error_p:+8.1f}\n")
                f.write(f"{'':30} {'DCMNet':<10} {exp_freq:8.1f}  {orig_peak_d:8.1f}  "
                       f"{fit_d['shift']:+8.1f}  {final_freq_d:8.1f}  {error_d:+8.1f}\n")
                f.write("-"*90 + "\n")
        
        # Frequency statistics
        f.write("\n📊 Frequency Statistics:\n\n")
        f.write("  PhysNet:\n")
        f.write(f"    Mean error:  {np.mean(errors_p):+7.1f} ± {np.std(errors_p):5.1f} cm⁻¹\n")
        f.write(f"    RMS error:   {np.sqrt(np.mean(np.array(errors_p)**2)):7.1f} cm⁻¹\n")
        f.write(f"    Max |error|: {np.max(np.abs(errors_p)):7.1f} cm⁻¹\n\n")
        
        f.write("  DCMNet:\n")
        f.write(f"    Mean error:  {np.mean(errors_d):+7.1f} ± {np.std(errors_d):5.1f} cm⁻¹\n")
        f.write(f"    RMS error:   {np.sqrt(np.mean(np.array(errors_d)**2)):7.1f} cm⁻¹\n")
        f.write(f"    Max |error|: {np.max(np.abs(errors_d)):7.1f} cm⁻¹\n\n")
        
        # Fitting parameters
        f.write("="*80 + "\n")
        f.write("FITTING PARAMETERS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Peak':<30} {'Model':<10} {'Shift':<12} {'FWHM':<12} {'Scale':<12} {'MSE':<10}\n")
        f.write(f"{'':30} {'':10} {'(cm⁻¹)':<12} {'(cm⁻¹)':<12} {'(factor)':<12} {'':10}\n")
        f.write("-"*80 + "\n")
        
        for (name_p, fit_p), (name_d, fit_d) in zip(peak_fits_physnet, peak_fits_dcmnet):
            f.write(f"{name_p:<30} {'PhysNet':<10} {fit_p['shift']:+7.1f}{' '*5} "
                   f"{fit_p['fwhm']:7.1f}{' '*5} {fit_p['scale']:8.2e}{' '*4} {fit_p['mse']:8.4f}\n")
            f.write(f"{'':30} {'DCMNet':<10} {fit_d['shift']:+7.1f}{' '*5} "
                   f"{fit_d['fwhm']:7.1f}{' '*5} {fit_d['scale']:8.2e}{' '*4} {fit_d['mse']:8.4f}\n")
            f.write("-"*80 + "\n")
        
        # Statistical summary
        shifts_p = [fit_p['shift'] for _, fit_p in peak_fits_physnet]
        fwhms_p = [fit_p['fwhm'] for _, fit_p in peak_fits_physnet]
        mses_p = [fit_p['mse'] for _, fit_p in peak_fits_physnet]
        
        shifts_d = [fit_d['shift'] for _, fit_d in peak_fits_dcmnet]
        fwhms_d = [fit_d['fwhm'] for _, fit_d in peak_fits_dcmnet]
        mses_d = [fit_d['mse'] for _, fit_d in peak_fits_dcmnet]
        
        f.write("\n📊 Statistical Summary:\n\n")
        f.write("  PhysNet:\n")
        f.write(f"    Mean shift:    {np.mean(shifts_p):+7.1f} ± {np.std(shifts_p):5.1f} cm⁻¹\n")
        f.write(f"    Mean FWHM:     {np.mean(fwhms_p):7.1f} ± {np.std(fwhms_p):5.1f} cm⁻¹\n")
        f.write(f"    Mean MSE:      {np.mean(mses_p):8.4f}\n")
        f.write(f"    Total MSE:     {np.sum(mses_p):8.4f}\n\n")
        
        f.write("  DCMNet:\n")
        f.write(f"    Mean shift:    {np.mean(shifts_d):+7.1f} ± {np.std(shifts_d):5.1f} cm⁻¹\n")
        f.write(f"    Mean FWHM:     {np.mean(fwhms_d):7.1f} ± {np.std(fwhms_d):5.1f} cm⁻¹\n")
        f.write(f"    Mean MSE:      {np.mean(mses_d):8.4f}\n")
        f.write(f"    Total MSE:     {np.sum(mses_d):8.4f}\n\n")
        
        # Key findings
        f.write("💡 Key Findings:\n")
        if np.mean(mses_p) < np.mean(mses_d):
            f.write("  • PhysNet has better average line shape match (lower MSE)\n")
        else:
            f.write("  • DCMNet has better average line shape match (lower MSE)\n")
        
        mean_shift_diff = abs(np.mean(shifts_p) - np.mean(shifts_d))
        if mean_shift_diff < 5:
            f.write(f"  • Both models have similar systematic frequency offsets (~{np.mean(shifts_p):.1f} cm⁻¹)\n")
        else:
            f.write(f"  • Models have different systematic offsets (PhysNet: {np.mean(shifts_p):+.1f}, DCMNet: {np.mean(shifts_d):+.1f} cm⁻¹)\n")
        
        mean_fwhm_diff = abs(np.mean(fwhms_p) - np.mean(fwhms_d))
        if mean_fwhm_diff < 10:
            f.write(f"  • Both models predict similar peak widths (~{np.mean(fwhms_p):.1f} cm⁻¹)\n")
        else:
            f.write(f"  • Models predict different peak widths (PhysNet: {np.mean(fwhms_p):.1f}, DCMNet: {np.mean(fwhms_d):.1f} cm⁻¹)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("✅ Analysis Complete!\n")
        f.write("="*80 + "\n")
    
    # Also print to console
    with open(output_file, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description='Comprehensive IR and Raman Spectrum Analysis')
    parser.add_argument('--md-spectrum', type=Path, default='md_ir_spectrum.npz',
                       help='MD-based IR spectrum NPZ file')
    parser.add_argument('--exp-spectrum', type=Path, default='../experimental_co2_ir.npz',
                       help='Experimental IR spectrum NPZ file')
    parser.add_argument('--raman-spectrum', type=Path, default=None,
                       help='MD-based Raman spectrum NPZ file (optional)')
    parser.add_argument('--output-dir', type=Path, default='./ir_analysis',
                       help='Output directory for figures and reports')
    parser.add_argument('--smooth-window', type=int, default=5,
                       help='Rolling average window size (1=no smoothing)')
    parser.add_argument('--freq-scale', type=float, default=1.0,
                       help='Frequency scaling factor for calculated spectrum')
    parser.add_argument('--include-raman', action='store_true',
                       help='Include Raman analysis (requires --raman-spectrum or Raman data in MD file)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE IR & RAMAN SPECTRUM ANALYSIS")
    print("="*70)
    print(f"\nInput files:")
    print(f"  MD spectrum:     {args.md_spectrum}")
    print(f"  Exp spectrum:    {args.exp_spectrum}")
    if args.raman_spectrum:
        print(f"  Raman spectrum:  {args.raman_spectrum}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"\nSettings:")
    print(f"  Smoothing:       {args.smooth_window}-point rolling average")
    print(f"  Freq scale:      {args.freq_scale}")
    print(f"  Include Raman:   {args.include_raman}")
    
    # Load MD data
    print(f"\n📂 Loading MD spectrum...")
    md_data = np.load(args.md_spectrum)
    print(f"   Keys: {list(md_data.keys())}")
    
    # Apply smoothing and scaling
    freqs = rolling_average(md_data['frequencies']) * args.freq_scale
    int_phys = rolling_average(md_data['intensity_physnet'])
    int_dcm = rolling_average(md_data['intensity_dcmnet'])
    
    # Load experimental data
    print(f"\n📂 Loading experimental spectrum...")
    exp_data = np.load(args.exp_spectrum, allow_pickle=True)
    exp_freq = exp_data['wavenumbers']
    exp_abs = exp_data['absorbance']
    
    # Create overview plots
    print(f"\n📊 Creating overview plots...")
    plot_overview(freqs, int_phys, int_dcm, exp_freq, exp_abs, EXP_CO2_IR_PEAKS, args.output_dir)
    
    # Fit individual peaks
    print(f"\n🔍 Fitting individual peaks...")
    peak_fits_physnet = []
    peak_fits_dcmnet = []
    
    for name, peak_freq in EXP_CO2_IR_PEAKS:
        print(f"\n  {name} ({peak_freq:.1f} cm⁻¹)...")
        
        # Determine window
        if 'combination' in name.lower() or peak_freq > 3000:
            window = 250
        elif 'asym' in name.lower():
            window = 200
        else:
            window = 150
        
        # Fit PhysNet
        fit_p = fit_individual_peak(freqs, int_phys, exp_freq, exp_abs, 
                                    peak_freq, window=window, smooth_window=args.smooth_window)
        if fit_p:
            print(f"    PhysNet: shift={fit_p['shift']:+.1f} cm⁻¹, FWHM={fit_p['fwhm']:.1f} cm⁻¹, MSE={fit_p['mse']:.4f}")
            peak_fits_physnet.append((name, fit_p))
        
        # Fit DCMNet
        fit_d = fit_individual_peak(freqs, int_dcm, exp_freq, exp_abs,
                                   peak_freq, window=window, smooth_window=args.smooth_window)
        if fit_d:
            print(f"    DCMNet:  shift={fit_d['shift']:+.1f} cm⁻¹, FWHM={fit_d['fwhm']:.1f} cm⁻¹, MSE={fit_d['mse']:.4f}")
            peak_fits_dcmnet.append((name, fit_d))
    
    # Create plots
    print(f"\n📊 Creating detailed plots...")
    plot_individual_peaks(peak_fits_physnet, peak_fits_dcmnet, args.output_dir)
    plot_zoomed_grid(EXP_CO2_IR_PEAKS, peak_fits_physnet, peak_fits_dcmnet, freqs, int_phys, int_dcm, args.output_dir)
    plot_combined_spectrum(peak_fits_physnet, peak_fits_dcmnet, exp_freq, exp_abs, EXP_CO2_IR_PEAKS, args.output_dir)
    
    # Print summary tables
    print(f"\n📝 Generating summary report...")
    summary_file = args.output_dir / 'analysis_summary.txt'
    print_summary_tables(EXP_CO2_IR_PEAKS, peak_fits_physnet, peak_fits_dcmnet, freqs, int_phys, int_dcm, summary_file)
    print(f"\n✅ Summary saved: {summary_file}")
    
    # Raman analysis (if requested and data available)
    raman_phys = None
    raman_dcm = None
    
    if args.include_raman:
        print(f"\n{'='*70}")
        print(f"RAMAN SPECTRUM ANALYSIS")
        print(f"{'='*70}")
        
        if args.raman_spectrum and args.raman_spectrum.exists():
            print(f"\n📂 Loading Raman spectrum from {args.raman_spectrum}...")
            raman_data = np.load(args.raman_spectrum)
            if 'intensity_physnet' in raman_data and 'intensity_dcmnet' in raman_data:
                raman_phys = rolling_average(raman_data['intensity_physnet'])
                raman_dcm = rolling_average(raman_data['intensity_dcmnet'])
                print(f"   ✓ Loaded Raman data")
        elif 'raman_intensity_physnet' in md_data and 'raman_intensity_dcmnet' in md_data:
            print(f"\n📂 Loading Raman data from MD spectrum file...")
            raman_phys = rolling_average(md_data['raman_intensity_physnet'])
            raman_dcm = rolling_average(md_data['raman_intensity_dcmnet'])
            print(f"   ✓ Loaded Raman data")
        else:
            print(f"\n⚠️  No Raman data found. Skipping Raman analysis.")
            print(f"   Provide --raman-spectrum or ensure MD file contains Raman data.")
        
        if raman_phys is not None and raman_dcm is not None:
            print(f"\n📊 Creating Raman plots...")
            print(f"   Note: CO₂ Raman-active modes: ν₁ (1388 cm⁻¹, strong), ν₂ (667 cm⁻¹, weak)")
            plot_raman_overview(freqs, raman_phys, raman_dcm, EXP_CO2_RAMAN_PEAKS, args.output_dir)
            plot_raman_comparison(freqs, raman_phys, raman_dcm, EXP_CO2_RAMAN_PEAKS, args.output_dir)
            plot_ir_raman_combined(freqs, int_phys, int_dcm, raman_phys, raman_dcm, 
                                  EXP_CO2_IR_PEAKS, args.output_dir)
            
            # Report Raman peak intensities
            print(f"\n📊 Raman Peak Analysis:")
            for name, peak_freq in EXP_CO2_RAMAN_PEAKS:
                window = 150
                mask = (freqs >= peak_freq - window) & (freqs <= peak_freq + window)
                if mask.sum() > 0:
                    max_idx_p = np.argmax(raman_phys[mask])
                    max_idx_d = np.argmax(raman_dcm[mask])
                    max_freq_p = freqs[mask][max_idx_p]
                    max_freq_d = freqs[mask][max_idx_d]
                    print(f"  {name} (exp: {peak_freq:.1f} cm⁻¹):")
                    print(f"    PhysNet:  {max_freq_p:7.1f} cm⁻¹  (Δ={max_freq_p-peak_freq:+6.1f}, I={raman_phys[mask][max_idx_p]:.3e})")
                    print(f"    DCMNet:   {max_freq_d:7.1f} cm⁻¹  (Δ={max_freq_d-peak_freq:+6.1f}, I={raman_dcm[mask][max_idx_d]:.3e})")
            
            print(f"\n✅ Raman analysis complete!")
    
    print(f"\n{'='*70}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  {args.output_dir / 'overview_spectra.png'}")
    print(f"  {args.output_dir / 'individual_peaks.png'}")
    print(f"  {args.output_dir / 'zoomed_peaks_grid.png'}")
    print(f"  {args.output_dir / 'combined_spectrum.png'}")
    print(f"  {args.output_dir / 'analysis_summary.txt'}")
    
    if args.include_raman and raman_phys is not None:
        print(f"  {args.output_dir / 'raman_spectra.png'}")
        print(f"  {args.output_dir / 'raman_comparison.png'}")
        print(f"  {args.output_dir / 'ir_raman_combined.png'}")


if __name__ == '__main__':
    main()


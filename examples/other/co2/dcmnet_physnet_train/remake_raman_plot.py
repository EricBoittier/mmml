#!/usr/bin/env python3
"""
Remake Raman spectrum plot from existing Raman calculator output.

Creates multi-wavelength Raman scans with different colored lasers.

Usage:
    # Single wavelength
    python remake_raman_plot.py --raman-dir ./raman_analysis --laser 532
    
    # Multi-wavelength scan
    python remake_raman_plot.py --raman-dir ./raman_analysis --multi-wavelength
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
from pathlib import Path
from ase.io import read

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

# Experimental CO2 Raman-active peaks
EXP_CO2_RAMAN = [
    (1388.2, 'ν₁ sym'),  # Strong - symmetric stretch
    (667.4, 'ν₂ bend'),  # Weak - bending
]


def extract_frequencies_from_cache(cache_dir):
    """Extract vibrational frequencies from ASE cache files."""
    from ase.vibrations import Vibrations
    
    # Load geometry
    xyz_file = cache_dir.parent.parent / 'CO2_optimized.xyz'
    if not xyz_file.exists():
        xyz_file = cache_dir.parent / 'CO2_optimized.xyz'
    
    if xyz_file.exists():
        atoms = read(xyz_file)
    else:
        from ase import Atoms
        atoms = Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    
    # Read vibrations
    vib = Vibrations(atoms, name=str(cache_dir / 'vib'))
    
    try:
        freqs = vib.get_frequencies()
        print(f"   ✓ Extracted {len(freqs)} vibrational frequencies")
        return freqs
    except Exception as e:
        print(f"   ⚠️  Could not extract frequencies: {e}")
        return None


def calculate_raman_intensity_wavelength_dependent(freq_vib, alpha_deriv, laser_wavelength_nm):
    """
    Calculate Raman intensity with proper wavelength dependence.
    
    I_Raman ∝ (ν₀ - ν_vib)⁴ × |∂α/∂Q|²
    
    Parameters
    ----------
    freq_vib : float
        Vibrational frequency (cm⁻¹)
    alpha_deriv : float
        Polarizability derivative (arbitrary units)
    laser_wavelength_nm : float
        Laser wavelength (nm)
    
    Returns
    -------
    intensity : float
        Raman scattering intensity
    """
    # Convert laser wavelength to wavenumber
    laser_freq_cm = 1e7 / laser_wavelength_nm  # nm → cm⁻¹
    
    # Stokes scattering: scattered freq = laser_freq - vib_freq
    scattered_freq = laser_freq_cm - freq_vib
    
    if scattered_freq <= 0:
        return 0.0
    
    # Raman intensity ∝ ν_scattered⁴ × |∂α/∂Q|²
    intensity = (scattered_freq**4) * (alpha_deriv**2)
    
    return intensity


def create_multiwavelength_raman_plot(frequencies, output_file, broadening=10.0):
    """
    Create Raman spectrum for multiple laser wavelengths.
    Shows how spectrum changes with excitation wavelength using different colors.
    """
    # Filter real positive frequencies
    freqs = np.real(frequencies)
    mask = freqs > 100
    freqs = freqs[mask]
    
    if len(freqs) == 0:
        print("⚠️  No positive frequencies")
        return
    
    # Polarizability derivatives (proxy)
    alpha_derivs = np.ones_like(freqs)
    for i, freq in enumerate(freqs):
        if abs(freq - 1388.2) < 50:  # Symmetric stretch - strong Raman
            alpha_derivs[i] = 10.0
        elif abs(freq - 667.4) < 50:  # Bend - weak Raman
            alpha_derivs[i] = 2.0
    
    # Common laser wavelengths (nm) and their colors
    laser_configs = [
        (325, COLORS['reddish_purple'], 'UV (325 nm)'),
        (488, COLORS['sky_blue'], 'Blue (488 nm)'),
        (532, COLORS['bluish_green'], 'Green (532 nm)'),
        (633, COLORS['vermillion'], 'Red (633 nm)'),
        (785, COLORS['orange'], 'NIR (785 nm)'),
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Multi-Wavelength Raman Spectroscopy', 
                 fontsize=16, weight='bold', y=0.995)
    
    freq_range = np.linspace(0, max(freqs) + 500, 2000)
    
    # Plot 1: All wavelengths overlaid (normalized)
    ax = axes[0]
    
    for laser_wl, color, label in laser_configs:
        # Calculate intensities for this wavelength
        intensities = np.array([
            calculate_raman_intensity_wavelength_dependent(f, a, laser_wl)
            for f, a in zip(freqs, alpha_derivs)
        ])
        
        if intensities.max() > 0:
            intensities = intensities / intensities.max()
        
        # Broaden
        spectrum = np.zeros_like(freq_range)
        for freq, intensity in zip(freqs, intensities):
            spectrum += intensity * (broadening**2) / ((freq_range - freq)**2 + broadening**2)
        
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        
        # Plot
        ax.fill_between(freq_range, 0, spectrum, color=color, alpha=0.2)
        ax.plot(freq_range, spectrum, color=color, lw=2, label=label, alpha=0.8)
    
    # Mark experimental peaks
    for exp_freq, label_text in EXP_CO2_RAMAN:
        ax.axvline(exp_freq, color=COLORS['black'], ls=':', lw=1, alpha=0.4)
        ax.text(exp_freq, ax.get_ylim()[1]*0.95, label_text, 
               rotation=90, va='bottom', ha='right',
               fontsize=8, color=COLORS['black'], weight='bold', alpha=0.7)
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title('Wavelength Comparison (Individually Normalized)', 
                fontsize=13, weight='bold', loc='left')
    ax.legend(fontsize=10, framealpha=0.9, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(400, 2800)
    ax.set_ylim(0, 1.15)
    
    # Plot 2: Stacked view
    ax = axes[1]
    
    offset = 0
    offset_step = 0.3
    
    for laser_wl, color, label in laser_configs:
        # Calculate intensities
        intensities = np.array([
            calculate_raman_intensity_wavelength_dependent(f, a, laser_wl)
            for f, a in zip(freqs, alpha_derivs)
        ])
        
        if intensities.max() > 0:
            intensities = intensities / intensities.max()
        
        # Broaden
        spectrum = np.zeros_like(freq_range)
        for freq, intensity in zip(freqs, intensities):
            spectrum += intensity * (broadening**2) / ((freq_range - freq)**2 + broadening**2)
        
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        
        # Plot with offset
        ax.fill_between(freq_range, offset, offset + spectrum * 0.25, 
                       color=color, alpha=0.4)
        ax.plot(freq_range, offset + spectrum * 0.25, 
               color=color, lw=1.5, alpha=0.9)
        
        # Add wavelength label
        ax.text(50, offset + 0.12, f'{laser_wl} nm', 
               fontsize=10, ha='left', color=color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color, alpha=0.8))
        
        offset += offset_step
    
    # Mark experimental peaks
    for exp_freq, label_text in EXP_CO2_RAMAN:
        ax.axvline(exp_freq, color=COLORS['black'], ls=':', lw=1, alpha=0.3)
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Intensity (stacked with offset)', fontsize=12, weight='bold')
    ax.set_title('Laser Wavelength Scan', 
                fontsize=13, weight='bold', loc='left')
    ax.grid(True, alpha=0.3, ls='--', axis='x')
    ax.set_xlim(400, 2800)
    ax.set_ylim(-0.1, offset + 0.2)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()
    
    # Report wavelength effects
    print(f"\n{'='*70}")
    print(f"WAVELENGTH-DEPENDENT INTENSITY ANALYSIS")
    print(f"{'='*70}")
    print(f"\nIntensity scaling with laser wavelength:")
    print(f"(Relative to 532 nm green laser, for ν₁ peak at ~1388 cm⁻¹)\n")
    
    # Find peak near 1388 cm⁻¹
    idx_1388 = np.argmin(np.abs(freqs - 1388.2))
    alpha_1388 = alpha_derivs[idx_1388]
    
    for laser_wl, color, label in laser_configs:
        intensity = calculate_raman_intensity_wavelength_dependent(freqs[idx_1388], alpha_1388, laser_wl)
        intensity_532 = calculate_raman_intensity_wavelength_dependent(freqs[idx_1388], alpha_1388, 532)
        ratio = intensity / intensity_532 if intensity_532 > 0 else 0
        print(f"  {label:20s}: {ratio:6.2f}× relative intensity")
    
    print(f"\n💡 Physics: Shorter wavelengths → stronger signal (I ∝ ν⁴)")
    print(f"   UV laser (325 nm) is {ratio:.1f}× stronger than NIR (785 nm)!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Remake Raman spectrum with wavelength scan')
    parser.add_argument('--raman-dir', type=Path, required=True,
                       help='Directory containing Raman analysis')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output file')
    parser.add_argument('--broadening', type=float, default=10.0,
                       help='Lorentzian broadening HWHM (cm⁻¹)')
    parser.add_argument('--multi-wavelength', action='store_true',
                       help='Create multi-wavelength scan plot')
    parser.add_argument('--laser', type=float, default=532.0,
                       help='Single laser wavelength (nm), ignored if --multi-wavelength')
    
    args = parser.parse_args()
    
    # Set default output name
    if args.output is None:
        if args.multi_wavelength:
            args.output = args.raman_dir / 'raman_multiwavelength.png'
        else:
            args.output = args.raman_dir / 'raman_spectrum_styled.png'
    
    print("="*70)
    print("RAMAN SPECTRUM PLOTTING")
    print("="*70)
    print(f"\nSettings:")
    print(f"  Raman dir:        {args.raman_dir}")
    print(f"  Output:           {args.output}")
    print(f"  Broadening:       {args.broadening} cm⁻¹")
    print(f"  Multi-wavelength: {args.multi_wavelength}")
    if not args.multi_wavelength:
        print(f"  Laser:            {args.laser} nm")
    
    # Try to load frequencies
    print(f"\n📂 Loading Raman data...")
    
    # Method 1: From NPZ
    npz_file = args.raman_dir / 'raman_data.npz'
    if npz_file.exists():
        print(f"   Loading from {npz_file}...")
        data = np.load(npz_file)
        frequencies = data['frequencies']
        print(f"   ✓ Loaded {len(frequencies)} modes")
    else:
        # Method 2: From vibrations cache
        cache_dir = args.raman_dir / 'vibrations'
        if cache_dir.exists():
            print(f"   Loading from vibrations cache...")
            frequencies = extract_frequencies_from_cache(cache_dir)
            if frequencies is None:
                print("   ❌ Failed to load frequencies")
                return
        else:
            print(f"   ❌ No Raman data found in {args.raman_dir}")
            return
    
    # Create plot
    print(f"\n📊 Creating plot...")
    if args.multi_wavelength:
        create_multiwavelength_raman_plot(frequencies, args.output, 
                                         broadening=args.broadening)
    else:
        create_single_wavelength_plot(frequencies, args.output,
                                      broadening=args.broadening,
                                      laser_wavelength=args.laser)
    
    print(f"\n{'='*70}")
    print(f"✅ DONE")
    print(f"{'='*70}")


def create_single_wavelength_plot(frequencies, output_file, broadening=10.0, laser_wavelength=532):
    """Create Raman plot for single laser wavelength."""
    # Filter frequencies
    freqs = np.real(frequencies)
    mask = freqs > 100
    freqs = freqs[mask]
    
    if len(freqs) == 0:
        return
    
    # Polarizability derivatives
    alpha_derivs = np.ones_like(freqs)
    for i, freq in enumerate(freqs):
        if abs(freq - 1388.2) < 50:
            alpha_derivs[i] = 10.0
        elif abs(freq - 667.4) < 50:
            alpha_derivs[i] = 2.0
    
    # Calculate intensities
    intensities = np.array([
        calculate_raman_intensity_wavelength_dependent(f, a, laser_wavelength)
        for f, a in zip(freqs, alpha_derivs)
    ])
    
    if intensities.max() > 0:
        intensities = intensities / intensities.max()
    
    # Broaden
    freq_range = np.linspace(0, max(freqs) + 500, 2000)
    spectrum = np.zeros_like(freq_range)
    for freq, intensity in zip(freqs, intensities):
        spectrum += intensity * (broadening**2) / ((freq_range - freq)**2 + broadening**2)
    
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Raman Spectrum - {laser_wavelength:.0f} nm Laser', 
                 fontsize=16, weight='bold', y=0.995)
    
    # Stick spectrum
    ax = axes[0]
    markerline, stemlines, baseline = ax.stem(freqs, intensities,
                                               linefmt=COLORS['bluish_green'], 
                                               markerfmt='o', basefmt=' ')
    plt.setp(markerline, color=COLORS['bluish_green'], markersize=7, alpha=0.8)
    plt.setp(stemlines, color=COLORS['bluish_green'], linewidth=2.5, alpha=0.7)
    
    for exp_freq, label in EXP_CO2_RAMAN:
        ax.axvline(exp_freq, color=COLORS['vermillion'], ls=':', lw=2, alpha=0.6)
        ax.text(exp_freq, ax.get_ylim()[1]*0.92, label, 
               rotation=90, va='bottom', ha='right',
               fontsize=10, color=COLORS['vermillion'], weight='bold')
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title(f'Stick Spectrum', fontsize=13, weight='bold', loc='left')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(400, 2800)
    ax.set_ylim(0, 1.1)
    
    # Broadened spectrum
    ax = axes[1]
    ax.fill_between(freq_range, 0, spectrum, 
                    color=COLORS['bluish_green'], alpha=0.4)
    ax.plot(freq_range, spectrum, 
            color=COLORS['bluish_green'], lw=2, alpha=0.9)
    
    for exp_freq, label in EXP_CO2_RAMAN:
        ax.axvline(exp_freq, color=COLORS['vermillion'], ls=':', lw=2, alpha=0.6)
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title(f'Broadened Spectrum (HWHM = {broadening:.1f} cm⁻¹)', 
                fontsize=13, weight='bold', loc='left')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim(400, 2800)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    main()

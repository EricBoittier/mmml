#!/usr/bin/env python3
"""
Helper functions for plotting IR and Raman spectra from compute_ir_raman.py output.

Usage:
    from plot_ir_raman import load_spectra, plot_ir, plot_raman, plot_combined
    
    # Load data
    data = load_spectra('ir_raman.npz')
    
    # Plot IR
    plot_ir(data['ir_frequencies'], data['ir_intensity'], 
            save_path='ir_spectrum.png', freq_range=(0, 4500))
    
    # Plot Raman
    plot_raman(data['raman_frequencies'], 
               data['raman_intensity_isotropic'],
               data['raman_intensity_anisotropic'],
               save_path='raman_spectrum.png')
    
    # Plot both
    plot_combined(data, save_path='combined_spectra.png')
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def apply_ir_corrections(
    frequencies: np.ndarray,
    intensity: np.ndarray,
    temperature: float = 300.0,
    apply_quantum: bool = True,
    apply_frequency: bool = True,
) -> np.ndarray:
    """
    Apply frequency-dependent corrections to IR intensity.
    
    Corrections applied:
    1. Frequency prefactor: ω (linear scaling)
    2. Quantum correction: (1 + n(ω)) where n(ω) is Bose-Einstein distribution
    
    I_corrected(ω) = ω × (1 + n(ω)) × I_raw(ω)
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array (cm⁻¹)
    intensity : np.ndarray
        Raw intensity (before corrections)
    temperature : float
        Temperature in K
    apply_quantum : bool
        Apply quantum (Bose-Einstein) correction
    apply_frequency : bool
        Apply frequency prefactor (ω)
    
    Returns
    -------
    np.ndarray
        Corrected intensity
    """
    intensity_corrected = intensity.copy()
    
    # Convert frequency to angular frequency (rad/s)
    c_cm_s = 2.99792458e10  # cm/s
    freq_hz = frequencies * c_cm_s
    omega = 2 * np.pi * freq_hz
    
    # Frequency prefactor: ω
    if apply_frequency:
        intensity_corrected = intensity_corrected * omega
    
    # Quantum correction: (1 + n(ω))
    if apply_quantum:
        hbar_eV = 6.582119569e-16  # eV·s
        kB_eV = 8.617333262e-5  # eV/K
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            n_bose = 1.0 / (np.exp(np.clip(hbar_eV * omega / (kB_eV * temperature), 0, 700)) - 1.0)
            n_bose = np.nan_to_num(n_bose, nan=0.0, posinf=0.0)
        
        intensity_corrected = intensity_corrected * (1.0 + n_bose)
    
    return intensity_corrected


def apply_raman_corrections(
    frequencies: np.ndarray,
    intensity: np.ndarray,
    temperature: float = 300.0,
    laser_freq: float = 18797.0,
    apply_quantum: bool = True,
    apply_frequency: bool = True,
) -> np.ndarray:
    """
    Apply frequency-dependent corrections to Raman intensity.
    
    Corrections applied:
    1. Frequency prefactor: (ω₀ - ω)⁴ (Stokes shift to 4th power)
    2. Quantum correction: (1 + n(ω)) where n(ω) is Bose-Einstein distribution
    
    I_corrected(ω) = (ω₀ - ω)⁴ × (1 + n(ω)) × I_raw(ω)
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array (cm⁻¹) - Raman shift
    intensity : np.ndarray
        Raw intensity (before corrections)
    temperature : float
        Temperature in K
    laser_freq : float
        Laser frequency in cm⁻¹ (default: 532 nm = 18797 cm⁻¹)
    apply_quantum : bool
        Apply quantum (Bose-Einstein) correction
    apply_frequency : bool
        Apply frequency prefactor (ω₀ - ω)⁴
    
    Returns
    -------
    np.ndarray
        Corrected intensity
    """
    intensity_corrected = intensity.copy()
    
    # Convert frequency to angular frequency (rad/s)
    c_cm_s = 2.99792458e10  # cm/s
    freq_hz = frequencies * c_cm_s
    omega = 2 * np.pi * freq_hz
    
    # Frequency prefactor: (ω₀ - ω)⁴ (Stokes shift)
    if apply_frequency:
        laser_freq_hz = laser_freq * c_cm_s
        laser_omega = 2 * np.pi * laser_freq_hz
        raman_freqs = laser_omega - omega  # Stokes shift
        raman_factor = raman_freqs**4
        raman_factor[raman_freqs < 0] = 0  # Only Stokes side
        intensity_corrected = intensity_corrected * raman_factor
    
    # Quantum correction: (1 + n(ω))
    if apply_quantum:
        hbar_eV = 6.582119569e-16  # eV·s
        kB_eV = 8.617333262e-5  # eV/K
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            n_bose = 1.0 / (np.exp(np.clip(hbar_eV * omega / (kB_eV * temperature), 0, 700)) - 1.0)
            n_bose = np.nan_to_num(n_bose, nan=0.0, posinf=0.0)
        
        intensity_corrected = intensity_corrected * (1.0 + n_bose)
    
    return intensity_corrected


def load_spectra(npz_path: Path | str) -> dict:
    """
    Load IR/Raman spectra from NPZ file.
    
    Parameters
    ----------
    npz_path : Path or str
        Path to NPZ file created by compute_ir_raman.py
    
    Returns
    -------
    dict
        Dictionary containing frequencies, intensities, etc.
    """
    data = np.load(npz_path)
    return {key: data[key] for key in data.keys()}


def find_peaks_in_spectrum(
    frequencies: np.ndarray,
    intensity: np.ndarray,
    min_height: float = 0.01,
    min_prominence: float = 0.01,
    min_distance: int = 10,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Find peaks in spectrum.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array (cm⁻¹)
    intensity : np.ndarray
        Intensity array
    min_height : float
        Minimum peak height (relative to max)
    min_prominence : float
        Minimum peak prominence
    min_distance : int
        Minimum distance between peaks (in frequency points)
    freq_range : tuple, optional
        (min_freq, max_freq) to restrict search
    
    Returns
    -------
    peaks : np.ndarray
        Indices of peaks
    properties : dict
        Peak properties (heights, prominences, etc.)
    """
    if freq_range is not None:
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_subset = frequencies[mask]
        intensity_subset = intensity[mask]
    else:
        freq_subset = frequencies
        intensity_subset = intensity
    
    # Normalize for peak finding
    max_intensity = np.max(intensity_subset)
    if max_intensity > 0:
        intensity_normalized = intensity_subset / max_intensity
        height_threshold = min_height
        prominence_threshold = min_prominence
    else:
        return np.array([]), {}
    
    peaks, properties = find_peaks(
        intensity_normalized,
        height=height_threshold,
        prominence=prominence_threshold,
        distance=min_distance,
    )
    
    return peaks, properties


def plot_ir(
    frequencies: np.ndarray,
    intensity: np.ndarray,
    save_path: Optional[Path | str] = None,
    freq_range: Tuple[float, float] = (0, 4500),
    title: str = "IR Spectrum",
    label: Optional[str] = None,
    color: str = "#2E86AB",
    linewidth: float = 1.5,
    alpha: float = 0.8,
    show_peaks: bool = True,
    peak_threshold: float = 0.05,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    apply_corrections: bool = False,
    temperature: float = 300.0,
    apply_quantum: bool = True,
    apply_frequency: bool = True,
) -> plt.Figure:
    """
    Plot IR spectrum.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array (cm⁻¹)
    intensity : np.ndarray
        Intensity array
    save_path : Path or str, optional
        Path to save figure
    freq_range : tuple
        (min_freq, max_freq) to plot
    title : str
        Plot title
    label : str, optional
        Label for legend
    color : str
        Line color
    linewidth : float
        Line width
    alpha : float
        Transparency
    show_peaks : bool
        Whether to mark peaks
    peak_threshold : float
        Minimum relative intensity for peak marking
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Resolution for saved figure
    ax : plt.Axes, optional
        Existing axes to plot on
    apply_corrections : bool
        Apply frequency-dependent corrections (quantum + frequency prefactor)
    temperature : float
        Temperature in K (for quantum correction)
    apply_quantum : bool
        Apply quantum (Bose-Einstein) correction
    apply_frequency : bool
        Apply frequency prefactor (ω)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Apply corrections if requested
    if apply_corrections:
        intensity = apply_ir_corrections(
            frequencies, intensity, temperature, apply_quantum, apply_frequency
        )
        # Renormalize after corrections
        intensity = intensity / (np.max(intensity) + 1e-10)
    
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    freq_plot = frequencies[mask]
    intensity_plot = intensity[mask]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.plot(freq_plot, intensity_plot, color=color, linewidth=linewidth, 
            alpha=alpha, label=label)
    ax.fill_between(freq_plot, 0, intensity_plot, color=color, alpha=0.3)
    
    # Find and mark peaks
    if show_peaks:
        peaks, props = find_peaks_in_spectrum(
            frequencies, intensity, min_height=peak_threshold, freq_range=freq_range
        )
        if len(peaks) > 0:
            peak_freqs = frequencies[mask][peaks]
            peak_intensities = intensity[mask][peaks]
            ax.scatter(peak_freqs, peak_intensities, color='red', s=50, 
                      zorder=5, marker='v', label='Peaks')
            
            # Label top peaks
            if len(peaks) > 0:
                top_n = min(5, len(peaks))
                top_indices = np.argsort(peak_intensities)[-top_n:]
                for idx in top_indices:
                    ax.annotate(
                        f'{peak_freqs[idx]:.0f}',
                        xy=(peak_freqs[idx], peak_intensities[idx]),
                        xytext=(5, 10),
                        textcoords='offset points',
                        fontsize=9,
                        color='red',
                        weight='bold',
                    )
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('IR Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(freq_range)
    
    if label:
        ax.legend(fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved IR spectrum to {save_path}")
    
    return fig


def plot_raman(
    frequencies: np.ndarray,
    intensity_isotropic: Optional[np.ndarray] = None,
    intensity_anisotropic: Optional[np.ndarray] = None,
    save_path: Optional[Path | str] = None,
    freq_range: Tuple[float, float] = (0, 4500),
    title: str = "Raman Spectrum",
    show_peaks: bool = True,
    peak_threshold: float = 0.05,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 300,
    ax: Optional[plt.Axes] = None,
    apply_corrections: bool = False,
    temperature: float = 300.0,
    laser_freq: float = 18797.0,
    apply_quantum: bool = True,
    apply_frequency: bool = True,
) -> plt.Figure:
    """
    Plot Raman spectrum (isotropic and/or anisotropic).
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array (cm⁻¹)
    intensity_isotropic : np.ndarray, optional
        Isotropic Raman intensity
    intensity_anisotropic : np.ndarray, optional
        Anisotropic Raman intensity
    save_path : Path or str, optional
        Path to save figure
    freq_range : tuple
        (min_freq, max_freq) to plot
    title : str
        Plot title
    show_peaks : bool
        Whether to mark peaks
    peak_threshold : float
        Minimum relative intensity for peak marking
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    ax : plt.Axes, optional
        Existing axes to plot on
    apply_corrections : bool
        Apply frequency-dependent corrections (quantum + frequency prefactor)
    temperature : float
        Temperature in K (for quantum correction)
    laser_freq : float
        Laser frequency in cm⁻¹ (for frequency prefactor)
    apply_quantum : bool
        Apply quantum (Bose-Einstein) correction
    apply_frequency : bool
        Apply frequency prefactor (ω₀ - ω)⁴
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Apply corrections if requested
    if apply_corrections:
        if intensity_isotropic is not None:
            intensity_isotropic = apply_raman_corrections(
                frequencies, intensity_isotropic, temperature, laser_freq,
                apply_quantum, apply_frequency
            )
            intensity_isotropic = intensity_isotropic / (np.max(intensity_isotropic) + 1e-10)
        
        if intensity_anisotropic is not None:
            intensity_anisotropic = apply_raman_corrections(
                frequencies, intensity_anisotropic, temperature, laser_freq,
                apply_quantum, apply_frequency
            )
            intensity_anisotropic = intensity_anisotropic / (np.max(intensity_anisotropic) + 1e-10)
    
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    freq_plot = frequencies[mask]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if intensity_isotropic is not None:
        iso_plot = intensity_isotropic[mask]
        ax.plot(freq_plot, iso_plot, color='#2E86AB', linewidth=1.5, 
                alpha=0.8, label='Isotropic', linestyle='-')
        ax.fill_between(freq_plot, 0, iso_plot, color='#2E86AB', alpha=0.3)
    
    if intensity_anisotropic is not None:
        aniso_plot = intensity_anisotropic[mask]
        ax.plot(freq_plot, aniso_plot, color='#A23B72', linewidth=1.5, 
                alpha=0.8, label='Anisotropic', linestyle='--')
        ax.fill_between(freq_plot, 0, aniso_plot, color='#A23B72', alpha=0.2)
    
    # Find and mark peaks (use isotropic if available, else anisotropic)
    if show_peaks:
        intensity_for_peaks = intensity_isotropic if intensity_isotropic is not None else intensity_anisotropic
        if intensity_for_peaks is not None:
            peaks, props = find_peaks_in_spectrum(
                frequencies, intensity_for_peaks, min_height=peak_threshold, freq_range=freq_range
            )
            if len(peaks) > 0:
                peak_freqs = frequencies[mask][peaks]
                peak_intensities = intensity_for_peaks[mask][peaks]
                ax.scatter(peak_freqs, peak_intensities, color='red', s=50, 
                          zorder=5, marker='v', label='Peaks')
                
                # Label top peaks
                top_n = min(5, len(peaks))
                top_indices = np.argsort(peak_intensities)[-top_n:]
                for idx in top_indices:
                    ax.annotate(
                        f'{peak_freqs[idx]:.0f}',
                        xy=(peak_freqs[idx], peak_intensities[idx]),
                        xytext=(5, 10),
                        textcoords='offset points',
                        fontsize=9,
                        color='red',
                        weight='bold',
                    )
    
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Raman Intensity (normalized)', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(freq_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved Raman spectrum to {save_path}")
    
    return fig


def plot_combined(
    data: dict,
    save_path: Optional[Path | str] = None,
    freq_range: Tuple[float, float] = (0, 4500),
    show_peaks: bool = True,
    peak_threshold: float = 0.05,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300,
    apply_corrections: bool = False,
    temperature: float = 300.0,
    laser_freq: float = 18797.0,
) -> plt.Figure:
    """
    Plot IR and Raman spectra together.
    
    Parameters
    ----------
    data : dict
        Dictionary from load_spectra() containing IR and/or Raman data
    save_path : Path or str, optional
        Path to save figure
    freq_range : tuple
        (min_freq, max_freq) to plot
    show_peaks : bool
        Whether to mark peaks
    peak_threshold : float
        Minimum relative intensity for peak marking
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    apply_corrections : bool
        Apply frequency-dependent corrections
    temperature : float
        Temperature in K (for quantum correction)
    laser_freq : float
        Laser frequency in cm⁻¹ (for Raman frequency prefactor)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    has_ir = 'ir_frequencies' in data and 'ir_intensity' in data
    has_raman = 'raman_frequencies' in data and (
        'raman_intensity_isotropic' in data or 'raman_intensity_anisotropic' in data
    )
    
    n_plots = sum([has_ir, has_raman])
    if n_plots == 0:
        raise ValueError("No IR or Raman data found in input dictionary")
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if has_ir:
        plot_ir(
            data['ir_frequencies'],
            data['ir_intensity'],
            freq_range=freq_range,
            title="IR Spectrum",
            show_peaks=show_peaks,
            peak_threshold=peak_threshold,
            ax=axes[plot_idx],
            apply_corrections=apply_corrections,
            temperature=temperature,
        )
        plot_idx += 1
    
    if has_raman:
        plot_raman(
            data['raman_frequencies'],
            data.get('raman_intensity_isotropic'),
            data.get('raman_intensity_anisotropic'),
            freq_range=freq_range,
            title="Raman Spectrum",
            show_peaks=show_peaks,
            peak_threshold=peak_threshold,
            ax=axes[plot_idx],
            apply_corrections=apply_corrections,
            temperature=temperature,
            laser_freq=laser_freq,
        )
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved combined spectra to {save_path}")
    
    return fig


def print_peak_summary(
    frequencies: np.ndarray,
    intensity: np.ndarray,
    spectrum_type: str = "IR",
    freq_range: Optional[Tuple[float, float]] = None,
    top_n: int = 10,
) -> None:
    """
    Print summary of peaks in spectrum.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array
    intensity : np.ndarray
        Intensity array
    spectrum_type : str
        Type of spectrum (for labeling)
    freq_range : tuple, optional
        Restrict to frequency range
    top_n : int
        Number of top peaks to print
    """
    peaks, props = find_peaks_in_spectrum(
        frequencies, intensity, min_height=0.01, freq_range=freq_range
    )
    
    if len(peaks) == 0:
        print(f"No peaks found in {spectrum_type} spectrum")
        return
    
    peak_freqs = frequencies[peaks]
    peak_intensities = intensity[peaks]
    
    # Sort by intensity
    sorted_indices = np.argsort(peak_intensities)[::-1]
    top_indices = sorted_indices[:top_n]
    
    print(f"\n{spectrum_type} Spectrum - Top {min(top_n, len(peaks))} Peaks:")
    print("-" * 50)
    print(f"{'Rank':<6} {'Frequency (cm⁻¹)':<20} {'Intensity':<15}")
    print("-" * 50)
    
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:<6} {peak_freqs[idx]:<20.2f} {peak_intensities[idx]:<15.6f}")


def main_with_args(args):
    """
    Main function that works with argparse.Namespace or any object with attributes.
    
    This allows the function to be called programmatically by spoofing CLI arguments.
    
    Parameters
    ----------
    args : object
        Object with attributes: input, output_dir, freq_range, no_peaks, peak_threshold, dpi
        Compatible with argparse.Namespace or types.SimpleNamespace
    """
    # Load data
    print(f"Loading spectra from {args.input}...")
    data = load_spectra(args.input)
    print(f"Available keys: {list(data.keys())}")
    
    # Determine output directory
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    freq_range = tuple(args.freq_range)
    
    # Get correction parameters from args (if available)
    apply_corrections = getattr(args, 'apply_corrections', False)
    temperature = getattr(args, 'temperature', 300.0)
    laser_freq = getattr(args, 'laser_freq', 18797.0)
    
    # Plot IR if available
    if 'ir_frequencies' in data and 'ir_intensity' in data:
        print("\nPlotting IR spectrum...")
        plot_ir(
            data['ir_frequencies'],
            data['ir_intensity'],
            save_path=output_dir / 'ir_spectrum.png',
            freq_range=freq_range,
            show_peaks=not args.no_peaks,
            peak_threshold=args.peak_threshold,
            dpi=args.dpi,
            apply_corrections=apply_corrections,
            temperature=temperature,
        )
        print_peak_summary(
            data['ir_frequencies'],
            data['ir_intensity'],
            spectrum_type="IR",
            freq_range=freq_range,
        )
    
    # Plot Raman if available
    if 'raman_frequencies' in data:
        print("\nPlotting Raman spectrum...")
        plot_raman(
            data['raman_frequencies'],
            data.get('raman_intensity_isotropic'),
            data.get('raman_intensity_anisotropic'),
            save_path=output_dir / 'raman_spectrum.png',
            freq_range=freq_range,
            show_peaks=not args.no_peaks,
            peak_threshold=args.peak_threshold,
            dpi=args.dpi,
            apply_corrections=apply_corrections,
            temperature=temperature,
            laser_freq=laser_freq,
        )
        if 'raman_intensity_isotropic' in data:
            print_peak_summary(
                data['raman_frequencies'],
                data['raman_intensity_isotropic'],
                spectrum_type="Raman (Isotropic)",
                freq_range=freq_range,
            )
    
    # Plot combined
    print("\nPlotting combined spectra...")
    plot_combined(
        data,
        save_path=output_dir / 'combined_spectra.png',
        freq_range=freq_range,
        show_peaks=not args.no_peaks,
        peak_threshold=args.peak_threshold,
        dpi=args.dpi,
        apply_corrections=apply_corrections,
        temperature=temperature,
        laser_freq=laser_freq,
    )
    
    print(f"\n✅ All plots saved to {output_dir}")


def main():
    """CLI interface using argparse."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot IR and Raman spectra")
    parser.add_argument("input", type=Path, help="Input NPZ file from compute_ir_raman.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--freq-range", type=float, nargs=2, default=[0, 4500],
                       metavar=("MIN", "MAX"), help="Frequency range (cm⁻¹)")
    parser.add_argument("--no-peaks", action="store_true", help="Don't mark peaks")
    parser.add_argument("--peak-threshold", type=float, default=0.05,
                       help="Minimum relative intensity for peaks")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution")
    parser.add_argument("--apply-corrections", action="store_true",
                       help="Apply frequency-dependent corrections (quantum + frequency prefactor)")
    parser.add_argument("--temperature", type=float, default=300.0,
                       help="Temperature in K (for quantum correction)")
    parser.add_argument("--laser-freq", type=float, default=18797.0,
                       help="Laser frequency in cm⁻¹ (for Raman, default: 532 nm)")
    
    args = parser.parse_args()
    main_with_args(args)


if __name__ == "__main__":
    main()


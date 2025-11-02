#!/usr/bin/env python3
"""
Match Harmonic Frequencies to MD IR Spectrum

Finds optimal scaling factor to align harmonic frequencies with MD IR peaks.
This corrects for anharmonic effects and systematic errors.

Common scaling factors:
- DFT (PBE): ~0.99
- DFT (B3LYP): ~0.96
- HF: ~0.89
- MP2: ~0.95

Usage:
    python match_harmonic_to_md.py \
        --harmonic-frequencies ./dynamics/frequencies.npy \
        --harmonic-intensities ./dynamics/ir_intensities.npy \
        --md-spectrum ./md_ir/ir_spectrum_md.npz \
        --output-dir ./ir_comparison
"""

import sys
from pathlib import Path
import numpy as np
import argparse
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_harmonic_data(freq_file: Path, intensity_file: Path = None):
    """
    Load harmonic frequencies and intensities.
    
    Can load from:
    - .npy files
    - .npz files with 'frequencies' and 'intensities' keys
    - Parse from dynamics output
    """
    if freq_file.suffix == '.npy':
        freqs = np.load(freq_file)
        if intensity_file:
            intensities = np.load(intensity_file)
        else:
            intensities = None
    elif freq_file.suffix == '.npz':
        data = np.load(freq_file)
        freqs = data['frequencies']
        intensities = data.get('intensities', None)
    else:
        raise ValueError(f"Unsupported file format: {freq_file.suffix}")
    
    # Filter out negative/low frequencies
    valid = freqs > 100  # cm⁻¹
    freqs = freqs[valid]
    if intensities is not None:
        intensities = intensities[valid]
    
    return freqs, intensities


def load_md_spectrum(spectrum_file: Path):
    """Load MD-based IR spectrum."""
    if spectrum_file.suffix == '.npz':
        data = np.load(spectrum_file)
        frequencies = data['frequencies']
        intensities = data['intensities']
        # Also load autocorrelation if available
        autocorr = data.get('autocorrelation', None)
        times = data.get('times', None)
        return frequencies, intensities, autocorr, times
    else:
        raise ValueError(f"Unsupported format: {spectrum_file.suffix}")


def find_spectrum_peaks(frequencies, intensities, min_height=0.1, min_distance=50):
    """Find peaks in spectrum."""
    # Normalize intensities
    intensities_norm = intensities / np.max(intensities)
    
    # Find peaks
    peaks, properties = find_peaks(
        intensities_norm,
        height=min_height,
        distance=min_distance
    )
    
    peak_freqs = frequencies[peaks]
    peak_heights = intensities[peaks]
    
    return peak_freqs, peak_heights


def compute_matching_score(scale_factor, harmonic_freqs, md_peak_freqs, method='min_dist'):
    """
    Compute how well scaled harmonic frequencies match MD peaks.
    
    Methods:
    - 'min_dist': Sum of minimum distances (good for matching)
    - 'peak_overlap': How many harmonic peaks are near MD peaks
    """
    scaled_freqs = harmonic_freqs * scale_factor
    
    if method == 'min_dist':
        # For each harmonic frequency, find closest MD peak
        total_distance = 0
        for hf in scaled_freqs:
            distances = np.abs(md_peak_freqs - hf)
            total_distance += np.min(distances)
        
        return total_distance
    
    elif method == 'peak_overlap':
        # Count how many harmonic peaks are within tolerance of MD peaks
        tolerance = 50  # cm⁻¹
        matches = 0
        for hf in scaled_freqs:
            if np.any(np.abs(md_peak_freqs - hf) < tolerance):
                matches += 1
        
        # Return negative (minimize_scalar minimizes, we want to maximize matches)
        return -matches
    
    else:
        raise ValueError(f"Unknown method: {method}")


def find_optimal_scale_factor(harmonic_freqs, md_peak_freqs, 
                              scale_range=(0.85, 1.05), method='min_dist'):
    """Find optimal scaling factor to match harmonic to MD."""
    
    result = minimize_scalar(
        lambda s: compute_matching_score(s, harmonic_freqs, md_peak_freqs, method),
        bounds=scale_range,
        method='bounded'
    )
    
    return result.x


def plot_comparison(harmonic_freqs, harmonic_intensities, 
                   md_frequencies, md_intensities,
                   scale_factor, output_file):
    """Plot comparison of harmonic vs MD IR spectra."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Normalize intensities
    if harmonic_intensities is not None:
        harm_int_norm = harmonic_intensities / np.max(harmonic_intensities)
    else:
        harm_int_norm = np.ones_like(harmonic_freqs)
    
    md_int_norm = md_intensities / np.max(md_intensities)
    
    # Plot 1: Original harmonic vs MD
    ax = axes[0]
    ax.plot(md_frequencies, md_int_norm, 'b-', alpha=0.7, linewidth=1.5, label='MD IR')
    ax.stem(harmonic_freqs, harm_int_norm, linefmt='r-', markerfmt='ro', 
            basefmt=' ', label='Harmonic (original)')
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Original Harmonic vs MD IR', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scaled harmonic vs MD
    ax = axes[1]
    scaled_freqs = harmonic_freqs * scale_factor
    ax.plot(md_frequencies, md_int_norm, 'b-', alpha=0.7, linewidth=1.5, label='MD IR')
    ax.stem(scaled_freqs, harm_int_norm, linefmt='g-', markerfmt='go',
            basefmt=' ', label=f'Harmonic (scaled ×{scale_factor:.4f})')
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title(f'Scaled Harmonic vs MD IR (scale = {scale_factor:.4f})', 
                fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Overlay both for comparison
    ax = axes[2]
    ax.plot(md_frequencies, md_int_norm, 'b-', alpha=0.7, linewidth=2, label='MD IR')
    ax.stem(harmonic_freqs, harm_int_norm * 0.9, linefmt='r--', markerfmt='ro',
            basefmt=' ', label='Original harmonic', alpha=0.5)
    ax.stem(scaled_freqs, harm_int_norm, linefmt='g-', markerfmt='go',
            basefmt=' ', label=f'Scaled harmonic (×{scale_factor:.4f})')
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Comparison: Original vs Scaled Harmonic vs MD', 
                fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {output_file}")
    plt.close()


def create_detailed_comparison_table(harmonic_freqs, harmonic_intensities,
                                     md_peak_freqs, md_peak_intensities,
                                     scale_factor):
    """Create detailed table comparing peaks."""
    scaled_freqs = harmonic_freqs * scale_factor
    
    print(f"\n{'='*70}")
    print("PEAK COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Harmonic':<12} {'Scaled':<12} {'Closest MD':<15} {'Δ (cm⁻¹)':<12} {'Assignment':<15}")
    print(f"{'-'*70}")
    
    assignments = []
    for i, (orig_freq, scaled_freq) in enumerate(zip(harmonic_freqs, scaled_freqs)):
        # Find closest MD peak
        distances = np.abs(md_peak_freqs - scaled_freq)
        closest_idx = np.argmin(distances)
        closest_md = md_peak_freqs[closest_idx]
        delta = scaled_freq - closest_md
        
        # Assign mode type for CO2
        if scaled_freq < 800:
            assignment = "Bend"
        elif 800 < scaled_freq < 1500:
            assignment = "Symmetric str"
        else:
            assignment = "Asymmetric str"
        
        print(f"{orig_freq:10.1f}  {scaled_freq:10.1f}  {closest_md:12.1f}  {delta:+10.1f}  {assignment:<15}")
        
        assignments.append({
            'harmonic_original': orig_freq,
            'harmonic_scaled': scaled_freq,
            'md_peak': closest_md,
            'delta': delta,
            'assignment': assignment,
        })
    
    return assignments


def main():
    parser = argparse.ArgumentParser(description='Match harmonic frequencies to MD IR')
    
    parser.add_argument('--harmonic-data', type=Path, default=None,
                       help='NPZ file with harmonic frequencies and intensities')
    parser.add_argument('--harmonic-frequencies', type=Path, default=None,
                       help='NPY file with harmonic frequencies (cm⁻¹)')
    parser.add_argument('--harmonic-intensities', type=Path, default=None,
                       help='NPY file with harmonic intensities')
    parser.add_argument('--md-spectrum', type=Path, required=True,
                       help='NPZ file with MD IR spectrum')
    
    parser.add_argument('--scale-range', type=float, nargs=2, default=[0.85, 1.05],
                       help='Range for scaling factor search (default: 0.85 1.05)')
    parser.add_argument('--initial-scale', type=float, default=None,
                       help='Initial guess for scaling factor (optional)')
    parser.add_argument('--method', type=str, default='min_dist',
                       choices=['min_dist', 'peak_overlap'],
                       help='Matching method')
    
    parser.add_argument('--output-dir', type=Path, default=Path('./ir_matching'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HARMONIC TO MD IR MATCHING")
    print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load harmonic data
    print(f"\n1. Loading harmonic frequencies...")
    if args.harmonic_data:
        harm_data = np.load(args.harmonic_data)
        harmonic_freqs = harm_data['frequencies']
        harmonic_intensities = harm_data.get('intensities', None)
        print(f"✅ Loaded from {args.harmonic_data}")
    elif args.harmonic_frequencies:
        harmonic_freqs, harmonic_intensities = load_harmonic_data(
            args.harmonic_frequencies, args.harmonic_intensities
        )
        print(f"✅ Loaded from {args.harmonic_frequencies}")
    else:
        # Try to find in common locations
        search_paths = [
            Path('./spectroscopy_suite/quick_analysis/harmonic_ir.npz'),
            Path('./dynamics_analysis/harmonic_ir.npz'),
            Path('./harmonic_ir.npz'),
        ]
        for p in search_paths:
            if p.exists():
                harm_data = np.load(p)
                harmonic_freqs = harm_data['frequencies']
                harmonic_intensities = harm_data.get('intensities', None)
                print(f"✅ Auto-found harmonic data: {p}")
                break
        else:
            print("❌ No harmonic data found. Specify --harmonic-data or --harmonic-frequencies")
            return
    
    print(f"   Harmonic modes: {len(harmonic_freqs)}")
    print(f"   Frequency range: {harmonic_freqs.min():.1f} - {harmonic_freqs.max():.1f} cm⁻¹")
    
    # Load MD spectrum
    print(f"\n2. Loading MD IR spectrum...")
    md_frequencies, md_intensities, md_autocorr, md_times = load_md_spectrum(args.md_spectrum)
    print(f"✅ Loaded from {args.md_spectrum}")
    print(f"   Spectrum points: {len(md_frequencies)}")
    print(f"   Frequency range: {md_frequencies.min():.1f} - {md_frequencies.max():.1f} cm⁻¹")
    
    # Find peaks in MD spectrum
    print(f"\n3. Finding peaks in MD spectrum...")
    md_peak_freqs, md_peak_heights = find_spectrum_peaks(md_frequencies, md_intensities)
    print(f"✅ Found {len(md_peak_freqs)} peaks in MD spectrum")
    print(f"   Peak frequencies: {md_peak_freqs}")
    
    # Find optimal scaling factor
    print(f"\n4. Finding optimal scaling factor...")
    print(f"   Method: {args.method}")
    print(f"   Search range: [{args.scale_range[0]}, {args.scale_range[1]}]")
    
    if args.initial_scale:
        print(f"   Initial guess: {args.initial_scale}")
    
    scale_factor = find_optimal_scale_factor(
        harmonic_freqs, md_peak_freqs,
        scale_range=tuple(args.scale_range),
        method=args.method
    )
    
    print(f"\n✅ Optimal scaling factor: {scale_factor:.6f}")
    
    # Compute quality metrics
    scaled_freqs = harmonic_freqs * scale_factor
    residuals = []
    for hf in scaled_freqs:
        distances = np.abs(md_peak_freqs - hf)
        residuals.append(np.min(distances))
    
    print(f"   Mean residual: {np.mean(residuals):.2f} cm⁻¹")
    print(f"   Max residual: {np.max(residuals):.2f} cm⁻¹")
    print(f"   RMS residual: {np.sqrt(np.mean(np.array(residuals)**2)):.2f} cm⁻¹")
    
    # Create detailed comparison
    assignments = create_detailed_comparison_table(
        harmonic_freqs, harmonic_intensities,
        md_peak_freqs, md_peak_heights,
        scale_factor
    )
    
    # Plot comparison
    print(f"\n5. Creating comparison plots...")
    plot_comparison(
        harmonic_freqs, harmonic_intensities,
        md_frequencies, md_intensities,
        scale_factor,
        args.output_dir / 'harmonic_md_comparison.png'
    )
    
    # Save results
    print(f"\n6. Saving results...")
    
    # Save scaled harmonic data
    np.savez(
        args.output_dir / 'harmonic_scaled.npz',
        frequencies_original=harmonic_freqs,
        frequencies_scaled=scaled_freqs,
        intensities=harmonic_intensities,
        scale_factor=scale_factor,
    )
    
    # Save peak assignments
    import json
    with open(args.output_dir / 'peak_assignments.json', 'w') as f:
        json.dump({
            'scale_factor': float(scale_factor),
            'method': args.method,
            'assignments': assignments,
            'quality_metrics': {
                'mean_residual_cm1': float(np.mean(residuals)),
                'max_residual_cm1': float(np.max(residuals)),
                'rms_residual_cm1': float(np.sqrt(np.mean(np.array(residuals)**2))),
            }
        }, f, indent=2)
    
    print(f"✅ Saved scaled data: {args.output_dir / 'harmonic_scaled.npz'}")
    print(f"✅ Saved assignments: {args.output_dir / 'peak_assignments.json'}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Optimal frequency scaling: {scale_factor:.6f}")
    print(f"\nTo use in calculations:")
    print(f"  scaled_freq = harmonic_freq × {scale_factor:.6f}")
    print(f"\nThis scaling corrects for:")
    print(f"  - Anharmonic effects")
    print(f"  - Systematic method errors")
    print(f"  - Temperature effects")
    
    # Suggest what method this corresponds to
    if 0.94 < scale_factor < 0.96:
        print(f"\n💡 Similar to MP2 scaling factor (~0.95)")
    elif 0.95 < scale_factor < 0.97:
        print(f"\n💡 Similar to B3LYP scaling factor (~0.96)")
    elif 0.98 < scale_factor < 1.00:
        print(f"\n💡 Similar to PBE scaling factor (~0.99)")
    elif 0.88 < scale_factor < 0.90:
        print(f"\n💡 Similar to HF scaling factor (~0.89)")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()


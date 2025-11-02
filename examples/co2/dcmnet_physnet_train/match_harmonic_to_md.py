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


def find_spectrum_peaks(frequencies, intensities, min_height=0.05, min_distance=20):
    """Find peaks in spectrum with improved detection."""
    # Focus on physically relevant range for CO2
    freq_mask = (frequencies > 200) & (frequencies < 4000)
    freq_range = frequencies[freq_mask]
    int_range = intensities[freq_mask]
    
    if len(int_range) == 0:
        return np.array([]), np.array([])
    
    # Normalize intensities
    intensities_norm = int_range / np.max(int_range)
    
    # Find peaks with relaxed criteria
    peaks, properties = find_peaks(
        intensities_norm,
        height=min_height,
        distance=min_distance,
        prominence=min_height/2,  # Require some prominence
    )
    
    if len(peaks) == 0:
        # Fallback: just find global maximum
        peaks = [np.argmax(intensities_norm)]
        print(f"   ⚠️  Using relaxed peak detection (only found {len(peaks)} peak)")
    
    peak_freqs = freq_range[peaks]
    peak_heights = int_range[peaks]
    
    # Sort by frequency
    sort_idx = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[sort_idx]
    peak_heights = peak_heights[sort_idx]
    
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
                   scale_factor, output_file, md_intensities_dcmnet=None):
    """Plot comparison of harmonic vs MD IR spectra."""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    
    # Normalize intensities
    if harmonic_intensities is not None:
        harm_int_norm = harmonic_intensities / np.max(harmonic_intensities)
    else:
        harm_int_norm = np.ones_like(harmonic_freqs)
    
    md_int_norm = md_intensities / np.max(md_intensities)
    
    # Experimental CO2 frequencies
    exp_freqs = {
        'ν2 bend': 667.4,
        'ν1 sym stretch': 1388.2,
        'ν3 asym stretch': 2349.2,
    }
    
    # Plot 0: Full MD spectrum with both PhysNet and DCMNet
    ax = axes[0]
    
    # Focus on CO2 range
    freq_mask = (md_frequencies > 100) & (md_frequencies < 3500)
    ax.plot(md_frequencies[freq_mask], md_int_norm[freq_mask], 'b-', 
            alpha=0.7, linewidth=1.5, label='MD IR (PhysNet)')
    
    if md_intensities_dcmnet is not None:
        md_int_dcm_norm = md_intensities_dcmnet / np.max(md_intensities_dcmnet)
        ax.plot(md_frequencies[freq_mask], md_int_dcm_norm[freq_mask], 'c--',
                alpha=0.7, linewidth=1.5, label='MD IR (DCMNet)')
    
    # Add experimental frequencies as vertical lines
    for name, freq in exp_freqs.items():
        ax.axvline(freq, color='red', linestyle=':', linewidth=2, alpha=0.6)
        ax.text(freq, ax.get_ylim()[1] * 0.9, name, rotation=90, 
                verticalalignment='bottom', fontsize=9, color='red')
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('MD IR Spectrum (Full Range)', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 3500)
    
    # Plot 1: Original harmonic vs MD
    ax = axes[1]
    ax.plot(md_frequencies[freq_mask], md_int_norm[freq_mask], 'b-', alpha=0.7, linewidth=1.5, label='MD IR (PhysNet)')
    if md_intensities_dcmnet is not None:
        ax.plot(md_frequencies[freq_mask], md_int_dcm_norm[freq_mask], 'c--',
                alpha=0.7, linewidth=1.5, label='MD IR (DCMNet)')
    ax.stem(harmonic_freqs, harm_int_norm, linefmt='r-', markerfmt='ro', 
            basefmt=' ', label='Harmonic (original)')
    
    # Add experimental lines
    for name, freq in exp_freqs.items():
        ax.axvline(freq, color='red', linestyle=':', linewidth=1, alpha=0.4)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Original Harmonic vs MD IR', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(500, 2800)
    
    # Plot 2: Scaled harmonic vs MD
    ax = axes[2]
    scaled_freqs = harmonic_freqs * scale_factor
    ax.plot(md_frequencies[freq_mask], md_int_norm[freq_mask], 'b-', alpha=0.7, linewidth=1.5, label='MD IR (PhysNet)')
    if md_intensities_dcmnet is not None:
        ax.plot(md_frequencies[freq_mask], md_int_dcm_norm[freq_mask], 'c--',
                alpha=0.7, linewidth=1.5, label='MD IR (DCMNet)')
    ax.stem(scaled_freqs, harm_int_norm, linefmt='g-', markerfmt='go',
            basefmt=' ', label=f'Harmonic (scaled ×{scale_factor:.4f})')
    
    # Add experimental lines
    for name, freq in exp_freqs.items():
        ax.axvline(freq, color='red', linestyle=':', linewidth=1, alpha=0.4)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title(f'Scaled Harmonic vs MD IR (scale = {scale_factor:.4f})', 
                fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(500, 2800)
    
    # Plot 3: Overlay both for comparison
    ax = axes[3]
    ax.plot(md_frequencies[freq_mask], md_int_norm[freq_mask], 'b-', alpha=0.7, linewidth=2, label='MD IR (PhysNet)')
    if md_intensities_dcmnet is not None:
        ax.plot(md_frequencies[freq_mask], md_int_dcm_norm[freq_mask], 'c--',
                alpha=0.7, linewidth=2, label='MD IR (DCMNet)')
    
    # Stem plots without alpha parameter
    markerline1, stemlines1, baseline1 = ax.stem(harmonic_freqs, harm_int_norm * 0.9, 
                                                  linefmt='r--', markerfmt='ro', basefmt=' ')
    stemlines1.set_alpha(0.5)
    markerline1.set_alpha(0.5)
    markerline1.set_label('Original harmonic')
    
    markerline2, stemlines2, baseline2 = ax.stem(scaled_freqs, harm_int_norm,
                                                  linefmt='g-', markerfmt='go', basefmt=' ')
    markerline2.set_label(f'Scaled harmonic (×{scale_factor:.4f})')
    
    # Add experimental lines
    for name, freq in exp_freqs.items():
        ax.axvline(freq, color='red', linestyle=':', linewidth=2, alpha=0.6, label=name if freq == 667.4 else '')
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Comparison: Original vs Scaled Harmonic vs MD vs Experimental', 
                fontsize=14, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(500, 2800)
    
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
    if args.harmonic_data and args.harmonic_data.exists():
        harm_data = np.load(args.harmonic_data)
        harmonic_freqs = harm_data['frequencies']
        harmonic_intensities = harm_data.get('intensities_physnet', 
                                            harm_data.get('intensities', None))
        print(f"✅ Loaded from {args.harmonic_data}")
    elif args.harmonic_frequencies and args.harmonic_frequencies.exists():
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
        found = False
        for p in search_paths:
            if p.exists():
                harm_data = np.load(p)
                harmonic_freqs = harm_data['frequencies']
                harmonic_intensities = harm_data.get('intensities_physnet',
                                                    harm_data.get('intensities', None))
                print(f"✅ Auto-found harmonic data: {p}")
                found = True
                break
        
        if not found:
            print("⚠️  No harmonic data found, using experimental CO2 frequencies")
            print("   (You can compute your own with: python dynamics_calculator.py ...)")
            
            # Use experimental CO2 frequencies
            harmonic_freqs = np.array([
                667.4,    # ν2 bending mode (doubly degenerate)
                667.4,    # ν2 bending mode
                1388.2,   # ν1 symmetric stretch
                2349.2,   # ν3 asymmetric stretch
            ])
            # Rough intensity estimates (asymmetric stretch is strongest)
            harmonic_intensities = np.array([1.0, 1.0, 0.0, 10.0])
            
            print(f"   Using experimental CO2 frequencies: {harmonic_freqs}")
            print(f"   Note: Intensities are approximate!")
    
    print(f"   Harmonic modes: {len(harmonic_freqs)}")
    print(f"   Frequency range: {harmonic_freqs.min():.1f} - {harmonic_freqs.max():.1f} cm⁻¹")
    
    # Load MD spectrum
    print(f"\n2. Loading MD IR spectrum...")
    md_spec = np.load(args.md_spectrum)
    md_frequencies = md_spec['frequencies']
    md_intensities_physnet = md_spec.get('intensity_physnet', md_spec.get('intensities'))
    md_intensities_dcmnet = md_spec.get('intensity_dcmnet', None)
    md_autocorr = md_spec.get('autocorrelation', None)
    md_times = md_spec.get('times', None)
    
    print(f"✅ Loaded from {args.md_spectrum}")
    print(f"   Spectrum points: {len(md_frequencies)}")
    print(f"   Frequency range: {md_frequencies.min():.1f} - {md_frequencies.max():.1f} cm⁻¹")
    
    # Use PhysNet spectrum for matching (primary)
    md_intensities = md_intensities_physnet
    
    # Find peaks in MD spectrum
    print(f"\n3. Finding peaks in MD spectrum...")
    md_peak_freqs, md_peak_heights = find_spectrum_peaks(md_frequencies, md_intensities)
    print(f"✅ Found {len(md_peak_freqs)} peaks in MD spectrum")
    print(f"   Peak frequencies: {md_peak_freqs}")
    
    # Diagnostic: Check spectrum in CO2-relevant range
    co2_mask = (md_frequencies > 500) & (md_frequencies < 3000)
    if np.any(co2_mask):
        print(f"\n   Spectrum in CO2 range (500-3000 cm⁻¹):")
        print(f"     Max intensity: {md_intensities[co2_mask].max():.6f}")
        print(f"     Mean intensity: {md_intensities[co2_mask].mean():.6f}")
        max_idx = np.argmax(md_intensities[co2_mask])
        max_freq = md_frequencies[co2_mask][max_idx]
        print(f"     Peak location: {max_freq:.1f} cm⁻¹")
        
        if len(md_peak_freqs) < 2:
            print(f"\n   ⚠️  Warning: Very few peaks found!")
            print(f"   This might indicate:")
            print(f"     - Trajectory too short (need >20 ps)")
            print(f"     - Timestep too small (try subsampling)")
            print(f"     - Dipoles not varying enough")
    
    if len(md_peak_freqs) == 0:
        print(f"\n❌ No peaks found in MD spectrum!")
        print(f"   Cannot perform matching. Check your MD IR spectrum.")
        return
    
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
        md_frequencies, md_intensities_physnet,
        scale_factor,
        args.output_dir / 'harmonic_md_comparison.png',
        md_intensities_dcmnet=md_intensities_dcmnet
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


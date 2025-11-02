#!/usr/bin/env python3
"""
Parse JCAMP-DX IR Spectrum Files

Reads experimental IR data in JCAMP-DX format (.jdx/.dx files).
Converts to absorbance and exports to NPZ for comparison with calculated spectra.

Usage:
    python parse_jdx_ir.py \
        --jdx ../124-38-9-IR.jdx \
        --output ./experimental_co2_ir.npz
"""

import sys
from pathlib import Path
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def parse_jcamp_dx(filepath: Path):
    """
    Parse JCAMP-DX file.
    
    Returns
    -------
    dict
        Metadata and spectral data
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    metadata = {}
    data_lines = []
    in_data_section = False
    
    for line in lines:
        line = line.strip()
        
        # Header fields
        if line.startswith('##'):
            if '=' in line:
                key, value = line[2:].split('=', 1)
                metadata[key] = value.strip()
            
            # Check for data section start
            if 'XYDATA' in line or 'XYPOINTS' in line:
                in_data_section = True
                continue
        
        # Data section
        elif in_data_section:
            if line.startswith('##'):
                in_data_section = False
            else:
                data_lines.append(line)
    
    # Parse data
    x_values = []
    y_values = []
    
    for line in data_lines:
        if not line or line.startswith('##'):
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        # First value is X, rest are Y values
        x_start = float(parts[0])
        y_vals = [float(v) for v in parts[1:]]
        
        # Get DELTAX from metadata
        delta_x = float(metadata.get('DELTAX', 1.0))
        
        # Generate X values
        for i, y in enumerate(y_vals):
            x_values.append(x_start + i * delta_x)
            y_values.append(y)
    
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    return {
        'metadata': metadata,
        'wavenumbers': x_values,
        'transmittance': y_values,
    }


def transmittance_to_absorbance(transmittance):
    """
    Convert transmittance to absorbance.
    
    A = -log10(T)
    
    Also handles T > 1 (noise) by clipping.
    """
    # Clip to valid range
    T = np.clip(transmittance, 1e-10, 1.0)
    
    # Convert
    absorbance = -np.log10(T)
    
    return absorbance


def plot_experimental_ir(data, output_file=None):
    """Plot experimental IR spectrum."""
    
    wavenumbers = data['wavenumbers']
    transmittance = data['transmittance']
    absorbance = transmittance_to_absorbance(transmittance)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Transmittance
    ax = axes[0]
    ax.plot(wavenumbers, transmittance, 'k-', linewidth=1, alpha=0.8)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Transmittance', fontsize=12, weight='bold')
    ax.set_title('Experimental IR: Transmittance', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Mark CO2 peaks
    co2_peaks = [667.4, 2349.2]  # Main IR-active modes
    for peak in co2_peaks:
        if wavenumbers.min() <= peak <= wavenumbers.max():
            ax.axvline(peak, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(peak, 0.05, f'{peak:.0f}', rotation=90, va='bottom',
                   fontsize=9, color='red', weight='bold')
    
    # Plot 2: Absorbance
    ax = axes[1]
    ax.plot(wavenumbers, absorbance, 'b-', linewidth=1, alpha=0.8)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, weight='bold')
    ax.set_ylabel('Absorbance', fontsize=12, weight='bold')
    ax.set_title('Experimental IR: Absorbance', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark CO2 peaks
    for peak in co2_peaks:
        if wavenumbers.min() <= peak <= wavenumbers.max():
            ax.axvline(peak, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(peak, ax.get_ylim()[1]*0.9, f'{peak:.0f}', rotation=90,
                   va='bottom', fontsize=9, color='red', weight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Parse JCAMP-DX IR spectrum')
    parser.add_argument('--jdx', type=Path, required=True,
                       help='JCAMP-DX file (.jdx or .dx)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output NPZ file (default: <jdx_name>.npz)')
    parser.add_argument('--plot', action='store_true',
                       help='Create plot of spectrum')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.jdx.parent / (args.jdx.stem + '.npz')
    
    print("="*70)
    print("JCAMP-DX IR SPECTRUM PARSER")
    print("="*70)
    
    # Parse file
    print(f"\nParsing {args.jdx}...")
    data = parse_jcamp_dx(args.jdx)
    
    # Print metadata
    print(f"\nMetadata:")
    key_fields = ['TITLE', 'CAS REGISTRY NO', 'STATE', 'XUNITS', 'YUNITS', 'NPOINTS']
    for key in key_fields:
        if key in data['metadata']:
            print(f"  {key}: {data['metadata'][key]}")
    
    # Print spectrum info
    wavenumbers = data['wavenumbers']
    transmittance = data['transmittance']
    absorbance = transmittance_to_absorbance(transmittance)
    
    print(f"\nSpectrum:")
    print(f"  Points: {len(wavenumbers)}")
    print(f"  Range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
    print(f"  Resolution: ~{np.median(np.diff(wavenumbers)):.2f} cm⁻¹")
    print(f"  Transmittance range: {transmittance.min():.3f} - {transmittance.max():.3f}")
    print(f"  Absorbance range: {absorbance.min():.3f} - {absorbance.max():.3f}")
    
    # Find peaks in absorbance (strong absorption = low transmittance)
    from scipy.signal import find_peaks
    
    peaks, properties = find_peaks(absorbance, height=0.3, prominence=0.1, distance=20)
    
    print(f"\nMajor absorption peaks:")
    if len(peaks) > 0:
        # Sort by intensity
        sorted_indices = np.argsort(properties['peak_heights'])[::-1]
        for i in sorted_indices[:10]:  # Top 10
            idx = peaks[i]
            print(f"  {wavenumbers[idx]:7.1f} cm⁻¹ (absorbance: {absorbance[idx]:.3f})")
    
    # Save to NPZ
    print(f"\nSaving to {args.output}...")
    np.savez(
        args.output,
        wavenumbers=wavenumbers,
        transmittance=transmittance,
        absorbance=absorbance,
        metadata=data['metadata'],
    )
    print(f"✅ Saved")
    
    # Plot if requested
    if args.plot:
        plot_file = args.output.parent / (args.output.stem + '.png')
        plot_experimental_ir(data, plot_file)
    
    print(f"\n{'='*70}")
    print("✅ DONE")
    print(f"{'='*70}")
    print(f"\nTo use in comparisons:")
    print(f"  exp = np.load('{args.output}')")
    print(f"  freqs = exp['wavenumbers']")
    print(f"  absorbance = exp['absorbance']")


if __name__ == '__main__':
    main()


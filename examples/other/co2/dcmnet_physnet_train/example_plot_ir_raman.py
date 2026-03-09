#!/usr/bin/env python3
"""
Quick example of using plot_ir_raman helper functions.

Usage:
    python example_plot_ir_raman.py ir_raman.npz
"""

import sys
from pathlib import Path

from plot_ir_raman import load_spectra, plot_ir, plot_raman, plot_combined, print_peak_summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_plot_ir_raman.py <ir_raman.npz>")
        sys.exit(1)
    
    npz_path = Path(sys.argv[1])
    
    # Load data
    print(f"Loading {npz_path}...")
    data = load_spectra(npz_path)
    
    output_dir = npz_path.parent
    
    # Plot IR
    if 'ir_frequencies' in data:
        print("\nPlotting IR spectrum...")
        plot_ir(
            data['ir_frequencies'],
            data['ir_intensity'],
            save_path=output_dir / 'ir_spectrum.png',
            freq_range=(0, 4500),
        )
        print_peak_summary(data['ir_frequencies'], data['ir_intensity'], "IR")
    
    # Plot Raman
    if 'raman_frequencies' in data:
        print("\nPlotting Raman spectrum...")
        plot_raman(
            data['raman_frequencies'],
            data.get('raman_intensity_isotropic'),
            data.get('raman_intensity_anisotropic'),
            save_path=output_dir / 'raman_spectrum.png',
            freq_range=(0, 4500),
        )
        if 'raman_intensity_isotropic' in data:
            print_peak_summary(
                data['raman_frequencies'],
                data['raman_intensity_isotropic'],
                "Raman"
            )
    
    # Plot combined
    print("\nPlotting combined spectra...")
    plot_combined(data, save_path=output_dir / 'combined_spectra.png')
    
    print(f"\n✅ Done! Check {output_dir} for plots.")


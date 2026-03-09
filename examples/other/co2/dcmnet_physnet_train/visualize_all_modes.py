#!/usr/bin/env python3
"""
Batch process all vibrational modes for charge visualization.

Creates visualizations for all normal modes automatically.

Usage:
    python visualize_all_modes.py \
        --checkpoint /path/to/checkpoint \
        --raman-dir ./raman_analysis \
        --n-frames 20
"""

import numpy as np
import argparse
from pathlib import Path
from ase.io import read
from ase.vibrations import Vibrations
import subprocess
import warnings


def main():
    parser = argparse.ArgumentParser(description='Visualize all normal modes')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--raman-dir', type=Path, required=True,
                       help='Directory containing Raman analysis')
    parser.add_argument('--n-frames', type=int, default=20,
                       help='Number of frames per mode')
    parser.add_argument('--amplitude-scale', type=float, default=1.0,
                       help='Scale factor for vibration amplitude')
    
    args = parser.parse_args()
    
    print("="*70)
    print("BATCH VISUALIZATION OF ALL NORMAL MODES")
    print("="*70)
    
    # Load geometry
    print(f"\n📂 Loading geometry...")
    xyz_file = args.raman_dir / 'CO2_optimized.xyz'
    if xyz_file.exists():
        atoms = read(xyz_file)
    else:
        from ase import Atoms
        atoms = Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    print(f"   ✓ Loaded {len(atoms)} atoms")
    
    # Load vibrations
    print(f"\n📂 Loading vibrations...")
    vib_cache = args.raman_dir / 'vibrations'
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vib = Vibrations(atoms, name=str(vib_cache / 'vib'))
    
    freqs = vib.get_frequencies()
    print(f"   ✓ Found {len(freqs)} modes")
    
    # Filter real positive frequencies
    mask = np.real(freqs) > 100
    freqs_filtered = np.real(freqs[mask])
    mode_indices = np.where(mask)[0]
    
    print(f"\n📊 Will visualize {len(mode_indices)} physical modes:")
    mode_info = []
    
    for mode_idx, freq in zip(mode_indices, freqs_filtered):
        # Classify and determine amplitude
        if abs(freq - 1388.2) < 50:
            mode_type = 'ν₁ Symmetric Stretch (Raman strong)'
            amplitude = 0.15 * args.amplitude_scale
        elif abs(freq - 2349.2) < 50:
            mode_type = 'ν₃ Asymmetric Stretch (IR strong)'
            amplitude = 0.10 * args.amplitude_scale
        elif abs(freq - 667.4) < 50:
            mode_type = 'ν₂ Bending (IR + Raman)'
            amplitude = 0.20 * args.amplitude_scale
        else:
            mode_type = 'Combination/Overtone'
            amplitude = 0.15 * args.amplitude_scale
        
        mode_info.append((mode_idx, freq, mode_type, amplitude))
        print(f"  Mode {mode_idx}: {freq:7.1f} cm⁻¹ - {mode_type} (amp={amplitude:.2f} Å)")
    
    # Process each mode
    print(f"\n{'='*70}")
    print(f"PROCESSING MODES")
    print(f"{'='*70}")
    
    for i, (mode_idx, freq, mode_type, amplitude) in enumerate(mode_info):
        print(f"\n[{i+1}/{len(mode_info)}] Mode {mode_idx}: {freq:.1f} cm⁻¹")
        print(f"{'='*70}")
        
        # Run visualization script
        cmd = [
            'python',
            'visualize_charges_during_vibration.py',
            '--checkpoint', str(args.checkpoint),
            '--raman-dir', str(args.raman_dir),
            '--mode-index', str(mode_idx),
            '--n-frames', str(args.n_frames),
            '--amplitude', f'{amplitude:.3f}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success")
        else:
            print(f"❌ Failed")
            print(f"   Error: {result.stderr[-500:]}")  # Last 500 chars of error
    
    print(f"\n{'='*70}")
    print(f"✅ ALL MODES COMPLETE!")
    print(f"{'='*70}")
    print(f"\nVisualization directories created:")
    for mode_idx, freq, mode_type, _ in mode_info:
        output_dir = args.raman_dir / f'charge_visualization_mode{mode_idx}'
        if output_dir.exists():
            print(f"  {output_dir.name}/  ({freq:.1f} cm⁻¹ - {mode_type})")
    
    print(f"\n💡 View all trajectories:")
    print(f"   ase gui {args.raman_dir}/charge_visualization_mode*/mode_*_vibration.traj")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()


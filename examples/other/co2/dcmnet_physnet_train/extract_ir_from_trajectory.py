#!/usr/bin/env python3
"""
Extract IR spectrum from existing trajectory NPZ file.

Usage:
    python extract_ir_from_trajectory.py \
        --trajectory ./md_ir_long/trajectory.npz \
        --output-dir ./md_ir_long
"""

import sys
from pathlib import Path
import numpy as np
import argparse

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

from dynamics_calculator import compute_ir_from_md
from scipy.signal import find_peaks


def main():
    parser = argparse.ArgumentParser(description='Extract IR from trajectory')
    parser.add_argument('--trajectory', type=Path, required=True,
                       help='Trajectory NPZ file')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: same as trajectory)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.trajectory.parent
    
    print("="*70)
    print("EXTRACT IR SPECTRUM FROM TRAJECTORY")
    print("="*70)
    
    # Load trajectory
    print(f"\nLoading trajectory from {args.trajectory}...")
    traj_data = np.load(args.trajectory)
    
    print(f"Keys in trajectory: {list(traj_data.keys())}")
    
    # Check for required data
    if 'dipoles_physnet' not in traj_data:
        print("\n❌ No dipole data in trajectory!")
        print("   The trajectory needs to be regenerated with dipole saving enabled.")
        return
    
    # Extract dipole data
    dipoles_physnet = traj_data['dipoles_physnet']
    dipoles_dcmnet = traj_data.get('dipoles_dcmnet', None)
    times = traj_data.get('times', None)
    
    if times is not None and len(times) > 1:
        timestep = times[1] - times[0]
    else:
        timestep = 0.01 # Default from run_production_md.py
        print(f"⚠️  No time data, assuming timestep = {timestep} fs")
    
    # Get speed of light constant
    c_cm_per_s = 2.99792458e10
    
    print(f"\n✅ Loaded trajectory:")
    print(f"  Frames: {len(dipoles_physnet)}")
    print(f"  Timestep: {timestep} fs (as stated in NPZ)")
    print(f"  Total time: {len(dipoles_physnet) * timestep / 1000:.2f} ps")
    
    # Diagnose dipole variation
    print(f"\n{'='*70}")
    print("DIPOLE VARIATION DIAGNOSTICS")
    print(f"{'='*70}")
    
    dipole_mag_phys = np.linalg.norm(dipoles_physnet, axis=1)
    dipole_mag_dcm = np.linalg.norm(dipoles_dcmnet, axis=1)
    
    print(f"\nPhysNet dipole magnitude:")
    print(f"  Mean: {dipole_mag_phys.mean():.6f} e·Å")
    print(f"  Std:  {dipole_mag_phys.std():.6f} e·Å")
    print(f"  Min:  {dipole_mag_phys.min():.6f} e·Å")
    print(f"  Max:  {dipole_mag_phys.max():.6f} e·Å")
    print(f"  Range: {dipole_mag_phys.max() - dipole_mag_phys.min():.6f} e·Å")
    print(f"  Coefficient of variation: {dipole_mag_phys.std() / dipole_mag_phys.mean():.4f}")
    
    print(f"\nDCMNet dipole magnitude:")
    print(f"  Mean: {dipole_mag_dcm.mean():.6f} e·Å")
    print(f"  Std:  {dipole_mag_dcm.std():.6f} e·Å")
    print(f"  Min:  {dipole_mag_dcm.min():.6f} e·Å")
    print(f"  Max:  {dipole_mag_dcm.max():.6f} e·Å")
    print(f"  Range: {dipole_mag_dcm.max() - dipole_mag_dcm.min():.6f} e·Å")
    print(f"  Coefficient of variation: {dipole_mag_dcm.std() / dipole_mag_dcm.mean():.4f}")
    
    # Check if dipoles are actually different
    dipole_diff = np.abs(dipoles_physnet - dipoles_dcmnet)
    print(f"\nPhysNet vs DCMNet difference:")
    print(f"  Mean |diff|: {np.mean(dipole_diff):.6f} e·Å")
    print(f"  Max |diff|: {np.max(dipole_diff):.6f} e·Å")
    
    # Check power spectrum of dipole (before autocorrelation)
    print(f"\nDipole frequency content (quick check):")
    # Sample first 10000 frames for quick FFT
    n_sample = min(10000, len(dipoles_physnet))
    dip_sample = dipoles_physnet[:n_sample] - dipoles_physnet[:n_sample].mean(axis=0)
    fft_dip = np.fft.rfft(dip_sample[:, 0])  # Just x-component
    freq_sample = np.fft.rfftfreq(n_sample, d=timestep * 1e-15) / c_cm_per_s
    power_sample = np.abs(fft_dip)**2
    
    # Find dominant frequency
    sig_mask = (freq_sample > 100) & (freq_sample < 4000)
    if np.any(sig_mask):
        dom_idx = np.argmax(power_sample[sig_mask])
        dom_freq = freq_sample[sig_mask][dom_idx]
        print(f"  Dominant frequency (first 10k frames): {dom_freq:.1f} cm⁻¹")
    
    # Check if molecule is actually vibrating
    # Look at position variations
    positions_traj = traj_data['trajectory']  # (n_frames, n_atoms, 3)
    
    # CO2 bond length variation
    # Atom 0 = C, Atoms 1,2 = O
    r1_traj = np.linalg.norm(positions_traj[:, 1, :] - positions_traj[:, 0, :], axis=1)
    r2_traj = np.linalg.norm(positions_traj[:, 2, :] - positions_traj[:, 0, :], axis=1)
    
    print(f"\nBond length variations:")
    print(f"  r1: {r1_traj.mean():.4f} ± {r1_traj.std():.4f} Å (range: {r1_traj.max()-r1_traj.min():.4f} Å)")
    print(f"  r2: {r2_traj.mean():.4f} ± {r2_traj.std():.4f} Å (range: {r2_traj.max()-r2_traj.min():.4f} Å)")
    
    if r1_traj.std() < 0.001:
        print(f"\n⚠️  WARNING: Bond lengths barely changing!")
        print(f"  Molecule is not vibrating enough for IR spectrum.")
        print(f"  Try:")
        print(f"    - Higher temperature (500-1000 K)")
        print(f"    - Longer simulation")
        print(f"    - Check if model forces are correct")
    
    # Prepare data for compute_ir_from_md
    md_data = {
        'dipoles_physnet': dipoles_physnet,
        'dipoles_dcmnet': dipoles_dcmnet if dipoles_dcmnet is not None else dipoles_physnet,
        'timestep': timestep,
    }
    
    # Compute IR spectrum
    print(f"\nComputing IR spectrum from autocorrelation...")
    ir_results = compute_ir_from_md(md_data, output_dir=args.output_dir)
    
    # Find and report peaks
    print(f"\n{'='*70}")
    print("PEAK DETECTION")
    print(f"{'='*70}")
    
    from scipy.signal import find_peaks
    
    freqs = ir_results['frequencies']
    int_physnet = ir_results['intensity_physnet']
    int_dcmnet = ir_results['intensity_dcmnet']
    
    # Find peaks in physically relevant range (200-4000 cm⁻¹)
    freq_mask = (freqs > 200) & (freqs < 4000)
    freqs_range = freqs[freq_mask]
    int_phys_range = int_physnet[freq_mask]
    int_dcm_range = int_dcmnet[freq_mask]
    
    # Normalize for peak finding
    int_phys_norm = int_phys_range / np.max(int_phys_range)
    int_dcm_norm = int_dcm_range / np.max(int_dcm_range)
    
    # Find peaks (PhysNet)
    peaks_phys, props_phys = find_peaks(int_phys_norm, height=0.1, distance=20, prominence=0.05)
    print(f"\nPhysNet IR peaks:")
    if len(peaks_phys) > 0:
        for i, (idx, height) in enumerate(zip(peaks_phys, props_phys['peak_heights'])):
            print(f"  Peak {i+1}: {freqs_range[idx]:7.1f} cm⁻¹ (intensity: {height:.3f})")
    else:
        # Find top 5 maxima
        top_indices = np.argsort(int_phys_norm)[-5:][::-1]
        print(f"  No clear peaks found. Top 5 maxima:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. {freqs_range[idx]:7.1f} cm⁻¹ (intensity: {int_phys_norm[idx]:.6f})")
    
    # Find peaks (DCMNet)
    peaks_dcm, props_dcm = find_peaks(int_dcm_norm, height=0.1, distance=20, prominence=0.05)
    print(f"\nDCMNet IR peaks:")
    if len(peaks_dcm) > 0:
        for i, (idx, height) in enumerate(zip(peaks_dcm, props_dcm['peak_heights'])):
            print(f"  Peak {i+1}: {freqs_range[idx]:7.1f} cm⁻¹ (intensity: {height:.3f})")
    else:
        # Find top 5 maxima
        top_indices = np.argsort(int_dcm_norm)[-5:][::-1]
        print(f"  No clear peaks found. Top 5 maxima:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. {freqs_range[idx]:7.1f} cm⁻¹ (intensity: {int_dcm_norm[idx]:.6f})")
    
    # Compare to experimental CO2
    print(f"\n{'='*70}")
    print("COMPARISON TO EXPERIMENTAL CO2")
    print(f"{'='*70}")
    exp_co2 = {
        'ν2 bend (2×)': 667.4,
        'ν1 symmetric stretch': 1388.2,
        'ν3 asymmetric stretch': 2349.2,
    }
    print(f"\nExperimental CO2 frequencies:")
    for name, freq in exp_co2.items():
        print(f"  {name:25s}: {freq:7.1f} cm⁻¹")
    
    # Test different timestep interpretations
    print(f"\n{'='*70}")
    print("TIMESTEP UNIT DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"\nGiven timestep value: {timestep}")
    print(f"\nTesting different unit interpretations:")
    
    test_units = [
        ('fs (as stated)', timestep, 1.0),
        ('ps (if mislabeled)', timestep, 1000.0),
        ('as (attoseconds)', timestep, 0.001),
    ]
    
    for unit_name, ts_value, conversion_to_fs in test_units:
        ts_fs = ts_value * conversion_to_fs
        # Estimate where peak should be
        # For CO2 asymmetric stretch at 2349 cm⁻¹
        # Total time = n_steps * ts_fs (in fs)
        total_time_fs = len(dipoles_physnet) * ts_fs
        total_time_s = total_time_fs * 1e-15
        # Frequency resolution
        df_hz = 1.0 / total_time_s
        df_cm1 = df_hz / c_cm_per_s
        # Expected peak location if CO2 stretch
        print(f"\n  If timestep = {ts_value} {unit_name}:")
        print(f"    → {ts_fs} fs per step")
        print(f"    → Total time: {total_time_fs/1000:.2f} ps")
        print(f"    → Freq resolution: {df_cm1:.2f} cm⁻¹")
        print(f"    → Nyquist freq: {0.5 / (ts_fs * 1e-15) / c_cm_per_s:.0f} cm⁻¹")
    
    print(f"\n{'='*70}")
    print("✅ DONE")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  - {args.output_dir}/md_ir_spectrum.npz")
    print(f"  - {args.output_dir}/ir_spectrum_md.png")


if __name__ == '__main__':
    main()


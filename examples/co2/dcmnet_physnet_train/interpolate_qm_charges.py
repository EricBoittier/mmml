#!/usr/bin/env python3
"""
QM Charge Interpolation Model

Creates an interpolator for QM charges as a function of geometry,
then applies it to MD trajectories to compute "QM-corrected" dipoles and IR.

Workflow:
1. Load merged QM charges (from merge_charge_files.py)
2. Train interpolator: geometry (r1, r2, angle) → charges
3. Load MD trajectory coordinates
4. Predict charges at each MD frame
5. Compute dipoles from charges + coordinates
6. Calculate IR spectrum from dipole autocorrelation
7. Compare ML dipoles vs QM-interpolated dipoles

Usage:
    # Train interpolator and apply to trajectory
    python interpolate_qm_charges.py \
        --qm-charges ./merged_charges.npz \
        --trajectory ./md_ir_long/trajectory.npz \
        --charge-method hirshfeld \
        --output-dir ./qm_corrected_ir
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse
from typing import Dict, Tuple

try:
    from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("❌ scipy required: pip install scipy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def compute_co2_geometry_features(positions: np.ndarray, atomic_numbers: np.ndarray) -> Dict:
    """Extract r1, r2, angle from CO2 geometry."""
    # Find C and O atoms
    c_idx = np.where(atomic_numbers == 6)[0][0]
    o_indices = np.where(atomic_numbers == 8)[0]
    
    c_pos = positions[c_idx]
    o1_pos = positions[o_indices[0]]
    o2_pos = positions[o_indices[1]]
    
    # Distances
    r1 = np.linalg.norm(o1_pos - c_pos)
    r2 = np.linalg.norm(o2_pos - c_pos)
    
    # Angle
    vec1 = o1_pos - c_pos
    vec2 = o2_pos - c_pos
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    
    return {'r1': r1, 'r2': r2, 'angle': angle}


class QMChargeInterpolator:
    """
    Interpolate QM charges as function of molecular geometry.
    
    Uses RBF interpolation for smooth, accurate predictions.
    """
    
    def __init__(self, method='rbf', kernel='thin_plate_spline'):
        """
        Initialize interpolator.
        
        Parameters
        ----------
        method : str
            'rbf' (smooth) or 'linear' (fast but less accurate)
        kernel : str
            RBF kernel: 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric'
        """
        self.method = method
        self.kernel = kernel
        self.interpolators = []  # One per atom
        self.geometry_data = None
        self.n_atoms = None
        
    def fit(self, geometries: np.ndarray, charges: np.ndarray):
        """
        Fit interpolator to QM data.
        
        Parameters
        ----------
        geometries : np.ndarray
            (n_samples, 3) - [r1, r2, angle]
        charges : np.ndarray
            (n_samples, n_atoms)
        """
        print(f"\nTraining charge interpolator...")
        print(f"  Method: {self.method}")
        if self.method == 'rbf':
            print(f"  Kernel: {self.kernel}")
        print(f"  Training points: {len(geometries)}")
        print(f"  Atoms: {charges.shape[1]}")
        
        self.n_atoms = charges.shape[1]
        self.geometry_data = geometries
        
        # Train one interpolator per atom
        for atom_idx in range(self.n_atoms):
            print(f"  Training atom {atom_idx+1}/{self.n_atoms}...", end='\r')
            
            if self.method == 'rbf':
                interp = RBFInterpolator(
                    geometries, 
                    charges[:, atom_idx],
                    kernel=self.kernel,
                    smoothing=0.0,  # No smoothing (exact interpolation)
                )
            elif self.method == 'linear':
                interp = LinearNDInterpolator(geometries, charges[:, atom_idx])
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.interpolators.append(interp)
        
        print(f"\n✅ Trained {self.n_atoms} interpolators")
    
    def predict(self, geometries: np.ndarray) -> np.ndarray:
        """
        Predict charges for new geometries.
        
        Parameters
        ----------
        geometries : np.ndarray
            (n_samples, 3) - [r1, r2, angle]
            
        Returns
        -------
        np.ndarray
            (n_samples, n_atoms) - predicted charges
        """
        n_samples = len(geometries)
        charges = np.zeros((n_samples, self.n_atoms))
        
        for atom_idx, interp in enumerate(self.interpolators):
            charges[:, atom_idx] = interp(geometries)
        
        return charges


def compute_dipole_from_charges(positions: np.ndarray, atomic_numbers: np.ndarray,
                                charges: np.ndarray) -> np.ndarray:
    """
    Compute molecular dipole from atomic charges.
    
    μ = Σ q_i * r_i (relative to COM)
    
    Parameters
    ----------
    positions : np.ndarray
        (n_atoms, 3) in Angstrom
    atomic_numbers : np.ndarray
        (n_atoms,)
    charges : np.ndarray
        (n_atoms,) in electron charge units
        
    Returns
    -------
    np.ndarray
        Dipole (3,) in e·Å
    """
    import ase.data
    
    # Compute COM
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    
    # Dipole relative to COM
    dipole = np.sum(charges[:, None] * (positions - com), axis=0)
    
    return dipole


def main():
    parser = argparse.ArgumentParser(description='QM charge interpolation')
    
    parser.add_argument('--qm-charges', type=Path, required=True,
                       help='Merged QM charges NPZ file')
    parser.add_argument('--trajectory', type=Path, required=True,
                       help='MD trajectory NPZ file')
    parser.add_argument('--charge-method', type=str, default='hirshfeld',
                       help='Charge method to use (hirshfeld, mk, adch, etc.)')
    parser.add_argument('--interpolation-method', type=str, default='rbf',
                       choices=['rbf', 'linear'],
                       help='Interpolation method')
    parser.add_argument('--rbf-kernel', type=str, default='thin_plate_spline',
                       choices=['thin_plate_spline', 'cubic', 'quintic', 'multiquadric'],
                       help='RBF kernel (if using RBF)')
    parser.add_argument('--subsample', type=int, default=None,
                       help='Subsample trajectory every N frames (for speed)')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("QM CHARGE INTERPOLATION → IR SPECTRUM")
    print("="*70)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load QM charges
    print(f"\n1. Loading QM charges from {args.qm_charges}...")
    qm_data = np.load(args.qm_charges, allow_pickle=True)
    
    print(f"   Available charge methods: {[k for k in qm_data.keys() if not k.startswith('_')]}")
    
    if args.charge_method not in qm_data:
        print(f"❌ Charge method '{args.charge_method}' not found!")
        print(f"   Available: {list(qm_data.keys())}")
        return
    
    charges_qm = qm_data[args.charge_method]
    r1_qm = qm_data['r1']
    r2_qm = qm_data['r2']
    angle_qm = qm_data['angle']
    
    print(f"✅ Loaded {args.charge_method} charges")
    print(f"   Samples: {len(charges_qm)}")
    print(f"   Atoms: {charges_qm.shape[1] if hasattr(charges_qm, 'shape') else 'variable'}")
    print(f"   r1 range: [{r1_qm.min():.3f}, {r1_qm.max():.3f}] Å")
    print(f"   r2 range: [{r2_qm.min():.3f}, {r2_qm.max():.3f}] Å")
    print(f"   Angle range: [{angle_qm.min():.1f}, {angle_qm.max():.1f}]°")
    
    # Prepare training data for interpolator
    geometries_qm = np.column_stack([r1_qm, r2_qm, angle_qm])
    
    # Handle object arrays (variable shapes)
    if charges_qm.dtype == object:
        # Find most common shape
        shapes = [c.shape for c in charges_qm]
        from collections import Counter
        most_common_shape = Counter(shapes).most_common(1)[0][0]
        print(f"   Using shape: {most_common_shape}")
        charges_qm = np.array([c for c in charges_qm if c.shape == most_common_shape])
        # Filter geometries too
        valid_indices = [i for i, c in enumerate(qm_data[args.charge_method]) if c.shape == most_common_shape]
        geometries_qm = geometries_qm[valid_indices]
    
    # Train interpolator
    interpolator = QMChargeInterpolator(method=args.interpolation_method, kernel=args.rbf_kernel)
    interpolator.fit(geometries_qm, charges_qm)
    
    # Load MD trajectory
    print(f"\n2. Loading MD trajectory from {args.trajectory}...")
    traj_data = np.load(args.trajectory)
    
    trajectory = traj_data['trajectory']  # (n_frames, n_atoms, 3)
    atomic_numbers = traj_data['atomic_numbers']
    timestep = float(traj_data['timestep'])
    
    n_frames = len(trajectory)
    print(f"✅ Loaded trajectory")
    print(f"   Frames: {n_frames}")
    print(f"   Atoms: {len(atomic_numbers)}")
    print(f"   Timestep: {timestep} fs")
    
    # Subsample if requested
    if args.subsample:
        print(f"\n📉 Subsampling every {args.subsample} frames...")
        trajectory = trajectory[::args.subsample]
        timestep_effective = timestep * args.subsample
        print(f"   Frames after subsampling: {len(trajectory)}")
        print(f"   Effective timestep: {timestep_effective} fs")
    else:
        timestep_effective = timestep
    
    # Extract geometries from trajectory
    print(f"\n3. Extracting geometries from trajectory...")
    geometries_md = np.zeros((len(trajectory), 3))
    
    for i, frame_pos in enumerate(trajectory):
        if i % 10000 == 0:
            print(f"   Processing frame {i}/{len(trajectory)}...", end='\r')
        geom = compute_co2_geometry_features(frame_pos, atomic_numbers)
        geometries_md[i] = [geom['r1'], geom['r2'], geom['angle']]
    
    print(f"\n✅ Extracted geometries")
    print(f"   r1 range: [{geometries_md[:, 0].min():.3f}, {geometries_md[:, 0].max():.3f}] Å")
    print(f"   r2 range: [{geometries_md[:, 1].min():.3f}, {geometries_md[:, 1].max():.3f}] Å")
    print(f"   Angle range: [{geometries_md[:, 2].min():.1f}, {geometries_md[:, 2].max():.1f}]°")
    
    # Check if MD geometries are within QM training range
    print(f"\n4. Checking interpolation validity...")
    r1_ok = (geometries_md[:, 0] >= r1_qm.min()) & (geometries_md[:, 0] <= r1_qm.max())
    r2_ok = (geometries_md[:, 1] >= r2_qm.min()) & (geometries_md[:, 1] <= r2_qm.max())
    ang_ok = (geometries_md[:, 2] >= angle_qm.min()) & (geometries_md[:, 2] <= angle_qm.max())
    valid = r1_ok & r2_ok & ang_ok
    
    print(f"   Frames within QM range: {np.sum(valid)}/{len(valid)} ({100*np.mean(valid):.1f}%)")
    if np.mean(valid) < 0.95:
        print(f"   ⚠️  Warning: {100*(1-np.mean(valid)):.1f}% of MD frames are outside QM training range")
        print(f"   Interpolation may be unreliable in those regions")
    
    # Predict charges
    print(f"\n5. Predicting charges for MD trajectory...")
    charges_pred = interpolator.predict(geometries_md)
    print(f"✅ Predicted charges for {len(charges_pred)} frames")
    print(f"   Charge range: [{charges_pred.min():.4f}, {charges_pred.max():.4f}] e")
    print(f"   Total charge per frame: {np.sum(charges_pred, axis=1).mean():.6f} ± {np.sum(charges_pred, axis=1).std():.6f} e")
    
    # Compute dipoles from charges
    print(f"\n6. Computing dipoles from QM-interpolated charges...")
    dipoles_qm = np.zeros((len(trajectory), 3))
    
    for i, frame_pos in enumerate(trajectory):
        if i % 10000 == 0:
            print(f"   Frame {i}/{len(trajectory)}...", end='\r')
        dipoles_qm[i] = compute_dipole_from_charges(frame_pos, atomic_numbers, charges_pred[i])
    
    print(f"\n✅ Computed QM-interpolated dipoles")
    print(f"   Dipole magnitude: {np.linalg.norm(dipoles_qm, axis=1).mean():.4f} ± {np.linalg.norm(dipoles_qm, axis=1).std():.4f} e·Å")
    
    # Load ML dipoles for comparison
    dipoles_ml_physnet = traj_data.get('dipoles_physnet', None)
    dipoles_ml_dcmnet = traj_data.get('dipoles_dcmnet', None)
    
    # Subsample ML dipoles if trajectory was subsampled
    if args.subsample and dipoles_ml_physnet is not None:
        dipoles_ml_physnet = dipoles_ml_physnet[::args.subsample]
        if dipoles_ml_dcmnet is not None:
            dipoles_ml_dcmnet = dipoles_ml_dcmnet[::args.subsample]
    
    # Compute IR spectra
    print(f"\n7. Computing IR spectra from autocorrelation...")
    
    from dynamics_calculator import compute_ir_from_md
    
    # QM-interpolated IR
    md_data_qm = {
        'dipoles_physnet': dipoles_qm,
        'dipoles_dcmnet': dipoles_qm,  # Use same for both
        'timestep': timestep_effective,
    }
    
    ir_qm = compute_ir_from_md(md_data_qm, output_dir=args.output_dir / 'qm_interpolated')
    
    # ML IR (if available)
    if dipoles_ml_physnet is not None:
        md_data_ml = {
            'dipoles_physnet': dipoles_ml_physnet,
            'dipoles_dcmnet': dipoles_ml_dcmnet if dipoles_ml_dcmnet is not None else dipoles_ml_physnet,
            'timestep': timestep_effective,
        }
        ir_ml = compute_ir_from_md(md_data_ml, output_dir=args.output_dir / 'ml_model')
    else:
        ir_ml = None
    
    # Create comparison plot
    print(f"\n8. Creating comparison plots...")
    plot_qm_ml_comparison(dipoles_qm, dipoles_ml_physnet, ir_qm, ir_ml, args.output_dir)
    
    # Save results
    print(f"\n9. Saving results...")
    np.savez(
        args.output_dir / 'qm_interpolated_dipoles.npz',
        dipoles=dipoles_qm,
        charges=charges_pred,
        geometries=geometries_md,
        timestep=timestep_effective,
        interpolation_method=args.interpolation_method,
        charge_method=args.charge_method,
    )
    
    print(f"\n{'='*70}")
    print("✅ DONE")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  QM-interpolated IR: {args.output_dir}/qm_interpolated/")
    if ir_ml:
        print(f"  ML model IR: {args.output_dir}/ml_model/")
    print(f"  Comparison plots: {args.output_dir}/")


def plot_qm_ml_comparison(dipoles_qm, dipoles_ml, ir_qm, ir_ml, output_dir):
    """Plot comparison of QM vs ML dipoles and IR."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Dipole magnitudes over time
    ax = axes[0, 0]
    dipole_mag_qm = np.linalg.norm(dipoles_qm, axis=1)
    ax.plot(dipole_mag_qm, 'g-', alpha=0.7, linewidth=0.5, label='QM-interpolated')
    if dipoles_ml is not None:
        dipole_mag_ml = np.linalg.norm(dipoles_ml, axis=1)
        ax.plot(dipole_mag_ml, 'b-', alpha=0.7, linewidth=0.5, label='ML (PhysNet)')
    ax.set_xlabel('Frame', fontsize=11, weight='bold')
    ax.set_ylabel('|Dipole| (e·Å)', fontsize=11, weight='bold')
    ax.set_title('Dipole Magnitude Time Series', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dipole correlation
    ax = axes[0, 1]
    if dipoles_ml is not None:
        ax.scatter(dipole_mag_ml, dipole_mag_qm, alpha=0.3, s=1)
        ax.plot([dipole_mag_ml.min(), dipole_mag_ml.max()],
                [dipole_mag_ml.min(), dipole_mag_ml.max()],
                'r--', label='Perfect agreement')
        ax.set_xlabel('ML Dipole (e·Å)', fontsize=11, weight='bold')
        ax.set_ylabel('QM-interpolated Dipole (e·Å)', fontsize=11, weight='bold')
        ax.set_title('Dipole Correlation', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No ML dipoles available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Dipole Correlation', fontsize=13, weight='bold')
    
    # IR spectra comparison
    ax = axes[1, 0]
    freqs_qm = ir_qm['frequencies']
    int_qm = ir_qm['intensity_physnet']
    mask = (freqs_qm > 500) & (freqs_qm < 3500)
    ax.plot(freqs_qm[mask], int_qm[mask], 'g-', linewidth=2, label='QM-interpolated', alpha=0.8)
    
    if ir_ml:
        freqs_ml = ir_ml['frequencies']
        int_ml = ir_ml['intensity_physnet']
        mask_ml = (freqs_ml > 500) & (freqs_ml < 3500)
        ax.plot(freqs_ml[mask_ml], int_ml[mask_ml], 'b-', linewidth=2, label='ML (PhysNet)', alpha=0.8)
    
    ax.set_xlabel('Frequency (cm⁻¹)', fontsize=11, weight='bold')
    ax.set_ylabel('Intensity (normalized)', fontsize=11, weight='bold')
    ax.set_title('IR Spectrum Comparison', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(500, 3500)
    
    # Autocorrelation comparison
    ax = axes[1, 1]
    times_qm = ir_qm['acf_times']
    acf_qm = ir_qm['acf_physnet']
    ax.plot(times_qm, acf_qm / acf_qm[0], 'g-', linewidth=2, label='QM-interpolated')
    
    if ir_ml:
        times_ml = ir_ml['acf_times']
        acf_ml = ir_ml['acf_physnet']
        ax.plot(times_ml, acf_ml / acf_ml[0], 'b-', linewidth=2, label='ML (PhysNet)')
    
    ax.set_xlabel('Time (fs)', fontsize=11, weight='bold')
    ax.set_ylabel('Normalized Autocorrelation', fontsize=11, weight='bold')
    ax.set_title('Dipole Autocorrelation', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(2000, times_qm.max()))
    
    plt.suptitle('QM-Interpolated vs ML Model Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(args.output_dir / 'qm_ml_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison plot: {args.output_dir / 'qm_ml_comparison.png'}")


if __name__ == '__main__':
    main()


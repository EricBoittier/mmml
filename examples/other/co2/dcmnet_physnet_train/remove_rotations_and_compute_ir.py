#!/usr/bin/env python3
"""
Remove Rotations/Translations and Compute IR from Pure Vibrations

Fast iteration script for IR analysis:
1. Load trajectory
2. Remove COM translation
3. Remove rotational motion (align to reference)
4. Compute dipoles from vibrations only
5. Generate IR spectrum

Usage:
    python remove_rotations_and_compute_ir.py \
        --trajectory ./md_ir_long/trajectory.npz \
        --checkpoint ./ckpts/model \
        --output-dir ./vib_only_ir
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from trainer import JointPhysNetDCMNet
from dynamics_calculator import compute_ir_from_md


def remove_com_motion(trajectory):
    """Remove center of mass translation."""
    # trajectory: (n_frames, n_atoms, 3)
    com = np.mean(trajectory, axis=1, keepdims=True)
    return trajectory - com


def align_to_reference(trajectory, masses, reference_frame=0):
    """
    Remove rotations by aligning all frames to a reference frame.
    
    Uses Kabsch algorithm for optimal rotation.
    """
    print(f"\nRemoving rotations (aligning to frame {reference_frame})...")
    
    ref = trajectory[reference_frame]
    aligned = np.zeros_like(trajectory)
    aligned[reference_frame] = ref
    
    # Weight by masses for proper alignment
    weights = masses / np.sum(masses)
    
    for i in range(len(trajectory)):
        if i % 10000 == 0:
            print(f"  Aligning frame {i}/{len(trajectory)}...", end='\r')
        
        if i == reference_frame:
            continue
        
        frame = trajectory[i]
        
        # Weighted covariance matrix
        H = (frame.T * weights) @ ref
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Optimal rotation
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply rotation
        aligned[i] = frame @ R
    
    print(f"\n✅ Removed rotations")
    return aligned


def compute_dipoles_from_trajectory(trajectory, atomic_numbers, model, params, batch_size=1000):
    """
    Recompute dipoles for each frame (FAST batched version).
    
    Processes multiple frames at once for major speedup.
    """
    print(f"\nRecomputing dipoles for vibrations-only (batched)...")
    
    n_frames = len(trajectory)
    n_atoms = len(atomic_numbers)
    
    dipoles_physnet = np.zeros((n_frames, 3))
    dipoles_dcmnet = np.zeros((n_frames, 3))
    
    # Build edge list once (CO2 is small)
    dst_list = []
    src_list = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                dst_list.append(i)
                src_list.append(j)
    
    dst_idx_single = np.array(dst_list, dtype=np.int32)
    src_idx_single = np.array(src_list, dtype=np.int32)
    n_edges = len(dst_idx_single)
    
    import ase.data
    masses_local = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    
    # Process in batches
    n_batches = (n_frames + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_frames)
        actual_batch_size = end_idx - start_idx
        
        print(f"  Batch {batch_idx+1}/{n_batches} ({actual_batch_size} frames)...", end='\r')
        
        # Prepare batched inputs
        positions_batch = trajectory[start_idx:end_idx]  # (batch, n_atoms, 3)
        
        # Flatten for model
        positions_flat = positions_batch.reshape(-1, 3)  # (batch*n_atoms, 3)
        atomic_numbers_batch = np.tile(atomic_numbers, actual_batch_size)  # (batch*n_atoms,)
        
        # Replicate edge lists for batch
        dst_idx_batch = np.concatenate([dst_idx_single + i*n_atoms for i in range(actual_batch_size)])
        src_idx_batch = np.concatenate([src_idx_single + i*n_atoms for i in range(actual_batch_size)])
        
        # Batch info
        batch_segments = np.repeat(np.arange(actual_batch_size), n_atoms)
        batch_mask = np.ones(len(dst_idx_batch), dtype=np.float32)
        atom_mask = np.ones(len(positions_flat), dtype=np.float32)
        
        # Run model on entire batch at once
        output = model.apply(
            params,
            atomic_numbers=jnp.array(atomic_numbers_batch),
            positions=jnp.array(positions_flat),
            dst_idx=jnp.array(dst_idx_batch),
            src_idx=jnp.array(src_idx_batch),
            batch_segments=jnp.array(batch_segments),
            batch_size=actual_batch_size,
            batch_mask=jnp.array(batch_mask),
            atom_mask=jnp.array(atom_mask),
        )
        
        # Extract dipoles - already batched
        dipoles_physnet[start_idx:end_idx] = np.array(output['dipoles'])
        
        # DCMNet dipoles - need to compute per frame
        mono_dist = np.array(output['mono_dist']).reshape(actual_batch_size, n_atoms, -1)
        dipo_dist = np.array(output['dipo_dist']).reshape(actual_batch_size, n_atoms, -1, 3)
        
        for i in range(actual_batch_size):
            # COM for this frame
            com = np.sum(positions_batch[i] * masses_local[:, None], axis=0) / masses_local.sum()
            dipo_rel_com = dipo_dist[i] - com[None, None, :]
            dipoles_dcmnet[start_idx + i] = np.sum(mono_dist[i, ..., None] * dipo_rel_com, axis=(0, 1))
    
    print(f"\n✅ Recomputed {n_frames} dipoles in {n_batches} batches")
    
    return dipoles_physnet, dipoles_dcmnet


def main():
    parser = argparse.ArgumentParser(description='Remove rotations and compute vibrational IR')
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Model checkpoint (to recompute dipoles)')
    parser.add_argument('--subsample', type=int, default=10,
                       help='Subsample trajectory for speed (default: 10)')
    parser.add_argument('--output-dir', type=Path, required=True)
    
    args = parser.parse_args()
    
    print("="*70)
    print("VIBRATIONAL IR (ROTATIONS REMOVED)")
    print("="*70)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory
    print(f"\n1. Loading trajectory...")
    traj_data = np.load(args.trajectory)
    
    trajectory = traj_data['trajectory']  # (n_frames, n_atoms, 3)
    atomic_numbers = traj_data['atomic_numbers']
    timestep = float(traj_data['timestep'])
    
    print(f"✅ Loaded: {len(trajectory)} frames, {len(atomic_numbers)} atoms")
    
    # Subsample for speed
    if args.subsample > 1:
        print(f"\n2. Subsampling by {args.subsample}...")
        trajectory = trajectory[::args.subsample]
        timestep_eff = timestep * args.subsample
        print(f"✅ {len(trajectory)} frames, timestep = {timestep_eff} fs")
    else:
        timestep_eff = timestep
    
    # Compute masses
    import ase.data
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    
    # Remove COM motion
    print(f"\n3. Removing COM translation...")
    trajectory_com_removed = remove_com_motion(trajectory)
    
    # Remove rotations
    print(f"\n4. Removing rotations...")
    trajectory_aligned = align_to_reference(trajectory_com_removed, masses, reference_frame=len(trajectory)//2)
    
    # Check residual motion
    com_check = np.mean(trajectory_aligned, axis=1)
    print(f"   Residual COM motion: {np.max(np.abs(com_check)):.6f} Å (should be ~0)")
    
    # Load model
    print(f"\n5. Loading model...")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    config['physnet_config']['natoms'] = len(atomic_numbers)
    
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    # Recompute dipoles
    print(f"\n6. Recomputing dipoles from aligned trajectory...")
    dipoles_physnet, dipoles_dcmnet = compute_dipoles_from_trajectory(
        trajectory_aligned, atomic_numbers, model, params
    )
    
    # Compute IR
    print(f"\n7. Computing IR from vibrational dipoles...")
    md_data = {
        'dipoles_physnet': dipoles_physnet,
        'dipoles_dcmnet': dipoles_dcmnet,
        'timestep': timestep_eff,
    }
    
    ir_results = compute_ir_from_md(md_data, output_dir=args.output_dir)
    
    # Peak detection
    print(f"\n8. Finding peaks...")
    freqs = ir_results['frequencies']
    int_physnet = ir_results['intensity_physnet']
    int_dcmnet = ir_results['intensity_dcmnet']
    
    freq_mask = (freqs > 300) & (freqs < 3500)
    freqs_range = freqs[freq_mask]
    int_phys_range = int_physnet[freq_mask]
    int_dcm_range = int_dcmnet[freq_mask]
    
    int_phys_norm = int_phys_range / np.max(int_phys_range)
    int_dcm_norm = int_dcm_range / np.max(int_dcm_range)
    
    peaks_phys, _ = find_peaks(int_phys_norm, height=0.1, prominence=0.05)
    peaks_dcm, _ = find_peaks(int_dcm_norm, height=0.1, prominence=0.05)
    
    print(f"\nPhysNet peaks (vibrations only):")
    if len(peaks_phys) > 0:
        for i, idx in enumerate(peaks_phys):
            print(f"  {freqs_range[idx]:7.1f} cm⁻¹")
    else:
        top5 = np.argsort(int_phys_norm)[-5:][::-1]
        print(f"  Top 5: {[f'{freqs_range[i]:.1f}' for i in top5]}")
    
    print(f"\nDCMNet peaks (vibrations only):")
    if len(peaks_dcm) > 0:
        for i, idx in enumerate(peaks_dcm):
            print(f"  {freqs_range[idx]:7.1f} cm⁻¹")
    else:
        top5 = np.argsort(int_dcm_norm)[-5:][::-1]
        print(f"  Top 5: {[f'{freqs_range[i]:.1f}' for i in top5]}")
    
    print(f"\n{'='*70}")
    print("✅ DONE - VIBRATIONS ONLY IR")
    print(f"{'='*70}")
    print(f"\nNote: Removed rotations and translations")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()

